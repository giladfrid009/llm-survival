# NOTE: works

"""Prepare the evaluation test set from a base dataset and optional fragments.

``make_mini_sample.py`` can generate many small datasets ("mini-sets") for the
same set of test prompts.  This script merges the original test split with any
number of these mini-sets and outputs several convenient representations used by
the evaluation scripts.

Supported output types
----------------------

``full``
    Retain the prompt text and all rating details.
``full_light``
    Keep the rating scores but drop the generated text to save space.
``prompt_only``
    Only store the prompts and whether or not they eventually produced a toxic
    output.
``surv_only``
    Save only the survival times (number of generations) for each prompt.

Example
-------

```
python prepare_test_set.py --base_dataset data/rtp_500/split_1_0.5_0.1_0.2_0.2/test.pkl \
    --fragments_dir mini_datasets --dataset_types prompt_only,surv_only
```

Run this script once with ``--dataset_types prompt_only`` to extract the base
test prompts used by ``make_mini_sample.py``.  After generating any desired
mini-sets, run it again with ``--fragments_dir`` to combine all fragments and
produce the files consumed by the experiment scripts.
"""

from typing import List
import argparse
import pickle
import numpy as np
import glob
import os

from src import utils
from src.survival_runner import SurvivalResult
from src.rating.base import RatingResult
from tqdm.auto import tqdm
from my_scripts import config


class DatasetType:
    FULL = "full"
    FULL_LIGHT = "full_light"
    PROMPT_ONLY = "prompt_only"
    SURV_ONLY = "surv_only"


def parse_args() -> argparse.Namespace:
    """Command line options for preparing the evaluation dataset."""
    parser = argparse.ArgumentParser(description="Prepare test set for evaluation")
    parser.add_argument(
        "--base_dataset",
        default=config.default_test_split_path,
        type=str,
        help="Pickle with base test data (use 'none' to skip)",
    )
    parser.add_argument(
        "--fragments_dir",
        default="mini_datasets",
        type=str,
        help="Directory containing additional mini-set pickles (use 'none' to skip)",
    )
    parser.add_argument(
        "--pattern",
        default="mini_set_*.pkl",
        help="Glob pattern for mini-set pickles inside fragments_dir",
    )
    parser.add_argument(
        "--dataset_types",
        default="prompt_only,surv_only",
        help="Comma-separated list of dataset types to save",
    )
    parser.add_argument("--output_full", default="data/test_full.pkl", help="Path for full dataset pickle")
    parser.add_argument("--output_full_light", default="data/test_full_light.pkl", help="Path for light-weight dataset pickle")
    parser.add_argument("--output_prompt_only", default=config.default_test_prompts_path, help="Path for prompt-only pickle")
    parser.add_argument("--output_surv_times", default=config.default_test_surv_time_path, help="Path for numpy survival-time array")
    
    parsed = parser.parse_args()
    
    # make all paths absolute
    parsed.base_dataset = utils.abs_path(parsed.base_dataset, ignore="none")
    parsed.fragments_dir = utils.abs_path(parsed.fragments_dir, ignore="none")
    parsed.output_full = utils.abs_path(parsed.output_full)
    parsed.output_full_light = utils.abs_path(parsed.output_full_light)
    parsed.output_prompt_only = utils.abs_path(parsed.output_prompt_only)
    parsed.output_surv_times = utils.abs_path(parsed.output_surv_times)
    
    # print all args
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    return parsed


def load_fragment(path: str) -> List[SurvivalResult]:
    """Return a single fragment loaded from ``path``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Fragment '{path}' not found")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_fragments(paths: List[str]) -> List[List[SurvivalResult]]:
    """Load pickled survival fragments from disk."""
    frags = []
    for frag in tqdm(paths, desc="Loading fragments"):
        data = load_fragment(frag)
        if frags and len(data) != len(frags[0]):
            raise ValueError(f"Fragment '{frag}' length {len(data)} does not match others ({len(frags[0])})")
        frags.append(data)
    return frags


def merge_fragments(fragments: List[List[SurvivalResult]]) -> List[SurvivalResult]:
    """Combine per-run survival data for each prompt into a single object."""
    combined: List[SurvivalResult] = []
    for i in tqdm(range(len(fragments[0])), desc="Joining fragments"):
        prompt = fragments[0][i].prompt
        ratings = []
        max_attempts = 0
        is_toxic = False
        for fr in fragments:
            assert fr[i].prompt == prompt
            ratings.extend(fr[i].ratings)
            max_attempts += fr[i].max_attempts
            is_toxic = is_toxic or fr[i].is_toxic
        combined.append(
            SurvivalResult(
                id=i,
                prompt=prompt,
                max_attempts=max_attempts,
                num_attempts=max_attempts,
                is_toxic=is_toxic,
                ratings=ratings,
            )
        )
    return combined


def save_dataset(data: List[SurvivalResult], dtype: str, args: argparse.Namespace) -> None:
    """Save ``data`` according to the requested representation ``dtype``."""
    if dtype == DatasetType.FULL:
        os.makedirs(os.path.dirname(args.output_full), exist_ok=True)
        with open(args.output_full, "wb") as f:
            pickle.dump(data, f)
    
    elif dtype == DatasetType.FULL_LIGHT:
        light = []
        for surv in data:
            new_ratings = [RatingResult(text="", scores=r.scores) for r in surv.ratings]
            light.append(
                SurvivalResult(
                    id=surv.id,
                    prompt=surv.prompt,
                    max_attempts=surv.max_attempts,
                    num_attempts=surv.num_attempts,
                    is_toxic=surv.is_toxic,
                    ratings=new_ratings,
                )
            )
        os.makedirs(os.path.dirname(args.output_full_light), exist_ok=True)
        with open(args.output_full_light, "wb") as f:
            pickle.dump(light, f)
            
    elif dtype == DatasetType.PROMPT_ONLY:
        prompt_only = [
            SurvivalResult(
                id=s.id,
                prompt=s.prompt,
                max_attempts=s.max_attempts,
                num_attempts=s.num_attempts,
                is_toxic=s.is_toxic,
            )
            for s in data
        ]
        os.makedirs(os.path.dirname(args.output_prompt_only), exist_ok=True)
        with open(args.output_prompt_only, "wb") as f:
            pickle.dump(prompt_only, f)
            
    elif dtype == DatasetType.SURV_ONLY:
        #TODO: surv time should be the index of th first toxic rating or the total number of ratings
        # here we imemdiatly set to the number of ratings, which assumes that we stopped generating once encountering a toxic rating
        # note that doesnt have to hold. Therefore, we need to find a robust way to get the survival time
        surv_times = np.array([len(s.ratings) for s in data])
        os.makedirs(os.path.dirname(args.output_surv_times), exist_ok=True)
        np.save(args.output_surv_times, surv_times)


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    base_dataset = None if args.base_dataset and args.base_dataset.lower() == "none" else args.base_dataset
    fragments_dir = None if args.fragments_dir and args.fragments_dir.lower() == "none" else args.fragments_dir

    fragment_paths: List[str] = []
    if base_dataset:
        fragment_paths.append(base_dataset)

    if fragments_dir and os.path.isdir(fragments_dir):
        glob_pattern = os.path.join(fragments_dir, args.pattern)
        fragment_paths.extend(sorted(glob.glob(glob_pattern)))
        
    print("Processing datasets:")
    for path in fragment_paths:
        print(f"  {path}")

    if not fragment_paths:
        raise ValueError("No fragments found. Provide --base_dataset or --fragments_dir")

    fragments = load_fragments(fragment_paths)
    combined = merge_fragments(fragments) if len(fragments) > 1 else fragments[0]

    dtypes = [d.strip() for d in args.dataset_types.split(",") if d.strip()]
    valid_types = {DatasetType.FULL, DatasetType.FULL_LIGHT, DatasetType.PROMPT_ONLY, DatasetType.SURV_ONLY}
    invalid = set(dtypes) - valid_types
    if invalid or not dtypes:
        raise ValueError(f"Invalid dataset types: {', '.join(invalid)}")

    for dtype in dtypes:
        save_dataset(combined, dtype, args)


if __name__ == "__main__":
    main()
