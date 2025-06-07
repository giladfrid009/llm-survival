# NOTE: WORKS.

"""Generate additional survival data for the evaluation test set.

This script repeatedly runs the survival generation pipeline over the test
prompts and stores each run as a separate *mini-set*.  Prompts can be supplied
directly via ``--prompts_path`` or extracted from ``--base_dataset``.  The
fragments can later be merged with :mod:`prepare_test_set` to obtain a large
evaluation set with multiple independent continuations per prompt.
"""

from src import utils
from src.rating.base import *
from src.generation.base import *
import torch
from huggingface_hub.hf_api import HfFolder
import numpy as np
import pickle
from src.survival_runner import SurvivalResult, generate_survival_results_generic
from src.datasets import PromptOnlyDataset
import os
import logging
import torch._dynamo
import argparse
import config

torch._dynamo.config.suppress_errors = True


###########
# CONFIG
###########


DEFAULT_FOLDER = "mini_datasets"
DEFAULT_FILE_NAME = "mini_set"
DEFAULT_START_IDX = 0


def parse_args() -> argparse.Namespace:
    """Command line arguments for mini dataset generation."""
    parser = argparse.ArgumentParser(description="Generate survival mini datasets")
    parser.add_argument(
        "--prompts_path", # NOTE: works!
        default=None,
        help=("Pickle file containing prompts to extend. If omitted, prompts are read from --base_dataset."),
    )
    parser.add_argument(
        "--base_dataset", # NOTE: now works!
        default=config.default_test_split_path,
        help=(
            "Pickle with base test dataset to read prompts from when --prompts_path"
            " is not provided. Use 'none' to disable loading a base dataset."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=config.default_batch_size, help="Batch size for generation")
    parser.add_argument("--max_attempts", type=int, default=100, help="Number of generations attempted per prompt")
    parser.add_argument("--total_datasets", type=int, default=100, help="Number of mini-sets to create")
    parser.add_argument("--mini_sample_folder", default=DEFAULT_FOLDER, help="Directory to store the generated fragments")
    parser.add_argument("--mini_sample_file_name", default=DEFAULT_FILE_NAME, help="Base name for each fragment (index and .pkl added)")
    parser.add_argument(
        "--start_idx",
        type=int,
        default=DEFAULT_START_IDX,
        help="Index to start numbering saved fragments from",
    )
    parser.add_argument("--model_name", default=config.default_model_name, help="Model name for generation")
    parser.add_argument("--hf_key_path", default=config.hf_key_path, help="Path to HuggingFace API key")
    parser.add_argument("--max_input_tokens", type=int, default=config.default_max_input_tokens)
    parser.add_argument("--max_output_tokens", type=int, default=config.default_max_output_tokens)
    
    # print all args
    parsed = parser.parse_args()
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    return parsed


###########
# FUNCTIONS
###########


def configure_logging(level=logging.ERROR):
    """Silence noisy libraries and set a global log level."""
    logging.captureWarnings(True)
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)

    # set level for all loggers
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)
            for h in logger.handlers:
                h.setLevel(level)

    os.environ["LOGLEVEL"] = logging.getLevelName(level)
    os.environ["VLLM_LOGGING_LEVEL"] = logging.getLevelName(level)
    logging.getLogger("lightning.pytorch").setLevel(level)
    torch._logging.set_logs(all=level)


def validate_save_path(save_path: str, start_idx: int) -> int:
    """Ensure the output directory exists and pick the next free index."""

    # make save_path absolute if it is not
    if not os.path.isabs(save_path):
        save_path = os.path.abspath(save_path)
    # Ensure directory exists
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        print(f"Results directory {directory} does not exist. Creating it.")
        os.makedirs(directory)
        print(f"Directory {directory} created.")

    i = start_idx
    while os.path.exists(save_path.format(i)):
        print(f"Results file {save_path.format(i)} already exists. Incrementing index.")
        i += 1
    return i


def create_datasets(args: argparse.Namespace):
    """Generate ``total_datasets`` fragments of the full prompt set."""

    # Load the prompts either directly or from the base dataset
    if args.prompts_path:
        prompts = PromptOnlyDataset(args.prompts_path)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_path}")
    else:
        if args.base_dataset is None:
            raise ValueError("Either --prompts_path or --base_dataset must be supplied")
        with open(args.base_dataset, "rb") as f:
            base_data = pickle.load(f)
        prompts = [res if isinstance(res, str) else res[0] for res in base_data]
        print(f"Loaded {len(prompts)} prompts from {args.base_dataset}")

    # validate the save path
    output_template = os.path.join(args.mini_sample_folder, f"{args.mini_sample_file_name}_{{}}.pkl")
    start = validate_save_path(output_template, args.start_idx)

    # run
    for i in range(start, args.total_datasets):

        print("-" * 40)
        print(f"Running iter ({i + 1} / {args.total_datasets} )")
        print("-" * 40)

        utils.clear_memory()

        survival_results = generate_survival_results_generic(
            prompts=prompts,
            prompt_ids=list(range(len(prompts))),
            prompt_attempts=[args.max_attempts] * len(prompts),
            generate_params={"batch_size": args.batch_size},
            generator_params=config.generator_params(
                model_name=args.model_name,
                hf_key=config.get_hf_key(args.hf_key_path),
                max_input_tokens=args.max_input_tokens,
                max_output_tokens=args.max_output_tokens,
            ),
            rater_params=config.rater_params(),
            max_attempts=args.max_attempts,
            toxicity_func="no_toxicity",
            text_prep_func="sentence_completion",
            conserve_memory_ratings=False,
            conserve_memory=False,
            multi_gpu=torch.cuda.device_count() > 1,
        )

        # Sort the results by the original order of the prompts
        ids = np.array([r.id for r in survival_results])
        sorted_indices = np.argsort(ids).flatten()
        survival_results = [survival_results[i] for i in sorted_indices]

        # Save final results to disk
        with open(output_template.format(i), "wb") as f:
            pickle.dump(survival_results, f)


def main() -> None:
    """Entry point for script execution."""
    args = parse_args()

    if args.base_dataset and args.base_dataset.lower() == "none":
        args.base_dataset = None

    utils.clear_memory()

    HfFolder.save_token(config.get_hf_key(args.hf_key_path))

    torch.multiprocessing.set_start_method("spawn", force=True)

    configure_logging(logging.ERROR)

    create_datasets(args)


if __name__ == "__main__":
    main()
