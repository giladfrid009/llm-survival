# NOTE: WORKS

"""Generate the large multisample dataset used for training and evaluation.

The script loads prompts (by default from the Real Toxicity Prompts dataset),
generates continuations with an LLM, rates them for toxicity, and stores the
resulting :class:`~src.survival_runner.SurvivalResult` objects.  The output is
a pickle file that can later be split into train/val/cal/test sets using
``data/make_split_ms.py``.
"""

import argparse
import pickle
import torch
import logging

from src import utils
from src.survival_runner import SurvivalResult, generate_survival_results_generic
from scripts import config


def parse_args() -> argparse.Namespace:
    """CLI arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate a large survival dataset")
    parser.add_argument("--prompts_path", default="hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl")
    parser.add_argument("--output", default="multisample_results.pkl")
    parser.add_argument("--batch_size", type=int, default=config.default_batch_size, help="Batch size for generation")
    parser.add_argument("--num_attempts", type=int, default=5000, help="Number of output generations per prompt")
    parser.add_argument("--model_name", default=config.default_model_name, help="Model name for generation")
    parser.add_argument("--hf_key_path", default=config.hf_key_path, help="Path to HuggingFace API key")
    parser.add_argument("--max_input_tokens", type=int, default=config.default_max_input_tokens, help="Maximum prompt tokens")
    parser.add_argument("--max_output_tokens", type=int, default=config.default_max_output_tokens, help="Maximum tokens to generate")
    
    parsed = parser.parse_args()

    # make all paths absolute
    parsed.prompts_path = utils.abs_path(parsed.prompts_path)
    parsed.output = utils.abs_path(parsed.output)
    parsed.hf_key_path = utils.abs_path(parsed.hf_key_path)

    # print all args
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    return parsed


def main() -> None:
    """Entry point: generate and save the multisample dataset."""
    args = parse_args()

    utils.set_seed(420)
    utils.clear_memory()

    # TODO: currently works for real toxicity prompts, but probably wont work for other datasets
    # we should specify the expected structure of the loaded file
    prompts = utils.load_jsonl_prompts(args.prompts_path)

    results = generate_survival_results_generic(
        prompts=prompts,
        prompt_ids=list(range(len(prompts))),
        prompt_attempts=[args.num_attempts] * len(prompts),
        generate_params={"batch_size": args.batch_size},
        generator_params=config.generator_params(
            model_name=args.model_name,
            hf_key=config.get_hf_key(args.hf_key_path),
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
        ),
        rater_params=config.rater_params(),
        max_attempts=args.num_attempts,
        toxicity_func="no_toxicity",
        text_prep_func="sentence_completion",
        conserve_memory_ratings=True,
        conserve_memory=False,
        multi_gpu=torch.cuda.device_count() > 1,
    )

    survival_list: list[SurvivalResult] = [res for res in results]
    with open(args.output, "wb") as f:
        pickle.dump(survival_list, f)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    utils.configure_logging(logging.WARNING)
    main()
