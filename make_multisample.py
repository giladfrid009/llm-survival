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

from src import utils
from src.survival_runner import SurvivalResult, generate_survival_results_generic
import config


def parse_args() -> argparse.Namespace:
    """CLI arguments for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate a large survival dataset")
    parser.add_argument("--prompts_path", default="hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl")
    parser.add_argument("--output", default="multisample_results.pkl")
    parser.add_argument("--batch_size", type=int, default=config.default_batch_size, help="Batch size for generation")
    parser.add_argument("--max_attempts", type=int, default=5000, help="Maximum attempts per prompt")
    parser.add_argument("--model_name", default=config.default_model_name, help="Model name for generation")
    parser.add_argument("--hf_key_path", default=config.hf_key_path, help="Path to HuggingFace API key")
    parser.add_argument("--max_input_tokens", type=int, default=config.default_max_input_tokens, help="Maximum prompt tokens")
    parser.add_argument("--max_output_tokens", type=int, default=config.default_max_output_tokens, help="Maximum tokens to generate")
    
    # print all args
    parsed = parser.parse_args()
    print("Command line arguments:")
    for arg, value in vars(parsed).items():
        print(f"  {arg}: {value}")
    return parsed


def main() -> None:
    """Entry point: generate and save the multisample dataset."""
    args = parse_args()

    utils.set_seed(42)
    utils.clear_memory()

    prompts = utils.load_jsonl_prompts(args.prompts_path)

    rater_params = config.rater_params()
    generator_params = config.generator_params(
        model_name=args.model_name,
        hf_key=config.get_hf_key(args.hf_key_path),
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
    )

    results = generate_survival_results_generic(
        prompts=prompts,
        prompt_attempts=None,
        generate_params={"batch_size": args.batch_size},
        generator_params=generator_params,
        rater_params=rater_params,
        max_attempts=args.max_attempts,
        toxicity_func="no_toxicity",
        text_prep_func="sentence_completion",
        conserve_memory_ratings=True,
        multi_gpu=True,
    )

    survival_list: list[SurvivalResult] = [res for res in results]
    with open(args.output, "wb") as f:
        pickle.dump(survival_list, f)


if __name__ == "__main__":
    main()
