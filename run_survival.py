"""Convenience CLI for running survival sampling on a set of prompts."""

from typing import List, Optional
import argparse
import os
import torch
import concurrent.futures
from src.survival_runner import SurvivalResult, SurvivalRunner
from src import utils
import config


def run_survival_analysis(
    prompts: List[str],
    prompt_attempts: Optional[List[int]] = None,
    *,
    seed: int = 42,
    model_name: str = config.default_model_name,
    hf_key: str | None = None,
    max_input_tokens: int = config.default_max_input_tokens,
    max_output_tokens: int = config.default_max_output_tokens,
    batch_size: int = config.default_batch_size,
    max_attempts: int = config.default_max_attempts,
) -> List[SurvivalResult]:
    """
    Run survival analysis on the given prompts using hardcoded backend configurations.

    Args:
        prompts: List of input prompts to process.
        prompt_attempts: Optional list of integers specifying the number of attempts for each prompt.

    Returns:
        List of SurvivalResult objects containing the analysis results.
    """
    from src.rating.detoxify import DetoxifyRater
    from src.generation.vanilla_model_vllm import VanillaGeneratorVLLM

    if hf_key is None:
        hf_key = config.get_hf_key()

    utils.set_seed(seed)
    utils.clear_memory()

    # Initialize rater and generator backends
    rater_backend = DetoxifyRater(**config.rater_params())
    generator_backend = VanillaGeneratorVLLM(
        **config.generator_params(
            model_name=model_name,
            hf_key=hf_key,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        ),
        max_batch_size=batch_size,
        gpu_memory_utilization=0.5,
    )

    # Set up survival runner
    survival_runner = SurvivalRunner(
        generator=generator_backend,
        rater=rater_backend,
        max_attempts=max_attempts,
        toxicity_func=(lambda gen: False),
        text_prep_func=lambda gen: gen.prompt + gen.output,
        conserve_memory=False,
    )

    # Generate results
    survival_results = survival_runner.generate(
        prompts=prompts,
        max_attempts=prompt_attempts,
        batch_size=batch_size,
    )

    return list(survival_results)


def worker(
    gpu_id: int,
    prompts_chunk: List[str],
    attempts_chunk: Optional[List[int]],
    params: dict,
) -> List[SurvivalResult]:
    """
    Process a chunk of prompts on a specific GPU.

    Args:
        gpu_id: The ID of the GPU to use.
        prompts_chunk: List of prompts to process.
        attempts_chunk: Optional list of attempts corresponding to the prompts.

    Returns:
        List of SurvivalResult objects.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return run_survival_analysis(prompts_chunk, attempts_chunk, **params)


def generate_survival_results(
    prompts: List[str],
    prompt_attempts: Optional[List[int]] = None,
    *,
    multi_gpu: bool = False,
    **params,
) -> List[SurvivalResult]:
    """
    Execute survival analysis on a list of prompts, with optional multi-GPU support.

    Args:
        prompts: List of input prompts to process.
        prompt_attempts: Optional list of integers specifying the number of attempts for each prompt.
                         Must match the length of prompts if provided.
        multi_gpu: If True, distribute prompts across available GPUs; if False, run on current process.

    Returns:
        List of SurvivalResult objects. Note: Order may not match input prompts when multi_gpu=True.

    Raises:
        ValueError: If prompt_attempts length does not match prompts length.
        RuntimeError: If multi_gpu is True but no CUDA devices are available.
        Exception: Propagates exceptions from worker processes with error details.
    """
    # Validate prompt_attempts length
    if prompt_attempts is not None and len(prompt_attempts) != len(prompts):
        raise ValueError("prompt_attempts must have the same length as prompts")

    # Single-GPU or CPU execution
    if not multi_gpu:
        return run_survival_analysis(prompts, prompt_attempts, **params)

    # Multi-GPU execution
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No CUDA devices available for multi-GPU processing.")

    # Split prompts and attempts into chunks for each GPU
    prompts_chunks = [prompts[i::n_gpus] for i in range(n_gpus)]
    attempts_chunks = [prompt_attempts[i::n_gpus] for i in range(n_gpus)] if prompt_attempts else [None] * n_gpus

    # Process chunks in parallel across GPUs
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = [
            executor.submit(worker, gpu_id, prompts_chunk, attempts_chunk, params)
            for gpu_id, prompts_chunk, attempts_chunk in zip(range(n_gpus), prompts_chunks, attempts_chunks)
        ]

        # Collect results and handle exceptions
        results = []
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error in worker process: {e}")
                raise

    # Combine results from all GPUs
    final_results = []
    for chunk_result in results:
        final_results.extend(chunk_result)

    return final_results


def parse_args() -> argparse.Namespace:
    """CLI options for :func:`generate_survival_results`."""
    parser = argparse.ArgumentParser(description="Run survival analysis on a list of prompts")
    parser.add_argument("prompts_path", help="Path to a JSONL file containing prompts under 'prompt.text'")
    parser.add_argument("--output", default="survival_results.pkl", help="Where to store the resulting pickle")
    parser.add_argument("--num_prompts", type=int, default=None, help="Optional limit on number of prompts to load")
    parser.add_argument("--multi_gpu", action="store_true", help="Use all available GPUs")
    parser.add_argument("--model_name", default=config.default_model_name, help="Model name to use for generation")
    parser.add_argument("--hf_key_path", default=config.hf_key_path, help="Path to HuggingFace API key")
    parser.add_argument("--max_input_tokens", type=int, default=config.default_max_input_tokens, help="Maximum number of prompt tokens")
    parser.add_argument("--max_output_tokens", type=int, default=config.default_max_output_tokens, help="Maximum number of tokens to generate")
    parser.add_argument("--batch_size", type=int, default=config.default_batch_size, help="Batch size for generation")
    parser.add_argument("--max_attempts", type=int, default=config.default_max_attempts, help="Maximum generations per prompt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the CLI and save the resulting pickle."""
    args = parse_args()

    prompts = utils.load_jsonl_prompts(args.prompts_path, limit=args.num_prompts)

    results = generate_survival_results(
        prompts,
        multi_gpu=args.multi_gpu,
        seed=args.seed,
        model_name=args.model_name,
        hf_key=config.get_hf_key(args.hf_key_path),
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
    )

    import pickle
    with open(args.output, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
