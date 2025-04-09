from typing import List, Optional
import os
import torch
import concurrent.futures
from src.survival_runner import SurvivalResult, SurvivalRunner


def run_survival_analysis(prompts: List[str], prompt_attempts: Optional[List[int]] = None) -> List[SurvivalResult]:
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
    from src import utils

    # Hardcoded parameters
    SEED = 42
    HF_KEY = utils.api_key_from_file("HF_KEY.txt")
    MAX_INPUT_TOKENS = 40 # length of input prompt in tokens
    MAX_OUTPUT_TOKENS = 30 # length of generated output in tokens
    BATCH_SIZE = 1500 # number of samples to process in each batch
    MAX_ATTEMPTS = 40 # global threshold of attempts for all samples, never exceed this value

    utils.set_seed(SEED)
    utils.clear_memory()

    # Initialize rater and generator backends
    rater_backend = DetoxifyRater(model_type="original", amp=True)
    generator_backend = VanillaGeneratorVLLM(
        model_name="meta-llama/Llama-3.2-1B",
        hf_token=HF_KEY,
        max_input_tokens=MAX_INPUT_TOKENS,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        max_batch_size=BATCH_SIZE,
        gpu_memory_utilization=0.5,
    )

    # Set up survival runner
    survival_runner = SurvivalRunner(
        generator=generator_backend,
        rater=rater_backend,
        max_attempts=MAX_ATTEMPTS,
        toxicity_func=(lambda gen: False),
        text_prep_func=lambda gen: gen.prompt + gen.output,
        conserve_memory=False,
    )

    # Generate results
    survival_results = survival_runner.generate(
        prompts=prompts,
        max_attempts=prompt_attempts,
        batch_size=BATCH_SIZE,
    )

    return list(survival_results)


def worker(gpu_id: int, prompts_chunk: List[str], attempts_chunk: Optional[List[int]]) -> List[SurvivalResult]:
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
    return run_survival_analysis(prompts_chunk, attempts_chunk)


def generate_survival_results(
    prompts: List[str], prompt_attempts: Optional[List[int]] = None, multi_gpu: bool = False
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
        return run_survival_analysis(prompts, prompt_attempts)

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
            executor.submit(worker, gpu_id, prompts_chunk, attempts_chunk)
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


if __name__ == "__main__":
    import pandas as pd
    import pickle

    # Load example data
    df = pd.read_json("hf://datasets/allenai/real-toxicity-prompts/prompts.jsonl", lines=True)
    input_dicts = df["prompt"].values.flatten().tolist()
    inputs = [d["text"] for d in input_dicts][: len(input_dicts) // 20]

    prompt_attempts = None
    results = generate_survival_results(inputs, prompt_attempts=prompt_attempts, multi_gpu=False)

    # Now save the results to a file
    # with open("survival_results.pkl", "wb") as f:
        # pickle.dump(results, f)
