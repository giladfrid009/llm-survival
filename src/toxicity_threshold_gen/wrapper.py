from typing import Any, Iterable, Iterator, List
from collections import OrderedDict

from tqdm import tqdm
from src.generation import GenerationResult, GenerationBackend
from src.rating import RatingBackend, RatingResult
from src.utils import batchify

class ToxicityThresholdGenerator(GenerationBackend):
    """
    A generation backend that repeatedly generates outputs for a given prompt until
    one of the outputs has a toxicity score above a specified threshold.
    """

    def __init__(
        self,
        generation_backend: GenerationBackend,
        rating_backend: RatingBackend,
        toxicity_threshold: float = 0.5,
        max_iterations: int = 10,
    ):
        """
        Args:
            generation_backend (GenerationBackend): The backend used to generate text.
            rating_backend (RatingBackend): The backend used to rate the generated text.
            toxicity_threshold (float, optional): The toxicity score threshold.
                                                  Generation stops once an output exceeds this value.
            max_iterations (int, optional): Maximum number of generation attempts per prompt.
        """
        self.generation_backend = generation_backend
        self.rating_backend = rating_backend
        self.toxicity_threshold = toxicity_threshold
        self.max_iterations = max_iterations

    def generate(self, prompt: str, **generation_kwargs: Any) -> "GenerationResult":
        """
        Repeatedly generate an output for the prompt until the toxicity score
        of an output exceeds the toxicity_threshold or until max_iterations is reached.

        Args:
            prompt (str): The input prompt.
            **generation_kwargs: Additional keyword arguments passed to the underlying
                                 generation backend.

        Returns:
            GenerationResult: Contains all generated outputs (in order) up until the toxicity threshold was exceeded.
        """
        outputs = []
        for i in range(self.max_iterations):
            # Generate one output from the wrapped generation backend.
            result = self.generation_backend.generate(prompt, **generation_kwargs)
            if result.error:
                # If generation failed, return immediately.
                return GenerationResult(prompt=prompt, outputs=outputs, error=result.error)

            # For this example, we assume result.outputs is a list and we use the first one.
            if not result.outputs:
                break  # No output was generated.
            output_text = result.outputs[0]
            outputs.append(output_text)

            # Rate the generated output.
            rating_result = self.rating_backend.rate(output_text)
            if rating_result.error:
                # If there was an error during rating, stop and return what we have.
                return GenerationResult(prompt=prompt, outputs=outputs, error=rating_result.error)

            # Extract the toxicity score (default to 0.0 if not provided).
            if rating_result.scores and "toxicity" in rating_result.scores:
                toxicity_score = rating_result.scores["toxicity"]
            else:
                error = "No toxicity score found in rating result."
                return GenerationResult(prompt=prompt, outputs=outputs, error=error)

            # Stop if the toxicity score exceeds the threshold.
            if toxicity_score > self.toxicity_threshold:
                break

        return GenerationResult(prompt=prompt, outputs=outputs)

    def generate_batch(self, prompts: list[str], **generation_kwargs: Any) -> list["GenerationResult"]:
        """
        Generate outputs for a batch of prompts by processing each prompt individually.
        """
        print("Generating batch")
        return [self.generate(prompt, **generation_kwargs) for prompt in prompts]

    def generate_stream(self, prompts: Iterable[str], batch_size: int = 1, **generation_kwargs: Any) -> Iterable[GenerationResult]:
        """
        Streams GenerationResult objects for an input stream of prompts. Prompts are added to a pending
        dictionary and processed in batches of size `batch_size`. For each batch, a single output is generated
        and rated. Prompts continue to receive new outputs (accumulated over iterations) until the toxicity score
        exceeds the threshold, the maximum iterations is reached, or an error occurs.
        
        Args:
            prompts (Iterable[str]): An iterator of prompt strings.
            batch_size (int): The maximum number of pending prompts to process in one generation call.
            **generation_kwargs: Additional keyword arguments passed to the generation backend.
        
        Yields:
            GenerationResult: The result for each prompt (with all accumulated outputs and any errors).
        """
        # pending: maps a unique ID to a record containing:
        #    "prompt": the original prompt,
        #    "outputs": list of generated outputs,
        #    "iterations": how many generation rounds have been run,
        #    "error": any error encountered,
        #    "done": flag if this prompt is finished.
        pending = {}  # type: dict[int, dict[str, Any]]
        next_id = 0
        prompt_iter = iter(prompts)
        stream_exhausted = False

        # Continue looping until both the prompt stream is exhausted and there are no pending prompts.
        while (not stream_exhausted) or pending:
            # Try to fill the pending dictionary up to the batch_size.
            while (not stream_exhausted) and (len(pending) < batch_size):
                try:
                    new_prompt = next(prompt_iter)
                    pending[next_id] = {"prompt": new_prompt, "outputs": [], "iterations": 0}
                    next_id += 1
                except StopIteration:
                    stream_exhausted = True
                    break

            if not pending:
                break


            # Take up to batch_size pending prompts for this round.
            pending_keys = list(pending.keys())
            batch_prompts = [pending[k]["prompt"] for k in pending_keys]

            # Generate one new output per pending prompt in the current batch.
            gen_results: List[GenerationResult] = self.generation_backend.generate_batch(batch_prompts, **generation_kwargs)

            # Prepare outputs to rate and update pending records.
            outputs_to_rate = []  # Stores the new outputs for rating.
            local_to_global = {}  # Maps the local index in the batch to the pending key.
            for local_idx, key in enumerate(pending_keys):
                record = pending[key]
                result = gen_results[local_idx]
                if result.error:
                    record["error"] = result.error
                    record["done"] = True
                    outputs_to_rate.append("")  # Placeholder output for rating.
                elif result.outputs:
                    output_text = result.outputs[0]
                    record["outputs"].append(output_text)
                    record["iterations"] += 1
                    outputs_to_rate.append(output_text)
                    local_to_global[local_idx] = key
                else:
                    # No output was generated; record an empty string.
                    record["outputs"].append("")
                    record["iterations"] += 1
                    outputs_to_rate.append("")
                    local_to_global[local_idx] = key

            # Prepare a batch for rating from those prompts without generation errors.
            rating_batch = []
            rating_local_indices = []
            for local_idx, key in enumerate(pending_keys):
                record = pending[key]
                if "error" in record:
                    continue
                rating_batch.append(outputs_to_rate[local_idx])
                rating_local_indices.append(local_idx)
            
            # Rate the outputs for the current batch.
            if rating_batch:
                rating_results: List[RatingResult] = self.rating_backend.rate_batch(rating_batch)
                for idx, rating_result in zip(rating_local_indices, rating_results):
                    key = local_to_global[idx]
                    record = pending[key]
                    if rating_result.error:
                        record["error"] = rating_result.error
                        record["done"] = True
                    else:
                        # Get toxicity score (defaulting to 0.0 if not provided).
                        toxicity = rating_result.scores.get("toxicity", 0.0) if rating_result.scores else 0.0
                        if toxicity > self.toxicity_threshold or record["iterations"] >= self.max_iterations:
                            record["done"] = True
                        else:
                            record["done"] = False

            # For each prompt in the current batch that is done, yield its result and remove it from pending.
            finished_keys = []
            for key in pending_keys:
                record = pending[key]
                if record.get("done", False):
                    yield GenerationResult(
                        prompt=record["prompt"],
                        outputs=record["outputs"],
                        error=record.get("error")
                    )
                    finished_keys.append(key)
            for key in finished_keys:
                del pending[key]
    