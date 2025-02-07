from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from collections import OrderedDict
from tqdm.auto import tqdm


@dataclass
class RatingResult:
    """
    Represents the rating result for a single prompt.

    Attributes:
        prompt (str): The input prompt.
        scores (OrderedDict[str, float], optional: The computed score per attribute,
            or None if an error occurred.
        error (str, optional): An error message if rating failed.
    """

    prompt: str
    scores: OrderedDict[str, float] | None = None
    error: str | None = None


class RatingBackend(ABC):
    """
    Abstract base class for a rating backend.

    New backends need only implement the `rate()` method (and may override `rate_batch()`
    for vectorized or batch processing if available).
    """

    @abstractmethod
    def rate(self, prompt: str) -> RatingResult:
        """
        Compute a score for the given prompt.

        Args:
            prompt (str): The prompt text to be rated.

        Returns:
            RatingResult: The result, containing scores or an error message.
        """
        pass

    def rate_batch(self, prompts: list[str]) -> list[RatingResult]:
        """
        Rate a batch of prompts sequentially by calling `rate()` for each.

        Args:
            prompts (list[str]): A list of prompt strings.

        Returns:
            list[RatingResult]: The results for each prompt.
        """
        return [self.rate(prompt) for prompt in prompts]


class RatingRunner:
    """
    Generic rating API that delegates scoring to the provided backend.
    """

    def __init__(self, backend: RatingBackend):
        """
        Args:
            backend (RatingBackend): The backend to use for rating.
        """
        self.backend = backend

    def rate(self, prompt: str) -> RatingResult:
        """
        Rate a single prompt.
        """
        return self.backend.rate(prompt)

    def rate_batch(self, prompts: list[str]) -> list[RatingResult]:
        """
        Rate a batch of prompts sequentially.
        """
        return self.backend.rate_batch(prompts)

    def rate_all(self, prompts: list[str], batch_size: int = 100, num_workers: int = 0) -> list[RatingResult]:
        """
        Rate all prompts using batching and optional parallel processing.

        This method splits the list of prompts into batches of size `batch_size`. Each batch is
        processed by calling the backend's `rate_batch()` method. If `num_workers` is greater than 0,
        batches are processed in parallel using a ThreadPoolExecutor. The results are returned in the
        same order as the input prompts.

        Args:
            prompts (list[str]): A list of prompt strings.
            batch_size (int): Number of prompts per batch.
            num_workers (int): Number of worker threads to use. If 0, processing is done sequentially.

        Returns:
            list[RatingResult]: The rating results for all prompts.
        """
        total_prompts = len(prompts)
        batches = [prompts[i : i + batch_size] for i in range(0, total_prompts, batch_size)]

        # Sequential processing.
        if num_workers <= 0:
            results = []
            for batch in tqdm(batches, desc="Processing", unit="batch"):
                results.extend(self.backend.rate_batch(batch))
            return results

        # Parallel processing.
        results_per_batch = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:

            # Submit each batch to the executor and record its index.
            future_to_index = {}
            for batch_index, batch in enumerate(batches):
                future = executor.submit(self.backend.rate_batch, batch)
                future_to_index[future] = batch_index

            # Process futures as they complete.
            for completed_future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing", unit="batch"):
                batch_index = future_to_index[completed_future]
                batch_results = completed_future.result()
                results_per_batch[batch_index] = batch_results

        # Flatten the list of lists.
        results = []
        for batch_result in results_per_batch:
            results.extend(batch_result)
        return results
