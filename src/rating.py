from abc import ABC, abstractmethod
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

    def rate_all(self, prompts: list[str], batch_size: int = 100) -> list[RatingResult]:
        """
        Rate all prompts using batching and optional parallel processing.

        This method splits the list of prompts into batches of size `batch_size`. Each batch is
        processed by calling the backend's `rate_batch()` method. The results are returned in the
        same order as the input prompts.

        Args:
            prompts (list[str]): A list of prompt strings.
            batch_size (int): Number of prompts per batch.

        Returns:
            list[RatingResult]: The rating results for all prompts.
        """
        total_prompts = len(prompts)
        batches = [prompts[i : i + batch_size] for i in range(0, total_prompts, batch_size)]

        results = []
        for batch in tqdm(batches, desc="Processing", unit="batch"):
            results.extend(self.backend.rate_batch(batch))
        return results
