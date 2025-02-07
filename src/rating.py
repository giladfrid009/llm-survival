from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import OrderedDict
from tqdm.auto import tqdm
from typing import Iterable

from src.utils import batchify


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

    def rate_single(self, prompt: str) -> RatingResult:
        """
        Rate a single prompt.

        Args:
            prompt (str): The prompt text to be rated.

        Returns:
            RatingResult: The result, containing scores or an error message.
        """
        return self.backend.rate(prompt)

    def rate_batch(self, prompts: list[str]) -> list[RatingResult]:
        """
        Rate a batch of prompts sequentially.

        Args:
            prompts (list[str]): A batch of prompt strings.

        Returns:
            list[RatingResult]: The results for each prompt.
        """
        return self.backend.rate_batch(prompts)

    def rate_stram(self, prompts: Iterable[str], batch_size: int = 100) -> Iterable[RatingResult]:
        """
        Rate a stream of prompt strings by processing them in batches.
        Args:
            prompts (Iterable[str]): An iterable of prompt strings to be rated.
            batch_size (int, optional): Number of prompts to process in each batch. Defaults to 100.
        Yields:
            RatingResult: The rating result for each processed prompt from the batch.
        """
        batches = batchify(prompts, batch_size)
        for batch in tqdm(batches, desc="Processing", unit="batch"):
            yield from self.rate_batch(batch)

    def rate_stream_batched(self, prompts: Iterable[Iterable[str]]) -> Iterable[list[RatingResult]]:
        """
        Rate a stream of batched prompts.
        This method processes an iterable containing batches of prompt strings. Each batch is
        ensured to be a list (if it isn't already) and then passed to the rate_batch method to
        generate ratings for that batch. The progress of processing is displayed using tqdm.
        Parameters:
            prompts (Iterable[Iterable[str]]): An iterable where each element is an iterable of
                                               prompt strings constituting a batch.
        Yields:
            list[RatingResult]: A list of RatingResult objects corresponding to the ratings of the
                                prompts in the batch.
        """
        for batch in tqdm(prompts, desc="Processing", unit="batch"):
            if not isinstance(batch, list):
                batch = list(batch)
            yield self.rate_batch(batch)
