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
        """
        return self.backend.rate(prompt)

    def rate_batch(self, prompts: list[str]) -> list[RatingResult]:
        """
        Rate a batch of prompts sequentially.
        """
        return self.backend.rate_batch(prompts)

    def rate_stram(self, prompts: Iterable[str], batch_size: int = 100) -> Iterable[RatingResult]:
        """
        Rate all prompts using batching.

        This method splits the list of prompts into batches of size `batch_size`. Each batch is
        processed by calling the backend's `rate_batch()` method. The results are returned in the
        same order as the input prompts.

        Args:
            prompts (list[str]): A list of prompt strings.
            batch_size (int): Number of prompts per batch.

        Returns:
            Iterable[RatingResult]: An iterable of results for each prompt.
        """
        batches = batchify(prompts, batch_size)
        for batch in tqdm(batches, desc="Processing", unit="batch"):
            yield from self.rate_batch(batch)

    def rate_stream_batched(self, prompts: Iterable[Iterable[str]]) -> Iterable[list[RatingResult]]:
        """
        Rate all prompts using batching.

        This method processes each batch of prompts by calling the backend's `rate_batch()` method.
        The results are returned in the same order as the input prompts.

        Args:
            prompts (Iterable[Iterable[str]]): An iterable of batches of prompt strings.

        Returns:
            Iterable[list[RatingResult]]: An iterable of results for each batch.
        """
        for batch in tqdm(prompts, desc="Processing", unit="batch"):
            if not isinstance(batch, list):
                batch = list(batch)
            yield self.rate_batch(batch)
