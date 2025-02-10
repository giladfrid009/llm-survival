from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import OrderedDict
from tqdm.auto import tqdm
from typing import Iterable

from src.utils import batchify


@dataclass
class RatingResult:
    """
    Represents the rating result for a single text.

    Attributes:
        text (str): The input text.
        scores (OrderedDict[str, float], optional: The computed score per attribute,
            or None if an error occurred.
        error (str, optional): An error message if rating failed.
    """

    text: str
    scores: OrderedDict[str, float] | None = None
    error: str | None = None


class RatingBackend(ABC):
    """
    Abstract base class for a rating backend.

    New backends need only implement the `rate()` method (and may override `rate_batch()`
    for vectorized or batch processing if available).
    """

    @abstractmethod
    def rate(self, text: str) -> RatingResult:
        """
        Compute a score for the given text.

        Args:
            text (str): The text text to be rated.

        Returns:
            RatingResult: The result, containing scores or an error message.
        """
        pass

    def rate_batch(self, texts: list[str]) -> list[RatingResult]:
        """
        Rate a batch of texts sequentially by calling `rate()` for each.

        Args:
            texts (list[str]): A list of text strings.

        Returns:
            list[RatingResult]: The results for each text.
        """
        return [self.rate(text) for text in texts]


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

    def rate_single(self, text: str) -> RatingResult:
        """
        Rate a single text.

        Args:
            text (str): The text text to be rated.

        Returns:
            RatingResult: The result, containing scores or an error message.
        """
        return self.backend.rate(text)

    def rate_batch(self, texts: list[str]) -> list[RatingResult]:
        """
        Rate a batch of texts.

        Args:
            texts (list[str]): A batch of text strings.

        Returns:
            list[RatingResult]: The results for each text.
        """
        return self.backend.rate_batch(texts)

    def rate_stram(self, texts: Iterable[str], batch_size: int = 100) -> Iterable[RatingResult]:
        """
        Rate a stream of text strings by processing them in batches.
        Args:
            texts (Iterable[str]): An iterable of text strings to be rated.
            batch_size (int, optional): Number of texts to process in each batch. Defaults to 100.
        Yields:
            RatingResult: The rating result for each processed text from the batch.
        """
        batches = batchify(texts, batch_size)
        for batch in tqdm(batches, desc="Rating", unit="batch"):
            yield from self.rate_batch(batch)

    def rate_stream_batched(self, texts: Iterable[Iterable[str]]) -> Iterable[list[RatingResult]]:
        """
        Rate a stream of batched texts.
        This method processes an iterable containing batches of text strings. Each batch is
        ensured to be a list (if it isn't already) and then passed to the rate_batch method to
        generate ratings for that batch. The progress of processing is displayed using tqdm.
        Parameters:
            texts (Iterable[Iterable[str]]): An iterable where each element is an iterable of
                                               text strings constituting a batch.
        Yields:
            list[RatingResult]: A list of RatingResult objects corresponding to the ratings of the
                                texts in the batch.
        """
        for batch in tqdm(texts, desc="Processing", unit="batch"):
            if not isinstance(batch, list):
                batch = list(batch)
            yield self.rate_batch(batch)
