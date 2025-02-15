from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator

from tqdm.auto import tqdm
from src.utils import batchify


@dataclass
class RatingResult:
    """
    Represents the rating result for a single text.

    Attributes:
        text (str): The input text.
        scores (dict[str, float]): The computed score per attribute.
    """

    text: str
    scores: dict[str, float]


class RatingBackend(ABC):
    """
    Abstract base class for a rating backend.

    New backends need only implement the `rate()` method (and may override `rate_batch()`
    for vectorized or batch processing if available).
    """

    @abstractmethod
    def rate(self, text: str, **kwargs) -> RatingResult:
        """
        Compute a score for the given text.

        Args:
            text (str): The text text to be rated.
            **kwargs: Additional keyword arguments for the rating backend.

        Returns:
            RatingResult: The result, containing scores or an error message.
        """
        pass

    def rate_batch(self, texts: list[str], **kwargs) -> list[RatingResult]:
        """
        Rate a batch of texts sequentially by calling `rate()` for each.

        Args:
            texts (list[str]): A list of text strings.
            **kwargs: Additional keyword arguments for the rating backend.

        Returns:
            list[RatingResult]: The results for each text.
        """
        return [self.rate(text, kwargs) for text in texts]


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

    def rate_single(self, text: str, **kwargs) -> RatingResult:
        """
        Rate a single text.

        Args:
            text (str): The text text to be rated.
            **kwargs: Additional keyword arguments for the rating backend.

        Returns:
            RatingResult: The result, containing scores or an error message.
        """
        return self.backend.rate(text, **kwargs)

    def rate_batch(self, texts: list[str], **kwargs) -> list[RatingResult]:
        """
        Rate a batch of texts.

        Args:
            texts (list[str]): A batch of text strings.
            **kwargs: Additional keyword arguments for the rating backend.

        Returns:
            list[RatingResult]: The results for each text.
        """
        if len(texts) == 0:
            return []
        return self.backend.rate_batch(texts, **kwargs)

    def rate_stram(self, texts: Iterable[str], batch_size: int = 100, **kwargs) -> Iterator[RatingResult]:
        """
        Rate a stream of text strings by processing them in batches.

        Args:
            texts (Iterable[str]): An iterable of text strings to be rated.
            batch_size (int, optional): Number of texts to process in each batch. Defaults to 100.
            **kwargs: Additional keyword arguments for the rating backend.

        Yields:
            RatingResult: The rating result for each processed text from the batch.
        """
        batches = batchify(texts, batch_size)
        for batch in tqdm(batches, desc="Rating", unit="batch"):
            yield from self.rate_batch(batch, **kwargs)

    def rate_stream_batched(self, texts: Iterable[Iterable[str]], **kwargs) -> Iterator[list[RatingResult]]:
        """
        Rate a stream of text strings by processing them in batches.
        
        Args:
            texts (Iterable[Iterable[str]]): An iterable of iterables of text strings to be rated.
            **kwargs: Additional keyword arguments for the rating backend.
            
        Yields:
            list[RatingResult]: The rating result for each processed batch of texts.
        """
        for batch in tqdm(texts, desc="Rating", unit="batch"):
            if not isinstance(batch, list):
                batch = list(batch)
            yield self.rate_batch(batch, **kwargs)
