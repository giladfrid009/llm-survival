from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from typing import Iterable

from src.utils import batchify


@dataclass
class GenerationResult:
    """
    Represents the generation result for a single prompt.

    Attributes:
        prompt (str): The input prompt.
        outputs (str): The generated text, or None if an error occurred.
        error (str, optional): An error message if generation failed.
    """

    prompt: str
    output: str | None = None
    error: str | None = None
    
    def is_valid(self) -> bool:
        return self.output is not None and self.error is None


class GenerationBackend(ABC):
    """
    Abstract base class for a generation backend.

    New backends need only implement the `generate()` method (and may override `generate_batch()`
    for vectorized or batch processing if available).
    """

    @abstractmethod
    def generate(self, prompt: str, max_len: int, **kwargs) -> GenerationResult:
        """
        Generate text for the given prompt.

        Args:
            prompt (str): The input prompt.
            max_len (int): The maximum length of the generated text.
            **kwargs: Additional keyword arguments passed to the generation method.

        Returns:
            GenerationResult: The result, containing generated text or an error message.
        """
        pass

    def generate_batch(self, prompts: list[str], max_len: int, **kwargs) -> list[GenerationResult]:
        """
        Generate an entire batch of prompts sequentially by calling `generate()` for each.

        Args:
            prompts (list[str]): A list of prompt strings.
            max_len (int): The maximum length of the generated text.
            **kwargs: Additional keyword arguments passed to the generation method.

        Returns:
            list[GenerationResult]: The results for each prompt.
        """
        return [self.generate(prompt, max_len, **kwargs) for prompt in prompts]


class GenerationRunner:
    """
    Generic API that delegates generation to the provided backend.
    """

    def __init__(self, backend: GenerationBackend):
        self.backend = backend

    def generate_single(self, prompt: str, max_len: int = 50, **kwargs) -> GenerationResult:
        """
        Generate text for the given prompt.

        Args:
            prompt (str): The input prompt.
            max_len (int): The maximum length of the generated text.
            **kwargs: Additional keyword arguments passed to the generation method.

        Returns:
            GenerationResult: The result, containing generated text or an error message.
        """
        return self.backend.generate(prompt, max_len, **kwargs)

    def generate_batch(self, prompts: list[str], max_len: int = 50, **kwargs) -> list[GenerationResult]:
        """
        Generate outputs for an entire batch of prompts.

        Args:
            prompts (list[str]): A list of prompt strings.
            max_len (int): The maximum length of the generated text.
            **kwargs: Additional keyword arguments passed to the generation method.

        Returns:
            list[GenerationResult]: The results for each prompt.
        """
        if len(prompts) == 0:
            return []
        return self.backend.generate_batch(prompts, max_len, **kwargs)

    def generate_stream(
        self,
        prompts: Iterable[str],
        max_len: int = 50,
        batch_size: int = 1,
        **kwargs,
    ) -> Iterable[GenerationResult]:
        """
        Generate outputs for an entire stream of prompts.

        Args:
            prompts (Iterable[str]): An iterable of prompt strings.
            max_len (int): The maximum length of the generated text.
            batch_size (int): The number of prompts to generate in parallel.
            **kwargs: Additional keyword arguments passed to the generation method.

        Yields:
            GenerationResult: The results for each prompt.
        """
        batches = batchify(prompts, batch_size)
        for batch in tqdm(batches, desc="Generating", unit="batch"):
            yield from self.generate_batch(batch, max_len, **kwargs)

    def generate_stream_batched(
        self,
        prompts: Iterable[str],
        max_len: int = 50,
        **kwargs,
    ) -> Iterable[list[GenerationResult]]:
        """
        Generate outputs for an entire stream of prompts in batches.

        Args:
            prompts (Iterable[str]): An iterable of prompt strings.
            max_len (int): The maximum length of the generated text.
            **kwargs: Additional keyword arguments passed to the generation method.

        Yields:
            list[GenerationResult]: The results for each batch of prompts.
        """
        for batch in tqdm(prompts, desc="Generating", unit="batch"):
            if not isinstance(batch, list):
                batch = list(batch)
            yield self.generate_batch(batch)
