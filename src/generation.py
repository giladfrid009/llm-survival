from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Iterable

from src.utils import batchify

@dataclass
class GenerationResult:
    """
    Represents the generation result for a single prompt.

    Attributes:
        prompt (str): The input prompt.
        outputs (list[str]): The generated outputs.
        error (str, optional): An error message if generation failed.
    """
    
    prompt: str
    outputs: list[str]
    error: str | None = None

class GenerationBackend(ABC):
    """
    Abstract base class for a generation backend.

    New backends need only implement the `generate()` method (and may override `generate_batch()`
    for vectorized or batch processing if available).
    """

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate text for the given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            GenerationResult: The result, containing generated text or an error message.
        """
        pass

    def generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        """
        Generate an entire batch of prompts sequentially by calling `generate()` for each.

        Args:
            prompts (list[str]): A list of prompt strings.
        """
        return [self.generate(prompt) for prompt in prompts]
    

class GenerationRunner:
    """
    Generic API that delegates generation to the provided backend.
    """

    def __init__(self, backend: GenerationBackend):
        self.backend = backend

    def generate_single(self, prompt: str) -> GenerationResult:
        """
        Generate text for the given prompt.

        Args:
            prompt (str): The input prompt.

        Returns:
            GenerationResult: The result, containing generated text or an error message.
        """
        return self.backend.generate(prompt)
    
    def generate_batch(self, prompts: list[str], batch_size: int = 1) -> list[GenerationResult]:
        """
        Generate outputs for an entire batch of prompts.

        Args:
            prompts (list[str]): A list of prompt strings.
            batch_size (int): The number of prompts to generate in parallel.

        Returns:
            list[GenerationResult]: The results for each prompt.
        """
        return self.backend.generate_batch(prompts)
    
    def generate_stream(self, prompts: Iterable[str], batch_size: int = 1) -> Iterable[GenerationResult]:
        """
        Generate outputs for an entire stream of prompts.

        Args:
            prompts (Iterable[str]): An iterable of prompt strings.
            batch_size (int): The number of prompts to generate in parallel.

        Returns:
            Iterable[GenerationResult]: The results for each prompt.
        """
        batches = batchify(prompts, batch_size)
        for batch in tqdm(batches, desc="Generating", unit="batch"):
            yield from self.generate_batch(batch)

    def generate_stream_batched(self, prompts: Iterable[str], batch_size: int = 1) -> Iterable[GenerationResult]:
        """
        Generate outputs for an entire stream of prompts in batches.

        Args:
            prompts (Iterable[str]): An iterable of prompt strings.
            batch_size (int): The number of prompts to generate in parallel.

        Returns:
            Iterable[GenerationResult]: The results for each prompt.
        """
        for batch in tqdm(prompts, desc="Generating", unit="batch"):
            if not isinstance(batch, list):
                batch = list(batch)
            yield self.generate_batch(batch)