from collections.abc import Iterable, Iterator
from typing import TypeVar, Generic, List
from pathlib import Path
import torch
import numpy
import random
import gc
import pandas as pd
import fsspec


import contextlib
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel

T = TypeVar("T")


from collections import deque


class RunningAverage:
    """
    Running average calculator. Supports both standard and windowed running averages.
    """

    def __init__(self, window_size: int = 10):
        """
        Args:
            window_size (int, optional): The size of the window for a fixed-size running average.
        """
        self._window_total = 0.0
        self._window_count = 0
        self._window_size = window_size
        self._window_values = deque(maxlen=window_size)

        self._global_total = 0.0
        self._global_count = 0

    def update(self, value: float, count: int = 1):
        """
        Updates the running average with a new value.

        Args:
            value (float): The new value to add.
            count (int): The number of times to add the value (default: 1).
        """
        # Update windown stats
        for _ in range(count):
            if len(self._window_values) == self._window_size:
                self._window_total -= self._window_values.popleft()
            self._window_values.append(value)
            self._window_total += value
        self._window_count = len(self._window_values)

        # Update global stats
        self._global_total += value * count
        self._global_count += count

    @property
    def global_count(self) -> int:
        """Returns the total count of values added."""
        return self._global_count

    @property
    def global_average(self) -> float:
        """Returns the global average."""
        return self._global_total / self._global_count if self._global_count > 0 else 0.0

    @property
    def window_average(self) -> float:
        """Returns the windowed average."""
        return self._window_total / self._window_count if self._window_count > 0 else 0.0

    @property
    def window_count(self) -> int:
        """Returns the count of values in the current window."""
        return self._window_count


class Batchifier(Generic[T]):
    def __init__(self, data: Iterable[T], batch_size: int):
        assert batch_size > 0, "batch_size must be > 0"
        self.data = data
        self.batch_size = batch_size

        if hasattr(data, "__len__"):
            self._length = (len(data) + batch_size - 1) // batch_size
        else:
            self._length = None

    def __iter__(self) -> Iterator[List[T]]:
        batch: List[T] = []
        for item in self.data:
            batch.append(item)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self) -> int:
        if self._length is None:
            raise AttributeError("Batchifier object has no attribute '__len__'")
        return self._length


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """
    Returns an iterable of batches.

    If the input data has a length (i.e. it is Sized), then the returned object
    also implements __len__ (giving the number of batches).
    """
    return Batchifier(data, batch_size)


def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.

    Args:
        seed (int): Seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)


def get_device() -> torch.device:
    """
    Get the device to use for computations.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_memory() -> None:
    """
    Frees unused memory by calling the garbage collector and clearing the CUDA cache.
    This helps prevent out-of-memory errors in GPU-limited environments.
    """
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()


def full_cleanup():    
    """
    Cleans up the environment by destroying distributed and model parallel environments,
    and clearing CUDA memory.
    """
    destroy_model_parallel()
    destroy_distributed_environment()

    with contextlib.suppress(AssertionError):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    clear_memory()


def api_key_from_file(path: str) -> str:
    """
    Read an API key from a file.

    Args:
        path (str): Path to the file containing the API key.

    Returns:
        str: The API key.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    key_file = Path(path)
    if key_file.exists():
        with key_file.open("r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        raise FileNotFoundError("API key file not found")


def load_jsonl_prompts(path: str, limit: int | None = None) -> list[str]:
    """Return a list of prompt strings from a JSONL file."""
    with fsspec.open(path, "r") as f:
        df = pd.read_json(f, lines=True)
    prompts = [d["text"] for d in df["prompt"].values.flatten().tolist()]
    return prompts[:limit] if limit else prompts
