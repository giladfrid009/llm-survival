from collections.abc import Iterable, Iterator
from typing import TypeVar, Generic, List
from pathlib import Path
import torch
import numpy
import random
import gc

T = TypeVar("T")


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
    torch.cuda.empty_cache()


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