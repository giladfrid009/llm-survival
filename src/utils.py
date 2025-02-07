from collections.abc import Iterable, Sized, Iterator
from typing import TypeVar, Generic, List

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
