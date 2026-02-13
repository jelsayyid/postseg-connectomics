"""Volume reader protocol and base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class VolumeReader(Protocol):
    """Protocol for reading segmentation volumes."""

    @property
    def shape(self) -> Tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...
    @property
    def resolution(self) -> Tuple[float, ...]: ...

    def read_chunk(
        self,
        offset: Tuple[int, ...],
        size: Tuple[int, ...],
    ) -> np.ndarray: ...

    def chunk_iterator(
        self,
        chunk_size: Tuple[int, ...],
        overlap: Tuple[int, ...],
    ) -> Iterator[Tuple[Tuple[int, ...], np.ndarray]]: ...


class BaseVolumeReader(ABC):
    """Base class providing shared chunk iteration logic."""

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]: ...

    @property
    @abstractmethod
    def dtype(self) -> np.dtype: ...

    @property
    @abstractmethod
    def resolution(self) -> Tuple[float, ...]: ...

    @abstractmethod
    def read_chunk(
        self,
        offset: Tuple[int, ...],
        size: Tuple[int, ...],
    ) -> np.ndarray: ...

    def chunk_iterator(
        self,
        chunk_size: Tuple[int, ...],
        overlap: Tuple[int, ...] = (0, 0, 0),
    ) -> Iterator[Tuple[Tuple[int, ...], np.ndarray]]:
        """Iterate over the volume in chunks with optional overlap.

        Args:
            chunk_size: Size of each chunk (z, y, x).
            overlap: Overlap between adjacent chunks (z, y, x).

        Yields:
            (offset, chunk_data) tuples.
        """
        shape = self.shape
        step = tuple(max(1, cs - ov) for cs, ov in zip(chunk_size, overlap))
        for z in range(0, shape[0], step[0]):
            for y in range(0, shape[1], step[1]):
                for x in range(0, shape[2], step[2]):
                    offset = (z, y, x)
                    actual_size = tuple(
                        min(cs, sh - o) for cs, sh, o in zip(chunk_size, shape, offset)
                    )
                    if all(s > 0 for s in actual_size):
                        chunk = self.read_chunk(offset, actual_size)
                        yield offset, chunk
