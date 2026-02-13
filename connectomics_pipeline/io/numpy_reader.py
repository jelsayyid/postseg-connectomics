"""NumPy array-backed volume reader for testing."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from connectomics_pipeline.io.volume_reader import BaseVolumeReader


class NumpyReader(BaseVolumeReader):
    """Volume reader wrapping an in-memory NumPy array.

    Useful for testing and small volumes.
    """

    def __init__(self, data: np.ndarray, resolution: Tuple[float, ...] = (30.0, 8.0, 8.0)):
        if data.ndim != 3:
            raise ValueError(f"Expected 3D array, got {data.ndim}D")
        self._data = data
        self._resolution = tuple(resolution)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def resolution(self) -> Tuple[float, ...]:
        return self._resolution

    def read_chunk(
        self,
        offset: Tuple[int, ...],
        size: Tuple[int, ...],
    ) -> np.ndarray:
        slices = tuple(slice(o, o + s) for o, s in zip(offset, size))
        return self._data[slices].copy()
