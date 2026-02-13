"""HDF5 volume reader."""

from __future__ import annotations

from typing import Tuple

import h5py
import numpy as np

from connectomics_pipeline.io.volume_reader import BaseVolumeReader


class HDF5Reader(BaseVolumeReader):
    """Read segmentation volumes from HDF5 files."""

    def __init__(
        self,
        path: str,
        dataset: str = "labels",
        resolution: Tuple[float, ...] = (30.0, 8.0, 8.0),
    ):
        self._path = path
        self._dataset = dataset
        self._resolution = tuple(resolution)
        with h5py.File(self._path, "r") as f:
            ds = f[self._dataset]
            self._shape = ds.shape
            self._dtype = ds.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def resolution(self) -> Tuple[float, ...]:
        return self._resolution

    def read_chunk(
        self,
        offset: Tuple[int, ...],
        size: Tuple[int, ...],
    ) -> np.ndarray:
        slices = tuple(slice(o, o + s) for o, s in zip(offset, size))
        with h5py.File(self._path, "r") as f:
            return f[self._dataset][slices]
