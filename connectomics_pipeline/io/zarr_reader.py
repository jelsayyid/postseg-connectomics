"""Zarr volume reader."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import zarr

from connectomics_pipeline.io.volume_reader import BaseVolumeReader


class ZarrReader(BaseVolumeReader):
    """Read segmentation volumes from Zarr arrays."""

    def __init__(
        self,
        path: str,
        dataset: str = "labels",
        resolution: Tuple[float, ...] = (30.0, 8.0, 8.0),
    ):
        self._path = path
        self._dataset = dataset
        self._resolution = tuple(resolution)
        store = zarr.open(self._path, mode="r")
        ds = store[self._dataset] if self._dataset else store
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
        store = zarr.open(self._path, mode="r")
        ds = store[self._dataset] if self._dataset else store
        return np.array(ds[slices])
