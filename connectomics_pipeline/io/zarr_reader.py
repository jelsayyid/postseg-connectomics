"""Zarr volume reader."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import zarr

from connectomics_pipeline.io.volume_reader import BaseVolumeReader


def _open_zarr_array(path: str, dataset: str) -> zarr.Array:
    """Open a Zarr array, navigating into a group dataset if needed."""
    if dataset:
        store = zarr.open_group(path, mode="r")
        arr = store[dataset]
    else:
        arr = zarr.open_array(path, mode="r")
    if not isinstance(arr, zarr.Array):
        raise TypeError(f"Expected zarr.Array at '{dataset}', got {type(arr).__name__}")
    return arr


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
        arr = _open_zarr_array(self._path, self._dataset)
        self._shape: Tuple[int, ...] = arr.shape
        self._dtype: np.dtype = arr.dtype

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
        arr = _open_zarr_array(self._path, self._dataset)
        return np.array(arr[slices])
