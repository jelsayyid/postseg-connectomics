"""Neuroglancer precomputed volume reader (optional dependency)."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from connectomics_pipeline.io.volume_reader import BaseVolumeReader

try:
    from cloudvolume import CloudVolume

    _HAS_CLOUDVOLUME = True
except ImportError:
    _HAS_CLOUDVOLUME = False


class PrecomputedReader(BaseVolumeReader):
    """Read segmentation volumes from Neuroglancer precomputed format.

    Requires the cloud-volume package: pip install cloud-volume
    """

    def __init__(
        self,
        path: str,
        resolution: Tuple[float, ...] = (30.0, 8.0, 8.0),
        mip: int = 0,
    ):
        if not _HAS_CLOUDVOLUME:
            raise ImportError(
                "cloud-volume is required for PrecomputedReader. "
                "Install it with: pip install cloud-volume"
            )
        self._vol = CloudVolume(path, mip=mip, use_https=True, fill_missing=True)
        self._resolution = tuple(resolution)

    @property
    def shape(self) -> Tuple[int, ...]:
        # CloudVolume uses (x, y, z) order; convert to (z, y, x)
        s = self._vol.shape[:3]
        return (s[2], s[1], s[0])

    @property
    def dtype(self) -> np.dtype:
        return self._vol.dtype

    @property
    def resolution(self) -> Tuple[float, ...]:
        return self._resolution

    def read_chunk(
        self,
        offset: Tuple[int, ...],
        size: Tuple[int, ...],
    ) -> np.ndarray:
        # Convert (z, y, x) to CloudVolume's (x, y, z) indexing
        z, y, x = offset
        dz, dy, dx = size
        data = self._vol[x : x + dx, y : y + dy, z : z + dz]
        # CloudVolume returns (x, y, z, channel), transpose to (z, y, x)
        return np.squeeze(data).transpose(2, 1, 0)
