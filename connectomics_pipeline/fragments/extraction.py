"""Connected component extraction from segmentation volumes."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import ndimage

from connectomics_pipeline.utils.config import FragmentConfig
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import BoundingBox, Fragment

logger = get_logger("fragments.extraction")


class FragmentExtractor:
    """Extract connected components from a labeled segmentation chunk."""

    def __init__(self, config: FragmentConfig, resolution: Tuple[float, ...] = (30.0, 8.0, 8.0)):
        self.config = config
        self.resolution = np.array(resolution)
        self._next_id = 0

    def extract_from_chunk(
        self,
        chunk: np.ndarray,
        chunk_origin: Tuple[int, ...],
    ) -> List[Fragment]:
        """Extract fragments from a single chunk.

        Args:
            chunk: 3D label array.
            chunk_origin: (z, y, x) voxel offset of this chunk in the full volume.

        Returns:
            List of Fragment objects.
        """
        fragments = []
        unique_labels = np.unique(chunk)

        for label_id in unique_labels:
            if label_id == 0:
                continue

            mask = chunk == label_id
            labeled, num_components = ndimage.label(mask)

            for comp_id in range(1, num_components + 1):
                comp_mask = labeled == comp_id
                voxel_count = int(np.sum(comp_mask))

                if voxel_count < self.config.min_voxel_count:
                    continue

                fragment = self._build_fragment(
                    comp_mask, int(label_id), voxel_count, chunk_origin, chunk.shape
                )
                fragments.append(fragment)

        logger.debug("Extracted %d fragments from chunk at %s", len(fragments), chunk_origin)
        return fragments

    def _build_fragment(
        self,
        mask: np.ndarray,
        label_id: int,
        voxel_count: int,
        chunk_origin: Tuple[int, ...],
        chunk_shape: Tuple[int, ...],
    ) -> Fragment:
        """Build a Fragment from a binary mask."""
        coords = np.argwhere(mask)  # (N, 3) in voxel coords

        # Bounding box in physical coordinates
        min_voxel = coords.min(axis=0) + np.array(chunk_origin)
        max_voxel = coords.max(axis=0) + np.array(chunk_origin)
        min_corner = min_voxel * self.resolution
        max_corner = (max_voxel + 1) * self.resolution

        # Centroid in physical coordinates
        centroid_voxel = coords.mean(axis=0) + np.array(chunk_origin)
        centroid = centroid_voxel * self.resolution

        # Check if fragment touches chunk boundary
        is_boundary = _touches_boundary(coords, chunk_shape)

        fragment_id = self._next_id
        self._next_id += 1

        return Fragment(
            fragment_id=fragment_id,
            label_id=label_id,
            voxel_count=voxel_count,
            bounding_box=BoundingBox(min_corner=min_corner, max_corner=max_corner),
            centroid=centroid,
            is_boundary=is_boundary,
            chunk_origin=chunk_origin,
        )

    def set_next_id(self, next_id: int) -> None:
        """Set the next fragment ID to assign."""
        self._next_id = next_id


def _touches_boundary(coords: np.ndarray, shape: Tuple[int, ...]) -> bool:
    """Check if any coordinate is on the chunk boundary."""
    for dim in range(3):
        if np.any(coords[:, dim] == 0) or np.any(coords[:, dim] == shape[dim] - 1):
            return True
    return False
