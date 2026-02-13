"""Surface mesh extraction from fragment volumes."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from skimage.measure import marching_cubes

from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import Mesh

logger = get_logger("fragments.mesh")


class MeshExtractor:
    """Extract surface meshes from fragment masks using marching cubes."""

    def __init__(self, resolution: Tuple[float, ...] = (30.0, 8.0, 8.0)):
        self.resolution = np.array(resolution)

    def extract(self, mask: np.ndarray) -> Mesh:
        """Extract a triangle mesh from a binary mask.

        Args:
            mask: 3D binary mask.

        Returns:
            Mesh dataclass with vertices in physical coordinates.
        """
        binary = (mask > 0).astype(float)
        if binary.sum() < 4:
            return Mesh(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=int),
            )

        # Pad to ensure closed mesh
        padded = np.pad(binary, 1, mode="constant", constant_values=0)

        try:
            verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(self.resolution))
            # Adjust for padding offset
            verts -= self.resolution
            return Mesh(vertices=verts, faces=faces.astype(int))
        except Exception:
            logger.warning("Marching cubes failed, returning empty mesh")
            return Mesh(
                vertices=np.zeros((0, 3)),
                faces=np.zeros((0, 3), dtype=int),
            )
