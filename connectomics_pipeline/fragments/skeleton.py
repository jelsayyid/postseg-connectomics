"""Skeletonization of fragment volumes."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from skimage.morphology import skeletonize

from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import Skeleton

logger = get_logger("fragments.skeleton")

try:
    import kimimaro

    _HAS_KIMIMARO = True  # pragma: no cover
except ImportError:
    _HAS_KIMIMARO = False


class Skeletonizer:
    """Extract topological skeletons from fragment masks."""

    def __init__(
        self,
        method: str = "teasar",
        resolution: Tuple[float, ...] = (30.0, 8.0, 8.0),
        params: Dict | None = None,
    ):
        self.method = method
        self.resolution = np.array(resolution)
        self.params = params or {}

    def skeletonize(self, mask: np.ndarray, label_id: int = 1) -> Skeleton:
        """Extract skeleton from a binary or labeled mask.

        Args:
            mask: 3D binary mask or labeled volume.
            label_id: Label to skeletonize if mask is labeled.

        Returns:
            Skeleton dataclass.
        """
        if self.method == "teasar" and _HAS_KIMIMARO:
            return self._teasar_skeleton(mask, label_id)
        return self._fallback_skeleton(mask)

    def _teasar_skeleton(self, mask: np.ndarray, label_id: int) -> Skeleton:
        """TEASAR skeletonization via kimimaro."""
        labels = mask if mask.max() > 1 else mask.astype(np.uint32) * label_id
        skels = kimimaro.skeletonize(
            labels,
            teasar_params={
                "scale": self.params.get("invalidation_d0", 10),
                "const": 0,
                "pdrf_scale": 100000,
                "pdrf_exponent": 4,
                "soma_acceptance_threshold": self.params.get("dust_threshold", 100),
                "soma_detection_threshold": 0,
                "soma_invalidation_const": 0,
                "soma_invalidation_scale": 0,
            },
            anisotropy=tuple(self.resolution),
            dust_threshold=self.params.get("dust_threshold", 100),
        )

        if label_id not in skels:
            return Skeleton(
                nodes=np.zeros((0, 3)),
                edges=np.zeros((0, 2), dtype=int),
                radii=np.zeros(0),
            )

        skel = skels[label_id]
        nodes = skel.vertices  # Already in physical coordinates
        edges = skel.edges
        radii = skel.radii.flatten() if skel.radii is not None else np.ones(len(nodes))

        return Skeleton(nodes=nodes, edges=edges, radii=radii)

    def _fallback_skeleton(self, mask: np.ndarray) -> Skeleton:
        """Fallback skeletonization using scikit-image."""
        binary = (mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            return Skeleton(
                nodes=np.zeros((0, 3)),
                edges=np.zeros((0, 2), dtype=int),
                radii=np.zeros(0),
            )

        skel_mask = skeletonize(binary)
        coords = np.argwhere(skel_mask > 0).astype(float)

        if len(coords) == 0:
            return Skeleton(
                nodes=np.zeros((0, 3)),
                edges=np.zeros((0, 2), dtype=int),
                radii=np.zeros(0),
            )

        # Convert to physical coordinates
        nodes = coords * self.resolution

        # Build edges by connecting nearest neighbors along the skeleton
        edges = _build_skeleton_edges(coords)
        radii = np.ones(len(nodes))

        return Skeleton(nodes=nodes, edges=edges, radii=radii)


def _build_skeleton_edges(coords: np.ndarray) -> np.ndarray:
    """Build edges connecting skeleton voxels by 26-connectivity."""
    if len(coords) < 2:
        return np.zeros((0, 2), dtype=int)

    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    edges_set: set[tuple[int, int]] = set()

    # 26-connectivity: max distance is sqrt(3) ~= 1.73
    for i, pt in enumerate(coords):
        neighbors = tree.query_ball_point(pt, r=1.75)
        for j in neighbors:
            if j != i:
                edge = (min(i, j), max(i, j))
                edges_set.add(edge)

    if not edges_set:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(edges_set), dtype=int)
