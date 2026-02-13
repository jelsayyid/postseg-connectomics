"""Fragment storage with spatial indexing."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy.spatial import cKDTree

from connectomics_pipeline.utils.types import BoundingBox, Fragment


class FragmentStore:
    """Dictionary-based fragment storage with KD-tree spatial index."""

    def __init__(self) -> None:
        self._fragments: dict[int, Fragment] = {}
        self._index: Optional[cKDTree] = None
        self._index_ids: list[int] = []

    def __len__(self) -> int:
        return len(self._fragments)

    def __contains__(self, fragment_id: int) -> bool:
        return fragment_id in self._fragments

    def __getitem__(self, fragment_id: int) -> Fragment:
        return self._fragments[fragment_id]

    def get(self, fragment_id: int) -> Optional[Fragment]:
        return self._fragments.get(fragment_id)

    def add(self, fragment: Fragment) -> None:
        """Add a fragment and rebuild the spatial index."""
        self._fragments[fragment.fragment_id] = fragment
        self._rebuild_index()

    def add_many(self, fragments: List[Fragment]) -> None:
        """Add multiple fragments and rebuild the spatial index once."""
        for f in fragments:
            self._fragments[f.fragment_id] = f
        self._rebuild_index()

    def remove(self, fragment_id: int) -> None:
        """Remove a fragment and rebuild index."""
        self._fragments.pop(fragment_id, None)
        self._rebuild_index()

    def all_fragments(self) -> List[Fragment]:
        """Return all stored fragments."""
        return list(self._fragments.values())

    def all_ids(self) -> List[int]:
        """Return all fragment IDs."""
        return list(self._fragments.keys())

    def query_radius(self, point: np.ndarray, radius: float) -> List[Fragment]:
        """Find all fragments whose centroids are within radius of a point.

        Args:
            point: Query point in physical coordinates (z, y, x).
            radius: Search radius in nm.

        Returns:
            List of matching fragments.
        """
        if self._index is None or len(self._index_ids) == 0:
            return []
        indices = self._index.query_ball_point(np.asarray(point), radius)
        return [self._fragments[self._index_ids[i]] for i in indices]

    def query_bbox(self, bbox: BoundingBox) -> List[Fragment]:
        """Find all fragments whose centroids fall within a bounding box.

        Args:
            bbox: Query bounding box.

        Returns:
            List of matching fragments.
        """
        results = []
        for frag in self._fragments.values():
            c = frag.centroid
            if np.all(c >= bbox.min_corner) and np.all(c <= bbox.max_corner):
                results.append(frag)
        return results

    def _rebuild_index(self) -> None:
        """Rebuild the KD-tree spatial index from fragment centroids."""
        if not self._fragments:
            self._index = None
            self._index_ids = []
            return

        self._index_ids = list(self._fragments.keys())
        centroids = np.array([self._fragments[fid].centroid for fid in self._index_ids])
        self._index = cKDTree(centroids)
