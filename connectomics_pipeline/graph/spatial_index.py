"""Spatial indexing for fragment endpoints."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from connectomics_pipeline.utils.types import Fragment


class EndpointIndex:
    """KD-tree index over all fragment endpoints for neighbor queries."""

    def __init__(self, fragments: List[Fragment]):
        self._points: List[np.ndarray] = []
        self._frag_ids: List[int] = []

        for f in fragments:
            for ep in f.endpoints:
                self._points.append(ep)
                self._frag_ids.append(f.fragment_id)

        if self._points:
            self._coords = np.array(self._points)
            self._tree = cKDTree(self._coords)
        else:
            self._coords = np.zeros((0, 3))
            self._tree = None

    def query_radius(self, point: np.ndarray, radius: float) -> List[Tuple[int, np.ndarray, float]]:
        """Find all endpoints within radius.

        Returns:
            List of (fragment_id, endpoint_coords, distance) tuples.
        """
        if self._tree is None:
            return []
        indices = self._tree.query_ball_point(np.asarray(point), radius)
        results = []
        for i in indices:
            dist = float(np.linalg.norm(self._coords[i] - point))
            results.append((self._frag_ids[i], self._coords[i], dist))
        return results

    def query_nearest(self, point: np.ndarray, k: int = 1) -> List[Tuple[int, np.ndarray, float]]:
        """Find k nearest endpoints.

        Returns:
            List of (fragment_id, endpoint_coords, distance) tuples.
        """
        if self._tree is None:
            return []
        distances, indices = self._tree.query(np.asarray(point), k=k)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)
        results = []
        for d, i in zip(distances, indices):
            if i < len(self._frag_ids):
                results.append((self._frag_ids[i], self._coords[i], float(d)))
        return results
