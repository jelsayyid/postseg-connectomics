"""Spatial indexing for fragment endpoints and skeleton nodes."""

from __future__ import annotations

from typing import Dict, List, Tuple

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


class SkeletonNodeIndex:
    """KD-tree index over ALL skeleton nodes for interior-split detection.

    Unlike EndpointIndex (which only indexes degree-1 TEASAR tips), this indexes
    every skeleton node so that splits occurring in the interior of long axons
    can be detected.  Falls back to endpoint positions for fragments without
    a skeleton (e.g. those processed via the PCA endpoint path).
    """

    def __init__(self, fragments: List[Fragment]):
        points: List[np.ndarray] = []
        frag_ids: List[int] = []

        for f in fragments:
            if f.skeleton is not None and f.skeleton.num_nodes > 0:
                for node in f.skeleton.nodes:
                    points.append(node)
                    frag_ids.append(f.fragment_id)
            else:
                # PCA-endpoint fragments: use their endpoints as proxy nodes
                for ep in f.endpoints:
                    points.append(ep)
                    frag_ids.append(f.fragment_id)

        self._frag_ids: List[int] = frag_ids
        if points:
            self._coords = np.array(points)
            self._tree = cKDTree(self._coords)
        else:
            self._coords = np.zeros((0, 3))
            self._tree = None

    def query_radius_batch(self, points: np.ndarray, radius: float) -> List[List[int]]:
        """Batch radius query — returns raw index lists into the internal array.

        Args:
            points: (N, 3) array of query positions.
            radius: Search radius in the same units as the node coordinates.

        Returns:
            List of N lists, each containing indices into self._coords /
            self._frag_ids for nodes within radius of the corresponding query.
        """
        if self._tree is None or len(points) == 0:
            return [[] for _ in range(len(points))]
        return self._tree.query_ball_point(np.asarray(points), radius)

    def frag_id_at(self, index: int) -> int:
        return self._frag_ids[index]

    def coord_at(self, index: int) -> np.ndarray:
        return self._coords[index]
