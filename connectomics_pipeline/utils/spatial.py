"""Spatial math utilities for the connectomics pipeline."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def estimate_tangent(nodes: np.ndarray, index: int, window: int = 3) -> np.ndarray:
    """Estimate tangent vector at a skeleton node using neighboring nodes.

    Args:
        nodes: (N, 3) array of skeleton node positions.
        index: Index of the node to estimate tangent at.
        window: Number of neighbors on each side to use.

    Returns:
        Unit tangent vector (3,).
    """
    n = len(nodes)
    start = max(0, index - window)
    end = min(n, index + window + 1)
    if end - start < 2:
        return np.array([1.0, 0.0, 0.0])
    segment = nodes[start:end]
    tangent = segment[-1] - segment[0]
    norm = np.linalg.norm(tangent)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return tangent / norm


def estimate_curvature(nodes: np.ndarray, index: int, window: int = 3) -> float:
    """Estimate curvature at a skeleton node.

    Uses the angle between consecutive tangent vectors as a curvature proxy.

    Args:
        nodes: (N, 3) array of skeleton node positions.
        index: Index of the node.
        window: Neighbor window for tangent estimation.

    Returns:
        Curvature estimate (radians). 0 = straight, pi = U-turn.
    """
    n = len(nodes)
    if index <= 0 or index >= n - 1:
        return 0.0
    t_before = estimate_tangent(nodes, max(0, index - 1), window)
    t_after = estimate_tangent(nodes, min(n - 1, index + 1), window)
    return angle_between(t_before, t_after)


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle between two vectors in radians.

    Returns:
        Angle in [0, pi].
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


class SpatialIndex:
    """KD-tree wrapper for spatial neighbor queries."""

    def __init__(self, points: np.ndarray):
        """Build spatial index from an array of points.

        Args:
            points: (N, 3) array of points.
        """
        self._points = np.asarray(points, dtype=float)
        self._tree = cKDTree(self._points) if len(self._points) > 0 else None

    @property
    def points(self) -> np.ndarray:
        return self._points

    def query_radius(self, point: np.ndarray, radius: float) -> list[int]:
        """Find all points within radius of a query point.

        Returns:
            List of indices into the original points array.
        """
        if self._tree is None:
            return []
        indices = self._tree.query_ball_point(np.asarray(point), radius)
        return list(indices)

    def query_nearest(self, point: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors.

        Returns:
            (distances, indices) arrays.
        """
        if self._tree is None:
            return np.array([]), np.array([], dtype=int)
        distances, indices = self._tree.query(np.asarray(point), k=k)
        return np.atleast_1d(distances), np.atleast_1d(indices)
