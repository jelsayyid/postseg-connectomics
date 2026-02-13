"""Continuity-based scoring for candidate connections."""

from __future__ import annotations

import numpy as np

from connectomics_pipeline.utils.types import Fragment


def compute_continuity_score(
    frag_a: Fragment,
    frag_b: Fragment,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
) -> float:
    """Score based on path continuity between fragments.

    Estimates curvature of the hypothetical connecting path and penalizes
    sharp turns. A straight connection scores highest.

    Args:
        frag_a: Source fragment.
        frag_b: Target fragment.
        endpoint_a: Connection point on fragment A.
        endpoint_b: Connection point on fragment B.

    Returns:
        Score in [0, 1]. 1.0 = perfectly smooth, 0.0 = extreme curvature.
    """
    # Get approach directions at each endpoint
    dir_a = _approach_direction(frag_a, endpoint_a)
    dir_b = _approach_direction(frag_b, endpoint_b)

    # Connection direction
    connection_dir = endpoint_b - endpoint_a
    conn_norm = np.linalg.norm(connection_dir)
    if conn_norm < 1e-12:
        return 1.0
    connection_dir = connection_dir / conn_norm

    # Curvature at junction A: angle between fragment direction and connection
    cos_a = np.clip(np.dot(dir_a, connection_dir), -1.0, 1.0)
    angle_a = np.arccos(cos_a)

    # Curvature at junction B: angle between connection and fragment direction
    cos_b = np.clip(np.dot(-connection_dir, dir_b), -1.0, 1.0)
    angle_b = np.arccos(cos_b)

    # Average curvature penalty
    max_angle = np.pi
    avg_angle = (angle_a + angle_b) / 2.0
    score = 1.0 - (avg_angle / max_angle)
    return float(np.clip(score, 0.0, 1.0))


def _approach_direction(fragment: Fragment, endpoint: np.ndarray) -> np.ndarray:
    """Estimate the direction of approach at an endpoint."""
    if fragment.skeleton is not None and fragment.skeleton.num_nodes > 1:
        nodes = fragment.skeleton.nodes
        dists = np.linalg.norm(nodes - endpoint, axis=1)
        nearest_idx = int(np.argmin(dists))

        from connectomics_pipeline.utils.spatial import estimate_tangent

        return estimate_tangent(nodes, nearest_idx)

    direction = endpoint - fragment.centroid
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return direction / norm
