"""Alignment-based scoring for candidate connections."""

from __future__ import annotations

import numpy as np

from connectomics_pipeline.utils.types import Fragment


def compute_alignment_score(
    frag_a: Fragment,
    frag_b: Fragment,
    endpoint_a: np.ndarray,
    endpoint_b: np.ndarray,
) -> float:
    """Score based on tangent vector alignment at connection endpoints.

    Measures how well the fragments' directions align at the connection point.
    Parallel fragments pointing towards each other score highest.

    Args:
        frag_a: Source fragment.
        frag_b: Target fragment.
        endpoint_a: Connection point on fragment A.
        endpoint_b: Connection point on fragment B.

    Returns:
        Score in [0, 1]. 1.0 = perfectly aligned, 0.0 = perpendicular or worse.
    """
    tangent_a = _estimate_endpoint_tangent(frag_a, endpoint_a)
    tangent_b = _estimate_endpoint_tangent(frag_b, endpoint_b)

    # Connection direction from A to B
    connection_dir = endpoint_b - endpoint_a
    conn_norm = np.linalg.norm(connection_dir)
    if conn_norm < 1e-12:
        return 1.0
    connection_dir = connection_dir / conn_norm

    # Fragment A's tangent should point toward B (positive dot with connection_dir)
    dot_a = np.dot(tangent_a, connection_dir)
    # Fragment B's tangent should point away from A (negative dot with connection_dir)
    dot_b = np.dot(tangent_b, -connection_dir)

    # Average alignment, mapped to [0, 1]
    alignment = (dot_a + dot_b) / 2.0
    return float(np.clip((alignment + 1.0) / 2.0, 0.0, 1.0))


def _estimate_endpoint_tangent(fragment: Fragment, endpoint: np.ndarray) -> np.ndarray:
    """Estimate tangent direction at an endpoint."""
    if fragment.skeleton is not None and fragment.skeleton.num_nodes > 1:
        nodes = fragment.skeleton.nodes
        # Find nearest skeleton node to endpoint
        dists = np.linalg.norm(nodes - endpoint, axis=1)
        nearest_idx = int(np.argmin(dists))

        from connectomics_pipeline.utils.spatial import estimate_tangent

        return estimate_tangent(nodes, nearest_idx)

    # Fallback: direction from centroid to endpoint
    direction = endpoint - fragment.centroid
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return direction / norm
