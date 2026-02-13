"""Fragment metadata computation."""

from __future__ import annotations

import numpy as np

from connectomics_pipeline.utils.types import Fragment


def compute_endpoints(fragment: Fragment) -> list[np.ndarray]:
    """Compute endpoints from skeleton terminal nodes.

    Terminal nodes are nodes with degree 1 (only one edge).

    Args:
        fragment: Fragment with skeleton attached.

    Returns:
        List of endpoint coordinates in physical space.
    """
    if fragment.skeleton is None or fragment.skeleton.num_nodes == 0:
        return [fragment.centroid.copy()]

    skeleton = fragment.skeleton
    if skeleton.num_edges == 0:
        return [skeleton.nodes[0].copy()]

    # Count degree of each node
    degree = np.zeros(skeleton.num_nodes, dtype=int)
    for e in skeleton.edges:
        degree[e[0]] += 1
        degree[e[1]] += 1

    terminal_mask = degree == 1
    if not np.any(terminal_mask):
        # No terminals (cycle?), use first and last nodes
        return [skeleton.nodes[0].copy(), skeleton.nodes[-1].copy()]

    return [skeleton.nodes[i].copy() for i in np.where(terminal_mask)[0]]


def refine_centroid(fragment: Fragment) -> np.ndarray:
    """Refine fragment centroid using skeleton midpoint if available.

    Args:
        fragment: Fragment with optional skeleton.

    Returns:
        Refined centroid in physical coordinates.
    """
    if fragment.skeleton is None or fragment.skeleton.num_nodes == 0:
        return fragment.centroid.copy()

    return fragment.skeleton.nodes.mean(axis=0)
