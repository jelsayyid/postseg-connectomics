"""Composite scoring for candidate connections."""

from __future__ import annotations

from typing import Dict


def compute_composite_score(
    proximity: float,
    alignment: float,
    continuity: float,
    size: float,
    weights: Dict[str, float],
) -> float:
    """Compute weighted composite score from individual factors.

    Args:
        proximity: Proximity score [0, 1].
        alignment: Alignment score [0, 1].
        continuity: Continuity score [0, 1].
        size: Size compatibility score [0, 1].
        weights: Weight dict with keys 'proximity', 'alignment', 'continuity', 'size'.

    Returns:
        Composite score in [0, 1].
    """
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0

    score = (
        weights.get("proximity", 0.35) * proximity
        + weights.get("alignment", 0.30) * alignment
        + weights.get("continuity", 0.25) * continuity
        + weights.get("size", 0.10) * size
    ) / total_weight

    return max(0.0, min(1.0, score))


def compute_size_score(radius_a: float, radius_b: float, max_ratio: float = 3.0) -> float:
    """Score based on size compatibility between two fragments.

    Args:
        radius_a: Estimated radius of fragment A.
        radius_b: Estimated radius of fragment B.
        max_ratio: Maximum acceptable radius ratio.

    Returns:
        Score in [0, 1]. 1.0 = same size, 0.0 = ratio >= max_ratio.
    """
    if radius_a <= 0 or radius_b <= 0:
        return 0.5
    ratio = max(radius_a, radius_b) / min(radius_a, radius_b)
    if ratio >= max_ratio:
        return 0.0
    return 1.0 - (ratio - 1.0) / (max_ratio - 1.0)
