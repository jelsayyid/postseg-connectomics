"""Proximity-based scoring for candidate connections."""

from __future__ import annotations

import numpy as np


def compute_proximity_score(distance: float, max_distance: float) -> float:
    """Score based on distance between endpoints.

    1.0 at distance=0, decaying toward zero beyond max_distance.
    Uses exponential decay for smooth falloff: exp(-3 * d/max_d).
    At d=max_d the score is exp(-3) ≈ 0.05; it continues to decay for d > max_d.

    Args:
        distance: Gap distance in nm.
        max_distance: Reference distance for decay rate in nm.

    Returns:
        Score in (0, 1].
    """
    if max_distance <= 0:
        return 0.0
    if distance <= 0:
        return 1.0
    # Exponential decay: score = exp(-3 * d/max_d) gives ~0.05 at d=max_d
    return float(np.exp(-3.0 * distance / max_distance))
