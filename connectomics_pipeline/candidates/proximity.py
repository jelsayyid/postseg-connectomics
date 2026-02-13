"""Proximity-based scoring for candidate connections."""

from __future__ import annotations

import numpy as np


def compute_proximity_score(distance: float, max_distance: float) -> float:
    """Score based on distance between endpoints.

    1.0 at distance=0, decaying to 0.0 at max_distance.
    Uses exponential decay for smooth falloff.

    Args:
        distance: Gap distance in nm.
        max_distance: Maximum endpoint distance in nm.

    Returns:
        Score in [0, 1].
    """
    if max_distance <= 0:
        return 0.0
    if distance <= 0:
        return 1.0
    if distance >= max_distance:
        return 0.0
    # Exponential decay: score = exp(-3 * d/max_d) gives ~0.05 at d=max_d
    return float(np.exp(-3.0 * distance / max_distance))
