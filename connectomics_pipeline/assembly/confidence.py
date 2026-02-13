"""Confidence computation for assembled structures."""

from __future__ import annotations

from typing import Dict, List

from connectomics_pipeline.utils.types import CandidateConnection


def compute_structure_confidence(
    accepted_cids: List[int],
    candidate_map: Dict[int, CandidateConnection],
) -> float:
    """Compute confidence for an assembled structure.

    Uses the minimum composite score across all accepted connections
    as the structure confidence (weakest-link principle).

    Args:
        accepted_cids: IDs of accepted connections in this structure.
        candidate_map: Mapping from candidate_id to CandidateConnection.

    Returns:
        Structure confidence in [0, 1].
    """
    if not accepted_cids:
        return 0.0

    scores = []
    for cid in accepted_cids:
        c = candidate_map.get(cid)
        if c is not None:
            scores.append(c.composite_score)

    if not scores:
        return 0.0

    return min(scores)
