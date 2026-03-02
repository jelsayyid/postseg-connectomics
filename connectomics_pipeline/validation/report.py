"""Helpers for building validation reports."""

from __future__ import annotations

from typing import Dict, List

from connectomics_pipeline.utils.types import (
    CandidateConnection,
    ConnectionStatus,
    ValidationReport,
    ValidationResult,
)


def build_report(
    results: Dict[int, List[ValidationResult]],
    candidates: List[CandidateConnection],
    accept_threshold: float = 0.8,
    reject_threshold: float = 0.3,
    long_range_distance_nm: float = 0.0,
    long_range_accept_threshold: float = 0.0,
) -> ValidationReport:
    """Build a ValidationReport from per-candidate results.

    Args:
        results: Mapping from candidate_id to list of rule results.
        candidates: All candidate connections.
        accept_threshold: Minimum mean confidence to accept.
        reject_threshold: Maximum mean confidence to reject.
        long_range_distance_nm: Gap distance above which long_range_accept_threshold
            applies.  0 disables the feature.
        long_range_accept_threshold: Accept threshold for long-range pairs (gap >
            long_range_distance_nm).  0 disables the feature.

    Returns:
        Populated ValidationReport.
    """
    use_long_range = long_range_distance_nm > 0 and long_range_accept_threshold > 0
    report = ValidationReport(results=results)

    for candidate in candidates:
        cid = candidate.candidate_id
        rule_results = results.get(cid, [])

        if not rule_results:
            report.ambiguous.append(cid)
            candidate.status = ConnectionStatus.AMBIGUOUS
            continue

        # Any hard REJECT -> REJECT
        has_reject = any(r.decision == ConnectionStatus.REJECTED for r in rule_results)
        if has_reject:
            report.rejected.append(cid)
            candidate.status = ConnectionStatus.REJECTED
            continue

        # Distance-conditioned accept threshold for long-range pairs
        if use_long_range and candidate.gap_distance > long_range_distance_nm:
            effective_accept = long_range_accept_threshold
        else:
            effective_accept = accept_threshold

        # All pass with high enough confidence -> ACCEPT
        mean_confidence = sum(r.confidence for r in rule_results) / len(rule_results)
        if mean_confidence >= effective_accept:
            report.accepted.append(cid)
            candidate.status = ConnectionStatus.ACCEPTED
        elif mean_confidence <= reject_threshold:
            report.rejected.append(cid)
            candidate.status = ConnectionStatus.REJECTED
        else:
            report.ambiguous.append(cid)
            candidate.status = ConnectionStatus.AMBIGUOUS

    return report
