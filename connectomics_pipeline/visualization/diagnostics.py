"""Diagnostic summary statistics for pipeline results."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.types import (
    AssembledStructure,
    CandidateConnection,
    ValidationReport,
)


def fragment_statistics(store: FragmentStore) -> Dict:
    """Compute summary statistics for fragments.

    Returns:
        Dict with count, size distribution, etc.
    """
    fragments = store.all_fragments()
    if not fragments:
        return {"count": 0}

    sizes = np.array([f.voxel_count for f in fragments])
    return {
        "count": len(fragments),
        "total_voxels": int(sizes.sum()),
        "mean_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "std_size": float(sizes.std()),
        "boundary_count": sum(1 for f in fragments if f.is_boundary),
        "with_skeleton": sum(1 for f in fragments if f.skeleton is not None),
    }


def connection_statistics(
    candidates: List[CandidateConnection],
    report: ValidationReport,
) -> Dict:
    """Compute summary statistics for connections.

    Returns:
        Dict with counts, score distributions, etc.
    """
    if not candidates:
        return {"total": 0}

    scores = np.array([c.composite_score for c in candidates])
    distances = np.array([c.gap_distance for c in candidates])

    return {
        "total": len(candidates),
        "accepted": len(report.accepted),
        "rejected": len(report.rejected),
        "ambiguous": len(report.ambiguous),
        "accept_rate": len(report.accepted) / max(len(candidates), 1),
        "mean_composite_score": float(scores.mean()),
        "median_composite_score": float(np.median(scores)),
        "mean_gap_distance": float(distances.mean()),
        "median_gap_distance": float(np.median(distances)),
    }


def structure_statistics(structures: List[AssembledStructure]) -> Dict:
    """Compute summary statistics for assembled structures.

    Returns:
        Dict with counts, size distributions, warnings, etc.
    """
    if not structures:
        return {"count": 0}

    frag_counts = np.array([len(s.fragment_ids) for s in structures])
    confidences = np.array([s.confidence for s in structures])

    return {
        "count": len(structures),
        "mean_fragments": float(frag_counts.mean()),
        "max_fragments": int(frag_counts.max()),
        "total_fragments": int(frag_counts.sum()),
        "mean_confidence": float(confidences.mean()),
        "min_confidence": float(confidences.min()),
        "with_ambiguous": sum(1 for s in structures if s.has_ambiguous_regions),
        "with_warnings": sum(1 for s in structures if s.topology_warnings),
        "total_path_length": float(sum(s.total_path_length for s in structures)),
    }
