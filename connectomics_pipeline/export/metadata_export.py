"""CSV export for fragment metadata, connections, and structure summaries."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import (
    AssembledStructure,
    CandidateConnection,
    ValidationReport,
)

logger = get_logger("export.metadata")


def export_fragment_metadata(store: FragmentStore, path: str | Path) -> None:
    """Export fragment metadata to CSV.

    Args:
        store: Fragment store.
        path: Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for frag in store.all_fragments():
        rows.append(
            {
                "fragment_id": frag.fragment_id,
                "label_id": frag.label_id,
                "voxel_count": frag.voxel_count,
                "centroid_z": frag.centroid[0],
                "centroid_y": frag.centroid[1],
                "centroid_x": frag.centroid[2],
                "bbox_min_z": frag.bounding_box.min_corner[0],
                "bbox_min_y": frag.bounding_box.min_corner[1],
                "bbox_min_x": frag.bounding_box.min_corner[2],
                "bbox_max_z": frag.bounding_box.max_corner[0],
                "bbox_max_y": frag.bounding_box.max_corner[1],
                "bbox_max_x": frag.bounding_box.max_corner[2],
                "is_boundary": frag.is_boundary,
                "has_skeleton": frag.skeleton is not None,
                "num_endpoints": len(frag.endpoints),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Exported %d fragment metadata records to %s", len(rows), path)


def export_connection_decisions(
    candidates: List[CandidateConnection],
    report: ValidationReport,
    path: str | Path,
) -> None:
    """Export connection decisions to CSV.

    Args:
        candidates: All candidate connections.
        report: Validation report.
        path: Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in candidates:
        rows.append(
            {
                "candidate_id": c.candidate_id,
                "fragment_a": c.fragment_a,
                "fragment_b": c.fragment_b,
                "gap_distance": c.gap_distance,
                "proximity_score": c.proximity_score,
                "alignment_score": c.alignment_score,
                "continuity_score": c.continuity_score,
                "size_score": c.size_score,
                "composite_score": c.composite_score,
                "status": c.status.value,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Exported %d connection decisions to %s", len(rows), path)


def export_structure_summaries(
    structures: List[AssembledStructure],
    path: str | Path,
) -> None:
    """Export structure summaries to CSV.

    Args:
        structures: Assembled structures.
        path: Output CSV file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in structures:
        rows.append(
            {
                "structure_id": s.structure_id,
                "num_fragments": len(s.fragment_ids),
                "num_accepted_connections": len(s.accepted_connections),
                "num_ambiguous_connections": len(s.ambiguous_connections),
                "confidence": s.confidence,
                "total_path_length": s.total_path_length,
                "num_branches": s.num_branches,
                "has_ambiguous_regions": s.has_ambiguous_regions,
                "num_warnings": len(s.topology_warnings),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Exported %d structure summaries to %s", len(rows), path)
