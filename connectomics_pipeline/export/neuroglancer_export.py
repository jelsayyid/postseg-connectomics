"""Neuroglancer precomputed annotation export (optional dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import AssembledStructure, CandidateConnection

logger = get_logger("export.neuroglancer")


def export_annotations(
    candidates: List[CandidateConnection],
    structures: List[AssembledStructure],
    output_dir: str | Path,
) -> None:
    """Export annotations as Neuroglancer-compatible JSON.

    Args:
        candidates: Candidate connections with status.
        structures: Assembled structures.
        output_dir: Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Connection annotations
    annotations = []
    for c in candidates:
        # Neuroglancer uses x, y, z order
        point_a = [float(c.endpoint_a[2]), float(c.endpoint_a[1]), float(c.endpoint_a[0])]
        point_b = [float(c.endpoint_b[2]), float(c.endpoint_b[1]), float(c.endpoint_b[0])]

        color = _status_color(c.status.value)

        annotations.append(
            {
                "type": "line",
                "pointA": point_a,
                "pointB": point_b,
                "id": str(c.candidate_id),
                "props": {
                    "status": c.status.value,
                    "composite_score": round(c.composite_score, 3),
                    "fragment_a": c.fragment_a,
                    "fragment_b": c.fragment_b,
                },
                "color": color,
            }
        )

    annotation_layer = {
        "type": "annotation",
        "annotations": annotations,
    }

    with open(output_dir / "connections.json", "w") as f:
        json.dump(annotation_layer, f, indent=2)

    logger.info("Exported %d annotations to %s", len(annotations), output_dir)


def _status_color(status: str) -> str:
    """Map connection status to hex color."""
    colors = {
        "accepted": "#00ff00",
        "rejected": "#ff0000",
        "ambiguous": "#ffff00",
        "proposed": "#808080",
    }
    return colors.get(status, "#ffffff")
