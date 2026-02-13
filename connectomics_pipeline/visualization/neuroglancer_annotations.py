"""Generate Neuroglancer annotation JSON layers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import AssembledStructure

logger = get_logger("visualization.neuroglancer")


def generate_structure_annotations(
    structures: List[AssembledStructure],
    store: FragmentStore,
    output_dir: str | Path,
) -> None:
    """Generate Neuroglancer annotation JSON for assembled structures.

    Each structure gets a different color annotation layer.

    Args:
        structures: Assembled structures.
        store: Fragment store for coordinate data.
        output_dir: Output directory for JSON files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = _generate_colors(len(structures))

    all_annotations = []
    for i, structure in enumerate(structures):
        color = colors[i % len(colors)]
        for frag_id in structure.fragment_ids:
            frag = store.get(frag_id)
            if frag is None:
                continue
            # Point annotation at centroid (x, y, z order for Neuroglancer)
            all_annotations.append(
                {
                    "type": "point",
                    "point": [
                        float(frag.centroid[2]),
                        float(frag.centroid[1]),
                        float(frag.centroid[0]),
                    ],
                    "id": f"s{structure.structure_id}_f{frag_id}",
                    "props": {
                        "structure_id": structure.structure_id,
                        "fragment_id": frag_id,
                    },
                    "color": color,
                }
            )

    layer = {
        "type": "annotation",
        "annotations": all_annotations,
    }

    with open(output_dir / "structures.json", "w") as f:
        json.dump(layer, f, indent=2)

    logger.info(
        "Generated %d annotations for %d structures",
        len(all_annotations),
        len(structures),
    )


def _generate_colors(n: int) -> List[str]:
    """Generate n distinct hex colors."""
    if n == 0:
        return ["#ffffff"]
    colors = []
    for i in range(max(n, 1)):
        hue = i / max(n, 1)
        r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV to RGB."""
    import colorsys

    return colorsys.hsv_to_rgb(h, s, v)
