"""Export assembled structures to SWC neuron morphology format."""

from __future__ import annotations

from pathlib import Path
from typing import List

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import AssembledStructure

logger = get_logger("export.swc")

# SWC type codes
SWC_UNDEFINED = 0
SWC_SOMA = 1
SWC_AXON = 2
SWC_DENDRITE = 3
SWC_APICAL = 4


def export_swc(
    structure: AssembledStructure,
    store: FragmentStore,
    path: str | Path,
    structure_type: int = SWC_AXON,
) -> None:
    """Export an assembled structure to SWC format.

    SWC format: id type x y z radius parent_id
    Coordinates are in physical units (nm).

    Args:
        structure: Assembled structure to export.
        store: Fragment store for skeleton data.
        path: Output file path.
        structure_type: SWC type code for nodes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append(f"# SWC export for structure {structure.structure_id}")
    lines.append(f"# Fragments: {len(structure.fragment_ids)}")
    lines.append(f"# Confidence: {structure.confidence:.3f}")
    lines.append(f"# Path length: {structure.total_path_length:.1f} nm")

    swc_id = 1
    node_id_map: dict[tuple[int, int], int] = {}  # (frag_id, node_idx) -> swc_id

    for frag_id in structure.fragment_ids:
        frag = store.get(frag_id)
        if frag is None or frag.skeleton is None:
            continue

        skel = frag.skeleton
        if skel.num_nodes == 0:
            continue

        # Map skeleton nodes to SWC IDs
        frag_start_id = swc_id
        for i in range(skel.num_nodes):
            node_id_map[(frag_id, i)] = swc_id
            swc_id += 1

        # Write nodes
        for i in range(skel.num_nodes):
            x, y, z = skel.nodes[i][2], skel.nodes[i][1], skel.nodes[i][0]  # ZYX -> XYZ
            radius = skel.radii[i] if i < len(skel.radii) else 1.0

            # Find parent from edges
            parent = -1
            for edge in skel.edges:
                if edge[1] == i:
                    parent = node_id_map.get((frag_id, edge[0]), -1)
                    break

            if parent == -1 and i > 0:
                # Default: connect to first node of this fragment
                parent = frag_start_id

            sid = node_id_map[(frag_id, i)]
            lines.append(f"{sid} {structure_type} {x:.3f} {y:.3f} {z:.3f} {radius:.3f} {parent}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Exported structure %d to %s", structure.structure_id, path)
