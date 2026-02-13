"""Assembly of validated connections into neuron structures."""

from __future__ import annotations

from typing import Dict, List, Optional

import networkx as nx

from connectomics_pipeline.assembly.confidence import compute_structure_confidence
from connectomics_pipeline.assembly.topology import check_topology
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.utils.config import AssemblyConfig
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import (
    AssembledStructure,
    CandidateConnection,
    ConnectionStatus,
    ValidationReport,
)

logger = get_logger("assembly.assembler")


class Assembler:
    """Assemble validated connections into reconstructed structures."""

    def __init__(self, config: AssemblyConfig):
        self.config = config

    def assemble(
        self,
        graph: FragmentGraph,
        candidates: List[CandidateConnection],
        report: ValidationReport,
        store: FragmentStore,
    ) -> List[AssembledStructure]:
        """Build assembled structures from accepted connections.

        Args:
            graph: Fragment adjacency graph.
            candidates: All candidate connections.
            report: Validation report with decisions.
            store: Fragment store.

        Returns:
            List of AssembledStructure objects.
        """
        # Build mapping from candidate_id to candidate
        cand_map: Dict[int, CandidateConnection] = {c.candidate_id: c for c in candidates}

        # Build assembly graph from accepted connections only
        assembly_graph = nx.Graph()
        for cid in report.accepted:
            c = cand_map.get(cid)
            if c is None:
                continue
            assembly_graph.add_edge(c.fragment_a, c.fragment_b, candidate_id=cid)

        # Find connected components
        structures: List[AssembledStructure] = []
        for sid, component in enumerate(nx.connected_components(assembly_graph)):
            fragment_ids = sorted(component)

            if len(fragment_ids) < self.config.min_structure_fragments:
                continue

            # Collect accepted connections within this component
            subg = assembly_graph.subgraph(component)
            accepted_cids = [subg.edges[e]["candidate_id"] for e in subg.edges()]

            # Find nearby ambiguous connections
            ambiguous_cids: List[int] = []
            if self.config.flag_ambiguous:
                for cid in report.ambiguous:
                    c = cand_map.get(cid)
                    if c and (c.fragment_a in component or c.fragment_b in component):
                        ambiguous_cids.append(cid)

            # Compute metrics
            confidence = compute_structure_confidence(accepted_cids, cand_map)
            topology_warnings = []
            num_branches = 0

            if self.config.detect_cycles:
                warnings, branches = check_topology(subg)
                topology_warnings = warnings
                num_branches = branches

            # Path length
            total_path_length = 0.0
            for fid in fragment_ids:
                frag = store.get(fid)
                if frag and frag.skeleton:
                    total_path_length += frag.skeleton.path_length()

            structure = AssembledStructure(
                structure_id=sid,
                fragment_ids=fragment_ids,
                accepted_connections=accepted_cids,
                ambiguous_connections=ambiguous_cids,
                confidence=confidence,
                total_path_length=total_path_length,
                num_branches=num_branches,
                has_ambiguous_regions=len(ambiguous_cids) > 0,
                topology_warnings=topology_warnings,
            )
            structures.append(structure)

        logger.info(
            "Assembled %d structures from %d accepted connections",
            len(structures),
            len(report.accepted),
        )
        return structures
