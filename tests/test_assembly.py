"""Tests for the assembly module."""

from __future__ import annotations

import numpy as np
import pytest

import networkx as nx

from connectomics_pipeline.assembly.assembler import Assembler
from connectomics_pipeline.assembly.confidence import compute_structure_confidence
from connectomics_pipeline.assembly.topology import check_topology, count_branch_order
from connectomics_pipeline.graph.builder import GraphBuilder
from connectomics_pipeline.utils.config import AssemblyConfig, GraphConfig
from connectomics_pipeline.utils.types import (
    CandidateConnection,
    ConnectionStatus,
    ValidationReport,
)


class TestTopology:
    def test_no_cycles(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3)])
        warnings, branches = check_topology(g)
        assert not any("cycle" in w.lower() for w in warnings)

    def test_cycle_detection(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        warnings, branches = check_topology(g)
        assert any("cycle" in w.lower() for w in warnings)

    def test_branch_points(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4)])
        warnings, branches = check_topology(g)
        assert branches == 1  # Node 1 has degree 3

    def test_branch_order(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (1, 3)])
        order = count_branch_order(g)
        assert order >= 1


class TestConfidence:
    def test_single_connection(self):
        candidate = CandidateConnection(
            candidate_id=0,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.zeros(3),
            endpoint_b=np.ones(3),
            composite_score=0.85,
        )
        conf = compute_structure_confidence([0], {0: candidate})
        assert abs(conf - 0.85) < 1e-6

    def test_weakest_link(self):
        c1 = CandidateConnection(
            candidate_id=0,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.zeros(3),
            endpoint_b=np.ones(3),
            composite_score=0.9,
        )
        c2 = CandidateConnection(
            candidate_id=1,
            fragment_a=1,
            fragment_b=2,
            endpoint_a=np.zeros(3),
            endpoint_b=np.ones(3),
            composite_score=0.6,
        )
        conf = compute_structure_confidence([0, 1], {0: c1, 1: c2})
        assert abs(conf - 0.6) < 1e-6

    def test_empty(self):
        assert compute_structure_confidence([], {}) == 0.0


class TestAssembler:
    def test_assemble_basic(self, fragment_store, sample_fragments):
        config = AssemblyConfig(min_structure_fragments=2)

        # Build a graph
        graph_config = GraphConfig(max_distance_nm=500.0)
        builder = GraphBuilder(graph_config)
        graph = builder.build(fragment_store)

        # Create a candidate connection between fragments 0 and 1
        candidate = CandidateConnection(
            candidate_id=0,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=sample_fragments[0].endpoints[1],
            endpoint_b=sample_fragments[1].endpoints[0],
            composite_score=0.85,
            status=ConnectionStatus.ACCEPTED,
        )

        report = ValidationReport(
            accepted=[0],
            rejected=[],
            ambiguous=[],
        )

        assembler = Assembler(config)
        structures = assembler.assemble(graph, [candidate], report, fragment_store)

        assert len(structures) >= 1
        # At least one structure should contain fragments 0 and 1
        found = any(0 in s.fragment_ids and 1 in s.fragment_ids for s in structures)
        assert found

    def test_assemble_skips_unknown_candidate_id(self, fragment_store, sample_fragments):
        """Line 57: candidate_id in report.accepted but absent from cand_map is skipped."""
        config = AssemblyConfig(min_structure_fragments=1)
        graph_config = GraphConfig(max_distance_nm=500.0)
        graph = GraphBuilder(graph_config).build(fragment_store)

        # report.accepted references cid=999 which does not exist in candidates list
        report = ValidationReport(accepted=[999], rejected=[], ambiguous=[])
        assembler = Assembler(config)
        structures = assembler.assemble(graph, [], report, fragment_store)
        # No valid edges → no structures
        assert structures == []

    def test_assemble_skips_small_components(self, fragment_store, sample_fragments):
        """Line 66: components below min_structure_fragments are skipped."""
        config = AssemblyConfig(min_structure_fragments=10)  # impossibly high
        graph_config = GraphConfig(max_distance_nm=500.0)
        graph = GraphBuilder(graph_config).build(fragment_store)

        candidate = CandidateConnection(
            candidate_id=0,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=sample_fragments[0].endpoints[1],
            endpoint_b=sample_fragments[1].endpoints[0],
            composite_score=0.85,
            status=ConnectionStatus.ACCEPTED,
        )
        report = ValidationReport(accepted=[0], rejected=[], ambiguous=[])
        assembler = Assembler(config)
        structures = assembler.assemble(graph, [candidate], report, fragment_store)
        assert structures == []


class TestTopologyMissingCoverage:
    def test_isolated_nodes_warning(self):
        """Line 35: isolated nodes trigger a warning."""
        g = nx.Graph()
        g.add_node(0)  # isolated
        g.add_node(1)  # isolated
        warnings, _ = check_topology(g)
        assert any("isolated" in w.lower() for w in warnings)

    def test_count_branch_order_empty_graph(self):
        """Line 51: count_branch_order returns 0 for an empty graph."""
        g = nx.Graph()
        assert count_branch_order(g) == 0


class TestConfidenceMissingCoverage:
    def test_all_cids_missing_from_map_returns_zero(self):
        """Line 36: if scores list is empty (all cids absent from map), return 0.0."""
        # cids [0, 1] don't exist in the empty map
        assert compute_structure_confidence([0, 1], {}) == 0.0
