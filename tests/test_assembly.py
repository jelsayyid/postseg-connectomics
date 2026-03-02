"""Tests for the assembly module."""

from __future__ import annotations

import numpy as np
import pytest

import networkx as nx

from connectomics_pipeline.assembly.assembler import Assembler, _apply_partner_limit
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


def _make_cand(cid, fa, fb, score):
    """Helper: create a minimal CandidateConnection for partner-limit tests."""
    return CandidateConnection(
        candidate_id=cid,
        fragment_a=fa,
        fragment_b=fb,
        endpoint_a=np.zeros(3),
        endpoint_b=np.ones(3),
        composite_score=score,
        status=ConnectionStatus.ACCEPTED,
    )


class TestPartnerLimit:
    """Unit tests for _apply_partner_limit (AND semantics)."""

    def test_empty_accepted_returns_empty(self):
        """Empty accepted list → empty result."""
        assert _apply_partner_limit([], {}, max_partners=2) == []

    def test_within_limit_no_change(self):
        """All fragments have ≤ max_partners accepted — nothing filtered."""
        c0 = _make_cand(0, fa=1, fb=2, score=0.9)
        c1 = _make_cand(1, fa=3, fb=4, score=0.8)
        cand_map = {0: c0, 1: c1}
        result = _apply_partner_limit([0, 1], cand_map, max_partners=2)
        assert set(result) == {0, 1}

    def test_excess_partners_filtered_by_score(self):
        """Fragment 1 has 3 partners; max=2 → lowest-scoring candidate dropped."""
        # frag 1 → [frag 2 (score=0.9), frag 3 (score=0.7), frag 4 (score=0.5)]
        # frag 2 → [frag 1 (score=0.9)]
        # frag 3 → [frag 1 (score=0.7)]
        # frag 4 → [frag 1 (score=0.5)]
        # With max=2, frag 1 keeps top-2: cids 0 (0.9) and 1 (0.7)
        # cid 2 (score=0.5) dropped because frag 1 doesn't include it in top-2
        c0 = _make_cand(0, fa=1, fb=2, score=0.9)
        c1 = _make_cand(1, fa=1, fb=3, score=0.7)
        c2 = _make_cand(2, fa=1, fb=4, score=0.5)
        cand_map = {0: c0, 1: c1, 2: c2}
        result = _apply_partner_limit([0, 1, 2], cand_map, max_partners=2)
        assert set(result) == {0, 1}
        assert 2 not in result

    def test_and_semantics_mutual_agreement_required(self):
        """AND semantics: cid kept only if BOTH fragments include it in top-N.

        Fragment A prefers [B (0.9), C (0.8)]; fragment B prefers [D (0.95), A (0.9)].
        Candidate A→C: A includes it (top-2), but C doesn't (C's only partner is A,
        so C includes it). Verify that the logic correctly keeps mutual top-N pairs.
        """
        # A→B: score 0.9  (A ranks B #1, B ranks A #2 → mutual top-2 ✓)
        # A→C: score 0.8  (A ranks C #2, C only has A → C ranks A #1 ✓)
        # B→D: score 0.95 (B ranks D #1, D only has B → D ranks B #1 ✓)
        # A→E: score 0.3  (A ranks E #3 — not in top-2 → dropped)
        c_ab = _make_cand(0, fa=10, fb=20, score=0.9)
        c_ac = _make_cand(1, fa=10, fb=30, score=0.8)
        c_bd = _make_cand(2, fa=20, fb=40, score=0.95)
        c_ae = _make_cand(3, fa=10, fb=50, score=0.3)
        cand_map = {0: c_ab, 1: c_ac, 2: c_bd, 3: c_ae}
        result = _apply_partner_limit([0, 1, 2, 3], cand_map, max_partners=2)
        # A→E (cid 3) dropped: score 0.3 puts it at rank 3 for fragment A
        assert 3 not in result
        # A→B (cid 0) and A→C (cid 1) kept: both in A's top-2
        assert 0 in result
        assert 1 in result
        # B→D (cid 2) kept: only two partners for B (A and D); D is top-1
        assert 2 in result

    def test_and_semantics_drops_non_mutual(self):
        """Candidate where one fragment doesn't include it in top-N is dropped."""
        # Fragment A: top-2 partners are B (0.9) and C (0.8)
        # Fragment D: top-2 partners are E (0.95) and F (0.85) — A is NOT in D's top-2
        # Candidate A→D (score 0.7): A ranks D at #3 → dropped from A's perspective
        c_ab = _make_cand(0, fa=1, fb=2, score=0.9)
        c_ac = _make_cand(1, fa=1, fb=3, score=0.8)
        c_ad = _make_cand(2, fa=1, fb=4, score=0.7)  # A's #3 — below max_partners=2
        c_de = _make_cand(3, fa=4, fb=5, score=0.95)
        c_df = _make_cand(4, fa=4, fb=6, score=0.85)
        cand_map = {0: c_ab, 1: c_ac, 2: c_ad, 3: c_de, 4: c_df}
        result = _apply_partner_limit([0, 1, 2, 3, 4], cand_map, max_partners=2)
        # A→D (cid 2) dropped: A's rank of D is #3 (below top-2)
        assert 2 not in result
        # Other candidates kept
        assert 0 in result and 1 in result and 3 in result and 4 in result

    def test_unmapped_candidates_pass_through(self):
        """Candidates absent from cand_map pass through unchanged."""
        # cid 99 not in cand_map → kept as-is
        result = _apply_partner_limit([99], {}, max_partners=2)
        assert result == [99]

    def test_assembler_applies_partner_limit(self, fragment_store, sample_fragments):
        """Assembler with max_partners_per_fragment=1 keeps only the best partner."""
        config = AssemblyConfig(min_structure_fragments=2, max_partners_per_fragment=1)
        graph_config = GraphConfig(max_distance_nm=500.0)
        graph = GraphBuilder(graph_config).build(fragment_store)

        # Fragment 0 is offered two partners: 1 (score=0.9) and 2 (score=0.5)
        c0 = CandidateConnection(
            candidate_id=0,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=sample_fragments[0].endpoints[1],
            endpoint_b=sample_fragments[1].endpoints[0],
            composite_score=0.9,
            status=ConnectionStatus.ACCEPTED,
        )
        c1 = CandidateConnection(
            candidate_id=1,
            fragment_a=0,
            fragment_b=2,
            endpoint_a=sample_fragments[0].endpoints[1],
            endpoint_b=sample_fragments[2].endpoints[0] if len(sample_fragments) > 2 else np.zeros(3),
            composite_score=0.5,
            status=ConnectionStatus.ACCEPTED,
        )

        report = ValidationReport(accepted=[0, 1], rejected=[], ambiguous=[])
        assembler = Assembler(config)
        assembler.assemble(graph, [c0, c1], report, fragment_store)

        # After partner limit: c1 (score=0.5) should be re-rejected because fragment 0
        # keeps only top-1 (c0, score=0.9).  c1's status must be REJECTED.
        assert c0.status == ConnectionStatus.ACCEPTED
        assert c1.status == ConnectionStatus.REJECTED
