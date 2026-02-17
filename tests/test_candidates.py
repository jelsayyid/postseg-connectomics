"""Tests for the candidates module."""

from __future__ import annotations

import numpy as np

from connectomics_pipeline.candidates.alignment import compute_alignment_score
from connectomics_pipeline.candidates.continuity import compute_continuity_score
from connectomics_pipeline.candidates.generator import CandidateGenerator
from connectomics_pipeline.candidates.proximity import compute_proximity_score
from connectomics_pipeline.candidates.scoring import compute_composite_score, compute_size_score
from connectomics_pipeline.graph.builder import GraphBuilder
from connectomics_pipeline.utils.config import CandidateConfig, GraphConfig


class TestProximityScore:
    def test_zero_distance(self):
        assert compute_proximity_score(0.0, 1000.0) == 1.0

    def test_max_distance(self):
        assert compute_proximity_score(1000.0, 1000.0) == 0.0

    def test_beyond_max(self):
        assert compute_proximity_score(1500.0, 1000.0) == 0.0

    def test_mid_range(self):
        score = compute_proximity_score(500.0, 1000.0)
        assert 0.0 < score < 1.0

    def test_monotonic_decay(self):
        scores = [compute_proximity_score(d, 1000.0) for d in range(0, 1001, 100)]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]


class TestAlignmentScore:
    def test_aligned_fragments(self, sample_fragments):
        frag_a, frag_b = sample_fragments[0], sample_fragments[1]
        score = compute_alignment_score(
            frag_a,
            frag_b,
            frag_a.endpoints[1],  # [50, 50, 200]
            frag_b.endpoints[0],  # [50, 50, 250]
        )
        assert score >= 0.5  # Should be well aligned

    def test_score_range(self, sample_fragments):
        frag_a, frag_b = sample_fragments[0], sample_fragments[1]
        score = compute_alignment_score(
            frag_a,
            frag_b,
            frag_a.endpoints[1],
            frag_b.endpoints[0],
        )
        assert 0.0 <= score <= 1.0


class TestContinuityScore:
    def test_straight_path(self, sample_fragments):
        frag_a, frag_b = sample_fragments[0], sample_fragments[1]
        score = compute_continuity_score(
            frag_a,
            frag_b,
            frag_a.endpoints[1],
            frag_b.endpoints[0],
        )
        assert score >= 0.5  # Straight path should score well

    def test_score_range(self, sample_fragments):
        frag_a, frag_c = sample_fragments[0], sample_fragments[2]
        score = compute_continuity_score(
            frag_a,
            frag_c,
            frag_a.endpoints[1],
            frag_c.endpoints[0],
        )
        assert 0.0 <= score <= 1.0


class TestCompositeScore:
    def test_equal_weights(self):
        weights = {"proximity": 0.25, "alignment": 0.25, "continuity": 0.25, "size": 0.25}
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, weights)
        assert abs(score - 1.0) < 1e-6

    def test_zero_scores(self):
        weights = {"proximity": 0.25, "alignment": 0.25, "continuity": 0.25, "size": 0.25}
        score = compute_composite_score(0.0, 0.0, 0.0, 0.0, weights)
        assert abs(score) < 1e-6

    def test_weighted(self):
        weights = {"proximity": 1.0, "alignment": 0.0, "continuity": 0.0, "size": 0.0}
        score = compute_composite_score(0.8, 0.0, 0.0, 0.0, weights)
        assert abs(score - 0.8) < 1e-6


class TestSizeScore:
    def test_equal_size(self):
        assert abs(compute_size_score(5.0, 5.0) - 1.0) < 1e-6

    def test_max_ratio(self):
        assert compute_size_score(1.0, 3.0, max_ratio=3.0) == 0.0

    def test_mid_range(self):
        score = compute_size_score(1.0, 2.0, max_ratio=3.0)
        assert 0.0 < score < 1.0


class TestCandidateGenerator:
    def test_generate_candidates(self, fragment_store, candidate_config):
        graph_config = GraphConfig(max_distance_nm=500.0)
        builder = GraphBuilder(graph_config)
        graph = builder.build(fragment_store)

        generator = CandidateGenerator(candidate_config, fragment_store)
        candidates = generator.generate(graph)

        assert isinstance(candidates, list)
        for c in candidates:
            assert 0.0 <= c.composite_score <= 1.0
            assert 0.0 <= c.proximity_score <= 1.0

    def test_generate_skips_missing_fragment(self, fragment_store, candidate_config):
        """Line 45: edge whose fragment is absent from store is skipped."""
        from connectomics_pipeline.graph.fragment_graph import FragmentGraph

        graph = FragmentGraph()
        # Add only one fragment to the graph; the other (id=99) is not in store
        graph.add_fragment(fragment_store.get(0))
        graph._graph.add_node(99)
        graph._graph.add_edge(0, 99, distance=100.0)

        generator = CandidateGenerator(candidate_config, fragment_store)
        candidates = generator.generate(graph)
        # The edge (0, 99) is skipped; no candidates produced for it
        cand_pairs = {(c.fragment_a, c.fragment_b) for c in candidates}
        assert (0, 99) not in cand_pairs and (99, 0) not in cand_pairs

    def test_generate_filters_low_composite_score(self, fragment_store):
        """Line 83: candidate with composite below min_composite_score is dropped."""
        # Set thresholds that almost nothing passes
        strict_config = CandidateConfig(
            max_endpoint_distance_nm=1.0,  # very short → proximity ~0
            min_alignment_score=0.0,
            min_composite_score=0.99,  # near-perfect required
        )
        graph_config = GraphConfig(max_distance_nm=500.0)
        graph = GraphBuilder(graph_config).build(fragment_store)
        generator = CandidateGenerator(strict_config, fragment_store)
        candidates = generator.generate(graph)
        for c in candidates:
            assert c.composite_score >= 0.99


class TestAlignmentScoreMissingCoverage:
    def test_endpoint_equals_centroid_fallback(self, sample_fragments):
        """Line 67: when endpoint == centroid, direction norm is ~0; returns [1,0,0]."""
        from connectomics_pipeline.utils.types import Fragment, BoundingBox, Skeleton

        # Fragment without a skeleton (forces centroid-fallback path)
        frag = Fragment(
            fragment_id=10,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(min_corner=np.zeros(3), max_corner=np.ones(3) * 10),
            centroid=np.array([5.0, 5.0, 5.0]),
            endpoints=[np.array([5.0, 5.0, 5.0])],
        )
        frag2 = sample_fragments[0]
        # Pass endpoint == centroid so norm < 1e-12 fallback fires
        score = compute_alignment_score(
            frag,
            frag2,
            endpoint_a=np.array([5.0, 5.0, 5.0]),  # same as centroid → zero direction
            endpoint_b=frag2.endpoints[0],
        )
        assert 0.0 <= score <= 1.0

    def test_no_skeleton_nonzero_direction_uses_fallback_return(self):
        """Line 67: fragment without skeleton, endpoint != centroid → direction/norm returned."""
        from connectomics_pipeline.utils.types import Fragment, BoundingBox

        frag_no_skel = Fragment(
            fragment_id=20,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(min_corner=np.zeros(3), max_corner=np.ones(3) * 20),
            centroid=np.array([0.0, 0.0, 0.0]),
            endpoints=[np.array([10.0, 0.0, 0.0])],
        )
        frag2 = Fragment(
            fragment_id=21,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(min_corner=np.ones(3) * 20, max_corner=np.ones(3) * 40),
            centroid=np.array([30.0, 0.0, 0.0]),
            endpoints=[np.array([30.0, 0.0, 0.0])],
        )
        # endpoint_a is far from centroid → norm > 1e-12 → hits line 67 (return direction/norm)
        score = compute_alignment_score(
            frag_no_skel,
            frag2,
            endpoint_a=np.array([10.0, 0.0, 0.0]),
            endpoint_b=np.array([30.0, 0.0, 0.0]),
        )
        assert 0.0 <= score <= 1.0
