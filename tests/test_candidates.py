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
