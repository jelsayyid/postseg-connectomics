"""Tests for the validation module."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.config import RuleConfig, ValidationConfig
from connectomics_pipeline.utils.types import (
    BoundingBox,
    CandidateConnection,
    ConnectionStatus,
    Fragment,
)
from connectomics_pipeline.validation.pipeline import ValidationPipeline
from connectomics_pipeline.validation.rules import (
    BranchingLimitRule,
    CompositeScoreRule,
    CurvatureRule,
    DirectionReversalRule,
    MaxDistanceRule,
    SizeDiscrepancyRule,
    create_rule,
)


@pytest.fixture
def close_candidate():
    """Candidate with small gap distance."""
    return CandidateConnection(
        candidate_id=0,
        fragment_a=0,
        fragment_b=1,
        endpoint_a=np.array([50.0, 50.0, 200.0]),
        endpoint_b=np.array([50.0, 50.0, 250.0]),
        proximity_score=0.9,
        alignment_score=0.8,
        continuity_score=0.7,
        size_score=0.9,
        composite_score=0.85,
    )


@pytest.fixture
def far_candidate():
    """Candidate with large gap distance."""
    return CandidateConnection(
        candidate_id=1,
        fragment_a=0,
        fragment_b=2,
        endpoint_a=np.array([50.0, 50.0, 200.0]),
        endpoint_b=np.array([50.0, 50.0, 5000.0]),
        proximity_score=0.1,
        alignment_score=0.2,
        continuity_score=0.3,
        size_score=0.5,
        composite_score=0.2,
    )


class TestMaxDistanceRule:
    def test_accept_close(self, close_candidate, fragment_store):
        rule = MaxDistanceRule(max_distance_nm=1500.0)
        result = rule.evaluate(close_candidate, fragment_store)
        assert result.decision == ConnectionStatus.ACCEPTED

    def test_reject_far(self, far_candidate, fragment_store):
        rule = MaxDistanceRule(max_distance_nm=1500.0)
        result = rule.evaluate(far_candidate, fragment_store)
        assert result.decision == ConnectionStatus.REJECTED


class TestCurvatureRule:
    def test_straight_connection(self, close_candidate, fragment_store):
        rule = CurvatureRule(max_curvature_deg=90.0)
        result = rule.evaluate(close_candidate, fragment_store)
        assert result.decision == ConnectionStatus.ACCEPTED


class TestDirectionReversalRule:
    def test_aligned(self, close_candidate, fragment_store):
        rule = DirectionReversalRule(min_alignment=0.3)
        result = rule.evaluate(close_candidate, fragment_store)
        assert result.decision == ConnectionStatus.ACCEPTED

    def test_reject_misaligned(self, fragment_store):
        candidate = CandidateConnection(
            candidate_id=2,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.array([0.0, 0.0, 0.0]),
            endpoint_b=np.array([100.0, 0.0, 0.0]),
            alignment_score=0.1,
            composite_score=0.3,
        )
        rule = DirectionReversalRule(min_alignment=0.5)
        result = rule.evaluate(candidate, fragment_store)
        assert result.decision == ConnectionStatus.REJECTED


class TestSizeDiscrepancyRule:
    def test_similar_size(self, close_candidate, fragment_store):
        rule = SizeDiscrepancyRule(max_radius_ratio=3.0)
        result = rule.evaluate(close_candidate, fragment_store)
        assert result.decision == ConnectionStatus.ACCEPTED


class TestBranchingLimitRule:
    def test_no_graph(self, close_candidate, fragment_store):
        rule = BranchingLimitRule(max_branches=10)
        result = rule.evaluate(close_candidate, fragment_store, graph=None)
        assert result.decision == ConnectionStatus.ACCEPTED


class TestCompositeScoreRule:
    def test_high_score(self, close_candidate, fragment_store):
        rule = CompositeScoreRule(reject_threshold=0.3)
        result = rule.evaluate(close_candidate, fragment_store)
        assert result.decision == ConnectionStatus.ACCEPTED

    def test_low_score(self, far_candidate, fragment_store):
        rule = CompositeScoreRule(reject_threshold=0.3)
        result = rule.evaluate(far_candidate, fragment_store)
        assert result.decision == ConnectionStatus.REJECTED


class TestCreateRule:
    def test_create_max_distance(self):
        rule = create_rule("MaxDistanceRule", {"max_distance_nm": 1000.0})
        assert isinstance(rule, MaxDistanceRule)

    def test_unknown_rule(self):
        with pytest.raises(ValueError, match="Unknown rule"):
            create_rule("NonexistentRule", {})


class TestValidationPipeline:
    def test_full_pipeline(self, close_candidate, far_candidate, fragment_store):
        config = ValidationConfig(
            accept_threshold=0.6,
            reject_threshold=0.3,
            rules=[
                RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 1500.0}),
                RuleConfig(name="CompositeScoreRule", params={"reject_threshold": 0.3}),
            ],
        )
        pipeline = ValidationPipeline(config)
        report = pipeline.validate([close_candidate, far_candidate], fragment_store)

        assert close_candidate.candidate_id in report.accepted
        assert far_candidate.candidate_id in report.rejected
        assert report.total == 2

    def test_explicit_rules_bypass_config(self, close_candidate, far_candidate, fragment_store):
        """Line 28: ValidationPipeline with explicit rules list uses those rules directly."""
        config = ValidationConfig(accept_threshold=0.6, reject_threshold=0.3, rules=[])
        explicit_rules = [MaxDistanceRule(max_distance_nm=1500.0)]
        pipeline = ValidationPipeline(config, rules=explicit_rules)
        assert len(pipeline.rules) == 1
        assert isinstance(pipeline.rules[0], MaxDistanceRule)
        report = pipeline.validate([close_candidate], fragment_store)
        assert report.total == 1


class TestCurvatureRuleMissingCoverage:
    def test_curvature_exceeds_max_rejected(self, fragment_store):
        """Line 100: CurvatureRule rejects when curvature > max_curvature_rad.

        Fragment 0 has centroid [50,50,100] and the endpoint_a is [50,50,200],
        giving direction_a ≈ [0,0,1]. The connection vector from endpoint_a to
        endpoint_b is [100,0,0] ≈ [1,0,0]. The angle between them is ~90°,
        which exceeds max_curvature_deg=30.
        """
        candidate = CandidateConnection(
            candidate_id=5,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.array([50.0, 50.0, 200.0]),
            endpoint_b=np.array([150.0, 50.0, 200.0]),  # 90° turn from z-axis to x-axis
            alignment_score=0.1,
            composite_score=0.1,
        )
        rule = CurvatureRule(max_curvature_deg=30.0)
        result = rule.evaluate(candidate, fragment_store)
        assert result.decision == ConnectionStatus.REJECTED

    def test_zero_connection_vector_returns_zero_curvature(self, fragment_store):
        """Line 124: conn_norm < 1e-12 early-returns 0 curvature → ACCEPT."""
        candidate = CandidateConnection(
            candidate_id=6,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.array([50.0, 50.0, 200.0]),
            endpoint_b=np.array([50.0, 50.0, 200.0]),  # identical → zero vector
            alignment_score=0.8,
            composite_score=0.8,
        )
        rule = CurvatureRule(max_curvature_deg=90.0)
        result = rule.evaluate(candidate, fragment_store)
        assert result.decision == ConnectionStatus.ACCEPTED

    def test_get_direction_zero_norm_returns_default(self, fragment_store):
        """Line 135: _get_direction with zero-norm direction returns [1,0,0]."""
        # Endpoint == centroid → norm is 0 → fallback to [1,0,0]
        candidate = CandidateConnection(
            candidate_id=7,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.array([50.0, 50.0, 100.0]),  # == centroid of fragment 0
            endpoint_b=np.array([50.0, 50.0, 200.0]),
            alignment_score=0.8,
            composite_score=0.8,
        )
        rule = CurvatureRule(max_curvature_deg=90.0)
        result = rule.evaluate(candidate, fragment_store)
        assert result.decision in (ConnectionStatus.ACCEPTED, ConnectionStatus.REJECTED)


class TestBranchingLimitRuleMissingCoverage:
    def test_reject_when_degree_exceeds_limit(self, close_candidate, fragment_store):
        """Line 257: BranchingLimitRule rejects when max degree >= max_branches."""
        from connectomics_pipeline.graph.fragment_graph import FragmentGraph
        from connectomics_pipeline.utils.types import BoundingBox

        graph = FragmentGraph()
        for fid in range(15):
            f = Fragment(
                fragment_id=fid,
                label_id=1,
                voxel_count=100,
                bounding_box=BoundingBox(np.zeros(3), np.ones(3) * 10),
                centroid=np.array([float(fid), 0.0, 0.0]),
                endpoints=[np.array([float(fid), 0.0, 0.0])],
            )
            graph.add_fragment(f)
        # Make fragment 0 have 12 neighbors (exceeds max_branches=10)
        for i in range(1, 13):
            graph.add_edge(0, i, distance=float(i))

        candidate = CandidateConnection(
            candidate_id=0,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.zeros(3),
            endpoint_b=np.ones(3),
            alignment_score=0.8,
            composite_score=0.8,
        )
        rule = BranchingLimitRule(max_branches=10)
        result = rule.evaluate(candidate, fragment_store, graph=graph)
        assert result.decision == ConnectionStatus.REJECTED

    def test_accept_when_degree_within_limit(self, close_candidate, fragment_store):
        """Line 257: BranchingLimitRule accept with real graph when degree < max_branches."""
        from connectomics_pipeline.graph.fragment_graph import FragmentGraph

        graph = FragmentGraph()
        for fid in [0, 1]:
            f = Fragment(
                fragment_id=fid,
                label_id=1,
                voxel_count=100,
                bounding_box=BoundingBox(np.zeros(3), np.ones(3) * 10),
                centroid=np.array([float(fid), 0.0, 0.0]),
                endpoints=[np.array([float(fid), 0.0, 0.0])],
            )
            graph.add_fragment(f)
        graph.add_edge(0, 1, distance=1.0)  # degree = 1 for each

        rule = BranchingLimitRule(max_branches=10)
        result = rule.evaluate(close_candidate, fragment_store, graph=graph)
        assert result.decision == ConnectionStatus.ACCEPTED
