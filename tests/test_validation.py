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
