"""Tests for validation report building."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.validation.report import build_report
from connectomics_pipeline.utils.types import (
    CandidateConnection,
    ConnectionStatus,
    ValidationResult,
)


def make_candidate(cid, score=0.5):
    return CandidateConnection(
        candidate_id=cid,
        fragment_a=0,
        fragment_b=1,
        endpoint_a=np.array([0, 0, 0.0]),
        endpoint_b=np.array([0, 0, 10.0]),
        composite_score=score,
    )


class TestBuildReport:
    def test_all_accepted(self):
        candidates = [make_candidate(0), make_candidate(1)]
        results = {
            0: [ValidationResult("rule1", ConnectionStatus.ACCEPTED, 0.95, "ok")],
            1: [ValidationResult("rule1", ConnectionStatus.ACCEPTED, 0.90, "ok")],
        }
        report = build_report(results, candidates, accept_threshold=0.8)
        assert len(report.accepted) == 2
        assert len(report.rejected) == 0
        assert len(report.ambiguous) == 0

    def test_hard_reject_overrides(self):
        """A single REJECT rule should reject the candidate regardless of confidence."""
        candidates = [make_candidate(0)]
        results = {
            0: [
                ValidationResult("rule1", ConnectionStatus.ACCEPTED, 0.95, "ok"),
                ValidationResult("rule2", ConnectionStatus.REJECTED, 0.1, "fail"),
            ],
        }
        report = build_report(results, candidates)
        assert 0 in report.rejected
        assert candidates[0].status == ConnectionStatus.REJECTED

    def test_low_confidence_rejects(self):
        candidates = [make_candidate(0)]
        results = {
            0: [ValidationResult("rule1", ConnectionStatus.ACCEPTED, 0.2, "meh")],
        }
        report = build_report(results, candidates, accept_threshold=0.8, reject_threshold=0.3)
        assert 0 in report.rejected

    def test_ambiguous_middle_confidence(self):
        candidates = [make_candidate(0)]
        results = {
            0: [ValidationResult("rule1", ConnectionStatus.ACCEPTED, 0.5, "uncertain")],
        }
        report = build_report(results, candidates, accept_threshold=0.8, reject_threshold=0.3)
        assert 0 in report.ambiguous
        assert candidates[0].status == ConnectionStatus.AMBIGUOUS

    def test_no_results_is_ambiguous(self):
        """Candidate with no rule results should be ambiguous."""
        candidates = [make_candidate(0)]
        results = {}
        report = build_report(results, candidates)
        assert 0 in report.ambiguous

    def test_total_count(self):
        candidates = [make_candidate(i) for i in range(3)]
        results = {
            0: [ValidationResult("r", ConnectionStatus.ACCEPTED, 0.95, "")],
            1: [ValidationResult("r", ConnectionStatus.REJECTED, 0.1, "")],
            2: [ValidationResult("r", ConnectionStatus.ACCEPTED, 0.5, "")],
        }
        report = build_report(results, candidates, accept_threshold=0.8, reject_threshold=0.3)
        assert report.total == 3

    def test_summary(self):
        candidates = [make_candidate(0)]
        results = {
            0: [ValidationResult("r", ConnectionStatus.ACCEPTED, 0.95, "")],
        }
        report = build_report(results, candidates, accept_threshold=0.8)
        s = report.summary()
        assert s["total"] == 1
        assert s["accepted"] == 1
        assert s["accept_rate"] == 1.0
