"""Unit tests for connectomics_pipeline.postprocess.ml_filter."""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest

from connectomics_pipeline.postprocess.ml_filter import MLFilter, _extract_features
from connectomics_pipeline.utils.types import (
    CandidateConnection,
    ConnectionStatus,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Picklable fake models (MagicMock is not picklable by joblib)
# ---------------------------------------------------------------------------


class _FakeModel:
    """Tiny picklable sklearn-API model that returns fixed probabilities."""

    def __init__(self, probas: List[float]):
        # probas[i] = P(class=1) for candidate i
        self._probas = np.array([[1.0 - p, p] for p in probas], dtype=np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probas


class _FakeFailModel:
    """Picklable model that always raises RuntimeError."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise RuntimeError("mock error")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cand(
    cid: int,
    gap: float = 500.0,
    proximity: float = 0.5,
    alignment: float = 0.5,
    continuity: float = 0.5,
    size: float = 0.5,
    composite: float = 0.5,
    frag_a: int = 0,
    frag_b: int = 1,
) -> CandidateConnection:
    ep_a = np.array([0.0, 0.0, 0.0])
    ep_b = np.array([gap, 0.0, 0.0])
    return CandidateConnection(
        candidate_id=cid,
        fragment_a=frag_a,
        fragment_b=frag_b,
        endpoint_a=ep_a,
        endpoint_b=ep_b,
        proximity_score=proximity,
        alignment_score=alignment,
        continuity_score=continuity,
        size_score=size,
        composite_score=composite,
        status=ConnectionStatus.ACCEPTED,
    )


def _make_report(accepted: list, rejected: list = None) -> ValidationReport:
    return ValidationReport(
        results={},
        accepted=list(accepted),
        rejected=list(rejected or []),
        ambiguous=[],
    )


def _make_filter(model: object, threshold: float = 0.5) -> MLFilter:
    """Save a picklable model artifact to a temp file and load an MLFilter."""
    try:
        import joblib
    except ImportError:
        pytest.skip("joblib not installed")

    artifact = {
        "model": model,
        "threshold": threshold,
        "features": [
            "gap_distance",
            "proximity_score",
            "alignment_score",
            "continuity_score",
            "size_score",
            "composite_score",
            "degree_a",
            "degree_b",
        ],
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        joblib.dump(artifact, f.name)
        tmp_path = f.name

    return MLFilter(model_path=tmp_path, threshold=0.0)


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_empty_returns_empty_array(self):
        result = _extract_features([])
        assert result.shape == (0, 8)

    def test_single_candidate_shape(self):
        c = _make_cand(
            1, gap=400.0, proximity=0.3, alignment=0.7, continuity=0.6, size=0.4, composite=0.5
        )
        X = _extract_features([c])
        assert X.shape == (1, 8)

    def test_feature_values_correct(self):
        c = _make_cand(
            1, gap=123.0, proximity=0.1, alignment=0.2, continuity=0.3, size=0.4, composite=0.5
        )
        X = _extract_features([c])
        # gap_distance is distance between ep_a and ep_b
        assert abs(X[0, 0] - 123.0) < 1e-4
        assert abs(X[0, 1] - 0.1) < 1e-4  # proximity
        assert abs(X[0, 2] - 0.2) < 1e-4  # alignment
        assert abs(X[0, 3] - 0.3) < 1e-4  # continuity
        assert abs(X[0, 4] - 0.4) < 1e-4  # size
        assert abs(X[0, 5] - 0.5) < 1e-4  # composite

    def test_multiple_candidates(self):
        cands = [_make_cand(i, gap=float(i * 100)) for i in range(5)]
        X = _extract_features(cands)
        assert X.shape == (5, 8)
        # gap_distance should be 0, 100, 200, 300, 400
        np.testing.assert_allclose(X[:, 0], [0, 100, 200, 300, 400], atol=1e-4)

    def test_dtype_is_float32(self):
        c = _make_cand(1)
        X = _extract_features([c])
        assert X.dtype == np.float32


# ---------------------------------------------------------------------------
# MLFilter.filter_report
# ---------------------------------------------------------------------------


class TestMLFilterFilterReport:
    def test_empty_accepted_unchanged(self):
        """Empty accepted list → report unchanged."""
        ml_filter = _make_filter(_FakeModel([]), threshold=0.5)
        cands = [_make_cand(1), _make_cand(2)]
        report = _make_report(accepted=[], rejected=[1, 2])
        result = ml_filter.filter_report(cands, report)
        assert result.accepted == []
        assert 1 in result.rejected
        assert 2 in result.rejected

    def test_all_above_threshold_kept(self):
        """All candidates above threshold → none moved to rejected."""
        ml_filter = _make_filter(_FakeModel([0.9, 0.8, 0.7]), threshold=0.5)
        cands = [_make_cand(i + 1) for i in range(3)]
        report = _make_report(accepted=[1, 2, 3])
        result = ml_filter.filter_report(cands, report)
        assert set(result.accepted) == {1, 2, 3}
        assert result.rejected == []

    def test_all_below_threshold_rejected(self):
        """All candidates below threshold → all moved to rejected."""
        ml_filter = _make_filter(_FakeModel([0.1, 0.2, 0.3]), threshold=0.5)
        cands = [_make_cand(i + 1) for i in range(3)]
        report = _make_report(accepted=[1, 2, 3])
        result = ml_filter.filter_report(cands, report)
        assert result.accepted == []
        assert set(result.rejected) == {1, 2, 3}

    def test_partial_filter(self):
        """Some above, some below threshold → correct split."""
        # cands 1,3 → prob 0.8/0.9 above 0.5; cand 2 → prob 0.2 below
        ml_filter = _make_filter(_FakeModel([0.8, 0.2, 0.9]), threshold=0.5)
        cands = [_make_cand(1), _make_cand(2), _make_cand(3)]
        report = _make_report(accepted=[1, 2, 3])
        result = ml_filter.filter_report(cands, report)
        assert set(result.accepted) == {1, 3}
        assert 2 in result.rejected

    def test_status_updated_on_rejected_candidates(self):
        """Candidates moved to rejected have status=REJECTED."""
        ml_filter = _make_filter(_FakeModel([0.1]), threshold=0.5)
        cand = _make_cand(1)
        report = _make_report(accepted=[1])
        ml_filter.filter_report([cand], report)
        assert cand.status == ConnectionStatus.REJECTED

    def test_status_unchanged_on_kept_candidates(self):
        """Candidates kept in accepted retain ACCEPTED status."""
        ml_filter = _make_filter(_FakeModel([0.9]), threshold=0.5)
        cand = _make_cand(1)
        report = _make_report(accepted=[1])
        ml_filter.filter_report([cand], report)
        assert cand.status == ConnectionStatus.ACCEPTED

    def test_pre_rejected_candidates_not_affected(self):
        """Candidates already rejected are not touched by the filter."""
        # Only the accepted cand (id=1) is scored; rejected cand (id=2) is ignored
        ml_filter = _make_filter(_FakeModel([0.9]), threshold=0.5)
        accepted_cand = _make_cand(1)
        rejected_cand = _make_cand(2)
        rejected_cand.status = ConnectionStatus.REJECTED
        report = _make_report(accepted=[1], rejected=[2])
        result = ml_filter.filter_report([accepted_cand, rejected_cand], report)
        assert 1 in result.accepted
        assert 2 in result.rejected

    def test_threshold_stored_in_model_used_when_zero(self):
        """threshold=0.0 (sentinel) → uses saved_threshold from artifact."""
        # Saved threshold=0.7; proba=0.6 → below threshold → rejected
        ml_filter = _make_filter(_FakeModel([0.6]), threshold=0.7)
        assert abs(ml_filter.saved_threshold - 0.7) < 1e-9
        assert abs(ml_filter.threshold - 0.7) < 1e-9  # loaded with threshold=0.0 sentinel

        cand = _make_cand(1)
        report = _make_report(accepted=[1])
        result = ml_filter.filter_report([cand], report)
        assert result.accepted == []
        assert 1 in result.rejected

    def test_predict_proba_failure_returns_unchanged(self):
        """If predict_proba raises, report is returned unchanged."""
        ml_filter = _make_filter(_FakeFailModel(), threshold=0.5)
        cand = _make_cand(1)
        report = _make_report(accepted=[1])
        result = ml_filter.filter_report([cand], report)
        # Report should be unchanged (failure is caught gracefully)
        assert 1 in result.accepted
        assert cand.status == ConnectionStatus.ACCEPTED
