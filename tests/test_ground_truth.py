"""Tests for connectomics_pipeline.evaluation.ground_truth."""

import pytest

from connectomics_pipeline.evaluation.ground_truth import evaluate_decisions
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.types import (
    BoundingBox,
    CandidateConnection,
    ConnectionStatus,
    Fragment,
)

import numpy as np


def _make_fragment(fid: int, label_id: int) -> Fragment:
    c = np.array([float(fid * 100), 50.0, 50.0])
    bb = BoundingBox(min_corner=c - 40, max_corner=c + 40)
    return Fragment(
        fragment_id=fid,
        label_id=label_id,
        voxel_count=100,
        bounding_box=bb,
        centroid=c,
    )


def _make_candidate(
    cid: int, frag_a: int, frag_b: int, status: ConnectionStatus
) -> CandidateConnection:
    return CandidateConnection(
        candidate_id=cid,
        fragment_a=frag_a,
        fragment_b=frag_b,
        endpoint_a=np.array([0.0, 0.0, 0.0]),
        endpoint_b=np.array([100.0, 0.0, 0.0]),
        proximity_score=0.9,
        alignment_score=0.8,
        continuity_score=0.8,
        size_score=0.9,
        composite_score=0.85,
        status=status,
    )


@pytest.fixture
def store_two_neurons():
    """Store with 4 fragments: frag 0,1 share label 1; frag 2,3 share label 2."""
    store = FragmentStore()
    store.add(_make_fragment(0, 1))
    store.add(_make_fragment(1, 1))
    store.add(_make_fragment(2, 2))
    store.add(_make_fragment(3, 2))
    return store


class TestEvaluateDecisions:
    def test_true_positive(self, store_two_neurons):
        """Accepting a same-label pair is a TP."""
        candidates = [_make_candidate(0, 0, 1, ConnectionStatus.ACCEPTED)]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["true_positives"] == 1
        assert result["false_positives"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_false_positive(self, store_two_neurons):
        """Accepting a cross-label pair is a FP."""
        candidates = [_make_candidate(0, 0, 2, ConnectionStatus.ACCEPTED)]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["false_positives"] == 1
        assert result["true_positives"] == 0
        assert result["precision"] == 0.0

    def test_true_negative(self, store_two_neurons):
        """Rejecting a cross-label pair is a TN."""
        candidates = [_make_candidate(0, 0, 2, ConnectionStatus.REJECTED)]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["true_negatives"] == 1
        assert result["false_negatives"] == 0

    def test_false_negative(self, store_two_neurons):
        """Rejecting a same-label pair is a FN."""
        candidates = [_make_candidate(0, 0, 1, ConnectionStatus.REJECTED)]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["false_negatives"] == 1
        assert result["true_negatives"] == 0
        assert result["recall"] == 0.0

    def test_ambiguous_same_label(self, store_two_neurons):
        """Ambiguous same-label candidate counted separately, not in metrics."""
        candidates = [_make_candidate(0, 0, 1, ConnectionStatus.AMBIGUOUS)]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["ambiguous_same_label"] == 1
        assert result["ambiguous_diff_label"] == 0
        assert result["true_positives"] == 0
        assert result["false_negatives"] == 0

    def test_ambiguous_diff_label(self, store_two_neurons):
        """Ambiguous cross-label candidate counted separately."""
        candidates = [_make_candidate(0, 0, 2, ConnectionStatus.AMBIGUOUS)]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["ambiguous_diff_label"] == 1
        assert result["ambiguous_same_label"] == 0

    def test_mixed_decisions(self, store_two_neurons):
        """Mixed candidates produce correct TP/FP/TN/FN tallies."""
        candidates = [
            _make_candidate(0, 0, 1, ConnectionStatus.ACCEPTED),  # TP
            _make_candidate(1, 2, 3, ConnectionStatus.ACCEPTED),  # TP
            _make_candidate(2, 0, 2, ConnectionStatus.REJECTED),  # TN
            _make_candidate(3, 1, 3, ConnectionStatus.REJECTED),  # TN
            _make_candidate(4, 0, 3, ConnectionStatus.ACCEPTED),  # FP (cross-label)
            _make_candidate(5, 0, 1, ConnectionStatus.REJECTED),  # FN (same-label rejected)
        ]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["true_positives"] == 2
        assert result["false_positives"] == 1
        assert result["true_negatives"] == 2
        assert result["false_negatives"] == 1
        precision = 2 / 3
        recall = 2 / 3
        f1 = 2 * precision * recall / (precision + recall)
        assert abs(result["precision"] - precision) < 1e-9
        assert abs(result["recall"] - recall) < 1e-9
        assert abs(result["f1"] - f1) < 1e-9

    def test_empty_candidates(self, store_two_neurons):
        """Empty input returns zeros and no errors."""
        result = evaluate_decisions([], store_two_neurons)
        assert result["true_positives"] == 0
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_all_ambiguous(self, store_two_neurons):
        """All-ambiguous input gives zero precision/recall (no accepted/rejected)."""
        candidates = [
            _make_candidate(0, 0, 1, ConnectionStatus.AMBIGUOUS),
            _make_candidate(1, 2, 3, ConnectionStatus.AMBIGUOUS),
        ]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["ambiguous_same_label"] == 2

    def test_missing_fragment_skipped(self):
        """Candidates referencing unknown fragment IDs are skipped gracefully."""
        store = FragmentStore()
        store.add(_make_fragment(0, 1))
        # fragment_id=99 does not exist
        candidates = [_make_candidate(0, 0, 99, ConnectionStatus.ACCEPTED)]
        result = evaluate_decisions(candidates, store)
        assert result["true_positives"] == 0
        assert result["false_positives"] == 0

    def test_perfect_precision_zero_recall(self, store_two_neurons):
        """Accepting nothing gives precision=0 (no denominator), recall=0."""
        candidates = [_make_candidate(0, 0, 2, ConnectionStatus.REJECTED)]
        result = evaluate_decisions(candidates, store_two_neurons)
        assert result["true_positives"] == 0
        assert result["recall"] == 0.0
