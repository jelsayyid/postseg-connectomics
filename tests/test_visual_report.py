"""Tests for visual report helper functions in scripts/generate_visual_report.py.

Covers:
- _gt_outcome: TP/FP/FN/TN/NA computation
- _sample_candidates: category sampling with ground-truth
- _write_gt_pair_audit: CSV export columns and content
- TP/FP/FN/TN logic is behaviorally unchanged
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# ---------------------------------------------------------------------------
# Import the script module directly (it lives in scripts/, not a package).
# The script calls matplotlib.use("Agg") and imports h5py at module level.
# matplotlib is a real dependency and may already be imported in the process
# (e.g. by test_visualization.py).  We must NOT replace sys.modules["matplotlib"]
# with a stub — that would break other tests.
#
# Instead we use the "Agg" backend (non-interactive, no display needed) which
# the script itself already requests.  h5py is also a real dependency.
# If either is missing the tests are skipped.
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "generate_visual_report.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_visual_report", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _mod = _load_script_module()
    _gt_outcome = _mod._gt_outcome
    _sample_candidates = _mod._sample_candidates
    _write_gt_pair_audit = _mod._write_gt_pair_audit
    SCRIPT_AVAILABLE = True
except Exception as _exc:
    SCRIPT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SCRIPT_AVAILABLE, reason="generate_visual_report not importable"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_frag_info(rows: list[tuple[int, int]]) -> pd.DataFrame:
    """rows = [(fragment_id, label_id), ...]"""
    return pd.DataFrame(rows, columns=["fragment_id", "label_id"]).set_index("fragment_id")


def _make_conn_row(
    candidate_id: int,
    fragment_a: int,
    fragment_b: int,
    status: str,
    composite_score: float = 0.5,
    gap_distance: float = 100.0,
) -> pd.Series:
    return pd.Series(
        {
            "candidate_id": candidate_id,
            "fragment_a": fragment_a,
            "fragment_b": fragment_b,
            "status": status,
            "composite_score": composite_score,
            "proximity_score": 0.5,
            "alignment_score": 0.5,
            "continuity_score": 0.5,
            "size_score": 0.5,
            "gap_distance": gap_distance,
        }
    )


# ---------------------------------------------------------------------------
# _gt_outcome tests
# ---------------------------------------------------------------------------


class TestGtOutcome:
    def test_tp(self):
        frag_info = _make_frag_info([(0, 10), (1, 20)])
        gt_pairs = {(10, 20)}
        row = _make_conn_row(0, 0, 1, "accepted")
        assert _gt_outcome(row, frag_info, gt_pairs) == "TP"

    def test_fp(self):
        frag_info = _make_frag_info([(0, 10), (1, 30)])
        gt_pairs = {(10, 20)}  # (10, 30) not in gt_pairs
        row = _make_conn_row(0, 0, 1, "accepted")
        assert _gt_outcome(row, frag_info, gt_pairs) == "FP"

    def test_fn(self):
        frag_info = _make_frag_info([(0, 10), (1, 20)])
        gt_pairs = {(10, 20)}
        row = _make_conn_row(0, 0, 1, "rejected")
        assert _gt_outcome(row, frag_info, gt_pairs) == "FN"

    def test_tn(self):
        frag_info = _make_frag_info([(0, 10), (1, 30)])
        gt_pairs = {(10, 20)}  # (10, 30) not in gt_pairs
        row = _make_conn_row(0, 0, 1, "rejected")
        assert _gt_outcome(row, frag_info, gt_pairs) == "TN"

    def test_na_when_no_gt(self):
        frag_info = _make_frag_info([(0, 10), (1, 20)])
        row = _make_conn_row(0, 0, 1, "accepted")
        assert _gt_outcome(row, frag_info, None) == "NA"

    def test_na_when_fragment_missing(self):
        frag_info = _make_frag_info([(0, 10)])  # frag 1 missing
        gt_pairs = {(10, 20)}
        row = _make_conn_row(0, 0, 1, "accepted")
        assert _gt_outcome(row, frag_info, gt_pairs) == "NA"

    def test_canonical_pair_order(self):
        """gt_pairs stores (min, max) — both orderings of label A/B should resolve correctly."""
        frag_info = _make_frag_info([(0, 20), (1, 10)])  # reversed labels vs gt_pairs
        gt_pairs = {(10, 20)}  # (min, max) canonical
        row = _make_conn_row(0, 0, 1, "accepted")
        assert _gt_outcome(row, frag_info, gt_pairs) == "TP"

    def test_outcome_logic_unchanged(self):
        """Regression: accepted+GT-positive=TP, rejected+GT-positive=FN."""
        frag_info = _make_frag_info([(0, 1), (1, 2), (2, 3), (3, 4)])
        gt_pairs = {(1, 2)}

        assert _gt_outcome(_make_conn_row(0, 0, 1, "accepted"), frag_info, gt_pairs) == "TP"
        assert _gt_outcome(_make_conn_row(1, 0, 1, "rejected"), frag_info, gt_pairs) == "FN"
        assert _gt_outcome(_make_conn_row(2, 2, 3, "accepted"), frag_info, gt_pairs) == "FP"
        assert _gt_outcome(_make_conn_row(3, 2, 3, "rejected"), frag_info, gt_pairs) == "TN"


# ---------------------------------------------------------------------------
# _write_gt_pair_audit tests
# ---------------------------------------------------------------------------


class TestWriteGtPairAudit:
    def _make_connections(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "candidate_id": 0,
                    "fragment_a": 0,
                    "fragment_b": 1,
                    "status": "accepted",
                    "composite_score": 0.9,
                    "alignment_score": 0.8,
                    "continuity_score": 0.7,
                    "gap_distance": 50.0,
                },
                {
                    "candidate_id": 1,
                    "fragment_a": 2,
                    "fragment_b": 3,
                    "status": "rejected",
                    "composite_score": 0.2,
                    "alignment_score": 0.3,
                    "continuity_score": 0.4,
                    "gap_distance": 200.0,
                },
            ]
        )

    def _make_fragments(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"fragment_id": 0, "label_id": 10},
                {"fragment_id": 1, "label_id": 20},
                {"fragment_id": 2, "label_id": 30},
                {"fragment_id": 3, "label_id": 40},
            ]
        )

    def test_expected_columns(self, tmp_path):
        out_pdf = str(tmp_path / "report.pdf")
        connections = self._make_connections()
        fragments = self._make_fragments()
        gt_pairs = {(10, 20)}

        _write_gt_pair_audit(connections, fragments, gt_pairs, out_pdf)

        audit_path = tmp_path / "gt_pair_audit.csv"
        assert audit_path.exists(), "gt_pair_audit.csv was not created"

        df = pd.read_csv(audit_path)
        expected_cols = {
            "candidate_id",
            "fragment_a",
            "fragment_b",
            "label_a",
            "label_b",
            "gt_should_merge",
            "gt_source",
            "status",
            "composite",
            "alignment",
            "continuity",
            "gap",
        }
        assert expected_cols.issubset(
            set(df.columns)
        ), f"Missing columns: {expected_cols - set(df.columns)}"

    def test_gt_should_merge_values(self, tmp_path):
        out_pdf = str(tmp_path / "report.pdf")
        connections = self._make_connections()
        fragments = self._make_fragments()
        gt_pairs = {(10, 20)}  # cand 0 (frags 0↔1, labels 10↔20) is GT-positive

        _write_gt_pair_audit(connections, fragments, gt_pairs, out_pdf)
        df = pd.read_csv(tmp_path / "gt_pair_audit.csv")

        row0 = df[df["candidate_id"] == 0].iloc[0]
        row1 = df[df["candidate_id"] == 1].iloc[0]
        assert row0["gt_should_merge"] == True
        assert row1["gt_should_merge"] == False

    def test_gt_source_skeleton_edge_crossing(self, tmp_path):
        out_pdf = str(tmp_path / "report.pdf")
        connections = self._make_connections()
        fragments = self._make_fragments()

        _write_gt_pair_audit(connections, fragments, {(10, 20)}, out_pdf)
        df = pd.read_csv(tmp_path / "gt_pair_audit.csv")
        assert (df["gt_source"] == "skeleton_edge_crossing").all()

    def test_gt_source_unavailable_when_no_gt(self, tmp_path):
        out_pdf = str(tmp_path / "report.pdf")
        connections = self._make_connections()
        fragments = self._make_fragments()

        _write_gt_pair_audit(connections, fragments, None, out_pdf)
        df = pd.read_csv(tmp_path / "gt_pair_audit.csv")
        assert (df["gt_source"] == "unavailable").all()

    def test_row_count_matches_connections(self, tmp_path):
        out_pdf = str(tmp_path / "report.pdf")
        connections = self._make_connections()
        fragments = self._make_fragments()

        _write_gt_pair_audit(connections, fragments, set(), out_pdf)
        df = pd.read_csv(tmp_path / "gt_pair_audit.csv")
        assert len(df) == len(connections)

    def test_status_column_preserved(self, tmp_path):
        out_pdf = str(tmp_path / "report.pdf")
        connections = self._make_connections()
        fragments = self._make_fragments()

        _write_gt_pair_audit(connections, fragments, set(), out_pdf)
        df = pd.read_csv(tmp_path / "gt_pair_audit.csv")
        assert df[df["candidate_id"] == 0].iloc[0]["status"] == "accepted"
        assert df[df["candidate_id"] == 1].iloc[0]["status"] == "rejected"


# ---------------------------------------------------------------------------
# _sample_candidates tests (GT semantics)
# ---------------------------------------------------------------------------


class TestSampleCandidatesGt:
    def _make_data(self):
        """Minimal connections + fragments for sampling tests."""
        connections = pd.DataFrame(
            [
                {
                    "candidate_id": i,
                    "fragment_a": i * 2,
                    "fragment_b": i * 2 + 1,
                    "status": "accepted" if i < 5 else "rejected",
                    "composite_score": float(10 - i),
                    "gap_distance": float(100 + i * 10),
                }
                for i in range(10)
            ]
        )
        fragments = pd.DataFrame([{"fragment_id": j, "label_id": j + 100} for j in range(20)])
        return connections, fragments

    def test_ground_truth_tp_category_present(self):
        connections, fragments = self._make_data()
        # Make frag 0 and 1 (labels 100, 101) a GT-positive pair
        gt_pairs = {(100, 101)}

        samples = _sample_candidates(connections, fragments, gt_pairs, n_samples=2, seed=0)
        categories = [cat for cat, _ in samples]
        assert "Ground-truth TP" in categories

    def test_gt_fn_category_present(self):
        connections, fragments = self._make_data()
        # frag 10 and 11 → labels 110, 111 → rejected candidate (cand_id=5)
        gt_pairs = {(110, 111)}

        samples = _sample_candidates(connections, fragments, gt_pairs, n_samples=2, seed=0)
        categories = [cat for cat, _ in samples]
        assert "GT FN" in categories

    def test_no_gt_omits_tp_fn_categories(self):
        connections, fragments = self._make_data()

        samples = _sample_candidates(connections, fragments, None, n_samples=2, seed=0)
        categories = [cat for cat, _ in samples]
        assert "Ground-truth TP" not in categories
        assert "GT FN" not in categories
