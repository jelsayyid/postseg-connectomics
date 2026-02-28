"""Tests for connectomics_pipeline.evaluation.xpress_ground_truth."""

import io
from pathlib import Path

import numpy as np
import pytest

from connectomics_pipeline.evaluation.xpress_ground_truth import (
    build_merge_oracle,
    evaluate_decisions_xpress,
    load_skeleton_graphs,
)
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.types import (
    BoundingBox,
    CandidateConnection,
    ConnectionStatus,
    Fragment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

try:
    import networkx as nx

    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not NX_AVAILABLE, reason="networkx not installed")


def _make_graph(edges: list, node_positions: dict) -> "nx.Graph":
    """Build a small NetworkX skeleton graph with 'position' attributes."""
    G = nx.Graph()
    for node, pos in node_positions.items():
        G.add_node(node, position=pos)
    G.add_edges_from(edges)
    return G


def _save_skeleton_npz(graphs: dict, tmp_path: Path) -> Path:
    """Save a dict {skel_id: nx.Graph} to .npz format matching XPRESS convention."""
    npz_path = tmp_path / "test_skels.npz"
    np.savez(npz_path, np.array(graphs, dtype=object))
    return npz_path


def _make_fragment(fid: int, label_id: int, centroid=None) -> Fragment:
    c = centroid if centroid is not None else np.array([float(fid * 100), 50.0, 50.0])
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


# ---------------------------------------------------------------------------
# load_skeleton_graphs
# ---------------------------------------------------------------------------


class TestLoadSkeletonGraphs:
    def test_load_dict_format(self, tmp_path):
        """Standard XPRESS format: arr_0 is a dict {skel_id: nx.Graph}."""
        G = _make_graph([(0, 1)], {0: (0.0, 0.0, 0.0), 1: (33.0, 0.0, 0.0)})
        npz_path = _save_skeleton_npz({42: G}, tmp_path)
        graphs = load_skeleton_graphs(npz_path)
        assert len(graphs) == 1
        assert isinstance(graphs[0], nx.Graph)

    def test_load_multiple_graphs(self, tmp_path):
        """Multiple axons in one .npz → multiple graphs returned."""
        G1 = _make_graph([(0, 1)], {0: (0.0, 0.0, 0.0), 1: (33.0, 0.0, 0.0)})
        G2 = _make_graph([(0, 1)], {0: (100.0, 0.0, 0.0), 1: (133.0, 0.0, 0.0)})
        npz_path = _save_skeleton_npz({1: G1, 2: G2}, tmp_path)
        graphs = load_skeleton_graphs(npz_path)
        assert len(graphs) == 2

    def test_node_positions_preserved(self, tmp_path):
        """Node 'position' attributes survive serialization round-trip."""
        pos = (330.0, 660.0, 990.0)
        G = _make_graph([], {0: pos})
        npz_path = _save_skeleton_npz({1: G}, tmp_path)
        graphs = load_skeleton_graphs(npz_path)
        node_attrs = dict(graphs[0].nodes(data=True))
        assert node_attrs[0]["position"] == pos


# ---------------------------------------------------------------------------
# build_merge_oracle
# ---------------------------------------------------------------------------


class TestBuildMergeOracle:
    def _simple_seg(self):
        """
        3×1×2 segmentation (z, y, x):
          [[[1, 2]],
           [[1, 2]],
           [[1, 3]]]
        Voxel size: 33nm isotropic.
        Positions are (x_nm, y_nm, z_nm).
        """
        seg = np.array([[[1, 2]], [[1, 2]], [[1, 3]]], dtype=np.uint64)
        return seg

    def test_edge_crossing_boundary_creates_pair(self):
        """Skeleton edge from seg 1 to seg 2 → (1, 2) in oracle."""
        seg = self._simple_seg()
        voxel_size = (33.0, 33.0, 33.0)
        # Node 0 maps to (z=0, y=0, x=0) → seg 1
        # Node 1 maps to (z=0, y=0, x=1) → seg 2
        # position = (x_nm, y_nm, z_nm): x=0→0nm, x=1→33nm; z=0→0nm
        G = _make_graph([(0, 1)], {0: (0.0, 0.0, 0.0), 1: (33.0, 0.0, 0.0)})
        oracle = build_merge_oracle([G], seg, voxel_size)
        assert (1, 2) in oracle

    def test_edge_within_same_segment_no_pair(self):
        """Skeleton edge entirely within segment 1 → no merge pair."""
        seg = self._simple_seg()
        voxel_size = (33.0, 33.0, 33.0)
        # Both nodes map to seg 1 (z=0, x=0) and (z=1, x=0)
        G = _make_graph([(0, 1)], {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 33.0)})
        oracle = build_merge_oracle([G], seg, voxel_size)
        assert len(oracle) == 0

    def test_background_label_excluded(self):
        """Nodes falling in background (seg=0) are not included in pairs."""
        seg = np.array([[[0, 2]]], dtype=np.uint64)
        voxel_size = (33.0, 33.0, 33.0)
        # Node 0 → seg 0 (background); Node 1 → seg 2
        G = _make_graph([(0, 1)], {0: (0.0, 0.0, 0.0), 1: (33.0, 0.0, 0.0)})
        oracle = build_merge_oracle([G], seg, voxel_size)
        assert len(oracle) == 0

    def test_out_of_bounds_nodes_ignored(self):
        """Nodes whose positions fall outside the segmentation volume are skipped."""
        seg = self._simple_seg()
        voxel_size = (33.0, 33.0, 33.0)
        # z=9999 → far out of bounds
        G = _make_graph([(0, 1)], {0: (0.0, 0.0, 0.0), 1: (0.0, 0.0, 9999.0)})
        oracle = build_merge_oracle([G], seg, voxel_size)
        # Node 1 is out of bounds → pair is incomplete → nothing added
        assert len(oracle) == 0

    def test_pairs_are_canonical_order(self):
        """Pairs are always stored as (min, max) regardless of edge direction."""
        seg = self._simple_seg()
        voxel_size = (33.0, 33.0, 33.0)
        # Edge from seg 2 to seg 1 (reversed direction)
        G = _make_graph([(0, 1)], {0: (33.0, 0.0, 0.0), 1: (0.0, 0.0, 0.0)})
        oracle = build_merge_oracle([G], seg, voxel_size)
        assert (1, 2) in oracle
        assert (2, 1) not in oracle

    def test_multiple_graphs(self):
        """Multiple skeleton graphs accumulate pairs across axons."""
        seg = np.array([[[1, 2, 3]]], dtype=np.uint64)
        voxel_size = (33.0, 33.0, 33.0)
        # Axon 1: node spans seg 1→2
        G1 = _make_graph([(0, 1)], {0: (0.0, 0.0, 0.0), 1: (33.0, 0.0, 0.0)})
        # Axon 2: node spans seg 2→3 (but these should NOT merge — different axons,
        #          they just happen to share a node. In a real scenario, the axon
        #          skeleton for axon2 passes through seg2 into seg3.)
        G2 = _make_graph([(0, 1)], {0: (33.0, 0.0, 0.0), 1: (66.0, 0.0, 0.0)})
        oracle = build_merge_oracle([G1, G2], seg, voxel_size)
        assert (1, 2) in oracle
        assert (2, 3) in oracle

    def test_seg_offset(self):
        """Segment offset shifts the coordinate mapping correctly."""
        seg = self._simple_seg()  # shape (3, 1, 2)
        voxel_size = (33.0, 33.0, 33.0)
        # With offset (1, 0, 0) in voxels, physical z=33nm maps to voxel z=0
        # Node 0: x=0nm, z=33nm → with oz=1: zi = round(33/33)-1 = 0 → seg[0,0,0]=1
        # Node 1: x=33nm, z=33nm → with oz=1: zi=0, xi=1 → seg[0,0,1]=2
        G = _make_graph([(0, 1)], {0: (0.0, 0.0, 33.0), 1: (33.0, 0.0, 33.0)})
        oracle = build_merge_oracle([G], seg, voxel_size, seg_offset_voxels=(1, 0, 0))
        assert (1, 2) in oracle

    def test_empty_graph(self):
        """Empty skeleton graph produces no pairs."""
        seg = self._simple_seg()
        voxel_size = (33.0, 33.0, 33.0)
        G = nx.Graph()
        oracle = build_merge_oracle([G], seg, voxel_size)
        assert len(oracle) == 0

    def test_no_graphs(self):
        """Empty graph list produces empty oracle."""
        seg = self._simple_seg()
        oracle = build_merge_oracle([], seg, (33.0, 33.0, 33.0))
        assert len(oracle) == 0


# ---------------------------------------------------------------------------
# evaluate_decisions_xpress
# ---------------------------------------------------------------------------


@pytest.fixture
def store_and_oracle():
    """
    4 fragments with label_ids matching segment IDs:
      frag 0 → seg 1, frag 1 → seg 2 (should merge, per oracle)
      frag 2 → seg 3, frag 3 → seg 4 (should NOT merge)
    Oracle says: (1, 2) should merge.
    """
    store = FragmentStore()
    store.add(_make_fragment(0, 1))
    store.add(_make_fragment(1, 2))
    store.add(_make_fragment(2, 3))
    store.add(_make_fragment(3, 4))
    oracle = {(1, 2)}
    return store, oracle


class TestEvaluateDecisionsXpress:
    def test_true_positive(self, store_and_oracle):
        store, oracle = store_and_oracle
        candidates = [_make_candidate(0, 0, 1, ConnectionStatus.ACCEPTED)]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        assert result["true_positives"] == 1
        assert result["false_positives"] == 0
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_false_positive(self, store_and_oracle):
        store, oracle = store_and_oracle
        # Accepting frag 2↔3 (segs 3,4) — not in oracle
        candidates = [_make_candidate(0, 2, 3, ConnectionStatus.ACCEPTED)]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        assert result["false_positives"] == 1
        assert result["true_positives"] == 0
        assert result["precision"] == 0.0

    def test_true_negative(self, store_and_oracle):
        store, oracle = store_and_oracle
        candidates = [_make_candidate(0, 2, 3, ConnectionStatus.REJECTED)]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        assert result["true_negatives"] == 1
        assert result["false_negatives"] == 0

    def test_false_negative(self, store_and_oracle):
        store, oracle = store_and_oracle
        # Rejecting frag 0↔1 (segs 1,2) — should have merged
        candidates = [_make_candidate(0, 0, 1, ConnectionStatus.REJECTED)]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        assert result["false_negatives"] == 1
        assert result["recall"] == 0.0

    def test_ambiguous_same(self, store_and_oracle):
        store, oracle = store_and_oracle
        candidates = [_make_candidate(0, 0, 1, ConnectionStatus.AMBIGUOUS)]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        assert result["ambiguous_same_label"] == 1
        assert result["ambiguous_diff_label"] == 0
        assert result["true_positives"] == 0

    def test_ambiguous_diff(self, store_and_oracle):
        store, oracle = store_and_oracle
        candidates = [_make_candidate(0, 2, 3, ConnectionStatus.AMBIGUOUS)]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        assert result["ambiguous_diff_label"] == 1
        assert result["ambiguous_same_label"] == 0

    def test_empty_oracle(self, store_and_oracle):
        """Empty oracle → any accepted candidate is a FP, any rejected is a TN."""
        store, _ = store_and_oracle
        candidates = [
            _make_candidate(0, 0, 1, ConnectionStatus.ACCEPTED),
            _make_candidate(1, 2, 3, ConnectionStatus.REJECTED),
        ]
        result = evaluate_decisions_xpress(candidates, store, set())
        assert result["false_positives"] == 1
        assert result["true_negatives"] == 1
        assert result["true_positives"] == 0

    def test_missing_fragment_skipped(self):
        """Candidates referencing unknown fragment IDs are skipped without error."""
        store = FragmentStore()
        store.add(_make_fragment(0, 1))
        oracle = {(1, 99)}
        candidates = [_make_candidate(0, 0, 99, ConnectionStatus.ACCEPTED)]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        # frag 99 not in store → skipped
        assert result["true_positives"] == 0
        assert result["false_positives"] == 0

    def test_mixed_decisions(self, store_and_oracle):
        store, oracle = store_and_oracle
        candidates = [
            _make_candidate(0, 0, 1, ConnectionStatus.ACCEPTED),  # TP (oracle has (1,2))
            _make_candidate(1, 2, 3, ConnectionStatus.REJECTED),  # TN
            _make_candidate(2, 0, 2, ConnectionStatus.ACCEPTED),  # FP (segs 1,3 not in oracle)
            _make_candidate(3, 0, 1, ConnectionStatus.REJECTED),  # FN
        ]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        assert result["true_positives"] == 1
        assert result["false_positives"] == 1
        assert result["true_negatives"] == 1
        assert result["false_negatives"] == 1
        assert abs(result["precision"] - 0.5) < 1e-9
        assert abs(result["recall"] - 0.5) < 1e-9

    def test_f1_formula(self, store_and_oracle):
        """F1 is harmonic mean of precision and recall."""
        store, oracle = store_and_oracle
        candidates = [
            _make_candidate(0, 0, 1, ConnectionStatus.ACCEPTED),  # TP
            _make_candidate(1, 2, 3, ConnectionStatus.ACCEPTED),  # FP
        ]
        result = evaluate_decisions_xpress(candidates, store, oracle)
        p = 0.5
        r = 1.0
        expected_f1 = 2 * p * r / (p + r)
        assert abs(result["f1"] - expected_f1) < 1e-9
