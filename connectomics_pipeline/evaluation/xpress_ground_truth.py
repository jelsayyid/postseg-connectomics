"""Ground truth evaluation using XPRESS skeleton graphs as merge oracle.

For the XPRESS challenge dataset (X-ray nanotomography of mouse white matter),
ground truth is provided as NetworkX skeleton graphs rather than voxel label IDs.
Two fragments should be merged when a skeleton edge spans their boundary — i.e.,
the axon skeleton passes from one segment into another, indicating a split error.

Reference:
  XPRESS Challenge — https://xpress.grand-challenge.org/
  Skeletons stored as NetworkX graphs in .npz files with node attribute 'position'
  containing (x, y, z) coordinates in nm.

Skeleton .npz format (from xray-challenge-eval):
  np.load(path, allow_pickle=True)['arr_0'] → dict {skel_id: nx.Graph}
  Each node in the graph has attrs: {'position': (x_nm, y_nm, z_nm), ...}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from connectomics_pipeline.fragments.store import FragmentStore
    from connectomics_pipeline.utils.types import CandidateConnection, ValidationReport

logger = logging.getLogger("evaluation.xpress_ground_truth")


def load_skeleton_graphs(npz_path: Union[str, Path]) -> List:
    """Load XPRESS skeleton graphs from a .npz file.

    The XPRESS .npz format stores either:
    - A dict mapping skeleton_id → nx.Graph (typical case, one graph per axon)
    - A single nx.Graph
    - A 0-d object array wrapping either of the above

    Each graph node has a 'position' attribute: (x_nm, y_nm, z_nm).

    Args:
        npz_path: Path to the .npz skeleton file (e.g. XPRESS_training_skels.npz).

    Returns:
        List of NetworkX graph objects, one per axon skeleton.

    Raises:
        ValueError: If the file format is not recognized.
        ImportError: If networkx is not installed.
    """
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("networkx is required for XPRESS evaluation: pip install networkx") from e

    # The actual XPRESS .npz file is a raw NumPy pickle (np.save, not np.savez),
    # so np.load returns the object directly rather than an NpzFile.
    data = np.load(npz_path, allow_pickle=True)

    # Case 1: raw pickle — np.load returns the Graph directly
    if isinstance(data, nx.Graph):
        # Combined graph with all axons; skeleton_id node attribute identifies each axon.
        # Edges only exist within an axon, so the whole graph can be treated as one unit
        # for edge-based oracle construction.
        return [data]

    # Case 2: NpzFile (np.savez format)
    key = list(data.keys())[0]
    obj = data[key]

    # Unwrap 0-d object arrays
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        obj = obj.item()

    if isinstance(obj, dict):
        # Dict format: {skel_id: nx.Graph}
        graphs = [v for v in obj.values() if isinstance(v, nx.Graph)]
        if not graphs:
            raise ValueError(f"Dict in {npz_path} contains no NetworkX graphs")
        return graphs
    elif isinstance(obj, nx.Graph):
        return [obj]
    elif hasattr(obj, "__iter__"):
        graphs = [g for g in obj if isinstance(g, nx.Graph)]
        if not graphs:
            raise ValueError(f"Iterable in {npz_path} contains no NetworkX graphs")
        return graphs
    else:
        raise ValueError(
            f"Unrecognized skeleton format in {npz_path}: {type(obj)}. "
            "Expected nx.Graph (raw pickle) or dict {{skel_id: nx.Graph}} in npz."
        )


def build_merge_oracle(
    skeleton_graphs: List,
    segmentation: np.ndarray,
    voxel_size_nm: Tuple[float, float, float],
    seg_offset_voxels: Tuple[int, int, int] = (0, 0, 0),
) -> Set[Tuple[int, int]]:
    """Build the set of fragment ID pairs that ground truth says should be merged.

    For each edge (u, v) in every skeleton graph, both node positions are mapped
    to voxel coordinates and looked up in the segmentation array. If u and v fall
    in different non-background segments, the pair is added to the merge oracle.

    This edge-based approach is conservative: it only marks pairs as positive
    when an axon skeleton directly crosses a segment boundary, which is the
    canonical definition of a split error in the XPRESS evaluation.

    Args:
        skeleton_graphs: List of NetworkX graphs from load_skeleton_graphs().
        segmentation: 3D integer array (z, y, x) of segment IDs. Background is 0.
        voxel_size_nm: (z_size, y_size, x_size) voxel dimensions in nm.
        seg_offset_voxels: (z, y, x) origin offset of the segmentation array
            within the full volume. Use this when running on a cropped sub-volume.

    Returns:
        Set of (seg_id_a, seg_id_b) pairs with seg_id_a < seg_id_b that should
        be merged according to the skeleton ground truth.
    """
    vz, vy, vx = voxel_size_nm
    oz, oy, ox = seg_offset_voxels
    vol_shape = segmentation.shape  # (Z, Y, X)

    def pos_to_voxel(pos) -> Optional[Tuple[int, int, int]]:
        """Convert (x_nm, y_nm, z_nm) position to (zi, yi, xi) voxel index."""
        try:
            x_nm, y_nm, z_nm = float(pos[0]), float(pos[1]), float(pos[2])
        except (TypeError, IndexError):
            return None
        zi = int(round(z_nm / vz)) - oz
        yi = int(round(y_nm / vy)) - oy
        xi = int(round(x_nm / vx)) - ox
        if 0 <= zi < vol_shape[0] and 0 <= yi < vol_shape[1] and 0 <= xi < vol_shape[2]:
            return (zi, yi, xi)
        return None

    merge_pairs: Set[Tuple[int, int]] = set()
    total_nodes = 0
    mapped_nodes = 0

    for graph in skeleton_graphs:
        # Pre-compute segment label for each node in this graph
        node_to_seg: Dict[object, int] = {}
        for node, attrs in graph.nodes(data=True):
            total_nodes += 1
            pos = attrs.get("position")
            if pos is None:
                continue
            voxel = pos_to_voxel(pos)
            if voxel is not None:
                seg_id = int(segmentation[voxel])
                if seg_id != 0:  # exclude background
                    node_to_seg[node] = seg_id
                    mapped_nodes += 1

        # Collect merge pairs from edges crossing segment boundaries
        for u, v in graph.edges():
            seg_u = node_to_seg.get(u)
            seg_v = node_to_seg.get(v)
            if seg_u is None or seg_v is None:
                continue
            if seg_u != seg_v:
                merge_pairs.add((min(seg_u, seg_v), max(seg_u, seg_v)))

    logger.info(
        "XPRESS oracle: %d skeleton graphs, %d/%d nodes mapped to volume, %d merge pairs",
        len(skeleton_graphs),
        mapped_nodes,
        total_nodes,
        len(merge_pairs),
    )
    return merge_pairs


def evaluate_decisions_xpress(
    candidates: List["CandidateConnection"],
    store: "FragmentStore",
    merge_oracle: Set[Tuple[int, int]],
) -> Dict[str, object]:
    """Score pipeline decisions against XPRESS skeleton ground truth.

    A candidate is a *positive* (should merge) when the (label_id_a, label_id_b)
    pair appears in the skeleton-derived merge oracle. Ambiguous decisions are
    tallied separately and excluded from precision/recall.

    This is the XPRESS analogue of evaluate_decisions() in ground_truth.py,
    replacing the label-ID oracle with a skeleton-edge-derived oracle.

    Args:
        candidates: All candidates produced by the pipeline (any status).
        store: Fragment store for looking up each fragment's label_id (= segment ID
               in the original XPRESS baseline segmentation).
        merge_oracle: Set of (seg_a, seg_b) pairs from build_merge_oracle().

    Returns:
        Dict with keys: true_positives, false_positives, true_negatives,
        false_negatives, ambiguous_same_label, ambiguous_diff_label,
        precision, recall, f1.
    """
    tp = fp = tn = fn = 0
    ambiguous_same = ambiguous_diff = 0
    skipped = 0

    for cand in candidates:
        frag_a = store.get(cand.fragment_a)
        frag_b = store.get(cand.fragment_b)
        if frag_a is None or frag_b is None:
            skipped += 1
            continue

        pair = (
            min(frag_a.label_id, frag_b.label_id),
            max(frag_a.label_id, frag_b.label_id),
        )
        should_merge = pair in merge_oracle
        status = cand.status.value  # "accepted" | "rejected" | "ambiguous"

        if status == "accepted":
            if should_merge:
                tp += 1
            else:
                fp += 1
        elif status == "rejected":
            if should_merge:
                fn += 1
            else:
                tn += 1
        else:  # ambiguous — preserve separately, exclude from metrics
            if should_merge:
                ambiguous_same += 1
            else:
                ambiguous_diff += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if skipped:
        logger.warning("Skipped %d candidates with missing fragments", skipped)

    logger.info(
        "XPRESS GT evaluation: precision=%.3f recall=%.3f F1=%.3f "
        "(TP=%d FP=%d TN=%d FN=%d ambiguous=%d/%d)",
        precision,
        recall,
        f1,
        tp,
        fp,
        tn,
        fn,
        ambiguous_same,
        ambiguous_diff,
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "ambiguous_same_label": ambiguous_same,
        "ambiguous_diff_label": ambiguous_diff,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fn_diagnosis(
    candidates: List["CandidateConnection"],
    store: "FragmentStore",
    merge_oracle: Set[Tuple[int, int]],
    report: Optional["ValidationReport"] = None,
) -> List[Dict[str, Any]]:
    """Diagnose false negative candidates: oracle pairs rejected by the pipeline.

    For each candidate that is (a) rejected and (b) in the merge oracle (ground
    truth says it should merge), extract which validation rule fired first with a
    REJECTED decision.  Useful for pinpointing the dominant blocker among the
    remaining FNs.

    Args:
        candidates: All candidate connections (any status).
        store: Fragment store for label_id lookup.
        merge_oracle: Set of (seg_a, seg_b) pairs from build_merge_oracle().
        report: Optional ValidationReport produced during validation.  When
            provided, per-rule outcomes are extracted for each FN.  If None,
            rule attribution is omitted.

    Returns:
        List of dicts sorted by gap_nm ascending, one per FN, with keys:
            candidate_id, fragment_a, fragment_b, label_pair,
            gap_nm, composite_score,
            first_reject_rule (None if rule info unavailable),
            first_reject_reason (None if rule info unavailable),
            all_rule_results (list of rule dicts, empty if report is None).
    """
    from connectomics_pipeline.utils.types import ConnectionStatus

    fn_records: List[Dict[str, Any]] = []
    for cand in candidates:
        if cand.status != ConnectionStatus.REJECTED:
            continue

        frag_a = store.get(cand.fragment_a)
        frag_b = store.get(cand.fragment_b)
        if frag_a is None or frag_b is None:
            continue

        pair = (
            min(frag_a.label_id, frag_b.label_id),
            max(frag_a.label_id, frag_b.label_id),
        )
        if pair not in merge_oracle:
            continue  # true negative — not a FN

        # Extract rule-level information from the ValidationReport
        rule_results_raw = []
        first_reject_rule: Optional[str] = None
        first_reject_reason: Optional[str] = None

        if report is not None:
            rule_results_raw = report.results.get(cand.candidate_id, [])
            first_reject = next(
                (r for r in rule_results_raw if r.decision == ConnectionStatus.REJECTED),
                None,
            )
            if first_reject is not None:
                first_reject_rule = first_reject.rule_name
                first_reject_reason = first_reject.reason

        fn_records.append(
            {
                "candidate_id": cand.candidate_id,
                "fragment_a": cand.fragment_a,
                "fragment_b": cand.fragment_b,
                "label_pair": pair,
                "gap_nm": round(cand.gap_distance, 1),
                "composite_score": round(cand.composite_score, 4),
                "first_reject_rule": first_reject_rule,
                "first_reject_reason": first_reject_reason,
                "all_rule_results": [
                    {
                        "rule": r.rule_name,
                        "decision": r.decision.value,
                        "confidence": round(r.confidence, 4),
                        "reason": r.reason,
                    }
                    for r in rule_results_raw
                ],
            }
        )

    fn_records.sort(key=lambda x: x["gap_nm"])

    logger.info(
        "FN diagnosis: %d false negatives among %d rejected candidates",
        len(fn_records),
        sum(1 for c in candidates if c.status == ConnectionStatus.REJECTED),
    )
    return fn_records
