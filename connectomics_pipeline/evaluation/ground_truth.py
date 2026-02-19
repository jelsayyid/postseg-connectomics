"""Ground truth evaluation using label IDs as merge oracle.

For datasets where label_id represents the true neuron identity (e.g. CREMI),
two fragments with the same label_id should be merged and two with different
label_ids should not. This module scores pipeline decisions against that oracle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

from connectomics_pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from connectomics_pipeline.fragments.store import FragmentStore
    from connectomics_pipeline.utils.types import CandidateConnection

logger = get_logger("evaluation.ground_truth")


def evaluate_decisions(
    candidates: List[CandidateConnection],
    store: FragmentStore,
) -> Dict[str, object]:
    """Score pipeline decisions against label-ID ground truth.

    A candidate is a *positive* (should merge) when both fragments share the
    same label_id and a *negative* (should not merge) when they differ.

    Ambiguous decisions are tallied separately and excluded from
    precision/recall so they don't artificially inflate either metric.

    Args:
        candidates: All candidates produced by the pipeline (any status).
        store: Fragment store used to look up label IDs.

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

        should_merge = frag_a.label_id == frag_b.label_id
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
        "Ground truth evaluation: precision=%.3f recall=%.3f F1=%.3f "
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
