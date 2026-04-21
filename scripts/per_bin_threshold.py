"""Per-bin composite-score threshold: Pareto-optimal precision ceiling.

The gap-stratified analysis (Experiment 26) showed that XPRESS per-bin
precision varies by ~30x.  This script quantifies the precision ceiling
achievable by tuning the composite accept threshold separately in each
gap bin, instead of using the single global threshold of Experiment 24.

Two strategies are evaluated:

  1. Global baseline: rank all accepted candidates by composite score
     (descending) and sweep an acceptance prefix. This is the current
     pipeline behavior after the fact.
  2. Per-bin Pareto: within each gap bin, order candidates by composite
     (descending); for each target global recall, use a Lagrangian sweep
     to find the allocation across bins that maximizes global precision.
     This is the precision ceiling for any pipeline variant whose only
     degree of freedom is a per-gap-bin composite threshold.

Outputs (written to out-dir):
  - per_bin_threshold_summary.csv
  - per_bin_threshold_pr.png
  - per_bin_threshold_curves.csv     full PR curves for both strategies
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from connectomics_pipeline.evaluation.xpress_ground_truth import (
    build_merge_oracle,
    load_skeleton_graphs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("per_bin_threshold")

DEFAULT_BINS_NM = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 20000]


def load_accepted_with_gt(
    output_dir: Path,
    seg_path: Path,
    seg_key: str,
    skel_path: Path,
    resolution_nm: float,
    seg_offset: tuple[int, int, int],
) -> pd.DataFrame:
    log.info("Loading skeletons + segmentation to build ground-truth oracle")
    graphs = load_skeleton_graphs(skel_path)
    with h5py.File(seg_path, "r") as f:
        seg = f[seg_key][...]
    gt_pairs = build_merge_oracle(
        graphs,
        seg,
        voxel_size_nm=(resolution_nm, resolution_nm, resolution_nm),
        seg_offset_voxels=seg_offset,
    )

    frags = pd.read_csv(output_dir / "fragments.csv", usecols=["fragment_id", "label_id"])
    frag_to_label = dict(zip(frags["fragment_id"], frags["label_id"]))

    conn = pd.read_csv(
        output_dir / "connections.csv",
        usecols=["fragment_a", "fragment_b", "gap_distance", "composite_score", "status"],
    )
    accepted = conn[conn["status"] == "accepted"].copy().reset_index(drop=True)
    accepted["label_a"] = accepted["fragment_a"].map(frag_to_label).astype("Int64")
    accepted["label_b"] = accepted["fragment_b"].map(frag_to_label).astype("Int64")
    accepted = accepted.dropna(subset=["label_a", "label_b"]).copy()
    accepted["label_a"] = accepted["label_a"].astype(np.int64)
    accepted["label_b"] = accepted["label_b"].astype(np.int64)
    lo = np.minimum(accepted["label_a"], accepted["label_b"])
    hi = np.maximum(accepted["label_a"], accepted["label_b"])
    accepted["gt_should_merge"] = [k in gt_pairs for k in zip(lo.tolist(), hi.tolist())]
    log.info(
        "Accepted: %d (%d TP, %d FP)",
        len(accepted),
        int(accepted["gt_should_merge"].sum()),
        int((~accepted["gt_should_merge"]).sum()),
    )
    return accepted


def global_pr_curve(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """PR curve for a global sort by score (descending)."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    cum_n = np.arange(1, len(sorted_labels) + 1)
    total_tp = sorted_labels.sum()
    precision = cum_tp / cum_n
    recall = cum_tp / total_tp if total_tp > 0 else np.zeros_like(cum_tp, dtype=float)
    return pd.DataFrame({"kept": cum_n, "tp": cum_tp, "precision": precision, "recall": recall})


def per_bin_pareto_pr_curve(
    labels: np.ndarray, scores: np.ndarray, bins: np.ndarray
) -> pd.DataFrame:
    """Pareto frontier achievable by independent per-bin thresholds.

    Within each bin, sort candidates by score descending.  The globally
    optimal allocation for any target recall is to merge all per-bin sorted
    streams and accept candidates in order of descending score --- i.e., the
    standard global ranking IS the Pareto frontier when features are
    bin-independent.  To get the *per-bin* ceiling we must instead merge
    streams by "cost per next TP": at each step, add the candidate from the
    bin whose next unpicked candidate is most likely to be a TP (i.e.
    whose within-bin rank has the highest observed TP probability).

    We approximate this by Lagrangian sweep: for each slope lambda, each bin
    keeps candidates whose observed within-bin marginal FP-per-TP rate is
    below lambda.  Sweeping lambda traces the Pareto frontier.

    In practice, because score is informative within each bin, we use a
    simpler construction: for each bin independently compute the sorted
    (cum_tp, cum_fp) arc, then greedy-merge across bins by next TP gain.
    """
    bin_ids = np.unique(bins)
    per_bin_sorted: list[tuple[int, np.ndarray, np.ndarray]] = []
    for b in bin_ids:
        mask = bins == b
        bscores = scores[mask]
        blabels = labels[mask]
        order = np.argsort(-bscores, kind="mergesort")
        per_bin_sorted.append((b, blabels[order], bscores[order]))

    # Greedy merge: at each step, among bins with remaining candidates, pick
    # the one whose NEXT candidate is a TP if any, else whose NEXT candidate
    # has the highest *remaining* TP-rate (i.e. future TPs / future candidates).
    pointers = {b: 0 for b, _, _ in per_bin_sorted}
    blabel_map = {b: lab for b, lab, _ in per_bin_sorted}
    blen_map = {b: len(lab) for b, lab, _ in per_bin_sorted}
    total_tp = int(labels.sum())

    kept_tp = 0
    kept_n = 0
    cum_tp_arr = []
    cum_n_arr = []
    bin_ids_list = [b for b, _, _ in per_bin_sorted]

    remaining_tps = {
        b: int(blabel_map[b].sum()) for b in bin_ids_list
    }

    while True:
        candidate_bins = [b for b in bin_ids_list if pointers[b] < blen_map[b]]
        if not candidate_bins:
            break

        # Prefer bins whose NEXT candidate is a TP (greedy: free precision gain).
        next_tp_bins = [b for b in candidate_bins if blabel_map[b][pointers[b]] == 1]
        if next_tp_bins:
            # Tie-break: take from the bin with highest remaining TP density
            chosen = max(
                next_tp_bins,
                key=lambda b: remaining_tps[b] / max(1, blen_map[b] - pointers[b]),
            )
        else:
            # No free TPs. Pick the bin whose remaining TP density (future TPs / future cands)
            # is HIGHEST --- i.e. where our next FP is least costly in expectation.
            chosen = max(
                candidate_bins,
                key=lambda b: remaining_tps[b] / max(1, blen_map[b] - pointers[b]),
            )

        lab = blabel_map[chosen][pointers[chosen]]
        pointers[chosen] += 1
        kept_tp += int(lab)
        kept_n += 1
        if lab == 1:
            remaining_tps[chosen] -= 1
        cum_tp_arr.append(kept_tp)
        cum_n_arr.append(kept_n)

    cum_tp_arr = np.array(cum_tp_arr)
    cum_n_arr = np.array(cum_n_arr)
    precision = cum_tp_arr / cum_n_arr
    recall = cum_tp_arr / total_tp if total_tp > 0 else np.zeros_like(cum_tp_arr, dtype=float)
    return pd.DataFrame(
        {"kept": cum_n_arr, "tp": cum_tp_arr, "precision": precision, "recall": recall}
    )


def precision_at_recall(curve: pd.DataFrame, target: float) -> tuple[float, int, int]:
    mask = curve["recall"].to_numpy() >= target
    if not mask.any():
        return float("nan"), 0, 0
    idx = int(np.argmax(mask))
    row = curve.iloc[idx]
    return float(row["precision"]), int(row["kept"]), int(row["tp"])


def make_figure(curves: dict[str, pd.DataFrame], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"global": "#1f77b4", "per_bin_pareto": "#d62728"}
    labels = {
        "global": "Global composite threshold (baseline)",
        "per_bin_pareto": "Per-bin oracle (greedy TP-density merge)",
    }
    for key, curve in curves.items():
        ax.plot(curve["recall"], curve["precision"], label=labels[key], color=colors[key], linewidth=1.4)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.02)
    ax.set_title(
        "Per-bin composite threshold precision ceiling\n"
        "(XPRESS Experiment 24 accepted candidates)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("output/xpress_training"))
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--seg-key", default="volumes/segmentation_0.550")
    ap.add_argument("--skel", type=Path, required=True)
    ap.add_argument("--resolution", type=float, default=33.0)
    ap.add_argument("--seg-offset", type=int, nargs=3, default=(0, 0, 0))
    ap.add_argument("--out-dir", type=Path, default=Path("docs"))
    ap.add_argument("--bin-edges", type=float, nargs="+", default=DEFAULT_BINS_NM)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    accepted = load_accepted_with_gt(
        args.output_dir, args.seg, args.seg_key, args.skel, args.resolution, tuple(args.seg_offset)
    )

    labels = accepted["gt_should_merge"].to_numpy().astype(int)
    scores = accepted["composite_score"].to_numpy()
    bin_ids = pd.cut(
        accepted["gap_distance"], bins=args.bin_edges, right=False, include_lowest=True, labels=False
    ).to_numpy()

    log.info("Computing global PR curve")
    global_curve = global_pr_curve(labels, scores)
    log.info("Computing per-bin Pareto PR curve")
    pareto_curve = per_bin_pareto_pr_curve(labels, scores, bin_ids)

    curves = {"global": global_curve, "per_bin_pareto": pareto_curve}

    rows = []
    for target in (0.99, 0.95, 0.90, 0.85, 0.80, 0.70):
        p_g, k_g, tp_g = precision_at_recall(global_curve, target)
        p_p, k_p, tp_p = precision_at_recall(pareto_curve, target)
        rel = (p_p - p_g) / p_g * 100 if p_g > 0 else float("nan")
        rows.append(
            {
                "recall_target": target,
                "global_precision": p_g,
                "global_kept": k_g,
                "global_tp": tp_g,
                "pareto_precision": p_p,
                "pareto_kept": k_p,
                "pareto_tp": tp_p,
                "relative_gain_pct": rel,
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = args.out_dir / "per_bin_threshold_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    log.info("Wrote %s", summary_path)

    combined = pd.concat(
        [
            global_curve.assign(strategy="global"),
            pareto_curve.assign(strategy="per_bin_pareto"),
        ]
    )
    curves_path = args.out_dir / "per_bin_threshold_curves.csv"
    combined.to_csv(curves_path, index=False)
    log.info("Wrote %s", curves_path)

    make_figure(curves, args.out_dir / "per_bin_threshold_pr.png")


if __name__ == "__main__":
    main()
