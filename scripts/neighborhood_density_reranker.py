"""Neighborhood-density re-ranker for XPRESS accepted candidates.

Hypothesis: a genuine same-axon split pair is locally isolated (its axon is the
primary structure in the neighborhood), while a cross-axon false-positive pair
sits in a crowd of other axons that makes many geometrically-plausible
alternatives to the pipeline's chosen partner.  A simple density count around
the candidate midpoint should therefore separate TPs from FPs even after the
composite score has already accepted both.

This script tests the hypothesis as a post-hoc re-ranker on the Experiment 24
accepted-candidate set.  No pipeline code is modified.

Features computed per candidate (a, b):
    mid_nm    = (centroid_a + centroid_b) / 2
    density_mid(R)  = |{c != a,b : dist(centroid_c, mid_nm) <= R}|
    density_max(R)  = max(|nbrs(a,R) \\ {a,b}|, |nbrs(b,R) \\ {a,b}|)

Outputs (written to out-dir):
    - density_summary.csv              per-radius, per-feature PR summary
    - density_pr_curve.png             precision-recall curves at R=1500nm
    - density_distribution.png         TP vs FP density histograms
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from connectomics_pipeline.evaluation.xpress_ground_truth import (
    build_merge_oracle,
    load_skeleton_graphs,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("density_reranker")

DEFAULT_RADII_NM = [500, 1000, 1500, 2000, 3000]


def load_gt_pairs(
    seg_path: Path,
    seg_key: str,
    skel_path: Path,
    resolution_nm: float,
    seg_offset: tuple[int, int, int] = (0, 0, 0),
) -> set[tuple[int, int]]:
    log.info("Loading skeleton graphs from %s", skel_path)
    graphs = load_skeleton_graphs(skel_path)
    log.info("Loading segmentation %s:%s", seg_path, seg_key)
    with h5py.File(seg_path, "r") as f:
        seg = f[seg_key][...]
    pairs = build_merge_oracle(
        graphs,
        seg,
        voxel_size_nm=(resolution_nm, resolution_nm, resolution_nm),
        seg_offset_voxels=seg_offset,
    )
    log.info("Ground-truth merge pairs: %d", len(pairs))
    return pairs


def compute_features(
    accepted: pd.DataFrame,
    frag_centroids_nm: np.ndarray,
    frag_id_to_row: dict[int, int],
    radii_nm: list[float],
) -> pd.DataFrame:
    tree = cKDTree(frag_centroids_nm)
    n = len(accepted)
    log.info("Computing density features for %d accepted candidates at %d radii", n, len(radii_nm))

    idx_a = accepted["fragment_a"].map(frag_id_to_row).to_numpy()
    idx_b = accepted["fragment_b"].map(frag_id_to_row).to_numpy()
    centroid_a = frag_centroids_nm[idx_a]
    centroid_b = frag_centroids_nm[idx_b]
    midpoints = 0.5 * (centroid_a + centroid_b)

    for r in radii_nm:
        counts_mid = tree.query_ball_point(midpoints, r=r, return_length=True)
        counts_a = tree.query_ball_point(centroid_a, r=r, return_length=True)
        counts_b = tree.query_ball_point(centroid_b, r=r, return_length=True)
        accepted[f"density_mid_{int(r)}"] = counts_mid - 2  # exclude a and b if they fall inside
        accepted[f"density_max_{int(r)}"] = np.maximum(counts_a, counts_b) - 2

    accepted[[c for c in accepted.columns if c.startswith("density_")]] = (
        accepted[[c for c in accepted.columns if c.startswith("density_")]].clip(lower=0)
    )
    return accepted


def pr_curve(labels: np.ndarray, scores_ascending: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (precision, recall) arrays for a monotone re-ranker.

    Candidates are taken in order of scores_ascending (lowest first) → accepted
    at each prefix size. labels: 1 = TP, 0 = FP.
    """
    order = np.argsort(scores_ascending, kind="mergesort")
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    cum_n = np.arange(1, len(sorted_labels) + 1)
    total_tp = sorted_labels.sum()
    precision = cum_tp / cum_n
    recall = cum_tp / total_tp if total_tp > 0 else np.zeros_like(cum_tp, dtype=float)
    return precision, recall


def precision_at_recall(precision: np.ndarray, recall: np.ndarray, target: float) -> tuple[float, int]:
    mask = recall >= target
    if not mask.any():
        return float("nan"), 0
    idx = int(np.argmax(mask))
    return float(precision[idx]), idx + 1


def summarize(
    accepted: pd.DataFrame, radii_nm: list[float], out_csv: Path
) -> pd.DataFrame:
    labels = accepted["gt_should_merge"].to_numpy().astype(int)
    rows = []

    # Baseline: composite score (descending = lower rank first)
    neg_composite = -accepted["composite_score"].to_numpy()
    p_comp, r_comp = pr_curve(labels, neg_composite)
    for target in (0.99, 0.95, 0.90, 0.85):
        p_at, n_kept = precision_at_recall(p_comp, r_comp, target)
        rows.append(
            {
                "feature": "composite_score (baseline)",
                "radius_nm": None,
                "recall_target": target,
                "precision": p_at,
                "candidates_kept": n_kept,
            }
        )

    # Density features: lower density first
    for r in radii_nm:
        for feat in (f"density_mid_{int(r)}", f"density_max_{int(r)}"):
            p_d, r_d = pr_curve(labels, accepted[feat].to_numpy().astype(float))
            for target in (0.99, 0.95, 0.90, 0.85):
                p_at, n_kept = precision_at_recall(p_d, r_d, target)
                rows.append(
                    {
                        "feature": feat,
                        "radius_nm": r,
                        "recall_target": target,
                        "precision": p_at,
                        "candidates_kept": n_kept,
                    }
                )

    # Combined: composite - alpha * log(density_mid + 1)
    for r in radii_nm:
        for alpha in (0.05, 0.10, 0.20):
            feat = (
                -accepted["composite_score"].to_numpy()
                + alpha * np.log1p(accepted[f"density_mid_{int(r)}"].to_numpy())
            )
            p_c, r_c = pr_curve(labels, feat)
            for target in (0.99, 0.95, 0.90):
                p_at, n_kept = precision_at_recall(p_c, r_c, target)
                rows.append(
                    {
                        "feature": f"composite + {alpha} * log(1+density_mid_{int(r)})",
                        "radius_nm": r,
                        "recall_target": target,
                        "precision": p_at,
                        "candidates_kept": n_kept,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    log.info("Wrote summary %s", out_csv)
    return df


def make_figures(accepted: pd.DataFrame, radius_nm: int, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = accepted["gt_should_merge"].to_numpy().astype(int)

    # PR curves
    fig, ax = plt.subplots(figsize=(8, 6))
    neg_composite = -accepted["composite_score"].to_numpy()
    p_comp, r_comp = pr_curve(labels, neg_composite)
    ax.plot(r_comp, p_comp, label="composite_score (baseline)", color="#1f77b4", linewidth=1.3)

    for feat_name, color in (
        (f"density_mid_{radius_nm}", "#d62728"),
        (f"density_max_{radius_nm}", "#2ca02c"),
    ):
        p_d, r_d = pr_curve(labels, accepted[feat_name].to_numpy().astype(float))
        ax.plot(r_d, p_d, label=feat_name, color=color, linewidth=1.3)

    # Combined feature
    alpha = 0.10
    feat = (
        -accepted["composite_score"].to_numpy()
        + alpha * np.log1p(accepted[f"density_mid_{radius_nm}"].to_numpy())
    )
    p_c, r_c = pr_curve(labels, feat)
    ax.plot(
        r_c,
        p_c,
        label=f"composite + {alpha}·log(1+density_mid_{radius_nm})",
        color="#9467bd",
        linewidth=1.3,
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_yscale("log")
    ax.set_xlim(0, 1.02)
    ax.set_title(f"Neighborhood-density re-ranker PR curves (XPRESS Exp 24, R={radius_nm} nm)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "density_pr_curve.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)

    # Distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, feat_name in zip(
        axes, (f"density_mid_{radius_nm}", f"density_max_{radius_nm}")
    ):
        tp = accepted.loc[accepted["gt_should_merge"], feat_name].to_numpy()
        fp = accepted.loc[~accepted["gt_should_merge"], feat_name].to_numpy()
        bins = np.linspace(0, max(np.percentile(fp, 99), 5), 40)
        ax.hist(fp, bins=bins, density=True, alpha=0.55, label=f"FP (n={len(fp):,})", color="#d62728")
        ax.hist(tp, bins=bins, density=True, alpha=0.7, label=f"TP (n={len(tp)})", color="#2ca02c")
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Density (normalised)")
        ax.set_title(
            f"{feat_name}: "
            f"TP mean={tp.mean():.1f}, FP mean={fp.mean():.1f}"
        )
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"TP vs FP neighborhood-density distributions (R={radius_nm} nm)")
    fig.tight_layout()
    out_path = out_dir / "density_distribution.png"
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
    ap.add_argument("--radii-nm", type=float, nargs="+", default=DEFAULT_RADII_NM)
    ap.add_argument("--fig-radius-nm", type=int, default=1500)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_pairs = load_gt_pairs(
        args.seg,
        args.seg_key,
        args.skel,
        args.resolution,
        tuple(args.seg_offset),
    )

    log.info("Loading %s", args.output_dir / "fragments.csv")
    frags = pd.read_csv(
        args.output_dir / "fragments.csv",
        usecols=["fragment_id", "label_id", "centroid_z", "centroid_y", "centroid_x"],
    )
    frag_centroids_nm = frags[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=np.float64)
    frag_id_to_row = {fid: i for i, fid in enumerate(frags["fragment_id"].to_numpy())}
    frag_to_label = dict(zip(frags["fragment_id"], frags["label_id"]))

    log.info("Loading %s", args.output_dir / "connections.csv")
    conn = pd.read_csv(
        args.output_dir / "connections.csv",
        usecols=["fragment_a", "fragment_b", "gap_distance", "composite_score", "status"],
    )
    accepted = conn[conn["status"] == "accepted"].copy().reset_index(drop=True)
    accepted["label_a"] = accepted["fragment_a"].map(frag_to_label)
    accepted["label_b"] = accepted["fragment_b"].map(frag_to_label)
    accepted = accepted.dropna(subset=["label_a", "label_b"]).copy()
    accepted["label_a"] = accepted["label_a"].astype(np.int64)
    accepted["label_b"] = accepted["label_b"].astype(np.int64)
    lo = np.minimum(accepted["label_a"], accepted["label_b"])
    hi = np.maximum(accepted["label_a"], accepted["label_b"])
    accepted["gt_should_merge"] = [k in gt_pairs for k in zip(lo.tolist(), hi.tolist())]

    log.info(
        "Accepted: %d (%d TP, %d FP); missing fragments dropped",
        len(accepted),
        int(accepted["gt_should_merge"].sum()),
        int((~accepted["gt_should_merge"]).sum()),
    )

    accepted = compute_features(accepted, frag_centroids_nm, frag_id_to_row, args.radii_nm)

    summary = summarize(accepted, args.radii_nm, args.out_dir / "density_summary.csv")

    # Pretty-print a focused view
    focus = summary[summary["recall_target"].isin([0.99, 0.95])].copy()
    focus["precision"] = focus["precision"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "NA")
    print("\n=== Precision at recall targets ===")
    print(focus.to_string(index=False))

    make_figures(accepted, args.fig_radius_nm, args.out_dir)

    # Write enriched candidate file for downstream inspection
    enriched_path = args.out_dir / "density_enriched_accepted.csv"
    accepted_out_cols = [
        "fragment_a",
        "fragment_b",
        "gap_distance",
        "composite_score",
        "gt_should_merge",
    ] + [c for c in accepted.columns if c.startswith("density_")]
    accepted[accepted_out_cols].to_csv(enriched_path, index=False)
    log.info("Wrote %s (%d rows)", enriched_path, len(accepted))


if __name__ == "__main__":
    main()
