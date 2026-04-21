"""Stratify XPRESS precision/recall by candidate gap distance.

Overall XPRESS Experiment 24 precision is ~0.003, which is a single aggregate
that hides large structure.  This script partitions the 369,570 candidates
into gap-distance bins and recomputes precision, recall, and candidate volume
within each bin.  The output pinpoints where the precision problem lives and
which bins contribute the bulk of the 1,175 TPs vs the ~367K FPs.

Outputs (written to docs/):
    - gap_stratified_precision.csv     per-bin table with all counts + metrics
    - gap_stratified_precision.png     two-panel figure (precision and TP/FP
                                       counts vs gap bin)

Reproducibility:
    python scripts/gap_stratified_precision.py \\
        --output-dir output/xpress_training \\
        --seg data/xpress/baseline_seg_training.h5 \\
        --seg-key volumes/segmentation_0.550 \\
        --skel data/xpress/XPRESS_training_skels.npz \\
        --resolution 33 \\
        --out-dir docs
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
log = logging.getLogger("gap_stratified")

DEFAULT_BINS_NM = [0, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 20000]


def load_gt_pairs(
    seg_path: Path,
    seg_key: str,
    skel_path: Path,
    resolution_nm: float,
    seg_offset: tuple[int, int, int] = (0, 0, 0),
) -> set[tuple[int, int]]:
    log.info("Loading skeleton graphs from %s", skel_path)
    graphs = load_skeleton_graphs(skel_path)
    log.info("Loaded %d skeleton graph(s)", len(graphs))

    log.info("Loading segmentation %s:%s", seg_path, seg_key)
    with h5py.File(seg_path, "r") as f:
        seg = f[seg_key][...]
    log.info("Segmentation shape=%s dtype=%s", seg.shape, seg.dtype)

    pairs = build_merge_oracle(
        graphs,
        seg,
        voxel_size_nm=(resolution_nm, resolution_nm, resolution_nm),
        seg_offset_voxels=seg_offset,
    )
    log.info("Ground-truth merge pairs: %d", len(pairs))
    return pairs


def join_candidates_with_gt(
    connections_csv: Path,
    fragments_csv: Path,
    gt_pairs: set[tuple[int, int]],
) -> pd.DataFrame:
    log.info("Loading %s", fragments_csv)
    frags = pd.read_csv(fragments_csv, usecols=["fragment_id", "label_id"])
    frag_to_label = dict(zip(frags["fragment_id"], frags["label_id"]))

    log.info("Loading %s", connections_csv)
    conn = pd.read_csv(
        connections_csv,
        usecols=["fragment_a", "fragment_b", "gap_distance", "composite_score", "status"],
    )
    conn["label_a"] = conn["fragment_a"].map(frag_to_label)
    conn["label_b"] = conn["fragment_b"].map(frag_to_label)
    missing = conn["label_a"].isna().sum() + conn["label_b"].isna().sum()
    if missing:
        log.warning("Dropped %d candidates with unknown fragment->label mapping", missing)
        conn = conn.dropna(subset=["label_a", "label_b"]).copy()
    conn["label_a"] = conn["label_a"].astype(np.int64)
    conn["label_b"] = conn["label_b"].astype(np.int64)

    lo = np.minimum(conn["label_a"], conn["label_b"])
    hi = np.maximum(conn["label_a"], conn["label_b"])
    keys = list(zip(lo.tolist(), hi.tolist()))
    conn["gt_should_merge"] = [k in gt_pairs for k in keys]
    log.info(
        "Candidates: %d total, %d accepted, %d rejected, %d GT-positive pairs represented",
        len(conn),
        (conn["status"] == "accepted").sum(),
        (conn["status"] == "rejected").sum(),
        conn["gt_should_merge"].sum(),
    )
    return conn


def stratify_by_gap(conn: pd.DataFrame, bin_edges_nm: list[float]) -> pd.DataFrame:
    bins = pd.cut(conn["gap_distance"], bins=bin_edges_nm, right=False, include_lowest=True)

    def agg(group: pd.DataFrame) -> pd.Series:
        acc = group["status"] == "accepted"
        rej = group["status"] == "rejected"
        gt = group["gt_should_merge"]
        tp = int((acc & gt).sum())
        fp = int((acc & ~gt).sum())
        fn = int((rej & gt).sum())
        tn = int((rej & ~gt).sum())
        n_candidates = len(group)
        n_accepted = int(acc.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        return pd.Series(
            {
                "candidates": n_candidates,
                "accepted": n_accepted,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
            }
        )

    table = conn.groupby(bins, observed=True).apply(agg).reset_index()
    table = table.rename(columns={"gap_distance": "gap_bin_nm"})
    return table


def make_figure(table: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(b) for b in table["gap_bin_nm"]]
    x = np.arange(len(labels))

    fig, (ax_prec, ax_cnt) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)

    ax_prec.bar(x, table["precision"], color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax_prec.set_ylabel("Precision (within bin)")
    ax_prec.set_title("XPRESS Experiment 24: precision and candidate volume by gap distance")
    ax_prec.set_yscale("log")
    ax_prec.grid(axis="y", alpha=0.3)
    for xi, p in zip(x, table["precision"]):
        if pd.notna(p) and p > 0:
            ax_prec.text(xi, p, f"{p:.3g}", ha="center", va="bottom", fontsize=8)

    width = 0.42
    ax_cnt.bar(x - width / 2, table["tp"], width, label="TP (accepted ∧ GT)", color="#2ca02c")
    ax_cnt.bar(x + width / 2, table["fp"], width, label="FP (accepted ∧ ¬GT)", color="#d62728")
    ax_cnt.set_yscale("symlog", linthresh=10)
    ax_cnt.set_ylabel("Candidates (symlog)")
    ax_cnt.set_xlabel("Gap distance bin (nm)")
    ax_cnt.set_xticks(x)
    ax_cnt.set_xticklabels(labels, rotation=30, ha="right")
    ax_cnt.legend(loc="upper left")
    ax_cnt.grid(axis="y", alpha=0.3)
    for xi, (tp, fp) in enumerate(zip(table["tp"], table["fp"])):
        if tp > 0:
            ax_cnt.text(xi - width / 2, tp, str(int(tp)), ha="center", va="bottom", fontsize=8)
        if fp > 0:
            ax_cnt.text(xi + width / 2, fp, f"{int(fp):,}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote figure %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("output/xpress_training"))
    ap.add_argument("--seg", type=Path, required=True)
    ap.add_argument("--seg-key", default="volumes/segmentation_0.550")
    ap.add_argument("--skel", type=Path, required=True)
    ap.add_argument("--resolution", type=float, default=33.0, help="Isotropic voxel size in nm")
    ap.add_argument("--seg-offset", type=int, nargs=3, default=(0, 0, 0))
    ap.add_argument("--out-dir", type=Path, default=Path("docs"))
    ap.add_argument(
        "--bin-edges",
        type=float,
        nargs="+",
        default=DEFAULT_BINS_NM,
        help="Bin edges in nm",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt_pairs = load_gt_pairs(
        args.seg,
        args.seg_key,
        args.skel,
        args.resolution,
        tuple(args.seg_offset),
    )

    conn = join_candidates_with_gt(
        args.output_dir / "connections.csv",
        args.output_dir / "fragments.csv",
        gt_pairs,
    )
    table = stratify_by_gap(conn, args.bin_edges)

    csv_path = args.out_dir / "gap_stratified_precision.csv"
    table.to_csv(csv_path, index=False)
    log.info("Wrote table %s", csv_path)
    print(table.to_string(index=False))

    total_tp = int(table["tp"].sum())
    total_fp = int(table["fp"].sum())
    total_fn = int(table["fn"].sum())
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    log.info(
        "Overall: TP=%d FP=%d FN=%d precision=%.4f recall=%.4f",
        total_tp,
        total_fp,
        total_fn,
        overall_precision,
        overall_recall,
    )

    make_figure(table, args.out_dir / "gap_stratified_precision.png")


if __name__ == "__main__":
    main()
