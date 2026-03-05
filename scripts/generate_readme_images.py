#!/usr/bin/env python3
"""Generate PNG showcase images for embedding in the GitHub README.

Produces two sets of images saved to docs/images/:
  candidate_showcase_accepted.png  — 3 high-confidence accepted pairs
  candidate_showcase_rejected.png  — 3 borderline rejected pairs

Each set shows a grid of 3 candidates × 3 panels:
  Panel 1 — Neutral segmentation (distinct muted colors, no A/B highlight)
             with skeleton node overlay (yellow dots) if --skel is provided.
  Panel 2 — Colorized prediction: Fragment A=red, Fragment B=blue,
             connection arrow, ACC/REJ badge, score breakdown.
  Panel 3 — Ground truth: same seg with oracle badge (TP/FP/FN/TN),
             highlighted skeleton paths for the relevant neurons,
             and merge verdict annotation.

Usage
-----
    python scripts/generate_readme_images.py \\
        --output-dir output/xpress_training \\
        --seg /tmp/xpress_full.h5 \\
        --seg-key volumes/segmentation_0.550 \\
        --skel data/xpress/XPRESS_training_skels.npz \\
        --resolution 33

    # CREMI with raw EM image
    python scripts/generate_readme_images.py \\
        --output-dir output/cremi_sample_a \\
        --seg data/cremi_crop.hdf \\
        --seg-key labels \\
        --raw data/sample_A_20160501.hdf \\
        --raw-key volumes/raw \\
        --resolution 40 4 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CROP_HALF = 100  # → 200×200 voxel crop per panel (spans ~6600nm, enough to show both fragments)


# ---------------------------------------------------------------------------
# Re-use helpers from generate_visual_report
# ---------------------------------------------------------------------------

def _load_crop(path, key, z, y0, y1, x0, x1, dtype=None):
    with h5py.File(path, "r") as f:
        dset = f[key]
        nz, ny, nx = dset.shape
        z_cl = int(np.clip(z, 0, nz - 1))
        ay0, ay1 = max(0, y0), min(ny, y1)
        ax0, ax1 = max(0, x0), min(nx, x1)
        crop = dset[z_cl, ay0:ay1, ax0:ax1]
    if dtype:
        crop = crop.astype(dtype)
    return crop, (ay0, ay1, ax0, ax1)


def _colorize_seg(crop, label_a, label_b):
    out = np.zeros((*crop.shape, 4), dtype=np.float32)
    out[crop == 0] = [0.07, 0.07, 0.07, 1.0]
    out[(crop != 0) & (crop != label_a) & (crop != label_b)] = [0.55, 0.55, 0.55, 1.0]
    out[crop == label_a] = [0.85, 0.18, 0.18, 1.0]
    out[crop == label_b] = [0.12, 0.47, 0.71, 1.0]
    return out


def _neutral_seg(crop):
    out = np.zeros((*crop.shape, 4), dtype=np.float32)
    out[crop == 0] = [0.07, 0.07, 0.07, 1.0]
    nonbg = crop != 0
    if nonbg.any():
        hues = (crop[nonbg].astype(np.uint64) * 2654435761 % (2**32)) / (2**32)
        hues = 0.08 + hues * 0.84
        r = np.clip(np.abs(hues * 6 - 3) - 1, 0, 1) * 0.5 + 0.35
        g = np.clip(2 - np.abs(hues * 6 - 2), 0, 1) * 0.5 + 0.35
        b = np.clip(2 - np.abs(hues * 6 - 4), 0, 1) * 0.5 + 0.35
        out[nonbg, 0] = r
        out[nonbg, 1] = g
        out[nonbg, 2] = b
        out[nonbg, 3] = 1.0
    return out


def _raw_to_rgba(raw_crop):
    gray = raw_crop.astype(np.float32)
    lo, hi = gray.min(), gray.max()
    if hi > lo:
        gray = (gray - lo) / (hi - lo)
    return np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)


def _load_skel(npz_path):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from connectomics_pipeline.evaluation.xpress_ground_truth import load_skeleton_graphs
    import networkx as nx
    graphs = load_skeleton_graphs(npz_path)
    if len(graphs) == 1:
        return graphs[0]
    combined = nx.Graph()
    for g in graphs:
        combined.update(g)
    return combined


def _build_oracle(seg_path, seg_key, skel_graph, res_z, res_y, res_x, seg_offset):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from connectomics_pipeline.evaluation.xpress_ground_truth import build_merge_oracle
    with h5py.File(seg_path, "r") as f:
        seg = f[seg_key][:]
    oracle = build_merge_oracle([skel_graph], seg, (res_z, res_y, res_x), seg_offset)
    del seg
    return oracle


def _skel_nodes_near(skel_graph, z_slice_vox, res_z, res_y, res_x, seg_offset, z_half=4):
    oz, oy, ox = seg_offset
    z_center_nm = (z_slice_vox + oz) * res_z
    lo, hi = z_center_nm - z_half * res_z, z_center_nm + z_half * res_z
    result = []
    for node, data in skel_graph.nodes(data=True):
        pos = data.get("position")
        if pos is None:
            continue
        x_nm, y_nm, z_nm = float(pos[0]), float(pos[1]), float(pos[2])
        if lo <= z_nm <= hi:
            result.append((y_nm, x_nm, data.get("skeleton_id", -1), node))
    return result


def _skel_edges_among(skel_graph, near_nodes):
    pos_map = {n[3]: (n[1], n[0]) for n in near_nodes}
    edges = []
    for u, v in skel_graph.edges():
        if u in pos_map and v in pos_map:
            edges.append((pos_map[u], pos_map[v]))
    return edges


def _nm_to_px(y_nm, x_nm, crop_y0, crop_x0, res_y, res_x, seg_offset):
    _, oy, ox = seg_offset
    y_vox = y_nm / res_y - oy
    x_vox = x_nm / res_x - ox
    return x_vox - crop_x0, y_vox - crop_y0


def _overlay_skel(ax, near_nodes, near_edges, crop_y0, crop_x0, res_y, res_x,
                  seg_offset, crop_h, crop_w, highlight_ids=None):
    # Edges
    for (x1, y1), (x2, y2) in near_edges:
        px1, py1 = _nm_to_px(y1, x1, crop_y0, crop_x0, res_y, res_x, seg_offset)
        px2, py2 = _nm_to_px(y2, x2, crop_y0, crop_x0, res_y, res_x, seg_offset)
        if not (max(px1, px2) < 0 or min(px1, px2) >= crop_w
                or max(py1, py2) < 0 or min(py1, py2) >= crop_h):
            ax.plot([px1, px2], [py1, py2], "-", color="#ffff00", lw=0.8, alpha=0.5, zorder=3)
    # Nodes
    for y_nm, x_nm, sid, _ in near_nodes:
        px, py = _nm_to_px(y_nm, x_nm, crop_y0, crop_x0, res_y, res_x, seg_offset)
        if 0 <= px < crop_w and 0 <= py < crop_h:
            highlighted = highlight_ids and sid in highlight_ids
            color = "#ffdd00" if highlighted else "white"
            alpha = 0.9 if highlighted else 0.45
            size = 4.5 if highlighted else 2.5
            ax.plot(px, py, "o", color=color, markersize=size, alpha=alpha,
                    markeredgewidth=0.0, zorder=4)


def _find_highlight_ids(skel_graph, label_a, label_b, seg_path, seg_key,
                        res_z, res_y, res_x, seg_offset, seg_shape):
    oz, oy, ox = seg_offset

    def _pos_to_vox(pos):
        x_nm, y_nm, z_nm = float(pos[0]), float(pos[1]), float(pos[2])
        zi = int(round(z_nm / res_z)) - oz
        yi = int(round(y_nm / res_y)) - oy
        xi = int(round(x_nm / res_x)) - ox
        nz, ny, nx = seg_shape
        if 0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx:
            return zi, yi, xi
        return None

    from collections import defaultdict
    by_skel = defaultdict(list)
    for node, data in skel_graph.nodes(data=True):
        pos = data.get("position")
        sid = data.get("skeleton_id", -1)
        if pos is not None:
            by_skel[sid].append(pos)

    found = set()
    with h5py.File(seg_path, "r") as f:
        dset = f[seg_key]
        for sid, positions in by_skel.items():
            for pos in positions[:20]:
                vox = _pos_to_vox(pos)
                if vox is None:
                    continue
                try:
                    lbl = int(dset[vox])
                except Exception:
                    continue
                if lbl == label_a or lbl == label_b:
                    found.add(sid)
                    break
    return found


def _oracle_outcome(cand_row, frag_info, oracle):
    if oracle is None:
        return "NA"
    fa, fb = int(cand_row["fragment_a"]), int(cand_row["fragment_b"])
    if fa not in frag_info.index or fb not in frag_info.index:
        return "NA"
    la = int(frag_info.loc[fa, "label_id"])
    lb = int(frag_info.loc[fb, "label_id"])
    should_merge = (min(la, lb), max(la, lb)) in oracle
    acc = cand_row["status"] == "accepted"
    if acc and should_merge:
        return "TP"
    if acc and not should_merge:
        return "FP"
    if not acc and should_merge:
        return "FN"
    return "TN"


OUTCOME_COLORS = {"TP": "#1a9641", "FP": "#d7191c", "FN": "#e07020", "TN": "#888888", "NA": "#cccccc"}
OUTCOME_LABELS = {
    "TP": "TP — Correct accept",
    "FP": "FP — False accept",
    "FN": "FN — Missed merge",
    "TN": "TN — Correct reject",
    "NA": "GT: N/A",
}


# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------

def _pick_candidates(connections, fragments, n, category, seg_path, seg_key,
                     res_z, res_y, res_x):
    frag_info = fragments[["fragment_id", "label_id", "centroid_z", "centroid_y", "centroid_x"]].set_index("fragment_id")
    frag_label = frag_info["label_id"]

    # Pre-filter same-label candidates at the DataFrame level so nlargest works correctly
    base = connections[connections["status"] == ("accepted" if category == "accepted" else "rejected")].copy()
    la_col = base["fragment_a"].map(frag_label)
    lb_col = base["fragment_b"].map(frag_label)
    base = base[(la_col != lb_col) & la_col.notna() & lb_col.notna()]
    pool = base.nlargest(500, "composite_score")

    chosen = []
    for _, row in pool.iterrows():
        if len(chosen) >= n:
            break
        fa, fb = int(row["fragment_a"]), int(row["fragment_b"])
        if fa not in frag_info.index or fb not in frag_info.index:
            continue
        info_a, info_b = frag_info.loc[fa], frag_info.loc[fb]
        label_a, label_b = int(info_a["label_id"]), int(info_b["label_id"])
        # Skip degenerate same-label candidates (gap=0, score=1.0)
        if label_a == label_b:
            continue

        cz_a, cy_a, cx_a = info_a["centroid_z"] / res_z, info_a["centroid_y"] / res_y, info_a["centroid_x"] / res_x
        cz_b, cy_b, cx_b = info_b["centroid_z"] / res_z, info_b["centroid_y"] / res_y, info_b["centroid_x"] / res_x

        mid_z = int(round((cz_a + cz_b) / 2))
        mid_y = int(round((cy_a + cy_b) / 2))
        mid_x = int(round((cx_a + cx_b) / 2))

        try:
            crop, (ay0, _, ax0, _) = _load_crop(
                seg_path, seg_key, mid_z,
                mid_y - CROP_HALF, mid_y + CROP_HALF,
                mid_x - CROP_HALF, mid_x + CROP_HALF,
                dtype=np.int64,
            )
        except Exception:
            continue

        if (crop == label_a).sum() == 0 or (crop == label_b).sum() == 0:
            continue

        chosen.append({
            "row": row,
            "frag_a": fa, "frag_b": fb,
            "label_a": label_a, "label_b": label_b,
            "cy_a": cy_a, "cx_a": cx_a,
            "cy_b": cy_b, "cx_b": cx_b,
            "cz_mid": mid_z,
            "crop": crop, "crop_y0": ay0, "crop_x0": ax0,
        })
    return chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output_dir, seg_path, seg_key, raw_path, raw_key, skel_path,
             seg_offset, resolution, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    connections = pd.read_csv(Path(output_dir) / "connections.csv")
    fragments = pd.read_csv(Path(output_dir) / "fragments.csv")
    frag_info = fragments[["fragment_id", "label_id", "centroid_z", "centroid_y", "centroid_x"]].set_index("fragment_id")

    res_z, res_y, res_x = (
        (resolution[0], resolution[1], resolution[2]) if len(resolution) == 3
        else (resolution[0],) * 3
    )

    # Skeleton + oracle
    skel_graph = None
    oracle = None
    if skel_path:
        print("Loading skeleton graph …", flush=True)
        skel_graph = _load_skel(skel_path)
        oracle = _build_oracle(seg_path, seg_key, skel_graph, res_z, res_y, res_x, seg_offset)

    with h5py.File(seg_path, "r") as f:
        seg_shape = tuple(f[seg_key].shape)

    for category, title_label, border_color, filename in [
        ("accepted", "High-confidence accepted", "#1a9641", "candidate_showcase_accepted.png"),
        ("rejected", "Borderline rejected",       "#d7191c", "candidate_showcase_rejected.png"),
    ]:
        candidates = _pick_candidates(
            connections, fragments, n=3, category=category,
            seg_path=seg_path, seg_key=seg_key,
            res_z=res_z, res_y=res_y, res_x=res_x,
        )

        if not candidates:
            print(f"  No suitable candidates for {category}")
            continue

        n = len(candidates)
        # 3 rows (one per candidate) × 3 cols (panels)
        fig, axes = plt.subplots(
            n, 3,
            figsize=(13, 4.2 * n),
            gridspec_kw={"hspace": 0.45, "wspace": 0.05},
            squeeze=False,
        )

        fig.suptitle(
            title_label,
            fontsize=13, fontweight="bold", color=border_color, y=1.01,
        )

        # Column headers (top row only)
        col_titles = ["Raw / Neutral + Skeleton", "Pipeline Prediction", "Ground Truth"]
        for col_i, col_title in enumerate(col_titles):
            axes[0, col_i].set_title(col_title, fontsize=8, pad=4, color="#333333")

        for row_i, cand in enumerate(candidates):
            crop = cand["crop"]
            label_a, label_b = cand["label_a"], cand["label_b"]
            crop_y0, crop_x0 = cand["crop_y0"], cand["crop_x0"]
            crop_h, crop_w = crop.shape
            mid_z = cand["cz_mid"]
            row = cand["row"]

            seg_rgba = _colorize_seg(crop, label_a, label_b)
            outcome = _oracle_outcome(row, frag_info, oracle)

            # Skeleton data for this slice
            near_nodes, near_edges, highlight_ids = [], [], set()
            if skel_graph is not None:
                near_nodes = _skel_nodes_near(
                    skel_graph, mid_z, res_z, res_y, res_x, seg_offset
                )
                near_edges = _skel_edges_among(skel_graph, near_nodes)
                highlight_ids = _find_highlight_ids(
                    skel_graph, label_a, label_b,
                    seg_path, seg_key, res_z, res_y, res_x, seg_offset, seg_shape,
                )

            # --- Panel 1: neutral/raw + skeleton ---
            ax1 = axes[row_i, 0]
            raw_rgba = None
            if raw_path:
                try:
                    raw_crop, _ = _load_crop(raw_path, raw_key, mid_z,
                                             crop_y0 - 0, crop_y0 + crop_h,
                                             crop_x0 - 0, crop_x0 + crop_w)
                    raw_rgba = _raw_to_rgba(raw_crop)
                except Exception:
                    raw_rgba = None
            panel1 = raw_rgba if raw_rgba is not None else _neutral_seg(crop)
            ax1.imshow(panel1, origin="upper", interpolation="nearest")
            if near_nodes:
                _overlay_skel(ax1, near_nodes, near_edges, crop_y0, crop_x0,
                              res_y, res_x, seg_offset, crop_h, crop_w, highlight_ids)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_ylabel(
                f"cand {int(row['candidate_id'])}\nfrag_id {cand['frag_a']}↔{cand['frag_b']}\nlbl {label_a}↔{label_b}\ngap {row['gap_distance']:.0f}nm",
                fontsize=6.0, rotation=90, labelpad=3, va="center",
            )
            for spine in ax1.spines.values():
                spine.set_edgecolor(border_color); spine.set_linewidth(2.5)
            if near_nodes:
                ax1.legend(handles=[mpatches.Patch(color="#ffdd00", label="Skeleton")],
                           fontsize=4.5, loc="lower left", framealpha=0.6,
                           handlelength=0.6, borderpad=0.2)

            # --- Panel 2: prediction ---
            ax2 = axes[row_i, 1]
            ax2.imshow(seg_rgba, origin="upper", interpolation="nearest")

            def _to_px(cy, cx):
                return (float(np.clip(cx - crop_x0, 0, crop_w - 1)),
                        float(np.clip(cy - crop_y0, 0, crop_h - 1)))

            px_a, py_a = _to_px(cand["cy_a"], cand["cx_a"])
            px_b, py_b = _to_px(cand["cy_b"], cand["cx_b"])

            if abs(px_b - px_a) + abs(py_b - py_a) > 2:
                ax2.annotate("", xy=(px_b, py_b), xytext=(px_a, py_a),
                             arrowprops=dict(arrowstyle="-|>", color="white",
                                             lw=2.0, mutation_scale=14), zorder=5)
            ax2.plot(px_a, py_a, "o", color="#ff6666", markersize=7,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)
            ax2.plot(px_b, py_b, "o", color="#66aaff", markersize=7,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=6)

            badge_color = "#1a9641" if row["status"] == "accepted" else "#d7191c"
            badge_text = "ACC" if row["status"] == "accepted" else "REJ"
            ax2.text(crop_w - 2, 3, badge_text, fontsize=7.5, color="white",
                     fontweight="bold", ha="right", va="top", zorder=7,
                     bbox=dict(facecolor=badge_color, alpha=0.88, pad=1.5, edgecolor="none"))
            ax2.set_xticks([])
            ax2.set_yticks([])

            score_line = (
                f"prox={row.get('proximity_score', float('nan')):.2f}  "
                f"align={row.get('alignment_score', float('nan')):.2f}  "
                f"cont={row.get('continuity_score', float('nan')):.2f}  "
                f"comp={row['composite_score']:.3f}"
            )
            if row_i == 0:
                ax2.set_title(f"Pipeline Prediction\n{score_line}", fontsize=6, pad=3)
            else:
                ax2.set_title(score_line, fontsize=6, pad=3)

            ax2.legend(handles=[
                mpatches.Patch(color="#dd3333", label="Frag A"),
                mpatches.Patch(color="#2277cc", label="Frag B"),
            ], fontsize=5, loc="lower right", framealpha=0.6,
               handlelength=0.8, borderpad=0.2, labelspacing=0.1)

            for spine in ax2.spines.values():
                spine.set_edgecolor(border_color); spine.set_linewidth(2.5)

            # --- Panel 3: ground truth ---
            ax3 = axes[row_i, 2]
            ax3.imshow(seg_rgba.copy(), origin="upper", interpolation="nearest")

            if near_nodes:
                _overlay_skel(ax3, near_nodes, near_edges, crop_y0, crop_x0,
                              res_y, res_x, seg_offset, crop_h, crop_w,
                              highlight_ids=highlight_ids)

            if outcome in ("TP", "FN"):
                ax3.plot([px_a, px_b], [py_a, py_b], "--",
                         color="#ffdd00", lw=1.8, alpha=0.9, zorder=5)

            badge_bg = OUTCOME_COLORS.get(outcome, "#cccccc")
            ax3.text(crop_w / 2, 4, OUTCOME_LABELS.get(outcome, outcome),
                     fontsize=7.5, color="white", fontweight="bold",
                     ha="center", va="top", zorder=8,
                     bbox=dict(facecolor=badge_bg, alpha=0.90, pad=2.0, edgecolor="none"))

            if outcome != "NA":
                should_merge = outcome in ("TP", "FN")
                verdict = "Oracle: SHOULD MERGE" if should_merge else "Oracle: NO MERGE"
                verdict_col = "#1a9641" if should_merge else "#d7191c"
                ax3.text(crop_w / 2, crop_h - 3, verdict, fontsize=5.5, color=verdict_col,
                         fontweight="bold", ha="center", va="bottom", zorder=8,
                         bbox=dict(facecolor="white", alpha=0.75, pad=1.0, edgecolor="none"))

            ax3.set_xticks([])
            ax3.set_yticks([])
            if row_i == 0:
                ax3.set_title("Ground Truth", fontsize=8, pad=3)

            for spine in ax3.spines.values():
                spine.set_edgecolor(OUTCOME_COLORS.get(outcome, "#cccccc"))
                spine.set_linewidth(3)

        # Shared legend
        shared_patches = [
            mpatches.Patch(color="#dd3333", label="Fragment A"),
            mpatches.Patch(color="#2277cc", label="Fragment B"),
            mpatches.Patch(color="#888888", label="Other fragments"),
            mpatches.Patch(color="#111111", label="Background (label 0)"),
            mpatches.Patch(color="#ffdd00", label="Skeleton (GT neuron path)"),
        ]
        outcome_patches = [
            mpatches.Patch(color="#1a9641", label="TP — correct accept"),
            mpatches.Patch(color="#d7191c", label="FP — false accept"),
            mpatches.Patch(color="#e07020", label="FN — missed merge"),
            mpatches.Patch(color="#888888", label="TN — correct reject"),
        ]
        fig.legend(
            handles=shared_patches + outcome_patches,
            ncol=5,
            loc="lower center",
            fontsize=7,
            framealpha=0.0,
            bbox_to_anchor=(0.5, -0.04),
            handlelength=1.0,
        )

        out_path = out_dir / filename
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seg", required=True)
    parser.add_argument("--seg-key", default="labels")
    parser.add_argument("--raw", default=None, help="Optional raw EM HDF5 volume")
    parser.add_argument("--raw-key", default="volumes/raw")
    parser.add_argument("--skel", default=None, help="Optional skeleton .npz for GT oracle")
    parser.add_argument("--seg-offset", nargs=3, type=int, default=[0, 0, 0],
                        metavar=("Z", "Y", "X"))
    parser.add_argument("--resolution", nargs="+", type=float, default=[33.0])
    parser.add_argument("--out-dir", default="docs/images")
    args = parser.parse_args()

    print(f"Generating README showcase images from {args.output_dir} ...")
    generate(
        output_dir=args.output_dir,
        seg_path=args.seg,
        seg_key=args.seg_key,
        raw_path=args.raw,
        raw_key=args.raw_key,
        skel_path=args.skel,
        seg_offset=tuple(args.seg_offset),
        resolution=args.resolution,
        out_dir=args.out_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
