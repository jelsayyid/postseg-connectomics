#!/usr/bin/env python3
"""Generate a visual validation report as a multi-page PDF.

Each sampled candidate occupies one row of THREE panels side-by-side:

  Panel 1 — Raw / Neutral:
      Grayscale EM image (if --raw is provided) or a neutral gray segmentation
      slice (no A/B coloring) with skeleton graph overlay showing neuron paths
      near the Z-slice. Gives context: what does the tissue actually look like?

  Panel 2 — Pipeline Prediction:
      Colorized segmentation (Fragment A = red, Fragment B = blue, others = gray)
      with the proposed connection arrow and an ACC/REJ decision badge.
      Also shows all five component scores.

  Panel 3 — Ground Truth:
      Same colorized segmentation with the oracle verdict overlaid:
        TP (accepted + should merge)   — green badge, skeleton crossing shown
        FP (accepted + should not)     — red badge
        FN (rejected + should merge)   — orange badge, skeleton crossing shown
        TN (rejected + should not)     — gray badge
      Skeleton nodes/edges for the fragment pair's neurons are highlighted.
      Without --skel the oracle overlay is omitted and "GT: N/A" is shown.

Candidates sampled (configurable via --n-samples):
  - High-confidence accepted  : highest composite score among accepted
  - Low-confidence accepted   : lowest  composite score among accepted
  - Borderline rejected       : highest composite score among rejected (hard cases)
  - Oracle FN sample          : false negatives from oracle (requires --skel)
  - Random rejected           : random sample of rejected

Usage
-----
    # XPRESS training — segmentation only
    python scripts/generate_visual_report.py \\
        --output-dir output/xpress_training \\
        --seg /tmp/xpress_full.h5 \\
        --seg-key volumes/segmentation_0.550 \\
        --resolution 33 \\
        --out output/xpress_training/visual_report.pdf

    # XPRESS + skeleton oracle overlay
    python scripts/generate_visual_report.py \\
        --output-dir output/xpress_training \\
        --seg /tmp/xpress_full.h5 \\
        --seg-key volumes/segmentation_0.550 \\
        --skel data/xpress/XPRESS_training_skels.npz \\
        --resolution 33 \\
        --out output/xpress_training/visual_report.pdf

    # CREMI (anisotropic, with raw EM image)
    python scripts/generate_visual_report.py \\
        --output-dir output/cremi_sample_a \\
        --seg data/cremi_crop.hdf \\
        --seg-key labels \\
        --raw data/sample_A_20160501.hdf \\
        --raw-key volumes/raw \\
        --resolution 40 4 4 \\
        --out output/cremi_sample_a/visual_report.pdf

    # XPRESS validation set (volume offset required)
    python scripts/generate_visual_report.py \\
        --output-dir output/xpress_validation \\
        --seg data/xpress/baseline_seg_validation.h5 \\
        --seg-key volumes/segmentation_0.550 \\
        --skel data/xpress/XPRESS_validation_skels.npz \\
        --seg-offset 252 252 252 \\
        --resolution 33 \\
        --out output/xpress_validation/visual_report.pdf

Arguments
---------
--output-dir    Pipeline output directory (contains connections.csv, fragments.csv).
--seg           HDF5 segmentation volume.
--seg-key       HDF5 dataset key for segmentation (default: labels).
--raw           Optional HDF5 volume for raw EM image (grayscale, Panel 1).
--raw-key       HDF5 dataset key for raw image (default: volumes/raw).
--skel          Optional skeleton .npz file for ground truth oracle overlay.
--seg-offset    Voxel offset of segmentation within full volume, z y x (default: 0 0 0).
--resolution    Voxel size in nm: one value (isotropic) or three values z y x.
--crop-half     Half-width of the 2D crop in voxels (default: 60 → 120×120 crop).
--z-half        Half-thickness (voxels) for skeleton node projection (default: 2).
--n-samples     Candidates per category (default: 5).
--out           Output PDF path (default: <output-dir>/visual_report.pdf).
--seed          Random seed (default: 42).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "Ground-truth TP":          "#2166ac",
    "High-confidence accepted": "#1a9641",
    "Low-confidence accepted":  "#fdae61",
    "Borderline rejected":      "#d7191c",
    "GT FN":                    "#e07020",
    "Random rejected":          "#999999",
}

OUTCOME_COLORS = {
    "TP": "#1a9641",  # green
    "FP": "#d7191c",  # red
    "FN": "#e07020",  # orange
    "TN": "#888888",  # gray
    "NA": "#cccccc",  # no oracle
}


# ---------------------------------------------------------------------------
# Volume helpers
# ---------------------------------------------------------------------------


def _load_crop(
    path: str,
    key: str,
    z: int,
    y0: int, y1: int,
    x0: int, x1: int,
    dtype=None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Load a 2D sub-region from a single Z-slice of an HDF5 volume.

    Returns the cropped array and the actual (y0, y1, x0, x1) bounds used
    after clamping to volume dimensions.
    """
    with h5py.File(path, "r") as f:
        dset = f[key]
        nz, ny, nx = dset.shape
        z_cl = int(np.clip(z, 0, nz - 1))
        ay0, ay1 = max(0, y0), min(ny, y1)
        ax0, ax1 = max(0, x0), min(nx, x1)
        crop = dset[z_cl, ay0:ay1, ax0:ax1]
    if dtype is not None:
        crop = crop.astype(dtype)
    return crop, (ay0, ay1, ax0, ax1)


def _colorize_seg(
    crop: np.ndarray,
    label_a: int,
    label_b: int,
) -> np.ndarray:
    """RGBA image: label_a=red, label_b=blue, others=mid-gray, bg=dark."""
    out = np.zeros((*crop.shape, 4), dtype=np.float32)
    out[crop == 0] = [0.07, 0.07, 0.07, 1.0]
    mask_other = (crop != 0) & (crop != label_a) & (crop != label_b)
    out[mask_other] = [0.55, 0.55, 0.55, 1.0]
    out[crop == label_a] = [0.85, 0.18, 0.18, 1.0]
    out[crop == label_b] = [0.12, 0.47, 0.71, 1.0]
    return out


def _neutral_seg(crop: np.ndarray) -> np.ndarray:
    """RGBA: all segments in distinct muted colors for context, bg=dark."""
    out = np.zeros((*crop.shape, 4), dtype=np.float32)
    out[crop == 0] = [0.07, 0.07, 0.07, 1.0]
    nonbg = crop != 0
    if nonbg.any():
        # Hash label IDs to hue (avoid pure red/blue reserved for A/B)
        hues = (crop[nonbg].astype(np.uint64) * 2654435761 % (2**32)) / (2**32)
        hues = 0.08 + hues * 0.84  # keep in [0.08, 0.92] to avoid pure red/blue
        # Convert hue → RGB (simple pastel palette)
        r = np.clip(np.abs(hues * 6 - 3) - 1, 0, 1) * 0.5 + 0.35
        g = np.clip(2 - np.abs(hues * 6 - 2), 0, 1) * 0.5 + 0.35
        b = np.clip(2 - np.abs(hues * 6 - 4), 0, 1) * 0.5 + 0.35
        out[nonbg, 0] = r
        out[nonbg, 1] = g
        out[nonbg, 2] = b
        out[nonbg, 3] = 1.0
    return out


def _raw_to_rgba(raw_crop: np.ndarray) -> np.ndarray:
    """Convert grayscale EM array to RGBA."""
    gray = raw_crop.astype(np.float32)
    lo, hi = gray.min(), gray.max()
    if hi > lo:
        gray = (gray - lo) / (hi - lo)
    out = np.stack([gray, gray, gray, np.ones_like(gray)], axis=-1)
    return out


# ---------------------------------------------------------------------------
# Skeleton helpers
# ---------------------------------------------------------------------------


def _load_combined_skeleton(npz_path: str):
    """Load all skeleton graphs from npz and return as a single combined graph."""
    import sys
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


def _build_oracle(
    seg_path: str,
    seg_key: str,
    skel_graph,
    res_z: float, res_y: float, res_x: float,
    seg_offset: tuple[int, int, int],
) -> set[tuple[int, int]]:
    """Build label-pair merge oracle from skeleton + segmentation.

    Loads the full segmentation into memory — only called once per report.
    Skeleton positions are (x_nm, y_nm, z_nm).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from connectomics_pipeline.evaluation.xpress_ground_truth import build_merge_oracle

    print("  Loading segmentation for oracle construction …", flush=True)
    with h5py.File(seg_path, "r") as f:
        seg = f[seg_key][:]

    oracle = build_merge_oracle(
        [skel_graph],
        seg,
        (res_z, res_y, res_x),
        seg_offset,
    )
    del seg
    print(f"  Oracle: {len(oracle):,} merge pairs", flush=True)
    return oracle


def _skel_nodes_near_slice(
    skel_graph,
    z_slice_vox: int,
    res_z: float, res_y: float, res_x: float,
    seg_offset: tuple[int, int, int],
    z_half: int = 2,
) -> list[tuple[float, float, int, object]]:
    """Return skeleton nodes within z_half voxels of z_slice_vox.

    Skeleton node positions are (x_nm, y_nm, z_nm).
    Returned tuples: (y_nm, x_nm, skeleton_id, node_id).
    """
    oz, oy, ox = seg_offset
    z_center_nm = (z_slice_vox + oz) * res_z
    z_lo = z_center_nm - z_half * res_z
    z_hi = z_center_nm + z_half * res_z

    result = []
    for node, data in skel_graph.nodes(data=True):
        pos = data.get("position")
        if pos is None:
            continue
        x_nm, y_nm, z_nm = float(pos[0]), float(pos[1]), float(pos[2])
        if z_lo <= z_nm <= z_hi:
            result.append((y_nm, x_nm, data.get("skeleton_id", -1), node))
    return result


def _skel_edges_among_nodes(
    skel_graph,
    near_nodes: list,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return skeleton edges where both endpoints are in near_nodes."""
    pos_map = {n[3]: (n[1], n[0]) for n in near_nodes}  # node_id → (x_nm, y_nm)
    edges = []
    for u, v in skel_graph.edges():
        if u in pos_map and v in pos_map:
            edges.append((pos_map[u], pos_map[v]))
    return edges


def _nm_to_crop_px(
    y_nm: float, x_nm: float,
    seg_crop_y0: int, seg_crop_x0: int,
    res_y: float, res_x: float,
    seg_offset: tuple[int, int, int],
) -> tuple[float, float]:
    """Convert (y_nm, x_nm) skeleton position to crop pixel coords."""
    _, oy, ox = seg_offset
    y_vox = y_nm / res_y - oy
    x_vox = x_nm / res_x - ox
    return x_vox - seg_crop_x0, y_vox - seg_crop_y0  # (px_x, px_y) for imshow


def _overlay_skeleton(
    ax: plt.Axes,
    near_nodes: list,
    near_edges: list,
    seg_crop_y0: int, seg_crop_x0: int,
    res_y: float, res_x: float,
    seg_offset: tuple[int, int, int],
    crop_h: int, crop_w: int,
    highlight_skel_ids: set | None = None,
    base_alpha: float = 0.55,
    highlight_alpha: float = 0.9,
    node_size: float = 3.0,
) -> None:
    """Draw skeleton nodes and edges onto an existing Axes."""
    # Assign colors per skeleton_id
    skel_ids = sorted({n[2] for n in near_nodes})
    cmap = plt.colormaps["tab20"].resampled(max(len(skel_ids), 1))
    id_to_color = {sid: cmap(i) for i, sid in enumerate(skel_ids)}

    # Draw edges first (behind nodes)
    for (x1, y1), (x2, y2) in near_edges:
        px1, py1 = _nm_to_crop_px(y1, x1, seg_crop_y0, seg_crop_x0, res_y, res_x, seg_offset)
        px2, py2 = _nm_to_crop_px(y2, x2, seg_crop_y0, seg_crop_x0, res_y, res_x, seg_offset)
        # Only draw if at least partially in-bounds
        if not (max(px1, px2) < 0 or min(px1, px2) >= crop_w
                or max(py1, py2) < 0 or min(py1, py2) >= crop_h):
            # (edge color: average the two endpoint skeleton colors, use yellow for highlighted)
            alpha = highlight_alpha if highlight_skel_ids and (
                # Approximate — edge color from first endpoint
                True
            ) else base_alpha
            ax.plot([px1, px2], [py1, py2], "-", color="#ffff00", lw=0.8, alpha=0.5, zorder=3)

    # Draw nodes
    for y_nm, x_nm, sid, _ in near_nodes:
        px, py = _nm_to_crop_px(y_nm, x_nm, seg_crop_y0, seg_crop_x0, res_y, res_x, seg_offset)
        if 0 <= px < crop_w and 0 <= py < crop_h:
            highlighted = highlight_skel_ids and sid in highlight_skel_ids
            color = "#ffdd00" if highlighted else id_to_color.get(sid, "white")
            alpha = highlight_alpha if highlighted else base_alpha
            size = node_size * 1.5 if highlighted else node_size
            ax.plot(px, py, "o", color=color, markersize=size, alpha=alpha,
                    markeredgewidth=0.0, zorder=4)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample_candidates(
    connections: pd.DataFrame,
    fragments: pd.DataFrame,
    oracle: set[tuple[int, int]] | None,
    n_samples: int,
    seed: int,
) -> list[tuple[str, pd.Series]]:
    """Return (category, row) tuples for the report sample."""
    rng = np.random.default_rng(seed)
    acc = connections[connections["status"] == "accepted"]
    rej = connections[connections["status"] == "rejected"]

    out: list[tuple[str, pd.Series]] = []

    # Pre-compute label_id lookup for same-label filtering (vectorized)
    frag_label = fragments[["fragment_id", "label_id"]].set_index("fragment_id")["label_id"]

    def _diff_label_mask(df: pd.DataFrame) -> pd.Series:
        la = df["fragment_a"].map(frag_label)
        lb = df["fragment_b"].map(frag_label)
        return (la != lb) & la.notna() & lb.notna()

    # Exclude degenerate same-label candidates (gap=0, score=1.0, trivially FP)
    acc_real = acc[_diff_label_mask(acc)]
    rej_real = rej[_diff_label_mask(rej)]

    def _take(df: pd.DataFrame, n: int, largest: bool, col: str = "composite_score") -> list:
        pool = df.nlargest(n * 5, col) if largest else df.nsmallest(n * 5, col)
        return list(pool.head(n).iterrows())

    # Ground-truth TP sample (accepted AND oracle-positive) — the main scientific check
    if oracle is not None:
        frag_info_lkp = fragments[["fragment_id", "label_id"]].set_index("fragment_id")
        tp_rows = []
        for _, row in acc_real.iterrows():
            fa, fb = int(row["fragment_a"]), int(row["fragment_b"])
            if fa not in frag_info_lkp.index or fb not in frag_info_lkp.index:
                continue
            la = int(frag_info_lkp.loc[fa, "label_id"])
            lb = int(frag_info_lkp.loc[fb, "label_id"])
            if (min(la, lb), max(la, lb)) in oracle:
                tp_rows.append(row)
        if tp_rows:
            tp_pool = pd.DataFrame(tp_rows)
            for _, row in _take(tp_pool, n_samples, largest=True):
                out.append(("Ground-truth TP", row))

    # High-confidence accepted
    for _, row in _take(acc_real, n_samples, largest=True):
        out.append(("High-confidence accepted", row))

    # Low-confidence accepted
    for _, row in _take(acc_real, n_samples, largest=False):
        out.append(("Low-confidence accepted", row))

    # Borderline rejected (hardest rejections)
    for _, row in _take(rej_real, n_samples, largest=True):
        out.append(("Borderline rejected", row))

    # GT FN sample (rejected but should merge) — requires oracle
    if oracle is not None:
        frag_info_lkp = fragments[["fragment_id", "label_id"]].set_index("fragment_id")
        fn_rows = []
        for _, row in rej_real.iterrows():
            fa, fb = int(row["fragment_a"]), int(row["fragment_b"])
            if fa not in frag_info_lkp.index or fb not in frag_info_lkp.index:
                continue
            la = int(frag_info_lkp.loc[fa, "label_id"])
            lb = int(frag_info_lkp.loc[fb, "label_id"])
            if (min(la, lb), max(la, lb)) in oracle:
                fn_rows.append(row)
        if fn_rows:
            fn_pool = pd.DataFrame(fn_rows)
            for _, row in _take(fn_pool, n_samples, largest=True):
                out.append(("GT FN", row))

    # Random rejected (excluding same-label)
    n_rand = min(n_samples, len(rej_real))
    idx = rng.choice(len(rej_real), size=n_rand, replace=False)
    for _, row in rej_real.iloc[idx].iterrows():
        out.append(("Random rejected", row))

    return out


# ---------------------------------------------------------------------------
# Oracle outcome per candidate
# ---------------------------------------------------------------------------


def _oracle_outcome(
    cand_row: pd.Series,
    frag_info: pd.DataFrame,
    oracle: set[tuple[int, int]] | None,
) -> str:
    """Return TP / FP / FN / TN / NA for this candidate."""
    if oracle is None:
        return "NA"
    fa, fb = int(cand_row["fragment_a"]), int(cand_row["fragment_b"])
    if fa not in frag_info.index or fb not in frag_info.index:
        return "NA"
    la = int(frag_info.loc[fa, "label_id"])
    lb = int(frag_info.loc[fb, "label_id"])
    should_merge = (min(la, lb), max(la, lb)) in oracle
    accepted = cand_row["status"] == "accepted"
    if accepted and should_merge:
        return "TP"
    if accepted and not should_merge:
        return "FP"
    if not accepted and should_merge:
        return "FN"
    return "TN"


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------


def _centroid_to_crop_px(
    cy_vox: float, cx_vox: float,
    crop_y0: int, crop_x0: int,
    crop_h: int, crop_w: int,
) -> tuple[float, float]:
    """Convert voxel centroid to crop pixel coords, clipped to crop bounds."""
    px = float(np.clip(cx_vox - crop_x0, 0, crop_w - 1))
    py = float(np.clip(cy_vox - crop_y0, 0, crop_h - 1))
    return px, py


def _draw_panel_raw(
    ax: plt.Axes,
    raw_rgba: np.ndarray,
    near_nodes: list,
    near_edges: list,
    seg_crop_y0: int, seg_crop_x0: int,
    res_y: float, res_x: float,
    seg_offset: tuple[int, int, int],
    title: str,
) -> None:
    """Panel 1: raw/neutral image with skeleton overlay."""
    h, w = raw_rgba.shape[:2]
    ax.imshow(raw_rgba, origin="upper", interpolation="nearest")
    if near_nodes:
        _overlay_skeleton(ax, near_nodes, near_edges,
                          seg_crop_y0, seg_crop_x0, res_y, res_x, seg_offset, h, w)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.5, pad=2)
    # Add legend for skeleton
    if near_nodes:
        dot = mpatches.Patch(color="#ffdd00", label="Skeleton nodes")
        ax.legend(handles=[dot], fontsize=4.5, loc="lower left",
                  framealpha=0.6, handlelength=0.6, borderpad=0.2)


def _draw_panel_prediction(
    ax: plt.Axes,
    seg_rgba: np.ndarray,
    cand_row: pd.Series,
    category: str,
    cy_a: float, cx_a: float,
    cy_b: float, cx_b: float,
    crop_y0: int, crop_x0: int,
    label_a: int, label_b: int,
) -> None:
    """Panel 2: colorized segmentation with pipeline decision overlay."""
    h, w = seg_rgba.shape[:2]
    ax.imshow(seg_rgba, origin="upper", interpolation="nearest")

    px_a, py_a = _centroid_to_crop_px(cy_a, cx_a, crop_y0, crop_x0, h, w)
    px_b, py_b = _centroid_to_crop_px(cy_b, cx_b, crop_y0, crop_x0, h, w)

    dist = np.sqrt((px_b - px_a) ** 2 + (py_b - py_a) ** 2)
    if dist > 2:
        ax.annotate(
            "",
            xy=(px_b, py_b),
            xytext=(px_a, py_a),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5, mutation_scale=12),
            zorder=5,
        )

    ax.plot(px_a, py_a, "o", color="#ff6666", markersize=6,
            markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax.plot(px_b, py_b, "o", color="#66aaff", markersize=6,
            markeredgecolor="white", markeredgewidth=0.5, zorder=6)

    # Decision badge (top-right corner)
    status = cand_row["status"]
    badge_color = "#1a9641" if status == "accepted" else "#d7191c"
    badge_text = "ACC" if status == "accepted" else "REJ"
    ax.text(w - 2, 3, badge_text, fontsize=7, color="white", fontweight="bold",
            ha="right", va="top", zorder=7,
            bbox=dict(facecolor=badge_color, alpha=0.85, pad=1.5, edgecolor="none"))

    # Category color bar at top
    ax.axhline(y=1, color=CATEGORY_COLORS.get(category, "#cccccc"), lw=4)

    ax.set_xticks([])
    ax.set_yticks([])

    # Score breakdown in title
    scores = (
        f"prox={cand_row.get('proximity_score', float('nan')):.2f}  "
        f"align={cand_row.get('alignment_score', float('nan')):.2f}  "
        f"cont={cand_row.get('continuity_score', float('nan')):.2f}\n"
        f"size={cand_row.get('size_score', float('nan')):.2f}  "
        f"comp={cand_row['composite_score']:.3f}  "
        f"gap={cand_row['gap_distance']:.0f}nm"
    )
    ax.set_title(scores, fontsize=5.5, pad=2, color="black")

    patches = [
        mpatches.Patch(color="#ff6666", label=f"A (lbl {label_a})"),
        mpatches.Patch(color="#66aaff", label=f"B (lbl {label_b})"),
    ]
    ax.legend(handles=patches, fontsize=4.5, loc="lower right",
              framealpha=0.6, handlelength=0.8, borderpad=0.2, labelspacing=0.15)


def _draw_panel_gt(
    ax: plt.Axes,
    seg_rgba: np.ndarray,
    outcome: str,
    cand_row: pd.Series,
    near_nodes: list,
    near_edges: list,
    highlight_skel_ids: set,
    seg_crop_y0: int, seg_crop_x0: int,
    res_y: float, res_x: float,
    seg_offset: tuple[int, int, int],
    cx_a: float, cy_a: float,
    cx_b: float, cy_b: float,
    crop_y0: int, crop_x0: int,
) -> None:
    """Panel 3: colorized segmentation with oracle ground truth overlay."""
    h, w = seg_rgba.shape[:2]
    ax.imshow(seg_rgba, origin="upper", interpolation="nearest")

    # Skeleton overlay (highlight relevant neurons)
    if near_nodes:
        _overlay_skeleton(ax, near_nodes, near_edges,
                          seg_crop_y0, seg_crop_x0, res_y, res_x, seg_offset, h, w,
                          highlight_skel_ids=highlight_skel_ids,
                          base_alpha=0.35, highlight_alpha=0.95)

    # For oracle-positive pairs, draw crossing indicator between fragment centroids
    if outcome in ("TP", "FN"):
        px_a, py_a = _centroid_to_crop_px(cy_a, cx_a, crop_y0, crop_x0, h, w)
        px_b, py_b = _centroid_to_crop_px(cy_b, cx_b, crop_y0, crop_x0, h, w)
        dist = np.sqrt((px_b - px_a) ** 2 + (py_b - py_a) ** 2)
        if dist > 2:
            ax.plot([px_a, px_b], [py_a, py_b], "--",
                    color="#ffdd00", lw=1.5, alpha=0.9, zorder=5,
                    label="Should merge")

    # Oracle outcome badge (large, center-top)
    badge_color = OUTCOME_COLORS.get(outcome, "#cccccc")
    outcome_label = {
        "TP": "TP — Correct accept",
        "FP": "FP — False accept",
        "FN": "FN — Missed merge",
        "TN": "TN — Correct reject",
        "NA": "GT: N/A (no ground-truth)",
    }.get(outcome, outcome)
    ax.text(w / 2, 4, outcome_label, fontsize=6.5, color="white", fontweight="bold",
            ha="center", va="top", zorder=8,
            bbox=dict(facecolor=badge_color, alpha=0.90, pad=2.0, edgecolor="none"))

    # Merge verdict text at bottom
    if outcome != "NA":
        should_merge = outcome in ("TP", "FN")
        verdict = "GT: SHOULD MERGE" if should_merge else "GT: NO MERGE"
        verdict_color = "#1a9641" if should_merge else "#d7191c"
        ax.text(w / 2, h - 3, verdict, fontsize=5.5, color=verdict_color,
                fontweight="bold", ha="center", va="bottom", zorder=8,
                bbox=dict(facecolor="white", alpha=0.75, pad=1.0, edgecolor="none"))

    ax.set_xticks([])
    ax.set_yticks([])

    title = "Ground Truth"
    if near_nodes and highlight_skel_ids:
        title += f"  (skel ids: {sorted(highlight_skel_ids)})"
    ax.set_title(title, fontsize=6.5, pad=2)

    if outcome in ("TP", "FN"):
        ax.legend(fontsize=4.5, loc="lower left",
                  framealpha=0.6, handlelength=0.8, borderpad=0.2)


# ---------------------------------------------------------------------------
# Skeleton-to-fragment lookup
# ---------------------------------------------------------------------------


def _find_skeleton_ids_for_labels(
    skel_graph,
    label_a: int,
    label_b: int,
    seg_path: str,
    seg_key: str,
    res_z: float, res_y: float, res_x: float,
    seg_offset: tuple[int, int, int],
    seg_shape: tuple[int, int, int],
) -> set[int]:
    """Find skeleton IDs whose nodes fall in label_a or label_b voxels.

    This is done on-demand by spot-checking a subset of skeleton nodes against
    the segment label at that position. Lazy — uses a cached segment dict.
    """
    # We do a lightweight check: for each skeleton node, look up label at its
    # voxel. We only load a sparse set of voxels (no full seg load).
    oz, oy, ox = seg_offset
    found: set[int] = set()

    def _pos_to_voxel(pos):
        x_nm, y_nm, z_nm = float(pos[0]), float(pos[1]), float(pos[2])
        zi = int(round(z_nm / res_z)) - oz
        yi = int(round(y_nm / res_y)) - oy
        xi = int(round(x_nm / res_x)) - ox
        nz, ny, nx = seg_shape
        if 0 <= zi < nz and 0 <= yi < ny and 0 <= xi < nx:
            return zi, yi, xi
        return None

    # Group nodes by skeleton_id, sample up to 20 nodes per skeleton
    from collections import defaultdict
    by_skel: dict[int, list] = defaultdict(list)
    for node, data in skel_graph.nodes(data=True):
        pos = data.get("position")
        sid = data.get("skeleton_id", -1)
        if pos is not None:
            by_skel[sid].append((node, pos))

    with h5py.File(seg_path, "r") as f:
        dset = f[seg_key]
        for sid, node_list in by_skel.items():
            # Sample up to 20 nodes to avoid slowness
            sample = node_list[:20]
            for _, pos in sample:
                vox = _pos_to_voxel(pos)
                if vox is None:
                    continue
                zi, yi, xi = vox
                try:
                    lbl = int(dset[zi, yi, xi])
                except Exception:
                    continue
                if lbl == label_a or lbl == label_b:
                    found.add(sid)
                    break  # found a match for this skeleton

    return found


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------


def generate_report(
    output_dir: str,
    seg_path: str,
    seg_key: str,
    raw_path: str | None,
    raw_key: str,
    skel_path: str | None,
    seg_offset: tuple[int, int, int],
    resolution: list[float],
    crop_half: int,
    z_half: int,
    n_samples: int,
    out_path: str,
    seed: int,
) -> None:
    out_dir = Path(output_dir)
    connections = pd.read_csv(out_dir / "connections.csv")
    fragments = pd.read_csv(out_dir / "fragments.csv")

    # Resolve resolution
    res_z, res_y, res_x = (
        (resolution[0], resolution[1], resolution[2])
        if len(resolution) == 3
        else (resolution[0],) * 3
    )

    frag_info = (
        fragments[["fragment_id", "label_id", "centroid_z", "centroid_y", "centroid_x"]]
        .set_index("fragment_id")
    )

    # Load skeleton + build oracle
    skel_graph = None
    oracle: set[tuple[int, int]] | None = None
    if skel_path:
        print("Loading skeleton graph …", flush=True)
        skel_graph = _load_combined_skeleton(skel_path)
        oracle = _build_oracle(seg_path, seg_key, skel_graph, res_z, res_y, res_x, seg_offset)

    # Get segmentation shape for skeleton-to-fragment lookup
    with h5py.File(seg_path, "r") as f:
        seg_shape = tuple(f[seg_key].shape)  # (Z, Y, X)

    # Sample candidates
    samples = _sample_candidates(connections, fragments, oracle, n_samples, seed)
    print(f"Generating report for {len(samples)} candidates → {out_path}", flush=True)

    with PdfPages(out_path) as pdf:
        # ---- Cover page ----
        _write_cover(pdf, output_dir, seg_path, skel_path, connections, samples, raw_path)

        # ---- Candidate rows (3 candidates per page) ----
        ROWS_PER_PAGE = 3
        n_pages = max(1, (len(samples) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)

        for page_i in range(n_pages):
            batch = samples[page_i * ROWS_PER_PAGE : (page_i + 1) * ROWS_PER_PAGE]
            n_rows = len(batch)

            fig, axes = plt.subplots(
                n_rows, 3,
                figsize=(11, 3.8 * n_rows),
                gridspec_kw={"hspace": 0.55, "wspace": 0.06},
                squeeze=False,
            )

            for row_i, (category, cand_row) in enumerate(batch):
                fa = int(cand_row["fragment_a"])
                fb = int(cand_row["fragment_b"])
                ax_raw, ax_pred, ax_gt = axes[row_i]

                if fa not in frag_info.index or fb not in frag_info.index:
                    for ax in (ax_raw, ax_pred, ax_gt):
                        ax.axis("off")
                        ax.set_title("missing fragment data", fontsize=7)
                    continue

                info_a = frag_info.loc[fa]
                info_b = frag_info.loc[fb]
                label_a = int(info_a["label_id"])
                label_b = int(info_b["label_id"])

                # Centroids in voxels
                cz_a = info_a["centroid_z"] / res_z
                cy_a = info_a["centroid_y"] / res_y
                cx_a = info_a["centroid_x"] / res_x
                cz_b = info_b["centroid_z"] / res_z
                cy_b = info_b["centroid_y"] / res_y
                cx_b = info_b["centroid_x"] / res_x

                # Midpoint for crop center
                mid_z = int(round((cz_a + cz_b) / 2))
                mid_y = int(round((cy_a + cy_b) / 2))
                mid_x = int(round((cx_a + cx_b) / 2))

                # Crop bounds
                cy0 = mid_y - crop_half
                cy1 = mid_y + crop_half
                cx0 = mid_x - crop_half
                cx1 = mid_x + crop_half

                try:
                    seg_crop, (ay0, ay1, ax0, ax1) = _load_crop(
                        seg_path, seg_key, mid_z, cy0, cy1, cx0, cx1, dtype=np.int64
                    )
                except Exception as exc:
                    for ax in (ax_raw, ax_pred, ax_gt):
                        ax.axis("off")
                        ax.set_title(f"load error: {exc}", fontsize=6)
                    continue

                crop_h, crop_w = seg_crop.shape

                # Colorized segmentation (shared by panels 2 and 3)
                seg_rgba = _colorize_seg(seg_crop, label_a, label_b)

                # Skeleton nodes near this slice
                near_nodes: list = []
                near_edges: list = []
                highlight_skel_ids: set[int] = set()
                if skel_graph is not None:
                    near_nodes = _skel_nodes_near_slice(
                        skel_graph, mid_z, res_z, res_y, res_x, seg_offset, z_half
                    )
                    near_edges = _skel_edges_among_nodes(skel_graph, near_nodes)
                    # Find which skeleton IDs belong to frag A and B
                    highlight_skel_ids = _find_skeleton_ids_for_labels(
                        skel_graph, label_a, label_b,
                        seg_path, seg_key,
                        res_z, res_y, res_x,
                        seg_offset, seg_shape,
                    )

                # Oracle outcome
                outcome = _oracle_outcome(cand_row, frag_info, oracle)

                # Row header text (left axis title area)
                row_header = (
                    f"cand {int(cand_row['candidate_id'])}  "
                    f"[frag_id {fa}↔{fb}]\n"
                    f"[lbl {label_a}↔{label_b}]  [{category}]"
                )

                # ---- Panel 1: Raw / Neutral ----
                raw_rgba: np.ndarray | None = None
                if raw_path:
                    try:
                        raw_crop, _ = _load_crop(
                            raw_path, raw_key, mid_z, cy0, cy1, cx0, cx1
                        )
                        raw_rgba = _raw_to_rgba(raw_crop)
                    except Exception:
                        raw_rgba = None

                if raw_rgba is not None:
                    panel1_rgba = raw_rgba
                    panel1_base = "Raw EM"
                else:
                    # Neutral segmentation (no A/B highlighting)
                    panel1_rgba = _neutral_seg(seg_crop)
                    panel1_base = "Segmentation (neutral)"
                panel1_title = (
                    f"{panel1_base}  +  skeleton overlay"
                    if near_nodes else panel1_base
                )

                _draw_panel_raw(
                    ax_raw, panel1_rgba, near_nodes, near_edges,
                    ay0, ax0, res_y, res_x, seg_offset,
                    title=panel1_title,
                )
                ax_raw.set_ylabel(row_header, fontsize=5.0, rotation=90,
                                  labelpad=3, va="center")
                _set_border(ax_raw, CATEGORY_COLORS.get(category, "#cccccc"))

                # ---- Panel 2: Prediction ----
                _draw_panel_prediction(
                    ax_pred, seg_rgba, cand_row, category,
                    cy_a, cx_a, cy_b, cx_b,
                    ay0, ax0, label_a, label_b,
                )
                # Always show column header on panel 2
                existing = ax_pred.get_title()
                ax_pred.set_title(
                    f"Pipeline Prediction\n{existing}", fontsize=5.5, pad=2
                )
                _set_border(ax_pred, CATEGORY_COLORS.get(category, "#cccccc"))

                # ---- Panel 3: Ground Truth ----
                _draw_panel_gt(
                    ax_gt, seg_rgba.copy(), outcome, cand_row,
                    near_nodes, near_edges, highlight_skel_ids,
                    ay0, ax0, res_y, res_x, seg_offset,
                    cx_a, cy_a, cx_b, cy_b,
                    ay0, ax0,
                )
                _set_border(ax_gt, OUTCOME_COLORS.get(outcome, "#cccccc"))

            # Category legend at page top
            cat_patches = [
                mpatches.Patch(color=c, label=cat)
                for cat, c in CATEGORY_COLORS.items()
            ]
            fig.legend(
                handles=cat_patches,
                loc="upper center",
                ncol=3,
                fontsize=6,
                framealpha=0.85,
                title="Category (row border / top bar)",
                title_fontsize=6,
                bbox_to_anchor=(0.5, 0.998),
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Report written to: {out_path}")


def _set_border(ax: plt.Axes, color: str, lw: float = 2.5) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------


def _write_cover(
    pdf: PdfPages,
    output_dir: str,
    seg_path: str,
    skel_path: str | None,
    connections: pd.DataFrame,
    samples: list,
    raw_path: str | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    n_acc = int((connections["status"] == "accepted").sum())
    n_rej = int((connections["status"] == "rejected").sum())

    # Count oracle outcomes from sample
    category_counts: dict[str, int] = {}
    for cat, _ in samples:
        category_counts[cat] = category_counts.get(cat, 0) + 1

    oracle_note = (
        f"Ground-truth: skeleton file loaded — TP/FP/FN/TN shown in Panel 3"
        if skel_path
        else "Ground-truth: no skeleton provided — Panel 3 shows GT: N/A"
    )
    raw_note = (
        f"Raw EM: {raw_path}" if raw_path
        else "Raw EM: not provided — Panel 1 shows neutral segmentation"
    )

    cover_text = (
        "Visual Validation Report\n"
        "Connectomics Split-Detection Pipeline\n\n"
        f"Source:  {output_dir}\n"
        f"Volume:  {seg_path}\n"
        f"{raw_note}\n"
        f"Skeleton: {skel_path or 'N/A'}\n\n"
        f"Candidates:  {len(connections):,} total  |  "
        f"Accepted: {n_acc:,}  |  Rejected: {n_rej:,}\n\n"
        f"Sample shown: {len(samples)} candidates\n"
    )
    for cat, n in category_counts.items():
        cover_text += f"  • {cat}: {n}\n"

    cover_text += (
        f"\n{oracle_note}\n\n"
        "─── Panel layout (3 columns per candidate row) ───────────────\n"
        "  Panel 1 — Raw / Neutral:\n"
        "    Grayscale EM (if available) or neutral segmentation.\n"
        "    Yellow dots/lines = skeleton nodes/edges near the Z-slice.\n"
        "    Use this to assess: what does the tissue look like?\n\n"
        "  Panel 2 — Pipeline Prediction:\n"
        "    Red = Fragment A,  Blue = Fragment B,  Gray = others.\n"
        "    White arrow = proposed merge direction A→B.\n"
        "    Badge: ACC (green) or REJ (red).\n"
        "    Title shows all 5 score components + gap distance.\n\n"
        "  Panel 3 — Ground Truth:\n"
        "    Same seg coloring + ground-truth overlay.\n"
        "    TP (green)  = correct accept  — pipeline & GT agree: merge\n"
        "    FP (red)    = false accept    — pipeline merged, GT says no\n"
        "    FN (orange) = missed merge    — pipeline rejected, GT says merge\n"
        "    TN (gray)   = correct reject  — pipeline & GT agree: no merge\n"
        "    Bright yellow skeleton = neuron(s) passing through fragment A or B.\n"
        "    Dashed yellow line = oracle crossing (for TP/FN pairs).\n\n"
        "─── How to inspect ───────────────────────────────────────────\n"
        "  1. Look at Panel 1 to understand the tissue context.\n"
        "  2. In Panel 2, check if fragments A and B plausibly belong\n"
        "     to the same neuron (do they touch? Are shapes continuous?).\n"
        "  3. In Panel 3, verify the oracle: does the skeleton cross\n"
        "     from one fragment into the other? For FP/FN cases, examine\n"
        "     whether the pipeline's score values explain the error.\n"
    )

    ax.text(
        0.5, 0.5, cover_text,
        ha="center", va="center",
        fontsize=8.5, fontfamily="monospace",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#aaaaaa"),
        linespacing=1.4,
    )

    # Outcome color chips
    for i, (label, color) in enumerate(OUTCOME_COLORS.items()):
        if label == "NA":
            continue
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.08, 0.05 - i * 0.035), 0.03, 0.022,
            boxstyle="round,pad=0.002",
            transform=ax.transAxes, color=color, clip_on=False,
        ))
        ax.text(0.13, 0.061 - i * 0.035, label,
                transform=ax.transAxes, fontsize=8, va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", required=True,
                        help="Pipeline output directory (connections.csv, fragments.csv)")
    parser.add_argument("--seg", required=True, help="HDF5 segmentation volume")
    parser.add_argument("--seg-key", default="labels", help="HDF5 key for segmentation (default: labels)")
    parser.add_argument("--raw", default=None, help="Optional HDF5 raw EM volume for Panel 1")
    parser.add_argument("--raw-key", default="volumes/raw", help="HDF5 key for raw image (default: volumes/raw)")
    parser.add_argument("--skel", default=None, help="Optional skeleton .npz for GT oracle overlay")
    parser.add_argument("--seg-offset", nargs=3, type=int, default=[0, 0, 0],
                        metavar=("Z", "Y", "X"),
                        help="Voxel offset of seg within full volume, z y x (default: 0 0 0)")
    parser.add_argument(
        "--resolution", nargs="+", type=float, default=[33.0],
        help="Voxel size in nm: one value (isotropic) or three (z y x). Default: 33",
    )
    parser.add_argument("--crop-half", type=int, default=100,
                        help="Crop half-width in voxels (default: 100 → 200×200 crop)")
    parser.add_argument("--z-half", type=int, default=4,
                        help="Skeleton node Z-projection half-thickness in voxels (default: 4)")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Candidates per category (default: 5)")
    parser.add_argument("--out", default=None,
                        help="Output PDF path (default: <output-dir>/visual_report.pdf)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out is None:
        args.out = str(Path(args.output_dir) / "visual_report.pdf")

    if len(args.resolution) not in (1, 3):
        print("Error: --resolution must have 1 (isotropic) or 3 (z y x) values", file=sys.stderr)
        sys.exit(1)

    generate_report(
        output_dir=args.output_dir,
        seg_path=args.seg,
        seg_key=args.seg_key,
        raw_path=args.raw,
        raw_key=args.raw_key,
        skel_path=args.skel,
        seg_offset=tuple(args.seg_offset),
        resolution=args.resolution,
        crop_half=args.crop_half,
        z_half=args.z_half,
        n_samples=args.n_samples,
        out_path=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
