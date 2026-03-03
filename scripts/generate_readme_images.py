#!/usr/bin/env python3
"""Generate PNG showcase images for embedding in the GitHub README.

Produces two images saved to docs/images/:
  candidate_showcase_accepted.png  — 3 high-confidence accepted pairs
  candidate_showcase_rejected.png  — 3 borderline rejected pairs

Each panel is a 2D Z-slice crop of the segmentation volume centered between
the two fragment centroids: Fragment A = red, Fragment B = blue,
context = gray, background = black, connection arrow in white.

Usage
-----
    python scripts/generate_readme_images.py \\
        --output-dir output/xpress_training \\
        --seg /tmp/xpress_full.h5 \\
        --resolution 33
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CROP_HALF = 55  # → 110×110 voxel crop per panel


# ---------------------------------------------------------------------------
# Core helpers (shared with generate_visual_report.py)
# ---------------------------------------------------------------------------


def _load_seg_crop(seg_path, seg_key, z, y_start, y_end, x_start, x_end):
    with h5py.File(seg_path, "r") as f:
        dset = f[seg_key]
        nz = dset.shape[0]
        z_clamped = int(np.clip(z, 0, nz - 1))
        y0, y1 = max(0, y_start), min(dset.shape[1], y_end)
        x0, x1 = max(0, x_start), min(dset.shape[2], x_end)
        crop = dset[z_clamped, y0:y1, x0:x1]
    return crop.astype(np.int64), (y0, y1, x0, x1)


def _colorize(crop, label_a, label_b):
    out = np.zeros((*crop.shape, 4), dtype=np.float32)
    out[crop == 0] = [0.07, 0.07, 0.07, 1.0]
    out[(crop != 0) & (crop != label_a) & (crop != label_b)] = [0.55, 0.55, 0.55, 1.0]
    out[crop == label_a] = [0.85, 0.18, 0.18, 1.0]
    out[crop == label_b] = [0.12, 0.47, 0.71, 1.0]
    return out


def _draw_panel(ax, rgba, cand_row, cy_a, cx_a, cy_b, cx_b, crop_y0, crop_x0, label_a, label_b):
    ax.imshow(rgba, origin="upper", interpolation="nearest")
    h, w = rgba.shape[:2]

    def to_img(vy, vx):
        return np.clip(vx - crop_x0, 0, w - 1), np.clip(vy - crop_y0, 0, h - 1)

    ix_a, iy_a = to_img(cy_a, cx_a)
    ix_b, iy_b = to_img(cy_b, cx_b)

    if abs(ix_b - ix_a) + abs(iy_b - iy_a) > 2:
        ax.annotate(
            "",
            xy=(ix_b, iy_b),
            xytext=(ix_a, iy_a),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0, mutation_scale=14),
        )

    ax.plot(ix_a, iy_a, "o", color="#ff6666", markersize=7, markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(ix_b, iy_b, "o", color="#66aaff", markersize=7, markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    gap_nm = cand_row["gap_distance"]
    score = cand_row["composite_score"]
    ax.set_title(
        f"gap {gap_nm:.0f} nm · score {score:.3f}\nfrags {int(label_a)} ↔ {int(label_b)}",
        fontsize=8,
        pad=3,
        color="black",
    )


def _pick_candidates(connections, fragments, n, category, seg_path, seg_key, res_z, res_y, res_x):
    """Pick n candidates from a category, preferring those where both fragments
    are visible in the crop (i.e., at least 1% of the crop is colored)."""
    frag_info = fragments[["fragment_id", "label_id", "centroid_z", "centroid_y", "centroid_x"]].set_index("fragment_id")

    if category == "accepted":
        pool = connections[connections["status"] == "accepted"].nlargest(50, "composite_score")
    else:
        pool = connections[connections["status"] == "rejected"].nlargest(50, "composite_score")

    chosen = []
    for _, row in pool.iterrows():
        if len(chosen) >= n:
            break
        fa, fb = int(row["fragment_a"]), int(row["fragment_b"])
        if fa not in frag_info.index or fb not in frag_info.index:
            continue
        info_a, info_b = frag_info.loc[fa], frag_info.loc[fb]
        label_a, label_b = int(info_a["label_id"]), int(info_b["label_id"])

        cz_a, cy_a, cx_a = info_a["centroid_z"] / res_z, info_a["centroid_y"] / res_y, info_a["centroid_x"] / res_x
        cz_b, cy_b, cx_b = info_b["centroid_z"] / res_z, info_b["centroid_y"] / res_y, info_b["centroid_x"] / res_x

        mid_z = int(round((cz_a + cz_b) / 2))
        mid_y = int(round((cy_a + cy_b) / 2))
        mid_x = int(round((cx_a + cx_b) / 2))

        try:
            crop, (ay0, _, ax0, _) = _load_seg_crop(
                seg_path, seg_key,
                mid_z, mid_y - CROP_HALF, mid_y + CROP_HALF,
                mid_x - CROP_HALF, mid_x + CROP_HALF,
            )
        except Exception:
            continue

        # Require both fragment labels to appear in the crop
        if (crop == label_a).sum() == 0 or (crop == label_b).sum() == 0:
            continue

        chosen.append({
            "row": row, "label_a": label_a, "label_b": label_b,
            "cy_a": cy_a, "cx_a": cx_a, "cy_b": cy_b, "cx_b": cx_b,
            "crop": crop, "crop_y0": ay0, "crop_x0": ax0,
        })
    return chosen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(output_dir, seg_path, seg_key, resolution, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    connections = pd.read_csv(Path(output_dir) / "connections.csv")
    fragments = pd.read_csv(Path(output_dir) / "fragments.csv")

    res_z, res_y, res_x = (
        (resolution[0], resolution[1], resolution[2]) if len(resolution) == 3
        else (resolution[0],) * 3
    )

    for category, label, border_color, filename in [
        ("accepted", "High-confidence accepted", "#1a9641", "candidate_showcase_accepted.png"),
        ("rejected", "Borderline rejected",       "#d7191c", "candidate_showcase_rejected.png"),
    ]:
        candidates = _pick_candidates(
            connections, fragments, n=3,
            category=category,
            seg_path=seg_path, seg_key=seg_key,
            res_z=res_z, res_y=res_y, res_x=res_x,
        )

        if not candidates:
            print(f"  No suitable candidates found for category: {category}")
            continue

        n = len(candidates)
        fig, axes = plt.subplots(
            1, n,
            figsize=(4.2 * n, 4.4),
            gridspec_kw={"wspace": 0.12},
        )
        if n == 1:
            axes = [axes]

        fig.suptitle(
            label,
            fontsize=12,
            fontweight="bold",
            color=border_color,
            y=1.01,
        )

        for ax, cand in zip(axes, candidates):
            rgba = _colorize(cand["crop"], cand["label_a"], cand["label_b"])
            _draw_panel(
                ax, rgba, cand["row"],
                cand["cy_a"], cand["cx_a"],
                cand["cy_b"], cand["cx_b"],
                cand["crop_y0"], cand["crop_x0"],
                cand["label_a"], cand["label_b"],
            )
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

        # Shared color legend below the panels
        patches = [
            mpatches.Patch(color="#dd3333", label="Fragment A"),
            mpatches.Patch(color="#2277cc", label="Fragment B"),
            mpatches.Patch(color="#888888", label="Other fragments"),
            mpatches.Patch(color="#111111", label="Background"),
        ]
        fig.legend(
            handles=patches,
            ncol=4,
            loc="lower center",
            fontsize=8,
            framealpha=0.0,
            bbox_to_anchor=(0.5, -0.06),
            handlelength=1.2,
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
    parser.add_argument("--resolution", nargs="+", type=float, default=[33.0])
    parser.add_argument("--out-dir", default="docs/images", help="Directory for output PNGs (default: docs/images)")
    args = parser.parse_args()

    print(f"Generating README showcase images from {args.output_dir} ...")
    generate(
        output_dir=args.output_dir,
        seg_path=args.seg,
        seg_key=args.seg_key,
        resolution=args.resolution,
        out_dir=args.out_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
