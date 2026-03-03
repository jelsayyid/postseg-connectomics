#!/usr/bin/env python3
"""Generate a visual validation report as a multi-page PDF.

For a representative sample of candidate pairs from pipeline output, produces
2D slice crops from the segmentation volume showing the two fragments
involved (color-coded red/blue), the proposed connection arrow, and per-panel
metadata. No oracle or external metrics are used — the report reflects only
what the pipeline itself produced.

Categories sampled:
  - High-confidence accepted  : accepted candidates with highest composite score
  - Low-confidence accepted   : accepted candidates with lowest composite score
  - Borderline rejected       : rejected candidates with highest composite score
  - Random rejected           : random sample of rejected candidates

Usage
-----
    python scripts/generate_visual_report.py \\
        --output-dir output/xpress_training \\
        --seg /tmp/xpress_full.h5 \\
        --resolution 33 \\
        --out output/xpress_training/visual_report.pdf

    # CREMI (anisotropic resolution)
    python scripts/generate_visual_report.py \\
        --output-dir output/cremi_sample_a \\
        --seg data/cremi_crop.hdf \\
        --resolution 40 4 4 \\
        --out output/cremi_sample_a/visual_report.pdf

Arguments
---------
--output-dir    Pipeline output directory (contains connections.csv, fragments.csv).
--seg           HDF5 segmentation volume used as input to the pipeline.
--seg-key       HDF5 dataset key (default: labels).
--resolution    Voxel size in nm: one value (isotropic) or three values z y x.
--crop-half     Half-width of the 2D crop in voxels (default: 50 → 100×100 crop).
--n-samples     Candidates per category (default: 6).
--out           Output PDF path.
--seed          Random seed for reproducible sampling (default: 42).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / no-display backend

import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "High-confidence accepted": "#1a9641",
    "Low-confidence accepted": "#fdae61",
    "Borderline rejected": "#d7191c",
    "Random rejected": "#999999",
}


def _load_seg_crop(
    seg_path: str,
    seg_key: str,
    z: int,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> np.ndarray:
    """Read a 2D sub-region from a single Z-slice of an HDF5 segmentation."""
    with h5py.File(seg_path, "r") as f:
        dset = f[seg_key]
        nz = dset.shape[0]
        z_clamped = int(np.clip(z, 0, nz - 1))
        y0, y1 = max(0, y_start), min(dset.shape[1], y_end)
        x0, x1 = max(0, x_start), min(dset.shape[2], x_end)
        crop = dset[z_clamped, y0:y1, x0:x1]
    return crop.astype(np.int64), (y0, y1, x0, x1)


def _colorize(
    crop: np.ndarray,
    label_a: int,
    label_b: int,
) -> np.ndarray:
    """Return an RGBA image colorizing label_a (red), label_b (blue), rest (gray)."""
    out = np.zeros((*crop.shape, 4), dtype=np.float32)

    # background (label 0)
    bg = crop == 0
    out[bg] = [0.07, 0.07, 0.07, 1.0]

    # other labels — mid gray
    other = (crop != 0) & (crop != label_a) & (crop != label_b)
    out[other] = [0.6, 0.6, 0.6, 1.0]

    # fragment A — warm red
    mask_a = crop == label_a
    out[mask_a] = [0.85, 0.18, 0.18, 1.0]

    # fragment B — blue
    mask_b = crop == label_b
    out[mask_b] = [0.12, 0.47, 0.71, 1.0]

    return out


def _nm_to_vox(nm: float, res: float) -> int:
    return int(round(nm / res))


def _vox_to_crop(vox: int, crop_start: int) -> float:
    return vox - crop_start


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_candidates(
    connections: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> list[tuple[str, pd.Series]]:
    """Return a list of (category_label, row) tuples."""
    rng = np.random.default_rng(seed)
    acc = connections[connections["status"] == "accepted"]
    rej = connections[connections["status"] == "rejected"]

    out: list[tuple[str, pd.Series]] = []

    if len(acc) >= n_samples:
        for _, row in acc.nlargest(n_samples, "composite_score").iterrows():
            out.append(("High-confidence accepted", row))
        for _, row in acc.nsmallest(n_samples, "composite_score").iterrows():
            out.append(("Low-confidence accepted", row))
    else:
        for _, row in acc.iterrows():
            out.append(("High-confidence accepted", row))

    if len(rej) >= n_samples:
        for _, row in rej.nlargest(n_samples, "composite_score").iterrows():
            out.append(("Borderline rejected", row))
        idx = rng.choice(len(rej), size=min(n_samples, len(rej)), replace=False)
        for _, row in rej.iloc[idx].iterrows():
            out.append(("Random rejected", row))
    else:
        for _, row in rej.iterrows():
            out.append(("Borderline rejected", row))

    return out


# ---------------------------------------------------------------------------
# Per-candidate panel
# ---------------------------------------------------------------------------


def _draw_candidate_panel(
    ax: plt.Axes,
    crop_rgba: np.ndarray,
    cand_row: pd.Series,
    category: str,
    cy_a: float,
    cx_a: float,
    cy_b: float,
    cx_b: float,
    crop_y0: int,
    crop_x0: int,
    crop_y1: int,
    crop_x1: int,
    label_a: int,
    label_b: int,
) -> None:
    ax.imshow(crop_rgba, origin="upper", interpolation="nearest")

    # Convert voxel centroids to crop-relative coordinates
    def to_crop(vy, vx):
        return _vox_to_crop(vx, crop_x0), _vox_to_crop(vy, crop_y0)

    cx_a_img, cy_a_img = to_crop(cy_a, cx_a)
    cx_b_img, cy_b_img = to_crop(cy_b, cx_b)

    # Clip to crop bounds (centroids may be outside crop if one fragment is large)
    h, w = crop_rgba.shape[:2]

    def clip_to_crop(x, y):
        return np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)

    cx_a_img, cy_a_img = clip_to_crop(cx_a_img, cy_a_img)
    cx_b_img, cy_b_img = clip_to_crop(cx_b_img, cy_b_img)

    # Draw connection arrow (from A to B)
    dx = cx_b_img - cx_a_img
    dy = cy_b_img - cy_a_img
    dist = max(np.sqrt(dx**2 + dy**2), 1e-6)
    if dist > 2:
        ax.annotate(
            "",
            xy=(cx_b_img, cy_b_img),
            xytext=(cx_a_img, cy_a_img),
            arrowprops=dict(
                arrowstyle="-|>",
                color="white",
                lw=1.5,
                mutation_scale=12,
            ),
        )

    # Draw centroid markers
    ax.plot(cx_a_img, cy_a_img, "o", color="#ff6666", markersize=6, markeredgecolor="white", lw=0.5)
    ax.plot(cx_b_img, cy_b_img, "o", color="#66aaff", markersize=6, markeredgecolor="white", lw=0.5)

    # Category color bar at top
    bar_color = CATEGORY_COLORS.get(category, "#cccccc")
    ax.axhline(y=1, color=bar_color, lw=4, xmin=0, xmax=1)

    ax.set_xticks([])
    ax.set_yticks([])

    # Title inside the panel as text (compact)
    status_short = "ACC" if "accepted" in category else "REJ"
    title = (
        f"cand {int(cand_row['candidate_id'])}  "
        f"[{int(label_a)}↔{int(label_b)}]\n"
        f"gap={cand_row['gap_distance']:.0f}nm  "
        f"comp={cand_row['composite_score']:.3f}  "
        f"{status_short}"
    )
    ax.set_title(title, fontsize=6.5, pad=2, color="black")

    # Legend patches
    patch_a = mpatches.Patch(color="#ff6666", label=f"frag {int(cand_row['fragment_a'])}")
    patch_b = mpatches.Patch(color="#66aaff", label=f"frag {int(cand_row['fragment_b'])}")
    ax.legend(
        handles=[patch_a, patch_b],
        fontsize=5,
        loc="lower right",
        framealpha=0.6,
        handlelength=0.8,
        borderpad=0.3,
        labelspacing=0.2,
    )


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------


def generate_report(
    output_dir: str,
    seg_path: str,
    seg_key: str,
    resolution: list[float],
    crop_half: int,
    n_samples: int,
    out_path: str,
    seed: int,
) -> None:
    out_dir = Path(output_dir)
    connections = pd.read_csv(out_dir / "connections.csv")
    fragments = pd.read_csv(out_dir / "fragments.csv")

    # Build lookup: fragment_id → (label_id, centroid_z_nm, centroid_y_nm, centroid_x_nm)
    frag_info = (
        fragments[["fragment_id", "label_id", "centroid_z", "centroid_y", "centroid_x"]]
        .set_index("fragment_id")
    )

    # Voxel size (z, y, x) — may be anisotropic
    res_z, res_y, res_x = (
        (resolution[0], resolution[1], resolution[2])
        if len(resolution) == 3
        else (resolution[0], resolution[0], resolution[0])
    )

    samples = sample_candidates(connections, n_samples, seed)
    print(f"Generating report for {len(samples)} candidates → {out_path}")

    # Grid layout: 3 panels per row
    ncols = 3
    nrows_per_page = 3  # 3 rows × 3 cols = 9 panels per page
    panels_per_page = ncols * nrows_per_page

    with PdfPages(out_path) as pdf:
        # ---- Cover page ----
        fig_cover, ax_cover = plt.subplots(figsize=(8.5, 11))
        ax_cover.axis("off")
        n_acc = int((connections["status"] == "accepted").sum())
        n_rej = int((connections["status"] == "rejected").sum())
        cover_text = (
            "Visual Validation Report\n"
            "Connectomics Split-Detection Pipeline\n\n"
            f"Source:  {output_dir}\n"
            f"Volume:  {seg_path}\n\n"
            f"Candidates:  {len(connections):,} total\n"
            f"  Accepted:  {n_acc:,}\n"
            f"  Rejected:  {n_rej:,}\n\n"
            f"Sample shown below: {len(samples)} candidates\n"
            f"  • High-confidence accepted  (highest composite score)\n"
            f"  • Low-confidence accepted   (lowest composite score)\n"
            f"  • Borderline rejected       (highest composite score among rejected)\n"
            f"  • Random rejected\n\n"
            "Color key per panel:\n"
            "  Red   = Fragment A (lower fragment_id)\n"
            "  Blue  = Fragment B (higher fragment_id)\n"
            "  Gray  = All other labeled fragments\n"
            "  Black = Background (label 0)\n"
            "  Arrow = Proposed connection direction A→B\n\n"
            "Crop: 2D Z-slice centered at midpoint between fragment centroids.\n"
            "Coordinates are in voxels derived from nm centroids in fragments.csv."
        )
        ax_cover.text(
            0.5, 0.6, cover_text,
            ha="center", va="center",
            fontsize=11, fontfamily="monospace",
            transform=ax_cover.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5", edgecolor="#aaaaaa"),
        )
        # Category color legend
        for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
            ax_cover.add_patch(
                mpatches.FancyBboxPatch(
                    (0.15, 0.12 - i * 0.04), 0.04, 0.025,
                    boxstyle="round,pad=0.002",
                    transform=ax_cover.transAxes,
                    color=color, clip_on=False,
                )
            )
            ax_cover.text(
                0.21, 0.134 - i * 0.04, cat,
                transform=ax_cover.transAxes, fontsize=9, va="center",
            )
        pdf.savefig(fig_cover, bbox_inches="tight")
        plt.close(fig_cover)

        # ---- Candidate panels ----
        page_samples = []
        all_groups: dict[str, list[tuple[str, pd.Series]]] = {}
        for cat, row in samples:
            all_groups.setdefault(cat, []).append((cat, row))

        # Order: by category
        ordered = []
        for cat in CATEGORY_COLORS:
            ordered.extend(all_groups.get(cat, []))

        n_pages = max(1, (len(ordered) + panels_per_page - 1) // panels_per_page)

        for page_i in range(n_pages):
            page_batch = ordered[page_i * panels_per_page : (page_i + 1) * panels_per_page]
            fig, axes = plt.subplots(
                nrows_per_page, ncols,
                figsize=(8.5, 11),
                gridspec_kw={"hspace": 0.5, "wspace": 0.08},
            )
            axes_flat = axes.flatten()

            for panel_i, (category, cand_row) in enumerate(page_batch):
                ax = axes_flat[panel_i]

                fa = int(cand_row["fragment_a"])
                fb = int(cand_row["fragment_b"])

                # Look up centroid info
                if fa not in frag_info.index or fb not in frag_info.index:
                    ax.axis("off")
                    ax.set_title(f"cand {int(cand_row['candidate_id'])}: missing fragment data", fontsize=7)
                    continue

                info_a = frag_info.loc[fa]
                info_b = frag_info.loc[fb]

                label_a = int(info_a["label_id"])
                label_b = int(info_b["label_id"])

                # Convert centroids from nm to voxels
                cz_a = info_a["centroid_z"] / res_z
                cy_a = info_a["centroid_y"] / res_y
                cx_a = info_a["centroid_x"] / res_x
                cz_b = info_b["centroid_z"] / res_z
                cy_b = info_b["centroid_y"] / res_y
                cx_b = info_b["centroid_x"] / res_x

                # Midpoint in voxel space
                mid_z = int(round((cz_a + cz_b) / 2))
                mid_y = int(round((cy_a + cy_b) / 2))
                mid_x = int(round((cx_a + cx_b) / 2))

                # Crop bounds
                y0 = mid_y - crop_half
                y1 = mid_y + crop_half
                x0 = mid_x - crop_half
                x1 = mid_x + crop_half

                try:
                    crop, (actual_y0, actual_y1, actual_x0, actual_x1) = _load_seg_crop(
                        seg_path, seg_key, mid_z, y0, y1, x0, x1
                    )
                except Exception as exc:
                    ax.axis("off")
                    ax.set_title(f"cand {int(cand_row['candidate_id'])}: load error\n{exc}", fontsize=6)
                    continue

                rgba = _colorize(crop, label_a, label_b)

                _draw_candidate_panel(
                    ax, rgba, cand_row, category,
                    cy_a, cx_a, cy_b, cx_b,
                    actual_y0, actual_x0, actual_y1, actual_x1,
                    label_a, label_b,
                )

                # Category color border on the axis
                border_color = CATEGORY_COLORS.get(category, "#cccccc")
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.5)

            # Hide unused panels
            for panel_i in range(len(page_batch), len(axes_flat)):
                axes_flat[panel_i].axis("off")

            # Page category legend at top
            cat_patches = [
                mpatches.Patch(color=c, label=cat)
                for cat, c in CATEGORY_COLORS.items()
            ]
            fig.legend(
                handles=cat_patches,
                loc="upper center",
                ncol=2,
                fontsize=7,
                framealpha=0.8,
                title="Panel border = category",
                title_fontsize=7,
                bbox_to_anchor=(0.5, 0.995),
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Report written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory (connections.csv, fragments.csv)")
    parser.add_argument("--seg", required=True, help="HDF5 segmentation volume")
    parser.add_argument("--seg-key", default="labels", help="HDF5 dataset key (default: labels)")
    parser.add_argument(
        "--resolution",
        nargs="+",
        type=float,
        default=[33.0],
        help="Voxel resolution in nm: one value (isotropic) or three (z y x). Default: 33",
    )
    parser.add_argument("--crop-half", type=int, default=50, help="Crop half-width in voxels (default: 50 → 100×100)")
    parser.add_argument("--n-samples", type=int, default=6, help="Candidates per category (default: 6)")
    parser.add_argument("--out", default=None, help="Output PDF path (default: <output-dir>/visual_report.pdf)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
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
        resolution=args.resolution,
        crop_half=args.crop_half,
        n_samples=args.n_samples,
        out_path=args.out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
