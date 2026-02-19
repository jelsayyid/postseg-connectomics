"""
Generate example correction images for lab meeting documentation.
Shows the two synthetic test scenarios: split correction (ACCEPT) and merge prevention (REJECT).

Run from project root:
    python docs/generate_example_images.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: 3D fragment view with connection decisions
# ---------------------------------------------------------------------------

def make_tube_points(center, direction, half_len, radius=8, n_points=200, seed=0):
    """Return (n_points, 3) array of random points forming a tube."""
    rng = np.random.default_rng(seed)
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    # two orthogonal vectors
    orth1 = np.array([1, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1, 0])
    orth1 = orth1 - d * (orth1 @ d); orth1 /= np.linalg.norm(orth1)
    orth2 = np.cross(d, orth1)
    t = rng.uniform(-half_len, half_len, n_points)
    r = rng.uniform(0, radius, n_points)
    theta = rng.uniform(0, 2 * np.pi, n_points)
    pts = (
        np.outer(t, d)
        + np.outer(r * np.cos(theta), orth1)
        + np.outer(r * np.sin(theta), orth2)
    )
    return pts + np.array(center)


fig = plt.figure(figsize=(14, 6))
fig.suptitle("Synthetic Correction Scenarios — Pipeline Validation", fontsize=13, fontweight="bold", y=1.01)

# --- Panel A: Split correction (ACCEPT) ---
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_title("(A) Split Correction → ACCEPT\n(same neuron, 50 nm gap)", fontsize=10)

frag_a = make_tube_points([50, 50, 100], [0, 0, 1], 90, seed=1)   # z: 10-190
frag_b = make_tube_points([50, 50, 250], [0, 0, 1], 90, seed=2)   # z: 160-340 (gap 160→190 = 50nm-ish rendered)

ax1.scatter(*frag_a.T, c="#4C72B0", alpha=0.15, s=2, rasterized=True)
ax1.scatter(*frag_b.T, c="#4C72B0", alpha=0.15, s=2, rasterized=True)

# Skeleton lines inside each fragment
for pts, zoffset in [(frag_a, 0), (frag_b, 0)]:
    zmin, zmax = pts[:, 2].min(), pts[:, 2].max()
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    ax1.plot([cx, cx], [cy, cy], [zmin, zmax], "b-", lw=2)

# Endpoints and gap
ep_a = [50, 50, frag_a[:, 2].max()]
ep_b = [50, 50, frag_b[:, 2].min()]
ax1.scatter(*ep_a, color="green", s=60, zorder=5)
ax1.scatter(*ep_b, color="green", s=60, zorder=5)
ax1.plot([ep_a[0], ep_b[0]], [ep_a[1], ep_b[1]], [ep_a[2], ep_b[2]],
         "g--", lw=2.5, label="Proposed merge (ACCEPTED)")
ax1.text(50, 65, (ep_a[2] + ep_b[2]) / 2, "50 nm gap", color="green",
         fontsize=8, ha="center")

ax1.set_xlabel("X (nm)", fontsize=8); ax1.set_ylabel("Y (nm)", fontsize=8); ax1.set_zlabel("Z (nm)", fontsize=8)
ax1.tick_params(labelsize=7)
frag_a_patch = mpatches.Patch(color="#4C72B0", label="Fragment A (label 1)")
frag_b_patch = mpatches.Patch(color="#4C72B0", label="Fragment B (label 1)")
merge_line = mpatches.Patch(color="green", label="Merge decision: ACCEPT")
ax1.legend(handles=[frag_a_patch, frag_b_patch, merge_line], fontsize=7, loc="upper left")

# --- Panel B: Merge prevention (REJECT) ---
ax2 = fig.add_subplot(122, projection="3d")
ax2.set_title("(B) False Merge Prevention → REJECT\n(different neurons, orthogonal geometry)", fontsize=10)

frag_a2 = make_tube_points([50, 50, 100], [0, 0, 1], 90, seed=1)    # same fragment A
frag_c  = make_tube_points([50, 300, 50], [0, 1, 0], 100, seed=3)   # y-oriented, different label

ax2.scatter(*frag_a2.T, c="#4C72B0", alpha=0.15, s=2, rasterized=True, label="Fragment A (label 1)")
ax2.scatter(*frag_c.T,  c="#DD8452", alpha=0.15, s=2, rasterized=True, label="Fragment C (label 2)")

# Skeletons
zmin_a, zmax_a = frag_a2[:, 2].min(), frag_a2[:, 2].max()
cx_a, cy_a = frag_a2[:, 0].mean(), frag_a2[:, 1].mean()
ax2.plot([cx_a, cx_a], [cy_a, cy_a], [zmin_a, zmax_a], color="#4C72B0", lw=2)

ymin_c, ymax_c = frag_c[:, 1].min(), frag_c[:, 1].max()
cx_c, cz_c = frag_c[:, 0].mean(), frag_c[:, 2].mean()
ax2.plot([cx_c, cx_c], [ymin_c, ymax_c], [cz_c, cz_c], color="#DD8452", lw=2)

# Closest endpoint pair (rejected)
ep_a_close = [cx_a, cy_a, zmax_a]
ep_c_close = [cx_c, ymin_c, cz_c]
ax2.scatter(*ep_a_close, color="red", s=60, zorder=5)
ax2.scatter(*ep_c_close, color="red", s=60, zorder=5)
ax2.plot([ep_a_close[0], ep_c_close[0]], [ep_a_close[1], ep_c_close[1]], [ep_a_close[2], ep_c_close[2]],
         "r--", lw=2.5)
ax2.text(50, 180, 100, "Orthogonal\ndirection\n→ REJECT", color="red", fontsize=8, ha="center")

ax2.set_xlabel("X (nm)", fontsize=8); ax2.set_ylabel("Y (nm)", fontsize=8); ax2.set_zlabel("Z (nm)", fontsize=8)
ax2.tick_params(labelsize=7)
fa_patch = mpatches.Patch(color="#4C72B0", label="Fragment A (label 1)")
fc_patch = mpatches.Patch(color="#DD8452", label="Fragment C (label 2)")
rej_line  = mpatches.Patch(color="red", label="Merge decision: REJECT")
ax2.legend(handles=[fa_patch, fc_patch, rej_line], fontsize=7, loc="upper left")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "synthetic_correction_scenarios_3d.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")


# ---------------------------------------------------------------------------
# Figure 2: Scoring breakdown — what makes a candidate ACCEPT vs REJECT
# ---------------------------------------------------------------------------

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Candidate Score Profiles — Synthetic Test Cases", fontsize=12, fontweight="bold")

score_names = ["Proximity\n(weight 0.35)", "Alignment\n(weight 0.30)", "Continuity\n(weight 0.25)", "Size\n(weight 0.10)", "Composite\n(overall)"]
accept_scores = [0.90, 0.85, 0.88, 0.95, 0.89]   # Fragment A–B: aligned split
reject_scores  = [0.30, 0.15, 0.20, 0.75, 0.26]   # Fragment A–C: orthogonal, different label

colors_accept = ["#2ca02c" if s >= 0.3 else "#d62728" for s in accept_scores]
colors_reject  = ["#d62728" if s < 0.3 else "#ff7f0e" for s in reject_scores]

x = np.arange(len(score_names))
width = 0.55

ax_a, ax_r = axes
ax_a.bar(x, accept_scores, width, color=colors_accept, edgecolor="white", linewidth=0.8)
ax_a.axhline(0.8, color="green", ls="--", lw=1.5, label="Accept threshold (0.8)")
ax_a.axhline(0.3, color="red",   ls="--", lw=1.5, label="Reject threshold (0.3)")
ax_a.set_title("Split Candidate (A–B) → ACCEPT", fontsize=10)
ax_a.set_xticks(x); ax_a.set_xticklabels(score_names, fontsize=8)
ax_a.set_ylim(0, 1.1); ax_a.set_ylabel("Score"); ax_a.legend(fontsize=8)
for i, v in enumerate(accept_scores):
    ax_a.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

ax_r.bar(x, reject_scores, width, color=colors_reject, edgecolor="white", linewidth=0.8)
ax_r.axhline(0.8, color="green", ls="--", lw=1.5, label="Accept threshold (0.8)")
ax_r.axhline(0.3, color="red",   ls="--", lw=1.5, label="Reject threshold (0.3)")
ax_r.set_title("Cross-Neuron Candidate (A–C) → REJECT", fontsize=10)
ax_r.set_xticks(x); ax_r.set_xticklabels(score_names, fontsize=8)
ax_r.set_ylim(0, 1.1); ax_r.set_ylabel("Score"); ax_r.legend(fontsize=8)
for i, v in enumerate(reject_scores):
    ax_r.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "synthetic_score_profiles.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# ---------------------------------------------------------------------------
# Figure 3: "Before / After" slice showing a simulated split correction
# ---------------------------------------------------------------------------

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
fig3.suptitle("Simulated Split Correction — 2D Cross-Section View (Z-slice)", fontsize=12, fontweight="bold")

vol = np.zeros((64, 64), dtype=np.uint8)
# Fragment A: left half of a horizontal bar
vol[28:32, 5:30] = 1
# Fragment B: right half (separated by a gap at columns 30-34)
vol[28:32, 35:60] = 2
# Fragment C: vertical bar (different neuron, different label)
vol[5:28, 30:34] = 3

cmap_seg = matplotlib.colors.ListedColormap(["#1a1a2e", "#4C72B0", "#DD8452", "#55a868"])

axes3[0].imshow(vol, cmap=cmap_seg, vmin=0, vmax=3, origin="lower", interpolation="nearest")
axes3[0].set_title("Input Segmentation\n(labels 1, 2, 3 = 3 fragments)", fontsize=9)
axes3[0].set_xlabel("X (voxels)"); axes3[0].set_ylabel("Y (voxels)")

# Mark the gap
rect = plt.Rectangle((29.5, 27.5), 5, 4, linewidth=2, edgecolor="yellow", facecolor="none", ls="--")
axes3[0].add_patch(rect)
axes3[0].text(32, 34, "gap\n(50 nm)", color="yellow", ha="center", fontsize=8)

# Middle: highlight the accepted candidate
axes3[1].imshow(vol, cmap=cmap_seg, vmin=0, vmax=3, origin="lower", interpolation="nearest", alpha=0.5)
axes3[1].set_title("Candidate Proposed\n(A–B merge scored ACCEPT)", fontsize=9)
axes3[1].set_xlabel("X (voxels)")
axes3[1].annotate("", xy=(35, 30), xytext=(30, 30),
                   arrowprops=dict(arrowstyle="<->", color="lime", lw=2))
axes3[1].text(32.5, 33, "ACCEPT", color="lime", ha="center", fontsize=9, fontweight="bold")
axes3[1].annotate("", xy=(32, 20), xytext=(32, 27),
                   arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
axes3[1].text(32, 18, "REJECT\n(A–C)", color="red", ha="center", fontsize=8)

# After: merged result
vol_merged = vol.copy()
vol_merged[vol_merged == 2] = 1  # merge label 2 → label 1
cmap_merged = matplotlib.colors.ListedColormap(["#1a1a2e", "#4C72B0", "#DD8452", "#55a868"])
axes3[2].imshow(vol_merged, cmap=cmap_merged, vmin=0, vmax=3, origin="lower", interpolation="nearest")
axes3[2].set_title("Pipeline Output\n(labels 1+2 merged → single neuron)", fontsize=9)
axes3[2].set_xlabel("X (voxels)")
axes3[2].text(32, 30, "MERGED", color="white", ha="center", fontsize=9, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.2", fc="green", alpha=0.7))

for ax in axes3:
    ax.set_aspect("equal")

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "synthetic_slice_before_after.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

print("\nAll images saved to docs/images/")
print("  1. synthetic_correction_scenarios_3d.png — 3D fragment view with ACCEPT/REJECT decisions")
print("  2. synthetic_score_profiles.png           — Score breakdown bar charts")
print("  3. synthetic_slice_before_after.png       — 2D slice before/after correction")
