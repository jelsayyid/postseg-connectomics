"""
Generate visualizations from the actual CREMI Sample A pipeline run.
Uses real output CSVs from output/cremi_sample_a/.

Run from project root:
    python docs/generate_cremi_visuals.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_DIR = "output/cremi_sample_a"
conns = pd.read_csv(os.path.join(DATA_DIR, "connections.csv"))
frags = pd.read_csv(os.path.join(DATA_DIR, "fragments.csv"))
structs = pd.read_csv(os.path.join(DATA_DIR, "structures.csv"))

STATUS_COLORS = {"accepted": "#2ca02c", "rejected": "#d62728", "ambiguous": "#ff7f0e"}
STATUS_LABELS = {"accepted": f"Accepted (n={len(conns[conns.status=='accepted'])})",
                 "rejected": f"Rejected (n={len(conns[conns.status=='rejected'])})",
                 "ambiguous": f"Ambiguous (n={len(conns[conns.status=='ambiguous'])})"}

frag_lookup = frags.set_index("fragment_id")

# ---------------------------------------------------------------------------
# Figure 1: 3D scatter of fragment centroids + accepted connection lines
# ---------------------------------------------------------------------------
fig1 = plt.figure(figsize=(13, 6))
fig1.suptitle("CREMI Sample A — Fragment Map with Pipeline Decisions\n"
              "(Drosophila brain EM, 64×256×256 voxels, 40×4×4 nm resolution)",
              fontsize=11, fontweight="bold")

ax = fig1.add_subplot(111, projection="3d")

# Plot all fragment centroids
ax.scatter(frags["centroid_x"], frags["centroid_y"], frags["centroid_z"],
           c="#aec7e8", alpha=0.5, s=10, label=f"Fragments (n={len(frags)})", zorder=1)

# Draw accepted connections as green lines
accepted = conns[conns["status"] == "accepted"]
for _, row in accepted.iterrows():
    try:
        fa = frag_lookup.loc[int(row["fragment_a"])]
        fb = frag_lookup.loc[int(row["fragment_b"])]
        ax.plot([fa["centroid_x"], fb["centroid_x"]],
                [fa["centroid_y"], fb["centroid_y"]],
                [fa["centroid_z"], fb["centroid_z"]],
                color="#2ca02c", lw=1.8, alpha=0.9, zorder=3)
    except KeyError:
        pass

ax.set_xlabel("X (nm)", fontsize=8)
ax.set_ylabel("Y (nm)", fontsize=8)
ax.set_zlabel("Z (nm)", fontsize=8)
ax.tick_params(labelsize=7)

frag_patch  = mpatches.Patch(color="#aec7e8", label=f"Fragments (n={len(frags)})")
accept_patch = mpatches.Patch(color="#2ca02c", label=f"Accepted merges (n={len(accepted)})")
ax.legend(handles=[frag_patch, accept_patch], fontsize=8, loc="upper left")

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "cremi_fragment_map_3d.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Figure 2: Score distributions — composite score by status + gap distance
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("CREMI Sample A — Score Distributions Across 3,369 Candidates",
              fontsize=12, fontweight="bold")

# Panel A: composite score histogram
ax_c = axes[0]
for status, color in STATUS_COLORS.items():
    subset = conns[conns["status"] == status]["composite_score"]
    if len(subset):
        ax_c.hist(subset, bins=30, color=color, alpha=0.7, label=STATUS_LABELS[status], edgecolor="white", lw=0.4)
ax_c.axvline(0.8, color="green", ls="--", lw=1.5, label="Accept threshold (0.8)")
ax_c.axvline(0.3, color="red",   ls="--", lw=1.5, label="Reject threshold (0.3)")
ax_c.set_xlabel("Composite Score"); ax_c.set_ylabel("Count")
ax_c.set_title("Composite Score Distribution")
ax_c.legend(fontsize=7)

# Panel B: gap distance histogram
ax_d = axes[1]
for status, color in STATUS_COLORS.items():
    subset = conns[conns["status"] == status]["gap_distance"]
    if len(subset):
        ax_d.hist(subset, bins=30, color=color, alpha=0.7, label=STATUS_LABELS[status], edgecolor="white", lw=0.4)
ax_d.axvline(400, color="purple", ls="--", lw=1.5, label="Max distance (400 nm)")
ax_d.set_xlabel("Gap Distance (nm)"); ax_d.set_ylabel("Count")
ax_d.set_title("Gap Distance Distribution")
ax_d.legend(fontsize=7)

# Panel C: per-score breakdown for accepted vs rejected (medians)
ax_s = axes[2]
score_cols = ["proximity_score", "alignment_score", "continuity_score", "size_score", "composite_score"]
score_labels = ["Proximity\n(0.35)", "Alignment\n(0.30)", "Continuity\n(0.25)", "Size\n(0.10)", "Composite"]
x = np.arange(len(score_cols))
width = 0.28

for i, (status, color) in enumerate(STATUS_COLORS.items()):
    subset = conns[conns["status"] == status]
    medians = [subset[col].median() for col in score_cols]
    ax_s.bar(x + (i - 1) * width, medians, width, color=color, alpha=0.85,
             label=status.capitalize(), edgecolor="white", lw=0.5)

ax_s.axhline(0.8, color="green", ls="--", lw=1.2, label="Accept threshold")
ax_s.axhline(0.3, color="red",   ls="--", lw=1.2, label="Reject threshold")
ax_s.set_xticks(x); ax_s.set_xticklabels(score_labels, fontsize=8)
ax_s.set_ylabel("Median Score"); ax_s.set_ylim(0, 1.1)
ax_s.set_title("Median Score by Status & Factor")
ax_s.legend(fontsize=7)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "cremi_score_distributions.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ---------------------------------------------------------------------------
# Figure 3: Pipeline outcome summary + assembled structure stats
# ---------------------------------------------------------------------------
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5))
fig3.suptitle("CREMI Sample A — Pipeline Outcome Summary", fontsize=12, fontweight="bold")

# Panel A: Candidate decision pie
ax_pie = axes3[0]
counts = conns["status"].value_counts()
labels_pie = [f"{s.capitalize()}\n{counts.get(s,0):,} ({100*counts.get(s,0)/len(conns):.1f}%)"
              for s in ["accepted", "ambiguous", "rejected"]]
colors_pie = [STATUS_COLORS["accepted"], STATUS_COLORS["ambiguous"], STATUS_COLORS["rejected"]]
wedge_vals = [counts.get(s, 0) for s in ["accepted", "ambiguous", "rejected"]]
ax_pie.pie(wedge_vals, labels=labels_pie, colors=colors_pie, startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5}, textprops={"fontsize": 9})
ax_pie.set_title(f"Candidate Decisions\n(total: {len(conns):,})", fontsize=10)

# Panel B: Structure sizes (fragments per structure)
ax_struct = axes3[1]
frag_counts = structs["num_fragments"]
bins = sorted(frag_counts.unique())
counts_by_size = frag_counts.value_counts().sort_index()
ax_struct.bar(counts_by_size.index.astype(str), counts_by_size.values,
              color="#5e81ac", edgecolor="white", lw=0.8)
ax_struct.set_xlabel("Fragments per Structure")
ax_struct.set_ylabel("Number of Structures")
ax_struct.set_title(f"Assembled Structures by Size\n(total: {len(structs)} structures)")
for bar, val in zip(ax_struct.patches, counts_by_size.values):
    ax_struct.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                   str(val), ha="center", fontsize=9, fontweight="bold")

# Panel C: Structure confidence distribution
ax_conf = axes3[2]
conf = structs["confidence"]
ax_conf.hist(conf, bins=15, color="#81a1c1", edgecolor="white", lw=0.8)
ax_conf.axvline(conf.median(), color="navy", ls="--", lw=1.5,
                label=f"Median: {conf.median():.3f}")
ax_conf.set_xlabel("Structure Confidence (weakest-link)")
ax_conf.set_ylabel("Count")
ax_conf.set_title("Assembled Structure Confidence")
ax_conf.legend(fontsize=8)

# Annotate: % with ambiguous regions
n_amb = structs["has_ambiguous_regions"].sum()
ax_conf.text(0.05, 0.88, f"{n_amb}/{len(structs)} structures\nhave ambiguous\nneighbors",
             transform=ax_conf.transAxes, fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ebcb8b", alpha=0.8))

plt.tight_layout()
out3 = os.path.join(OUT_DIR, "cremi_outcome_summary.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

print("\nAll CREMI images saved to docs/images/")
print("  1. cremi_fragment_map_3d.png     — 3D centroid map with accepted merge lines")
print("  2. cremi_score_distributions.png — composite score + gap distance + per-factor medians")
print("  3. cremi_outcome_summary.png     — pie chart, structure sizes, confidence histogram")
