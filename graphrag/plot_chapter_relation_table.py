"""
Per-chapter triple frequency heatmap for gen4 filtered_and_validated seed KG.
Run: python3 graphrag/plot_chapter_relation_table.py
"""

from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[1]
OUT  = Path(__file__).parent / "chapter_relation_table.png"

# ── Load counts ───────────────────────────────────────────────────────────────
counts = defaultdict(lambda: defaultdict(int))
for ch in range(1, 9):
    with open(REPO / f"gen4_triplets/filtered_and_validated/validated_ch{ch}.csv") as f:
        for row in csv.DictReader(f):
            counts[row["relation_type"]][ch] += 1

totals = {r: sum(counts[r].values()) for r in counts}
rels = sorted(counts, key=lambda r: -totals[r])
chapters = list(range(1, 9))

# Matrix: rows=relations, cols=chapters
mat = np.array([[counts[r][ch] for ch in chapters] for r in rels], dtype=float)
row_totals = mat.sum(axis=1)
col_totals = mat.sum(axis=0)

n_rels = len(rels)
n_ch   = len(chapters)

# ── Build extended matrix with total column and total row ─────────────────────
# Add "Total" column on right and "Total" row on bottom
mat_ext = np.zeros((n_rels + 1, n_ch + 1))
mat_ext[:n_rels, :n_ch] = mat
mat_ext[:n_rels,  n_ch] = row_totals   # right column
mat_ext[ n_rels, :n_ch] = col_totals   # bottom row
mat_ext[ n_rels,  n_ch] = mat.sum()    # grand total

cmap = LinearSegmentedColormap.from_list("wt", ["#f7f7f7", "#2a9d8f"])

fig, ax = plt.subplots(figsize=(13, 9))

# Only colour the data cells, not the totals
im = ax.imshow(mat_ext[:n_rels, :n_ch], aspect="auto", cmap=cmap,
               vmin=0, vmax=mat.max(),
               extent=[-0.5, n_ch - 0.5, n_rels - 0.5, -0.5])

# Draw total columns/rows with a neutral grey background
for i in range(n_rels + 1):
    ax.axhline(i - 0.5, color="white", linewidth=0.4)
for j in range(n_ch + 1):
    ax.axvline(j - 0.5, color="white", linewidth=0.4)

# Cell annotations — data cells
for i in range(n_rels):
    for j in range(n_ch):
        v = int(mat[i, j])
        text_color = "white" if v > mat.max() * 0.55 else "#333333"
        ax.text(j, i, str(v) if v > 0 else "–",
                ha="center", va="center", fontsize=8, color=text_color)

# Total column (right)
for i in range(n_rels):
    ax.text(n_ch, i, str(int(row_totals[i])),
            ha="center", va="center", fontsize=8,
            color="#1a1a1a", fontweight="bold",
            bbox=dict(fc="#e8e8e8", ec="none", pad=1.5))

# Total row (bottom)
for j in range(n_ch):
    ax.text(j, n_rels, str(int(col_totals[j])),
            ha="center", va="center", fontsize=8,
            color="#1a1a1a", fontweight="bold",
            bbox=dict(fc="#e8e8e8", ec="none", pad=1.5))

# Grand total
ax.text(n_ch, n_rels, str(int(mat.sum())),
        ha="center", va="center", fontsize=8.5,
        color="white", fontweight="bold",
        bbox=dict(fc="#2a9d8f", ec="none", pad=1.5))

# Axes ticks
ax.set_xlim(-0.5, n_ch + 0.5)
ax.set_ylim(n_rels + 0.5, -0.5)
ax.set_xticks(range(n_ch + 1))
ax.set_xticklabels([f"Ch{c}" for c in chapters] + ["Total"],
                   fontsize=9.5)
ax.set_yticks(range(n_rels + 1))
ax.set_yticklabels(rels + ["Total"], fontsize=8.5)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

ax.set_title("Per-Chapter Triple Frequency — Gen4 filtered_and_validated",
             fontsize=12, fontweight="bold", pad=20)

plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label="Triple count")
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
