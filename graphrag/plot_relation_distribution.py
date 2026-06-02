"""
Frequency distribution of relation types in gen4 filtered_and_validated seed KG.
Run: python3 graphrag/plot_relation_distribution.py
"""

from collections import defaultdict
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT  = Path(__file__).parent / "relation_distribution.png"

seed = defaultdict(int)
for ch in range(1, 9):
    with open(REPO / f"gen4_triplets/filtered_and_validated/validated_ch{ch}.csv") as f:
        for row in csv.DictReader(f):
            seed[row["relation_type"]] += 1

rels, counts = zip(*sorted(seed.items(), key=lambda x: -x[1]))

THRESHOLD = 75
colors = ["#2a9d8f" if c >= THRESHOLD else "#e76f51" for c in counts]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(len(rels)), counts, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

# Value labels on bars
for i, (bar, c) in enumerate(zip(bars, counts)):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(c), ha="center", va="bottom", fontsize=7.5, color="#333333")

# Threshold line
ax.axhline(THRESHOLD, color="#e76f51", linewidth=1.2, linestyle="--", zorder=2, alpha=0.8)
ax.text(len(rels) - 0.4, THRESHOLD + 3, f"threshold = {THRESHOLD}",
        ha="right", fontsize=8.5, color="#e76f51")

ax.set_xticks(range(len(rels)))
ax.set_xticklabels(rels, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Seed triple count", fontsize=11)
ax.set_title("Relation Distribution - all chapters",
             fontsize=12, fontweight="bold")
ax.set_xlim(-0.6, len(rels) - 0.4)
ax.set_ylim(0, max(counts) * 1.12)
ax.grid(axis="y", alpha=0.3, linewidth=0.6)
ax.set_axisbelow(True)

legend = [
    mpatches.Patch(fc="#2a9d8f", label=f"≥ {THRESHOLD} triples"),
    mpatches.Patch(fc="#e76f51", label=f"< {THRESHOLD} triples"),
]
ax.legend(handles=legend, fontsize=9, framealpha=0.9)

ax.text(0.99, 0.97, f"Total: {sum(counts)} triples  |  {len(rels)} relations",
        transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555555")

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
