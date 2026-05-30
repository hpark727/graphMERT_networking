"""
Relation frequency stats across all validated chapter KGs.

Reads:  gen1_triplets/validated/validated_ch{1-8}.csv
Prints: relation vocab size + per-relation counts
Saves:  gen1_triplets/validated/relation_distribution.png

Run from repo root:
    python3 graphrag/relation_stats.py
"""

from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

import argparse

_REPO = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--val-dir", type=Path, default=_REPO / "gen1_triplets/validated",
                help="Directory containing validated_ch{N}.csv files")
args = ap.parse_args()
_VAL_DIR = args.val_dir

frames = []
for ch in range(1, 9):
    p = _VAL_DIR / f"validated_ch{ch}.csv"
    if p.exists():
        frames.append(pd.read_csv(p))
    else:
        print(f"  [skip] {p.name} not found")

df = pd.concat(frames, ignore_index=True)
counts = Counter(df["relation_type"].str.strip())

print(f"\nTotal validated triples : {len(df)}")
print(f"Relation vocab size     : {len(counts)}\n")
print(f"{'Relation':<25} {'Count':>6}  {'%':>6}")
print("-" * 42)
for rel, cnt in counts.most_common():
    print(f"{rel:<25} {cnt:>6}  {cnt/len(df)*100:>5.1f}%")

# --- bar chart ---
rels, freqs = zip(*counts.most_common())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(rels, freqs, color="steelblue", edgecolor="white")
ax.bar_label(bars, padding=3, fontsize=8)
ax.set_xlabel("Relation type")
ax.set_ylabel("Triple count")
ax.set_title(f"Relation distribution — all chapters ({len(df)} validated triples)")
ax.set_xticks(range(len(rels)))
ax.set_xticklabels(rels, rotation=40, ha="right", fontsize=9)
ax.margins(y=0.12)
plt.tight_layout()

out = _VAL_DIR / "relation_distribution.png"
_VAL_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"\nSaved → {out}")
plt.show()
