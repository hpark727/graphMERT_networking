"""
Find candidate vocabulary bridges between unmatched seed KG heads and
discovered entities from head_positions datasets.

For each seed KG head that never appears in any chunk's head_positions,
find the most similar discovered entities using substring and token-overlap
scoring. Writes a CSV for manual review.

Run: python3 graphrag/generate_vocab_bridge.py
Output:
  gen4_triplets/vocab_bridge_candidates.csv  — review + confirm this file
  gen4_triplets/vocab_bridge.json            — edit this to confirm bridges
"""

import csv, json
from collections import defaultdict
from pathlib import Path
from datasets import load_from_disk

REPO = Path(__file__).resolve().parents[1]
OUT_CANDIDATES = REPO / "gen4_triplets/vocab_bridge_candidates.csv"
OUT_BRIDGE     = REPO / "gen4_triplets/vocab_bridge.json"


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def score(seed_head: str, candidate: str) -> float:
    s = seed_head.lower()
    c = candidate.lower()
    # Exact substring containment
    if s == c:
        return 1.0
    if s in c or c in s:
        return 0.85
    # Token overlap (Jaccard)
    return token_overlap(s, c)


# ── Load seed KG heads ────────────────────────────────────────────────────────
seed_heads = set()
for ch in range(1, 9):
    with open(REPO / f"gen4_triplets/filtered_and_validated/validated_ch{ch}.csv") as f:
        for row in csv.DictReader(f):
            seed_heads.add(row["head"].strip().lower())

# ── Load all discovered entities ──────────────────────────────────────────────
discovered_raw: set[str] = set()
for ch in range(1, 9):
    if ch == 1:
        p = REPO / "json_data/entity_discovery_output_gpt-oss-120b_all"
    else:
        p = REPO / f"json_data/entity_discovery_output/ch{ch}_gpt-oss-120b_all"
    ds = load_from_disk(str(p))
    import json as _json
    for ex in ds:
        for ent in _json.loads(ex["head_positions"]):
            discovered_raw.add(ent.strip().lower())

# ── Find unmatched seed heads ─────────────────────────────────────────────────
unmatched = sorted(seed_heads - discovered_raw)
print(f"Seed heads:          {len(seed_heads)}")
print(f"Discovered entities: {len(discovered_raw)}")
print(f"Unmatched:           {len(unmatched)}")

# ── Score candidates for each unmatched head ──────────────────────────────────
TOP_K = 5
MIN_SCORE = 0.2

rows = []
for head in unmatched:
    candidates = []
    for d in discovered_raw:
        s = score(head, d)
        if s >= MIN_SCORE:
            candidates.append((s, d))
    candidates.sort(reverse=True)
    top = candidates[:TOP_K]

    rows.append({
        "seed_head":       head,
        "confirmed_bridge": "",  # user fills this in
        "score_1": f"{top[0][0]:.2f}" if len(top) > 0 else "",
        "candidate_1":     top[0][1]  if len(top) > 0 else "",
        "score_2": f"{top[1][0]:.2f}" if len(top) > 1 else "",
        "candidate_2":     top[1][1]  if len(top) > 1 else "",
        "score_3": f"{top[2][0]:.2f}" if len(top) > 2 else "",
        "candidate_3":     top[2][1]  if len(top) > 2 else "",
        "score_4": f"{top[3][0]:.2f}" if len(top) > 3 else "",
        "candidate_4":     top[3][1]  if len(top) > 3 else "",
        "score_5": f"{top[4][0]:.2f}" if len(top) > 4 else "",
        "candidate_5":     top[4][1]  if len(top) > 4 else "",
    })

# ── Write candidates CSV ──────────────────────────────────────────────────────
fieldnames = ["seed_head", "confirmed_bridge",
              "score_1", "candidate_1",
              "score_2", "candidate_2",
              "score_3", "candidate_3",
              "score_4", "candidate_4",
              "score_5", "candidate_5"]
with open(OUT_CANDIDATES, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows → {OUT_CANDIDATES}")

# ── Write empty bridge JSON (populate confirmed_bridge in CSV first,
#    then run: python3 graphrag/generate_vocab_bridge.py --finalize)
print(f"\nNext steps:")
print(f"  1. Open {OUT_CANDIDATES}")
print(f"  2. For each row, paste the best candidate into 'confirmed_bridge'")
print(f"     (or leave blank to skip that seed head)")
print(f"  3. Run: python3 graphrag/generate_vocab_bridge.py --finalize")

# ── --finalize: read confirmed CSV and write vocab_bridge.json ─────────────────
import sys
if "--finalize" in sys.argv:
    bridge: dict[str, str] = {}
    with open(OUT_CANDIDATES, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cb = row["confirmed_bridge"].strip().lower()
            if cb:
                bridge[row["seed_head"]] = cb
    with open(OUT_BRIDGE, "w", encoding="utf-8") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(bridge)} bridges → {OUT_BRIDGE}")
