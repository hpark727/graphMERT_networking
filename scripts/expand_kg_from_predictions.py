"""
Map single predicted tokens → canonical seed-KG entity names, then emit ranked
candidate triples that are not already in the seed KG.

Usage (on Della):
  python3 scripts/expand_kg_from_predictions.py \
      --predictions  outputs/kg_expansion_bert_init_stage2/train/top_15 \
      --seed_kg_dir  gen4_triplets/seed_kg \
      --output       outputs/kg_expansion_bert_init_stage2/train/candidate_triples.csv \
      [--min_score -5.0] [--min_evidence 1] [--top_k 2000]

Optional: compare two prediction sets:
  python3 scripts/expand_kg_from_predictions.py \
      --predictions  outputs/kg_expansion_bert_init_stage2/train/top_15 \
      --compare      outputs/kg_expansion_bert_init/train/top_15 \
      --seed_kg_dir  gen4_triplets/seed_kg \
      --output       outputs/kg_expansion_bert_init_stage2/train/candidate_triples.csv
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

import numpy as np
from datasets import load_from_disk


# ── Seed-KG helpers ──────────────────────────────────────────────────────────

def load_seed_kg(seed_kg_dir):
    """Return (entities, existing_triples) from all seed_kg_ch*.csv files."""
    entities = set()
    existing = set()
    for path in sorted(glob.glob(os.path.join(seed_kg_dir, "seed_kg_ch*.csv"))):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                head = row["head"].strip()
                rel  = row["relation_type"].strip()
                tail = row["tail"].strip()
                entities.add(head)
                entities.add(tail)
                existing.add((head, rel, tail))
    return entities, existing


def build_token_index(entities):
    """Build word-token → set of entities index (word-level split of entity names)."""
    idx = defaultdict(set)
    for entity in entities:
        for tok in entity.split():
            idx[tok].add(entity)
    return idx


# ── Scoring ──────────────────────────────────────────────────────────────────

def aggregate_scores(ds, token_idx):
    """
    For each (head, relation, entity) triple candidate, accumulate log-prob
    evidence across all rows in the dataset.

    Returns:
        scores  : dict[(head, rel, entity)] → sum of log(prob)
        counts  : dict[(head, rel, entity)] → number of supporting rows
        token_used : dict[(head, rel, entity)] → set of matched tokens
    """
    scores    = defaultdict(float)
    counts    = defaultdict(int)
    token_hit = defaultdict(set)

    for row in ds:
        head = row["head"]
        rel  = row["relation"]
        tokens = row["predictions"].split()
        probs  = row["probabilities"]

        for tok, prob in zip(tokens, probs):
            # Skip subword fragments (##...) — not mappable to full entities
            if tok.startswith("##"):
                continue
            candidates = token_idx.get(tok)
            if not candidates:
                continue
            log_p = math.log(prob + 1e-12)
            for entity in candidates:
                key = (head, rel, entity)
                scores[key]    += log_p
                counts[key]    += 1
                token_hit[key].add(tok)

    return scores, counts, token_hit


# ── Output ───────────────────────────────────────────────────────────────────

def write_candidates(path, ranked, scores, counts, token_hit, existing):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["head", "relation", "tail", "score", "n_evidence",
                         "matched_tokens", "in_seed_kg"])
        for key in ranked:
            head, rel, entity = key
            writer.writerow([
                head, rel, entity,
                f"{scores[key]:.4f}",
                counts[key],
                " | ".join(sorted(token_hit[key])),
                "yes" if key in existing else "no",
            ])
    print(f"Wrote {len(ranked)} rows → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",  required=True,
                        help="Path to top_k HuggingFace dataset from predict_tails")
    parser.add_argument("--seed_kg_dir",  required=True,
                        help="Directory containing seed_kg_ch*.csv files")
    parser.add_argument("--output",       required=True,
                        help="Output CSV path")
    parser.add_argument("--compare",      default=None,
                        help="Optional second predictions dataset for side-by-side output")
    parser.add_argument("--min_score",    type=float, default=-8.0,
                        help="Drop candidates with total log-prob below this threshold")
    parser.add_argument("--min_evidence", type=int,   default=1,
                        help="Minimum number of supporting rows")
    parser.add_argument("--top_k",        type=int,   default=2000,
                        help="Maximum number of candidate triples to output")
    args = parser.parse_args()

    # Load seed KG
    print("Loading seed KG …")
    entities, existing = load_seed_kg(args.seed_kg_dir)
    print(f"  {len(entities):,} entities, {len(existing):,} existing triples")

    token_idx = build_token_index(entities)
    print(f"  {len(token_idx):,} unique word-tokens in entity index")

    # Load & score primary predictions
    print(f"\nLoading predictions from {args.predictions} …")
    ds = load_from_disk(args.predictions)
    print(f"  {len(ds):,} rows")

    scores, counts, token_hit = aggregate_scores(ds, token_idx)
    print(f"  {len(scores):,} raw (head, rel, entity) candidates before filtering")

    # Filter
    new_only = {k for k in scores
                if k not in existing
                and k[0] != k[2]             # drop circular (head == tail)
                and scores[k] >= args.min_score
                and counts[k] >= args.min_evidence}
    print(f"  {len(new_only):,} novel candidates after filtering")

    ranked_new = sorted(new_only, key=lambda k: scores[k], reverse=True)[:args.top_k]
    ranked_all = sorted(scores,   key=lambda k: scores[k], reverse=True)[:args.top_k]

    # Print top-30 novel candidates
    print(f"\n=== Top 30 novel candidate triples (not in seed KG) ===")
    for key in ranked_new[:30]:
        head, rel, entity = key
        print(f"  {head:30s} --{rel:25s}--> {entity:30s}"
              f"  score={scores[key]:6.2f}  n={counts[key]}"
              f"  tok=[{' | '.join(sorted(token_hit[key]))}]")

    # Also show how many new entities (tails not already in seed KG entity set)
    new_tail_entities = {k[2] for k in new_only if k[2] not in entities}
    known_tails = {k[2] for k in new_only if k[2] in entities}
    print(f"\n  Tail entity breakdown:")
    print(f"    Known seed-KG entities used as tails : {len(known_tails):,}")
    print(f"    Entirely new tail entities            : {len(new_tail_entities):,}")

    # Write primary output (novel only)
    write_candidates(args.output, ranked_new, scores, counts, token_hit, existing)

    # Optional comparison
    if args.compare:
        print(f"\nLoading comparison predictions from {args.compare} …")
        ds2 = load_from_disk(args.compare)
        scores2, counts2, token_hit2 = aggregate_scores(ds2, token_idx)
        new2 = {k for k in scores2
                if k not in existing
                and k[0] != k[2]
                and scores2[k] >= args.min_score
                and counts2[k] >= args.min_evidence}
        ranked2 = sorted(new2, key=lambda k: scores2[k], reverse=True)[:args.top_k]

        novel_in_s2_not_s1 = set(ranked_new[:500]) - set(ranked2[:500])
        novel_in_s1_not_s2 = set(ranked2[:500]) - set(ranked_new[:500])
        print(f"\n  Top-500 comparison:")
        print(f"    In stage-2 but not stage-1 : {len(novel_in_s2_not_s1)}")
        print(f"    In stage-1 but not stage-2 : {len(novel_in_s1_not_s2)}")

        compare_path = args.output.replace(".csv", "_compare_s1.csv")
        write_candidates(compare_path, ranked2, scores2, counts2, token_hit2, existing)

    print("\nDone.")


if __name__ == "__main__":
    main()
