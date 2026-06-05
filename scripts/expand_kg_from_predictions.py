"""
Map single predicted tokens → canonical seed-KG entity names, then emit ranked
candidate triples that are not already in the seed KG.

Usage (on Della):
  # Single split
  python3 scripts/expand_kg_from_predictions.py \
      --predictions  outputs/kg_expansion_bert_init_stage2/train/top_15 \
      --seed_kg_dir  gen4_triplets/seed_kg \
      --output       outputs/kg_expansion_bert_init_stage2/full_kg/candidate_triples.csv

  # Full textbook: combine train (ch1-ch7) + eval (ch8)
  python3 scripts/expand_kg_from_predictions.py \
      --predictions  outputs/kg_expansion_bert_init_stage2/train/top_15 \
                     outputs/kg_expansion_bert_init_stage2/eval/top_15 \
      --seed_kg_dir  gen4_triplets/seed_kg \
      --output       outputs/kg_expansion_bert_init_stage2/full_kg/candidate_triples.csv \
      --per_pair_top_k 20

--per_pair_top_k (default 20): for each unique (head, relation) pair, take the
  top-N candidate entities ranked by specificity-weighted score.  This ensures
  every predicted pair gets representation rather than high-frequency tokens
  dominating the global ranking.

Optional comparison:
  python3 scripts/expand_kg_from_predictions.py \
      --predictions  outputs/kg_expansion_bert_init_stage2/train/top_15 \
                     outputs/kg_expansion_bert_init_stage2/eval/top_15 \
      --compare      outputs/kg_expansion_bert_init/train/top_15 \
                     outputs/kg_expansion_bert_init/eval/top_15 \
      --seed_kg_dir  gen4_triplets/seed_kg \
      --output       outputs/kg_expansion_bert_init_stage2/full_kg/candidate_triples.csv
"""

import argparse
import csv
import glob
import math
import os
from collections import defaultdict

from datasets import load_from_disk, concatenate_datasets


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
    """
    Build word-token → set of entities index (word-level split of entity names).
    Also returns fan_out: token → number of entities that contain it (used for
    specificity weighting — lower fan-out = more specific prediction).
    """
    idx = defaultdict(set)
    for entity in entities:
        for tok in entity.split():
            idx[tok].add(entity)
    fan_out = {tok: len(ents) for tok, ents in idx.items()}
    return idx, fan_out


# ── Scoring ──────────────────────────────────────────────────────────────────

def aggregate_scores(ds, token_idx, fan_out):
    """
    For each (head, relation, entity) triple candidate, accumulate a
    specificity-weighted log-prob score across all rows in the dataset.

    The raw log-prob for a (head, rel, tok→entity) hit is divided by the
    token's fan-out (number of entities it maps to).  A token like "chrome"
    that maps to exactly 1 entity contributes its full log-prob; "delay"
    mapping to 17 entities contributes 1/17 of its log-prob.  This ensures
    specific signals dominate the ranking rather than frequent broad tokens.

    Returns:
        scores    : dict[(head, rel, entity)] → weighted score
        counts    : dict[(head, rel, entity)] → number of supporting rows
        token_hit : dict[(head, rel, entity)] → set of matched tokens
    """
    scores    = defaultdict(float)
    counts    = defaultdict(int)
    token_hit = defaultdict(set)

    for row in ds:
        head   = row["head"]
        rel    = row["relation"]
        tokens = row["predictions"].split()
        probs  = row["probabilities"]

        for tok, prob in zip(tokens, probs):
            if tok.startswith("##"):
                continue
            candidates = token_idx.get(tok)
            if not candidates:
                continue
            n_entities = fan_out[tok]
            log_p = math.log(prob + 1e-12) / n_entities  # specificity weight
            for entity in candidates:
                key = (head, rel, entity)
                scores[key]    += log_p
                counts[key]    += 1
                token_hit[key].add(tok)

    return scores, counts, token_hit


def token_density(entity, tok):
    """Fraction of the entity's words that equal tok (higher = more specific match)."""
    words = entity.split()
    return words.count(tok) / len(words) if words else 0.0


def select_candidates(scores, counts, token_hit, existing,
                      min_score, min_evidence, per_pair_top_k, per_token_top_k=1):
    """
    Filter candidates and apply per-(head, relation) budget.

    To eliminate fan-out ties (a single broad token like "packet" mapping to 32
    entities and filling the entire budget), we first select the top
    per_token_top_k entities per (head, relation, token) group ranked by token
    density (fraction of the entity's words that equal the predicted token).
    This means "packet" → "packet" (density 1.0) beats "udp packet" (0.5) beats
    "deep packet inspection" (0.33).  After this reduction, the per-pair budget
    is applied across distinct token signals.

    Returns a list of keys sorted by score (descending).
    """
    # Global filter
    valid = {k for k in scores
             if k not in existing
             and k[0] != k[2]
             and scores[k] >= min_score
             and counts[k] >= min_evidence}

    # Per-(head, rel, token) reduction: keep top per_token_top_k entities by density
    by_token_group = defaultdict(list)
    for key in valid:
        for tok in token_hit[key]:
            by_token_group[(key[0], key[1], tok)].append(key)

    reduced = set()
    for group_keys in by_token_group.values():
        group_keys.sort(key=lambda k: token_density(k[2], next(iter(token_hit[k]))), reverse=True)
        reduced.update(group_keys[:per_token_top_k])

    # Per-(head, relation) top-K across distinct token representatives
    if per_pair_top_k is not None:
        by_pair = defaultdict(list)
        for k in reduced:
            by_pair[(k[0], k[1])].append(k)
        selected = []
        for pair_keys in by_pair.values():
            pair_keys.sort(key=lambda k: scores[k], reverse=True)
            selected.extend(pair_keys[:per_pair_top_k])
        reduced = set(selected)

    return sorted(reduced, key=lambda k: scores[k], reverse=True)


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
    print(f"Wrote {len(ranked):,} rows → {path}")


def print_summary(ranked, scores, counts, token_hit, existing, label=""):
    n = len(ranked)
    unique_pairs = len({(k[0], k[1]) for k in ranked})
    unique_heads = len({k[0] for k in ranked})
    print(f"\n{label}Summary: {n:,} candidates | {unique_pairs:,} (head,rel) pairs | {unique_heads:,} unique heads")

    print(f"\n=== Top 30 novel candidates {label}===")
    for key in ranked[:30]:
        head, rel, entity = key
        print(f"  {head:30s} --{rel:20s}--> {entity:35s}"
              f"  score={scores[key]:6.3f}  n={counts[key]}"
              f"  tok=[{' | '.join(sorted(token_hit[key]))}]")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions",    required=True, nargs="+",
                        help="One or more paths to top_k HuggingFace datasets from predict_tails "
                             "(e.g. train/top_15 eval/top_15 — they are concatenated before scoring)")
    parser.add_argument("--seed_kg_dir",    required=True,
                        help="Directory containing seed_kg_ch*.csv files")
    parser.add_argument("--output",         required=True,
                        help="Output CSV path")
    parser.add_argument("--compare",        default=None, nargs="+",
                        help="Optional second set of prediction paths for side-by-side output")
    parser.add_argument("--per_pair_top_k",  type=int,   default=20,
                        help="Max candidates per (head, relation) pair (default 20)")
    parser.add_argument("--per_token_top_k", type=int,   default=1,
                        help="Max entities per (head, relation, token) group, ranked by token "
                             "density — 1 = only the most specific entity per token signal, "
                             "higher values trade quality for more candidates (default 1)")
    parser.add_argument("--min_score",       type=float, default=-5.0,
                        help="Min specificity-weighted log-prob score (default -5.0)")
    parser.add_argument("--min_evidence",    type=int,   default=1,
                        help="Min number of supporting prediction rows (default 1)")
    args = parser.parse_args()

    # Load seed KG
    print("Loading seed KG …")
    entities, existing = load_seed_kg(args.seed_kg_dir)
    print(f"  {len(entities):,} entities, {len(existing):,} existing triples")

    token_idx, fan_out = build_token_index(entities)
    print(f"  {len(token_idx):,} unique word-tokens in entity index")

    # Load & score primary predictions
    print(f"\nLoading predictions from: {args.predictions} …")
    ds = concatenate_datasets([load_from_disk(p) for p in args.predictions])
    print(f"  {len(ds):,} rows total ({len(args.predictions)} split(s))")

    scores, counts, token_hit = aggregate_scores(ds, token_idx, fan_out)
    print(f"  {len(scores):,} raw (head, rel, entity) candidates before filtering")

    ranked = select_candidates(scores, counts, token_hit, existing,
                               args.min_score, args.min_evidence,
                               args.per_pair_top_k, args.per_token_top_k)

    print_summary(ranked, scores, counts, token_hit, existing)
    write_candidates(args.output, ranked, scores, counts, token_hit, existing)

    # Optional comparison
    if args.compare:
        print(f"\nLoading comparison predictions from: {args.compare} …")
        ds2 = concatenate_datasets([load_from_disk(p) for p in args.compare])
        scores2, counts2, token_hit2 = aggregate_scores(ds2, token_idx, fan_out)
        ranked2 = select_candidates(scores2, counts2, token_hit2, existing,
                                    args.min_score, args.min_evidence,
                                    args.per_pair_top_k, args.per_token_top_k)
        print_summary(ranked2, scores2, counts2, token_hit2, existing, label="(compare) ")
        compare_path = args.output.replace(".csv", "_compare.csv")
        write_candidates(compare_path, ranked2, scores2, counts2, token_hit2, existing)

        top_s2 = set(ranked[:500])
        top_c  = set(ranked2[:500])
        print(f"\n  Top-500 overlap: {len(top_s2 & top_c)} shared, "
              f"{len(top_s2 - top_c)} only in primary, {len(top_c - top_s2)} only in compare")

    print("\nDone.")


if __name__ == "__main__":
    main()
