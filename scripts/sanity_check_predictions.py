"""
Quick sanity check for predict_tails output.

Usage:
  python3 scripts/sanity_check_predictions.py \
      --path outputs/kg_expansion_bert_init/train/top_15

Optional comparison against a second model:
  python3 scripts/sanity_check_predictions.py \
      --path outputs/kg_expansion_bert_init/train/top_15 \
      --compare outputs/kg_expansion/train/top_15
"""

import argparse
import numpy as np
from collections import defaultdict
from datasets import load_from_disk


def aggregate(ds):
    scores = defaultdict(float)
    counts = defaultdict(int)
    for row in ds:
        tokens = row['predictions'].split()
        for t, p in zip(tokens, row['probabilities']):
            key = (row['head'], row['relation'], t)
            scores[key] += np.log(p + 1e-12)
            counts[key] += 1
    return scores, counts


def print_top(scores, counts, n=30, label=""):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== Top {n} aggregated predictions {label}===")
    for (head, rel, tok), score in ranked[:n]:
        print(f"  {head:20s} --{rel:25s}--> {tok:15s}  score={score:.1f}  n={counts[(head, rel, tok)]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to top_15 dataset")
    parser.add_argument("--compare", default=None, help="Optional second dataset to compare")
    args = parser.parse_args()

    ds = load_from_disk(args.path)
    print(f"\nLoaded {len(ds)} rows from {args.path}")
    print(f"Columns: {ds.column_names}")

    # 1. Raw spot-check
    print("\n=== Raw spot-check (first 5 rows) ===")
    for row in ds.select(range(min(5, len(ds)))):
        tokens = row['predictions'].split()
        probs  = row['probabilities']
        print(f"  head={row['head']!r:20s}  rel={row['relation']!r:25s}")
        for t, p in zip(tokens[:5], probs[:5]):
            print(f"    {t:20s} {p:.4f}")

    # 2. Head frequency
    head_counts = defaultdict(int)
    for row in ds:
        head_counts[row['head']] += 1
    print("\n=== Most frequent head entities ===")
    for head, n in sorted(head_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {head:30s}  {n:4d}")

    # 3. Aggregated top predictions
    scores, counts = aggregate(ds)
    print_top(scores, counts, label=f"({args.path.split('/')[-3] if '/' in args.path else ''}) ")

    # 4. Optional comparison
    if args.compare:
        ds2 = load_from_disk(args.compare)
        scores2, counts2 = aggregate(ds2)
        print_top(scores2, counts2, label=f"(compare: {args.compare.split('/')[-3] if '/' in args.compare else ''}) ")

        # Triples in bert-init top-500 not in from-scratch top-500
        top_bert = {k for k, _ in sorted(scores.items(), key=lambda x: -x[1])[:500]}
        top_base = {k for k, _ in sorted(scores2.items(), key=lambda x: -x[1])[:500]}
        novel = top_bert - top_base
        print(f"\n=== {len(novel)} triples in BERT-init top-500 not in from-scratch top-500 ===")
        novel_sorted = sorted(novel, key=lambda k: scores[k], reverse=True)
        for (head, rel, tok) in novel_sorted[:20]:
            print(f"  {head:20s} --{rel:25s}--> {tok}")


if __name__ == "__main__":
    main()
