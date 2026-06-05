"""
Build the dataset required by predict_tails.py from the existing training data.

predict_tails.py needs columns:
  - input_ids       : tokenized sentence (already in heads_train)
  - attention_mask  : 1 for all tokens (no padding in this dataset)
  - head_positions  : JSON {head: token_pos} (already in heads_train)
  - cleaned_response: JSON {head: [relation_type, ...]} (built from injections CSV)
  - id              : sentence index (already in heads_train)

Heads with no entries in the injection CSV are silently dropped from cleaned_response
(predict_tails.py skips examples where cleaned_response is empty anyway).
"""

import argparse
import json
import logging
from collections import defaultdict

import pandas as pd
from datasets import load_from_disk, concatenate_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_head_relations(injections_csv_path: str) -> dict[int, dict[str, list[str]]]:
    """Return {sentence_id: {head: [relation_type, ...]}} from injections CSV."""
    df = pd.read_csv(injections_csv_path)
    mapping: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        sid = int(row["id"])
        head = str(row["head"]).lower().strip()
        rel = str(row["relation_type"]).strip()
        if rel not in mapping[sid][head]:
            mapping[sid][head].append(rel)
    return {sid: dict(heads) for sid, heads in mapping.items()}


def add_cleaned_response(example, head_relations):
    sid = example["id"]
    head_positions = json.loads(example["head_positions"])

    # Only keep heads that both appear in head_positions and have known relations.
    cleaned = {}
    for head, rels in head_relations.get(sid, {}).items():
        if head in head_positions:
            cleaned[head] = rels

    example["cleaned_response"] = json.dumps(cleaned)
    return example


def add_attention_mask(example):
    # Sequences are all 128 tokens with no padding tokens.
    example["attention_mask"] = [1] * len(example["input_ids"])
    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads_dataset", required=True,
                        help="Path to heads_train or heads_eval dataset (HF arrow format)")
    parser.add_argument("--injections_csv", required=True,
                        help="Path to injections_train.csv or injections_eval.csv")
    parser.add_argument("--output_path", required=True,
                        help="Where to save the prepared dataset")
    args = parser.parse_args()

    logger.info("Loading heads dataset from %s", args.heads_dataset)
    ds = load_from_disk(args.heads_dataset)
    logger.info("Loaded %d examples", len(ds))

    logger.info("Building head→relations map from %s", args.injections_csv)
    head_relations = build_head_relations(args.injections_csv)
    logger.info("Found entries for %d sentence ids", len(head_relations))

    logger.info("Adding attention_mask ...")
    ds = ds.map(add_attention_mask, desc="Adding attention_mask")

    logger.info("Adding cleaned_response ...")
    ds = ds.map(
        add_cleaned_response,
        fn_kwargs={"head_relations": head_relations},
        desc="Adding cleaned_response",
    )

    # Drop sentences where no (head, relation) pair is known — predict_tails
    # would skip them anyway, but filtering now keeps the dataset clean.
    before = len(ds)
    ds = ds.filter(lambda ex: ex["cleaned_response"] != "{}", desc="Filtering empty")
    logger.info("Kept %d / %d examples with at least one (head, relation) pair", len(ds), before)

    ds.save_to_disk(args.output_path)
    logger.info("Saved to %s", args.output_path)
    print(ds)


if __name__ == "__main__":
    main()
