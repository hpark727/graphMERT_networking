"""
Build gen4_expanded injection files by augmenting gen4 injection files with
the newly validated triples from the expanded KG.

Each new triple is injected into every sentence where its head entity appears
(matching the gen4 pipeline behaviour), not just the sentence it was derived from.

Input:
  outputs/kg_expansion_bert_init_stage2/full_kg/validated_triples.csv
  json_data/entity_discovery_output_gpt-oss-120b_all        (ch1 heads)
  json_data/entity_discovery_output/ch{2-8}_gpt-oss-120b_all (ch2-8 heads)
  gen4_triplets/injections/injections_ch{1-8}.csv

Output:
  gen4_expanded_triplets/injections/injections_ch{1-8}.csv

Run from repo root:
    python3 graphrag/build_gen5_injections.py
"""

import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]

VALIDATED_PATH        = _REPO / "outputs/kg_expansion_bert_init_stage2/full_kg/validated_triples.csv"
GEN4_INJECTIONS_DIR   = _REPO / "gen4_triplets/injections"
OUTPUT_INJECTIONS_DIR = _REPO / "gen4_expanded_triplets/injections"

CHAPTERS = list(range(1, 9))


def heads_dataset_path(ch: int) -> Path:
    if ch == 1:
        return _REPO / "json_data/entity_discovery_output_gpt-oss-120b_all"
    return _REPO / f"json_data/entity_discovery_output/ch{ch}_gpt-oss-120b_all"


def build_head_index(ch: int) -> dict[str, list[int]]:
    """Return {head_entity: [raw_sentence_id, ...]} for a chapter."""
    ds = load_from_disk(str(heads_dataset_path(ch)))
    index: dict[str, list[int]] = defaultdict(list)
    for row in ds:
        positions = json.loads(row["head_positions"])
        for head in positions:
            index[head].append(row["id"])
    return index


def main() -> None:
    logger.info(f"Loading validated triples from {VALIDATED_PATH}")
    val_df = pd.read_csv(VALIDATED_PATH)
    before = len(val_df)
    val_df = val_df[(val_df["verdict_qwen"] == "yes") & (val_df["verdict_gpt"] == "yes")]
    logger.info(f"  {len(val_df):,} both-yes triples (filtered from {before:,})")
    val_df = val_df[["head", "relation", "tail"]]

    OUTPUT_INJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    total_new_rows = 0
    for ch in CHAPTERS:
        logger.info(f"Building head index for ch{ch} …")
        head_index = build_head_index(ch)

        gen4_path   = GEN4_INJECTIONS_DIR   / f"injections_ch{ch}.csv"
        output_path = OUTPUT_INJECTIONS_DIR / f"injections_ch{ch}.csv"

        # Load existing gen4 injections; track (id, head, rel, tail) to avoid duplicates
        existing_rows: list[dict] = []
        existing_keys: set[tuple] = set()
        with open(gen4_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing_rows.append(row)
                existing_keys.add((row["id"], row["head"], row["relation_type"], row["tail"]))

        # Fan out each validated triple across all sentences where its head appears
        new_rows: list[dict] = []
        for _, triple in val_df.iterrows():
            head     = triple["head"]
            relation = triple["relation"]
            tail     = triple["tail"]
            sentence_ids = head_index.get(head, [])
            for sid in sentence_ids:
                key = (str(sid), head, relation, tail)
                if key not in existing_keys:
                    new_rows.append({"id": sid, "head": head, "relation_type": relation, "tail": tail})
                    existing_keys.add(key)

        total_new_rows += len(new_rows)
        logger.info(
            f"ch{ch}: {len(existing_rows):,} gen4 + {len(new_rows):,} new rows "
            f"= {len(existing_rows) + len(new_rows):,} total"
        )

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "head", "relation_type", "tail"])
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(new_rows)

    logger.info(f"Done. {total_new_rows:,} new injection rows added → {OUTPUT_INJECTIONS_DIR}")


if __name__ == "__main__":
    main()
