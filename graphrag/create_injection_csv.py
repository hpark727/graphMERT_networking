"""
Create the injection CSV for GraphMERT training from the validated gen4 KG.

Reads:
  - gen4_triplets/validated_both/validated_ch{N}.csv  (head, relation_type, tail)
  - head_positions dataset for chapter N

Outputs:
  - gen4_triplets/validated_both/injections_ch{N}.csv  (id, head, relation_type, tail)

The id column matches the chunk 'id' field in the head_positions dataset.
Only triples whose head appears in a chunk's head_positions dict are emitted.
The same triple may appear for multiple chunk IDs if the head is found in
multiple chunks.

Run from repo root:
    python3 graphrag/create_injection_csv.py --chapter 1
    python3 graphrag/create_injection_csv.py --all
"""

import argparse
import csv
import json
import logging
import sys
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


def main(chapter: int) -> None:
    seed_csv = _REPO / f"gen4_triplets/validated_both/validated_ch{chapter}.csv"
    if not seed_csv.exists():
        raise FileNotFoundError(f"Validated KG not found: {seed_csv}")

    if chapter == 1:
        heads_path = _REPO / "json_data/entity_discovery_output_gpt-oss-120b_all"
    else:
        heads_path = _REPO / f"json_data/entity_discovery_output/ch{chapter}_gpt-oss-120b_all"

    out_dir = _REPO / "gen4_triplets/validated_both"
    out_csv = out_dir / f"injections_ch{chapter}.csv"

    # Load seed KG — index by head entity
    logger.info(f"Loading seed KG from {seed_csv}")
    kg_df = pd.read_csv(seed_csv)
    # Group by head: head → list of (relation_type, tail)
    head_to_triples: dict[str, list[tuple[str, str]]] = {}
    for _, row in kg_df.iterrows():
        h = str(row["head"]).strip().lower()
        head_to_triples.setdefault(h, []).append(
            (str(row["relation_type"]).strip(), str(row["tail"]).strip())
        )
    logger.info(f"Seed KG: {len(kg_df)} triples, {len(head_to_triples)} unique heads")

    # Load head_positions dataset
    logger.info(f"Loading head_positions dataset from {heads_path}")
    ds = load_from_disk(str(heads_path))
    logger.info(f"Dataset: {ds}")

    rows_written = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "head", "relation_type", "tail"])
        for example in ds:
            chunk_id = int(example["id"])
            head_positions = json.loads(example["head_positions"])
            # head_positions: {entity_text: token_position}
            for entity_text in head_positions:
                entity_lower = entity_text.strip().lower()
                if entity_lower in head_to_triples:
                    for rel, tail in head_to_triples[entity_lower]:
                        writer.writerow([chunk_id, entity_text, rel, tail])
                        rows_written += 1

    logger.info(f"Wrote {rows_written} injection rows → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--chapter", type=int, help="Single chapter (1-8)")
    grp.add_argument("--all", action="store_true", help="All chapters 1-8")
    args = ap.parse_args()
    chapters = list(range(1, 9)) if args.all else [args.chapter]
    for ch in chapters:
        main(ch)
