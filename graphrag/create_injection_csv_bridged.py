"""
Create bridged injection CSVs using vocab_bridge.json to map unmatched
seed KG heads to their discovered-entity equivalents.

Reads:
  gen4_triplets/filtered_and_validated/validated_ch{N}.csv
  gen4_triplets/vocab_bridge.json          {seed_head: discovered_entity}
  json_data/.../head_positions datasets

Outputs:
  gen4_triplets/injections_bridged/injections_ch{N}.csv

The bridge extends coverage: when a chunk contains a discovered entity
that is mapped to a seed head, that seed head's triples are injected as
if the seed head itself had appeared. The injected 'head' column uses the
discovered entity text (what actually appears in the chunk).

Run:
  python3 graphrag/create_injection_csv_bridged.py --chapter 1
  python3 graphrag/create_injection_csv_bridged.py --all
"""

import argparse, csv, json, logging, sys
from pathlib import Path
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]


def main(chapter: int) -> None:
    seed_csv    = _REPO / f"gen4_triplets/filtered_and_validated/validated_ch{chapter}.csv"
    bridge_path = _REPO / "gen4_triplets/vocab_bridge.json"
    out_dir     = _REPO / "gen4_triplets/injections_bridged"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv     = out_dir / f"injections_ch{chapter}.csv"

    if chapter == 1:
        heads_path = _REPO / "json_data/entity_discovery_output_gpt-oss-120b_all"
    else:
        heads_path = _REPO / f"json_data/entity_discovery_output/ch{chapter}_gpt-oss-120b_all"

    # Load seed KG
    kg_df: dict[str, list[tuple[str, str]]] = {}
    with open(seed_csv) as f:
        for row in csv.DictReader(f):
            h = row["head"].strip().lower()
            kg_df.setdefault(h, []).append(
                (row["relation_type"].strip(), row["tail"].strip())
            )
    logger.info(f"Ch{chapter}: seed KG {sum(len(v) for v in kg_df.values())} triples, "
                f"{len(kg_df)} unique heads")

    # Load vocab bridge: {discovered_entity → seed_head}
    # (bridge file is {seed_head → discovered_entity}, invert it)
    bridged_lookup: dict[str, str] = {}  # discovered_entity → seed_head
    if bridge_path.exists():
        with open(bridge_path) as f:
            bridge = json.load(f)
        for seed_head, discovered in bridge.items():
            bridged_lookup[discovered.strip().lower()] = seed_head.strip().lower()
        logger.info(f"Loaded {len(bridged_lookup)} vocab bridges")
    else:
        logger.warning(f"No vocab_bridge.json found at {bridge_path} — running without bridge")

    # Combined lookup: direct match OR bridged match → triples
    def get_triples(entity_lower: str) -> tuple[list[tuple[str, str]], bool]:
        """Returns (triples, was_bridged)."""
        if entity_lower in kg_df:
            return kg_df[entity_lower], False
        seed_head = bridged_lookup.get(entity_lower)
        if seed_head and seed_head in kg_df:
            return kg_df[seed_head], True
        return [], False

    # Load head_positions dataset
    ds = load_from_disk(str(heads_path))
    logger.info(f"Dataset: {len(ds)} chunks")

    direct_rows = bridged_rows = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "head", "relation_type", "tail"])
        for example in ds:
            chunk_id = int(example["id"])
            head_positions = json.loads(example["head_positions"])
            for entity_text in head_positions:
                triples, bridged = get_triples(entity_text.strip().lower())
                for rel, tail in triples:
                    writer.writerow([chunk_id, entity_text, rel, tail])
                    if bridged:
                        bridged_rows += 1
                    else:
                        direct_rows += 1

    logger.info(f"Ch{chapter}: {direct_rows} direct + {bridged_rows} bridged "
                f"= {direct_rows + bridged_rows} total rows → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--chapter", type=int)
    grp.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for ch in (range(1, 9) if args.all else [args.chapter]):
        main(ch)
