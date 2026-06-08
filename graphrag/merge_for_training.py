"""
Merge all 8 chapters' head-position datasets and injection CSVs into
train/eval splits ready for run_dataset_preprocessing.py.

Chapters 1-7 → train, Chapter 8 → eval.

Outputs (written to --output_dir, default gen4_triplets/training_data/):
  - heads_train/   (concatenated HF dataset)
  - heads_eval/    (ch8 dataset)
  - injections_train.csv
  - injections_eval.csv

Run from repo root:
    # gen4 (default)
    python3 graphrag/merge_for_training.py

    # gen5 (expanded KG)
    python3 graphrag/merge_for_training.py \
        --injections_dir gen5_triplets/injections \
        --output_dir     gen5_triplets/training_data
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

from datasets import load_from_disk, concatenate_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]

TRAIN_CHAPTERS = list(range(1, 8))
EVAL_CHAPTERS  = [8]


def heads_path(ch: int) -> Path:
    if ch == 1:
        return _REPO / "json_data/entity_discovery_output_gpt-oss-120b_all"
    return _REPO / f"json_data/entity_discovery_output/ch{ch}_gpt-oss-120b_all"


def merge_datasets(chapters: list[int]) -> object:
    datasets = []
    for ch in chapters:
        p = heads_path(ch)
        logger.info(f"Loading ch{ch} heads from {p}")
        ds = load_from_disk(str(p))
        offset = (ch - 1) * 100_000
        ds = ds.map(lambda ex, idx: {"id": ex["id"] + offset}, with_indices=True)
        datasets.append(ds)
        logger.info(f"  ch{ch}: {len(ds)} rows")
    combined = concatenate_datasets(datasets)
    logger.info(f"Combined: {len(combined)} rows")
    return combined


def merge_injections(
    chapters: list[int],
    out_csv: Path,
    id_offsets: dict[int, int],
    injections_dir: Path,
) -> None:
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "head", "relation_type", "tail"])
        for ch in chapters:
            offset = id_offsets[ch]
            src = injections_dir / f"injections_ch{ch}.csv"
            logger.info(f"Reading injections ch{ch} from {src}")
            with open(src, newline="", encoding="utf-8") as g:
                reader = csv.DictReader(g)
                for row in reader:
                    writer.writerow([int(row["id"]) + offset, row["head"], row["relation_type"], row["tail"]])
    logger.info(f"Saved {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--injections_dir", type=Path,
        default=_REPO / "gen4_triplets/injections",
        help="Directory containing injections_ch{1-8}.csv (default: gen4_triplets/injections)",
    )
    ap.add_argument(
        "--output_dir", type=Path,
        default=_REPO / "gen4_triplets/training_data",
        help="Output directory for merged training data (default: gen4_triplets/training_data)",
    )
    args = ap.parse_args()

    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_offsets = {ch: (ch - 1) * 100_000 for ch in TRAIN_CHAPTERS}
    eval_offsets  = {ch: (ch - 1) * 100_000 for ch in EVAL_CHAPTERS}

    logger.info("=== Merging TRAIN datasets ===")
    train_ds = merge_datasets(TRAIN_CHAPTERS)
    train_out = OUT_DIR / "heads_train"
    train_ds.save_to_disk(str(train_out))
    logger.info(f"Saved train heads → {train_out} ({len(train_ds)} rows)")

    logger.info("=== Merging EVAL datasets ===")
    eval_ds = merge_datasets(EVAL_CHAPTERS)
    eval_out = OUT_DIR / "heads_eval"
    eval_ds.save_to_disk(str(eval_out))
    logger.info(f"Saved eval heads → {eval_out} ({len(eval_ds)} rows)")

    logger.info("=== Merging TRAIN injections ===")
    merge_injections(TRAIN_CHAPTERS, OUT_DIR / "injections_train.csv", train_offsets, args.injections_dir)

    logger.info("=== Merging EVAL injections ===")
    merge_injections(EVAL_CHAPTERS, OUT_DIR / "injections_eval.csv", eval_offsets, args.injections_dir)

    logger.info("=== Done ===")
    logger.info(f"Output dir: {OUT_DIR}")
