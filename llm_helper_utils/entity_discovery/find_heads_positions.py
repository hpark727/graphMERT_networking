import os
import re
import argparse
import torch
from datasets import load_from_disk, concatenate_datasets
import logging
import json

from transformers import AutoTokenizer
import yaml

# ── Model config (stable across chapters) ─────────────────────────────────────
ECONF_PATH = os.path.join(os.path.dirname(__file__), 'entity_discovery_args.yaml')
if not os.path.exists(ECONF_PATH):
    raise FileNotFoundError(f"Missing config: {ECONF_PATH}")
with open(ECONF_PATH, 'r', encoding='utf-8') as _f:
    _cfg = yaml.safe_load(_f)

tokenizer = AutoTokenizer.from_pretrained(_cfg['model']['tokenizer_path'])
MODEL_NAME = _cfg['model']['model_name']

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ── Normalization helpers ──────────────────────────────────────────────────────

_DASH_CHARS = '‐‑‒–—―−﹘﹣－'
_DASH_RE = re.compile(f'[{re.escape(_DASH_CHARS)}]')


def _normalize(s: str) -> str:
    """Lowercase, replace Unicode dashes with ASCII hyphen, collapse whitespace."""
    s = _DASH_RE.sub('-', s.lower())
    return re.sub(r'\s+', ' ', s).strip()


def _candidate_variants(head: str) -> list[str]:
    """
    Return a list of tokenization-friendly variants to try in order:
      0. raw lowercased (preserves original Unicode dashes — matches if text also has them)
      1. normalized as-is (Unicode dashes → ASCII hyphen)
      2. parenthetical abbreviation stripped  e.g. "fiber to the home ( ftth )" → "fiber to the home"
      3. spaces added around hyphens
      4. spaces removed around hyphens
    Combinations of 2+3 and 2+4 are also included.
    """
    # Candidate 0: preserve original Unicode dashes; only lowercase + collapse whitespace
    raw_lower = re.sub(r'\s+', ' ', head.lower()).strip()

    base = _normalize(head)
    # Strip trailing parenthetical: "foo ( bar )" or "foo (bar)"
    stripped = re.sub(r'\s*[\(\[].*?[\)\]]\s*$', '', base).strip()

    seeds = [base]
    if stripped and stripped != base:
        seeds.append(stripped)

    variants: list[str] = []
    seen: set[str] = set()

    # Try raw lowercased first so original dash tokenization is preserved
    if raw_lower and raw_lower not in seen:
        seen.add(raw_lower)
        variants.append(raw_lower)

    for s in seeds:
        for v in (
            s,
            re.sub(r'\s*-\s*', ' - ', s),   # add spaces around hyphens
            re.sub(r'\s*-\s*', '-', s),      # remove spaces around hyphens
        ):
            v = re.sub(r'\s+', ' ', v).strip()
            if v and v not in seen:
                seen.add(v)
                variants.append(v)

    # Plural/singular fallback: try appending 's' (entity is singular, text may be plural)
    # and stripping trailing 's' (entity is plural, text may be singular).
    # Operate on the base normalized form only.
    extra: list[str] = []
    if base and not base.endswith('s'):
        extra.append(base + 's')
    elif base.endswith('s') and len(base) > 4:
        extra.append(base[:-1])
    for v in extra:
        if v not in seen:
            seen.add(v)
            variants.append(v)

    return variants


def _search(input_ids: list, head_token_ids: list) -> int:
    """Return start index of head_token_ids in input_ids, or -1."""
    n, m = len(input_ids), len(head_token_ids)
    for i in range(n - m + 1):
        if input_ids[i:i + m] == head_token_ids:
            return i
    return -1


def find_head_positions(example, idx):
    input_ids = list(example["input_ids"])
    response_list = example["response"]

    head_positions = {}

    for head in response_list:
        match_index = -1
        matched_variant_ids = None

        for variant in _candidate_variants(head):
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue
            pos = _search(input_ids, token_ids)
            if pos != -1:
                match_index = pos
                matched_variant_ids = token_ids
                break

        if match_index == -1:
            logger.info(f"{idx}: head '{_normalize(head)}' not found.")
        else:
            span = input_ids[match_index: match_index + len(matched_variant_ids)]
            matched_text = tokenizer.decode(span, skip_special_tokens=True).strip()
            head_positions[matched_text] = match_index

    example["head_positions"] = json.dumps(head_positions)
    return example


# ── Dataset loading ────────────────────────────────────────────────────────────

def unite_output(dataset_path: str, model_name: str, subset_size=None) -> object:
    """Scan dataset_path for chunk datasets and concatenate them."""
    if subset_size is not None:
        prefix = f"{model_name}_subset_{subset_size}_"
    else:
        prefix = f"{model_name}_"

    entries = sorted(os.listdir(dataset_path))
    chunk_dirs = [
        e for e in entries
        if e.startswith(prefix) and not e.endswith('_all')
    ]

    datasets = []
    for d in chunk_dirs:
        path = os.path.join(dataset_path, d)
        if os.path.isdir(path):
            ds = load_from_disk(path)
            logger.info(f'loaded from {path}')
            datasets.append(ds)

    assert datasets, f"No chunk datasets found in {dataset_path!r} with prefix {prefix!r}"
    return concatenate_datasets(datasets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter', type=int, default=1, help='Chapter number (1-8)')
    parser.add_argument('--subset', type=int, default=None, help='Subset size for testing')
    args = parser.parse_args()

    ch = args.chapter
    subset_size = args.subset

    # ch1 uses legacy flat paths; ch2+ use per-chapter subdirectories
    if ch == 1:
        dataset_path = 'json_data/entity_discovery_output'
        output_path = f'json_data/entity_discovery_output_{MODEL_NAME}_all'
    else:
        dataset_path = f'json_data/entity_discovery_output/ch{ch}'
        output_path = f'json_data/entity_discovery_output/ch{ch}_{MODEL_NAME}_all'

    dataset_heads = unite_output(dataset_path, MODEL_NAME, subset_size)
    logger.info(f"Whole dataset:\n{dataset_heads}")

    if subset_size:
        dataset_heads = dataset_heads.select(range(min(subset_size, len(dataset_heads))))

    dataset_heads_with_positions = dataset_heads.map(
        find_head_positions, num_proc=4, with_indices=True
    )
    dataset_heads_with_positions = dataset_heads_with_positions.remove_columns(["response"])

    dataset_heads_with_positions = dataset_heads_with_positions.map(
        lambda example, idx: {"id": idx}, with_indices=True,
        desc='Indexing dataset',
        num_proc=4,
    )

    path_to_save = f'{output_path}_subset_{subset_size}' if subset_size else output_path
    dataset_heads_with_positions.save_to_disk(path_to_save)
    print(f'saved to {path_to_save}')
    print(dataset_heads_with_positions)
