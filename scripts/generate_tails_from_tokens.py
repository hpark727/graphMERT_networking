"""
Paper's tail generation pipeline — Steps 2–5.

  Step 2: Helper LLM (Qwen3-32B via Tinker) combines top-20 predicted tokens into
          multi-token tails using only the tokens GraphMERT predicted.
  Step 3: Hallucination filter — any tail whose BERT tokens aren't all in the top-20
          list is dropped.
  Step 4: Embedding similarity filter (β = 0.67) — cosine similarity of the triple
          "{head} {relation} {tail}" against the source sentence must exceed β.
  Step 5: Deduplication across all sentences; filter triples already in the seed KG.

Input:  One or more HuggingFace datasets saved by predict_tails.py.
        Required columns: head, relation, predictions, probabilities, input_ids, id.
Output: CSV with columns: head, relation, tail, source_sentence, similarity, sentence_id.

Usage (on Della):
    python3 scripts/generate_tails_from_tokens.py \\
        --predictions  outputs/kg_expansion_bert_init_stage2/train/top_20 \\
                       outputs/kg_expansion_bert_init_stage2/eval/top_20 \\
        --seed_kg_dir  gen4_triplets/seed_kg \\
        --tokenizer_path /path/to/bert-base-uncased \\
        --output       outputs/kg_expansion_bert_init_stage2/full_kg/generated_tails.csv

Resume an interrupted run by re-running the same command — rows already in the output
file are skipped automatically (matched on sentence_id + head + relation).
"""

import os
import sys
import csv
import glob
import asyncio
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

from datasets import load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

import tinker

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise RuntimeError("TINKER_API_KEY not set in .env")

MODEL_HELPER = "Qwen/Qwen3-32B"
TEMPERATURE  = 0.0
MAX_TOKENS   = 256
MAX_CONCURRENT = 50

_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    return _sem


# ── Seed-KG helpers ───────────────────────────────────────────────────────────

def load_seed_kg(seed_kg_dir: str) -> set[tuple[str, str, str]]:
    existing = set()
    for path in sorted(glob.glob(os.path.join(seed_kg_dir, "seed_kg_ch*.csv"))):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                existing.add((row["head"].strip(), row["relation_type"].strip(), row["tail"].strip()))
    return existing


# ── LLM call ─────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a knowledge graph assistant. "
    "Given a source sentence, a head entity, a relation, and a list of allowed tokens, "
    "generate tail entity phrases for the KG triple. "
    "CRITICAL: every word in every tail you output must be one of the allowed tokens — "
    "do not invent any word not in that list. "
    "Output one tail phrase per line, nothing else."
)


def _build_prompt(head: str, relation: str, sentence: str, tokens: list[str]) -> list[dict]:
    token_str = ", ".join(f'"{t}"' for t in tokens if not t.startswith("##"))
    user_msg = (
        f"Head: {head}\n"
        f"Relation: {relation}\n"
        f"Source sentence: {sentence}\n"
        f"Allowed tokens (use ONLY these): {token_str}\n\n"
        "Generate 1–5 tail entity phrases using only the allowed tokens. "
        "One phrase per line."
    )
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": user_msg},
    ]


async def _llm_call(
    client,
    tokenizer_llm,
    messages: list[dict],
    max_retries: int = 3,
) -> str:
    prompt_text = tokenizer_llm.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids   = tokenizer_llm.encode(prompt_text)
    model_input = tinker.types.ModelInput.from_ints(input_ids)
    sp          = tinker.types.SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    loop        = asyncio.get_event_loop()
    async with _get_sem():
        for attempt in range(1, max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: client.sample(
                            prompt=model_input, num_samples=1, sampling_params=sp
                        ).result(),
                    ),
                    timeout=90.0,
                )
                raw = tokenizer_llm.decode(resp.sequences[0].tokens, skip_special_tokens=True)
                # strip <think>…</think> block if present
                import re
                think_m = re.search(r"</think>(.*)", raw, re.DOTALL)
                return think_m.group(1).strip() if think_m else raw.strip()
            except Exception as e:
                logger.warning(f"LLM attempt {attempt}/{max_retries} failed: {e!r}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                else:
                    return ""


# ── Hallucination filter ──────────────────────────────────────────────────────

def passes_hallucination_filter(
    tail: str,
    top_k_token_set: set[str],
    bert_tokenizer,
) -> bool:
    """All BERT tokens in the tail must appear in the top-k predicted token set."""
    tokens = bert_tokenizer.tokenize(tail)
    return bool(tokens) and all(t in top_k_token_set for t in tokens)


# ── Parse LLM output ─────────────────────────────────────────────────────────

def _parse_tails(raw: str) -> list[str]:
    """Extract one tail phrase per line; skip blank/meta lines."""
    tails = []
    for line in raw.splitlines():
        line = line.strip().strip("-•*").strip()
        if not line or len(line) > 80:
            continue
        # drop lines that look like explanations (contain colon or are very long)
        if ":" in line and len(line) > 40:
            continue
        tails.append(line)
    return tails


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def generate(
    predictions: list[str],
    seed_kg_dir: str,
    tokenizer_path: str,
    output_path: Path,
    beta: float,
    embed_model_name: str,
) -> None:
    # Load seed KG
    existing = load_seed_kg(seed_kg_dir)
    logger.info(f"Seed KG: {len(existing):,} existing triples")

    # Load predict_tails datasets
    ds = concatenate_datasets([load_from_disk(p) for p in predictions])
    logger.info(f"Loaded {len(ds):,} prediction rows from {len(predictions)} dataset(s)")

    # BERT tokenizer — used to decode input_ids → source sentence and for hallucination filter
    bert_tok = AutoTokenizer.from_pretrained(tokenizer_path)

    # Embedding model for similarity filter
    logger.info(f"Loading embedding model: {embed_model_name}")
    embed_model = SentenceTransformer(embed_model_name)

    # Tinker helper LLM
    service_client  = tinker.ServiceClient(api_key=TINKER_API_KEY)
    helper_client   = service_client.create_sampling_client(base_model=MODEL_HELPER)
    helper_tok      = helper_client.get_tokenizer()

    # Resume: load already-processed (sentence_id, head, relation) combos
    done_keys: set[tuple] = set()
    if output_path.exists():
        done_df = pd.read_csv(output_path)
        for _, row in done_df.iterrows():
            done_keys.add((str(row["sentence_id"]), str(row["head"]), str(row["relation"])))
        logger.info(f"Resuming — {len(done_keys):,} rows already processed, skipping")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_file = open(output_path, "a", newline="", encoding="utf-8")
    writer   = csv.writer(out_file)
    if write_header:
        writer.writerow(["head", "relation", "tail", "source_sentence", "similarity", "sentence_id"])

    # Track seen (head, relation, tail) for deduplication across sentences
    seen_triples: set[tuple[str, str, str]] = set()
    if output_path.exists() and not write_header:
        for _, row in pd.read_csv(output_path).iterrows():
            seen_triples.add((str(row["head"]), str(row["relation"]), str(row["tail"])))

    # Filter to pending rows
    pending = [
        row for row in ds
        if (str(row["id"]), str(row["head"]), str(row["relation"])) not in done_keys
    ]
    logger.info(f"{len(pending):,} rows to process")

    if not pending:
        logger.info("Nothing to do.")
        out_file.close()
        return

    bar = tqdm(total=len(pending), desc="Generating", unit="row", dynamic_ncols=True)
    kept = 0

    async def _process(row) -> None:
        nonlocal kept

        sentence_id = str(row["id"])
        head        = str(row["head"]).strip()
        relation    = str(row["relation"]).strip()
        top_k_tokens: list[str] = str(row["predictions"]).split()
        top_k_set   = set(top_k_tokens)

        # Decode source sentence from input_ids
        source_sentence = bert_tok.decode(row["input_ids"], skip_special_tokens=True)

        # Step 2: call helper LLM
        messages = _build_prompt(head, relation, source_sentence, top_k_tokens)
        raw = await _llm_call(helper_client, helper_tok, messages)
        candidates = _parse_tails(raw)

        # Step 3: hallucination filter
        candidates = [
            t for t in candidates
            if passes_hallucination_filter(t, top_k_set, bert_tok)
        ]

        if not candidates:
            bar.update(1)
            return

        # Step 4: embedding similarity filter
        triple_texts  = [f"{head} {relation} {t}" for t in candidates]
        triple_embeds = embed_model.encode(triple_texts, convert_to_tensor=True, show_progress_bar=False)
        sent_embed    = embed_model.encode(source_sentence, convert_to_tensor=True, show_progress_bar=False)
        sims          = util.cos_sim(triple_embeds, sent_embed).squeeze(1).tolist()

        for tail, sim in zip(candidates, sims):
            if sim < beta:
                continue
            triple_key = (head, relation, tail)
            # Step 5: dedup and seed-KG filter
            if triple_key in seen_triples or triple_key in existing:
                continue
            seen_triples.add(triple_key)
            writer.writerow([head, relation, tail, source_sentence, f"{sim:.4f}", sentence_id])
            out_file.flush()
            kept += 1

        bar.update(1)
        bar.set_postfix(kept=kept)

    await asyncio.gather(*[_process(row) for row in pending])
    bar.close()
    out_file.close()
    logger.info(f"Done. {kept:,} new triples written → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--predictions", required=True, nargs="+",
        help="One or more paths to HuggingFace datasets from predict_tails.py (top_20)",
    )
    ap.add_argument(
        "--seed_kg_dir", required=True,
        help="Directory containing seed_kg_ch*.csv files",
    )
    ap.add_argument(
        "--tokenizer_path", required=True,
        help="Path to bert-base-uncased tokenizer (used to decode source sentences and hallucination filter)",
    )
    ap.add_argument(
        "--output", type=Path,
        default=_REPO / "outputs/kg_expansion_bert_init_stage2/full_kg/generated_tails.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--beta", type=float, default=0.67,
        help="Embedding similarity threshold β (default 0.67, matching the paper)",
    )
    ap.add_argument(
        "--embed_model", default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for embedding similarity filter (default all-MiniLM-L6-v2)",
    )
    ap.add_argument(
        "--max_concurrent", type=int, default=MAX_CONCURRENT,
        help=f"Max concurrent LLM requests (default {MAX_CONCURRENT})",
    )
    args = ap.parse_args()
    MAX_CONCURRENT = args.max_concurrent
    asyncio.run(generate(
        predictions=args.predictions,
        seed_kg_dir=args.seed_kg_dir,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output,
        beta=args.beta,
        embed_model_name=args.embed_model,
    ))
