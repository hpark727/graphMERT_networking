"""
Seed KG extraction for computer networking using Tinker.

Reads the chapter .txt file, splits into overlapping word windows, and uses Tinker
to extract (head, relation_type, tail) triples from each window.

Output: <out_dir>/seed_kg_ch{N}.csv
Columns: head, relation_type, tail

Run from repo root:
    # gen1 (14-relation schema, default config)
    python3 graphrag/extract_kg_tinker.py --chapter 1

    # gen2 (30-relation schema)
    python3 graphrag/extract_kg_tinker.py --chapter 1 \
        --config graphrag/extraction_config_networking_v2.yaml \
        --out-dir gen2_triplets/seed_kg
"""

import os
import sys
import re
import csv
import asyncio
import logging
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import tinker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise RuntimeError("TINKER_API_KEY not set in .env")

MODEL          = "openai/gpt-oss-120b"
MAX_CONCURRENT = 20

_semaphore: asyncio.Semaphore | None = None

def _get_sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


# ── Text windowing ─────────────────────────────────────────────────────────────

def make_windows(text: str, window_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    windows = []
    step = window_words - overlap_words
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + window_words])
        if len(chunk.split()) >= 20:
            windows.append(chunk)
    return windows


# ── Tinker inference ──────────────────────────────────────────────────────────

async def tinker_call(
    sampling_client, tokenizer, system_prompt: str, few_shots: list[dict],
    user_query: str, temperature: float, max_tokens: int, max_retries: int = 3,
) -> str:
    messages = (
        [{"role": "system", "content": system_prompt}]
        + few_shots
        + [{"role": "user", "content": user_query}]
    )
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids   = tokenizer.encode(prompt_text)
    model_input = tinker.types.ModelInput.from_ints(input_ids)
    sp          = tinker.types.SamplingParams(temperature=temperature, max_tokens=max_tokens)
    loop        = asyncio.get_event_loop()
    async with _get_sem():
        for attempt in range(1, max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: sampling_client.sample(
                            prompt=model_input, num_samples=1, sampling_params=sp,
                        ).result(),
                    ),
                    timeout=120.0,
                )
                return tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Tinker attempt {attempt}/{max_retries} failed: {e!r}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                else:
                    logger.error("All retries exhausted.")
                    return ""


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_response(raw: str, valid_relations: set[str],
                   tuple_delim: str, record_delim: str) -> list[tuple[str, str, str]]:
    think_match = re.search(r"</think>(.*)", raw, re.DOTALL)
    text = think_match.group(1).strip() if think_match else raw.strip()

    triples: list[tuple[str, str, str]] = []
    for record in text.split(record_delim):
        record = re.sub(r"^\(|\)$", "", record.strip())
        parts  = [p.strip().strip('"').strip("'") for p in record.split(tuple_delim)]
        if len(parts) >= 4 and parts[0].lower() == "relationship":
            source   = parts[1].lower().strip()
            target   = parts[2].lower().strip()
            relation = parts[3].lower().strip()
            if relation in valid_relations and source and target and source != target:
                triples.append((source, relation, target))
    return triples


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(chapter: int, cfg: dict, out_dir: Path,
               window_words: int, overlap_words: int) -> None:
    txt_path = _REPO / f"json_data/ch{chapter}_text_only_no_equations.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"Chapter text not found: {txt_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"seed_kg_ch{chapter}.csv"

    # ── Build prompt pieces from config ───────────────────────────────────────
    td = cfg["tuple_delimiter"]
    rd = cfg["record_delimiter"]
    cd = cfg["completion_delimiter"]

    system_prompt = cfg["prompt_template"].format(
        completion_delimiter=cd,
        tuple_delimiter=td,
        record_delimiter=rd,
        entity_types=", ".join(cfg["entity_types"]),
        entity_types_examples=cfg["entity_types_examples"],
        relation_types=", ".join(cfg["relation_types"]),
        relation_types_examples=cfg["relation_types_examples"],
    )

    few_shots: list[dict] = [
        {"role": "user",      "content": "I will provide example inputs and outputs."},
        {"role": "assistant", "content": "Understood — I will extract entities and relationships in the specified format."},
    ]
    for ex in cfg.get("examples", []):
        few_shots.append({"role": "user", "content": ex["user"]})
        few_shots.append({
            "role": "assistant",
            "content": ex["assistant"].format(
                completion_delimiter=cd, tuple_delimiter=td, record_delimiter=rd,
            ),
        })
    few_shots.append({"role": "user", "content": "**End of examples**.\nNow process the actual input:"})

    valid_relations = set(cfg["relation_types"])
    temperature     = cfg["llm_config"]["temperature"]
    max_tokens      = cfg["llm_config"]["max_tokens"]
    user_tmpl       = cfg["user_prompt"]

    # ── Tinker client ─────────────────────────────────────────────────────────
    service_client  = tinker.ServiceClient(api_key=TINKER_API_KEY)
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer       = sampling_client.get_tokenizer()

    # ── Extract ───────────────────────────────────────────────────────────────
    text    = txt_path.read_text(encoding="utf-8")
    windows = make_windows(text, window_words, overlap_words)
    logger.info(f"Chapter {chapter}: {len(windows)} windows | config relation types: {len(valid_relations)}")

    queries = [user_tmpl.format(input_text=w) for w in windows]
    tasks   = [
        asyncio.ensure_future(
            tinker_call(sampling_client, tokenizer, system_prompt, few_shots,
                        q, temperature, max_tokens)
        )
        for q in queries
    ]

    all_triples: set[tuple[str, str, str]] = set()
    bar = tqdm(asyncio.as_completed(tasks), total=len(tasks),
               desc=f"Ch{chapter}", unit="win", dynamic_ncols=True)
    for coro in bar:
        raw     = await coro
        triples = parse_response(raw, valid_relations, td, rd)
        all_triples.update(triples)
        bar.set_postfix(unique_triples=len(all_triples))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head", "relation_type", "tail"])
        for head, rel, tail in sorted(all_triples):
            writer.writerow([head, rel, tail])

    logger.info(f"Saved {len(all_triples)} unique triples → {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", type=int, default=1, help="Chapter number (1-8)")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "extraction_config_networking.yaml",
        help="Path to extraction config YAML",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "json_data/seed_kg",
        help="Output directory for seed_kg_ch{N}.csv (default: json_data/seed_kg)",
    )
    ap.add_argument("--window-words",  type=int, default=500,
                    help="Words per extraction window (default 500)")
    ap.add_argument("--overlap-words", type=int, default=50,
                    help="Overlap words between windows (default 50)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as _f:
        cfg = yaml.safe_load(_f)

    asyncio.run(main(
        chapter      = args.chapter,
        cfg          = cfg,
        out_dir      = args.out_dir,
        window_words = args.window_words,
        overlap_words= args.overlap_words,
    ))
