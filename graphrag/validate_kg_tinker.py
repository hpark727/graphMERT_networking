"""
Dual-model validation of seed KG triples using Tinker.

Calls two LLM models concurrently for every triple.
A triple is kept only when BOTH models return [yes].

Models:
  - Qwen/Qwen3.6-35B-A3B
  - openai/gpt-oss-120b

Input:  gen1_triplets/seed_kg/seed_kg_ch{N}.csv
Output: gen1_triplets/validated/validated_ch{N}.csv  (head, relation_type, tail)

Run from repo root:
    python3 graphrag/validate_kg_tinker.py --chapter 1
    python3 graphrag/validate_kg_tinker.py --all
"""

import os
import sys
import re
import csv
import asyncio
import logging
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import tinker

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from llm_evaluation_scores.prompts_scores import system_prompt_validity_networking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise RuntimeError("TINKER_API_KEY not set in .env")

MODEL_QWEN = "Qwen/Qwen3.6-35B-A3B"
MODEL_GPT  = "openai/gpt-oss-120b"

TEMPERATURE = 0.0
MAX_TOKENS  = 512
MAX_CONCURRENT_PER_MODEL = 10

_sem_qwen: asyncio.Semaphore | None = None
_sem_gpt:  asyncio.Semaphore | None = None

def _sem(model_key: str) -> asyncio.Semaphore:
    global _sem_qwen, _sem_gpt
    if model_key == "qwen":
        if _sem_qwen is None:
            _sem_qwen = asyncio.Semaphore(MAX_CONCURRENT_PER_MODEL)
        return _sem_qwen
    else:
        if _sem_gpt is None:
            _sem_gpt = asyncio.Semaphore(MAX_CONCURRENT_PER_MODEL)
        return _sem_gpt


def _build_messages(head: str, relation: str, tail: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt_validity_networking},
        {"role": "user",   "content": f"Triple: {head} | {relation} | {tail}\nOutput:"},
    ]


async def _model_call(
    sampling_client,
    tokenizer,
    model_key: str,
    messages: list[dict],
    max_retries: int = 3,
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer.encode(prompt_text)
    model_input = tinker.types.ModelInput.from_ints(input_ids)
    sp = tinker.types.SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    loop = asyncio.get_event_loop()
    async with _sem(model_key):
        for attempt in range(1, max_retries + 1):
            try:
                resp = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: sampling_client.sample(
                            prompt=model_input, num_samples=1, sampling_params=sp
                        ).result(),
                    ),
                    timeout=90.0,
                )
                return tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"[{model_key}] attempt {attempt}/{max_retries} failed: {e!r}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                else:
                    return ""


def _extract_verdict(raw: str) -> str:
    """Return 'yes', 'no', 'maybe', or '' if unparseable."""
    think_match = re.search(r"</think>(.*)", raw, re.DOTALL)
    text = think_match.group(1).strip() if think_match else raw.strip()
    start, end = text.rfind("["), text.rfind("]")
    if start == -1 or end == -1 or start >= end:
        return ""
    candidate = text[start + 1 : end].strip().lower()
    return candidate if candidate in ("yes", "no", "maybe") else ""


async def _validate_triple(
    qwen_client, qwen_tok, gpt_client, gpt_tok,
    head: str, relation: str, tail: str,
) -> tuple[str, str]:
    """Returns (qwen_verdict, gpt_verdict)."""
    messages = _build_messages(head, relation, tail)
    qwen_raw, gpt_raw = await asyncio.gather(
        _model_call(qwen_client, qwen_tok, "qwen", messages),
        _model_call(gpt_client,  gpt_tok,  "gpt",  messages),
    )
    return _extract_verdict(qwen_raw), _extract_verdict(gpt_raw)


def _passes(qv: str, gv: str, threshold: str) -> bool:
    if threshold == "both":   return qv == "yes" and gv == "yes"
    if threshold == "either": return qv == "yes" or gv == "yes"
    if threshold == "soft":   return "yes" in (qv, gv) and "no" not in (qv, gv)
    return False


async def validate_chapter(chapter: int, seed_dir: Path, out_dir: Path,
                            threshold: str = "both") -> None:
    seed_csv = seed_dir / f"seed_kg_ch{chapter}.csv"
    if not seed_csv.exists():
        raise FileNotFoundError(f"Seed KG not found: {seed_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"validated_ch{chapter}.csv"

    df = pd.read_csv(seed_csv)
    logger.info(f"Ch{chapter}: {len(df)} triples to validate")

    service_client = tinker.ServiceClient(api_key=TINKER_API_KEY)
    qwen_client = service_client.create_sampling_client(base_model=MODEL_QWEN)
    gpt_client  = service_client.create_sampling_client(base_model=MODEL_GPT)
    qwen_tok = qwen_client.get_tokenizer()
    gpt_tok  = gpt_client.get_tokenizer()

    triples = list(df.itertuples(index=False))
    bar = tqdm(total=len(triples), desc=f"Ch{chapter}", unit="triple", dynamic_ncols=True)
    kept = 0

    async def _run(row) -> tuple[str, str]:
        nonlocal kept
        result = await _validate_triple(
            qwen_client, qwen_tok, gpt_client, gpt_tok,
            str(row.head).strip(), str(row.relation_type).strip(), str(row.tail).strip(),
        )
        qv, gv = result
        if _passes(qv, gv, threshold):
            kept += 1
        bar.update(1)
        bar.set_postfix(kept=kept)
        return result

    verdicts: list[tuple[str, str]] = await asyncio.gather(*[_run(row) for row in triples])
    bar.close()

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["head", "relation_type", "tail", "verdict_qwen", "verdict_gpt"])
        for row, (qv, gv) in zip(triples, verdicts):
            if _passes(qv, gv, threshold):
                writer.writerow([row.head, row.relation_type, row.tail, qv, gv])

    logger.info(
        f"Ch{chapter}: {kept}/{len(triples)} triples passed dual validation → {out_csv}"
    )


async def main(chapters: list[int], seed_dir: Path, out_dir: Path,
               threshold: str) -> None:
    for ch in chapters:
        await validate_chapter(ch, seed_dir, out_dir, threshold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--chapter", type=int, help="Single chapter (1-8)")
    grp.add_argument("--chapters", type=int, nargs="+", metavar="N", help="One or more chapter numbers")
    grp.add_argument("--all", action="store_true", help="All chapters 1-8")
    ap.add_argument(
        "--seed-dir", type=Path, default=_REPO / "gen1_triplets/seed_kg",
        help="Directory containing seed_kg_ch{N}.csv files (default: gen1_triplets/seed_kg)",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=_REPO / "gen1_triplets/validated",
        help="Output directory for validated_ch{N}.csv files (default: gen1_triplets/validated)",
    )
    ap.add_argument(
        "--threshold", choices=["both", "either", "soft"], default="either",
        help="both=both yes (default); either=at least one yes; soft=one yes and no hard no",
    )
    args = ap.parse_args()

    if args.all:
        chapters = list(range(1, 9))
    elif args.chapters:
        chapters = args.chapters
    else:
        chapters = [args.chapter]
    asyncio.run(main(chapters, args.seed_dir, args.out_dir, args.threshold))
