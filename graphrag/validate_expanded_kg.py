"""
Dual-model validation of GraphMERT-expanded KG triples using Tinker.

Validates triples from expand_kg_from_predictions.py output.
A triple is kept only when BOTH models return [yes].

Models:
  - Qwen/Qwen3.6-35B-A3B
  - openai/gpt-oss-120b

Input:  outputs/kg_expansion_bert_init_stage2/full_kg/candidate_triples.csv
Output: outputs/kg_expansion_bert_init_stage2/full_kg/validated_triples.csv

Run from repo root:
    python3 graphrag/validate_expanded_kg.py \
        --input  outputs/kg_expansion_bert_init_stage2/full_kg/candidate_triples.csv \
        --output outputs/kg_expansion_bert_init_stage2/full_kg/validated_triples.csv

Resume an interrupted run by rerunning the same command — already-validated
triples in the output file are skipped automatically.
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
MAX_CONCURRENT_PER_MODEL = 50  # overridden by --max_concurrent

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
    messages = _build_messages(head, relation, tail)
    qwen_raw, gpt_raw = await asyncio.gather(
        _model_call(qwen_client, qwen_tok, "qwen", messages),
        _model_call(gpt_client,  gpt_tok,  "gpt",  messages),
    )
    return _extract_verdict(qwen_raw), _extract_verdict(gpt_raw)


async def validate(input_path: Path, output_path: Path, threshold: str) -> None:
    df = pd.read_csv(input_path)
    # Support both 'relation' (from expand script) and 'relation_type' (seed KG format)
    if "relation_type" in df.columns and "relation" not in df.columns:
        df = df.rename(columns={"relation_type": "relation"})
    logger.info(f"Loaded {len(df):,} candidate triples from {input_path}")

    # Resume: skip triples already in the output file
    already_done: set[tuple[str, str, str]] = set()
    if output_path.exists():
        done_df = pd.read_csv(output_path)
        for _, row in done_df.iterrows():
            already_done.add((str(row["head"]), str(row["relation"]), str(row["tail"])))
        logger.info(f"Resuming — {len(already_done):,} triples already validated, skipping")

    pending = df[~df.apply(
        lambda r: (str(r["head"]), str(r["relation"]), str(r["tail"])) in already_done,
        axis=1,
    )]
    logger.info(f"{len(pending):,} triples to validate")

    if pending.empty:
        logger.info("Nothing to do.")
        return

    service_client = tinker.ServiceClient(api_key=TINKER_API_KEY)
    qwen_client = service_client.create_sampling_client(base_model=MODEL_QWEN)
    gpt_client  = service_client.create_sampling_client(base_model=MODEL_GPT)
    qwen_tok = qwen_client.get_tokenizer()
    gpt_tok  = gpt_client.get_tokenizer()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in append mode so partial results survive interruptions
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    out_file = open(output_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(out_file)
    if write_header:
        writer.writerow(["head", "relation", "tail", "verdict_qwen", "verdict_gpt"])

    bar = tqdm(total=len(pending), desc="Validating", unit="triple", dynamic_ncols=True)
    kept = len(already_done)  # count already-kept triples for display
    total_seen = 0

    def _passes(qv: str, gv: str) -> bool:
        if threshold == "both":   return qv == "yes" and gv == "yes"
        if threshold == "either": return qv == "yes" or gv == "yes"
        if threshold == "soft":   return "yes" in (qv, gv) and "no" not in (qv, gv)
        return False

    async def _run(row: pd.Series) -> None:
        nonlocal kept, total_seen
        qv, gv = await _validate_triple(
            qwen_client, qwen_tok, gpt_client, gpt_tok,
            str(row["head"]).strip(), str(row["relation"]).strip(), str(row["tail"]).strip(),
        )
        if _passes(qv, gv):
            writer.writerow([row["head"], row["relation"], row["tail"], qv, gv])
            out_file.flush()
            kept += 1
        total_seen += 1
        bar.update(1)
        bar.set_postfix(kept=kept, rate=f"{kept/total_seen:.0%}" if total_seen else "—")

    await asyncio.gather(*[_run(row) for _, row in pending.iterrows()])
    bar.close()
    out_file.close()

    logger.info(f"Done. {kept:,} triples passed dual validation → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path,
        default=_REPO / "outputs/kg_expansion_bert_init_stage2/full_kg/candidate_triples.csv",
        help="Path to candidate_triples.csv from expand_kg_from_predictions.py",
    )
    ap.add_argument(
        "--output", type=Path,
        default=_REPO / "outputs/kg_expansion_bert_init_stage2/full_kg/validated_triples.csv",
        help="Output path for validated triples",
    )
    ap.add_argument(
        "--threshold", choices=["both", "either", "soft"], default="both",
        help="both=both yes (strictest, default); either=at least one yes; soft=one yes and no hard no",
    )
    ap.add_argument(
        "--max_concurrent", type=int, default=MAX_CONCURRENT_PER_MODEL,
        help=f"Max simultaneous requests per model (default {MAX_CONCURRENT_PER_MODEL})",
    )
    args = ap.parse_args()
    MAX_CONCURRENT_PER_MODEL = args.max_concurrent
    asyncio.run(validate(args.input, args.output, args.threshold))
