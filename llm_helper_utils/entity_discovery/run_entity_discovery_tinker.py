"""
Entity discovery — Tinker (Thinking Machines) backend.

Uses create_sampling_client() for inference-only; no fine-tuning required.
No RPM/RPD limits — concurrency is only credit-bounded.

Run from the repo root:
    python3 llm_helper_utils/entity_discovery/run_entity_discovery_tinker.py
"""

import os
import sys
import json
import ast
import re
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

import argparse

import tinker

from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer as HFAutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from entity_discovery_prompts import (
    SYSTEM_CONTEXT,
    example_user_1, example_assistant_1, example_explanation_1,
    example_user_2, example_assistant_2, example_explanation_2,
    example_user_3, example_assistant_3, example_explanation_3,
    example_user_negative_1, example_assistant_negative_1, example_explanation_negative_1,
)

# ── CLI args ──────────────────────────────────────────────────────────────────
_ap = argparse.ArgumentParser()
_ap.add_argument("--chapter", type=int, default=1, help="Chapter number (1-8)")
_cli = _ap.parse_args()
_CH = _cli.chapter

# ── config ────────────────────────────────────────────────────────────────────
import hashlib as _hashlib
def _tokenized_dir(ch: int) -> Path:
    if ch == 1:
        return Path("json_data/tokenized/ch1_txt")
    txt = f"json_data/ch{ch}_text_only_no_equations.txt"
    base = txt.split("/")[-1].split(".")[0]
    h = _hashlib.md5(txt.encode()).hexdigest()[:8]
    return Path(f"json_data/tokenized/{base}_{h}_tokenized")

if _CH == 1:
    DATASET_DIR  = _tokenized_dir(1)
    PATH_TO_SAVE = Path("json_data/entity_discovery_output")
else:
    DATASET_DIR  = _tokenized_dir(_CH)
    PATH_TO_SAVE = Path(f"json_data/entity_discovery_output/ch{_CH}")

MODEL      = "openai/gpt-oss-120b"   
MODEL_NAME = "gpt-oss-120b"

TAKE_SUBSET    = False
SUBSET_SIZE    = 5
SAVE_CHUNK     = 1000
BATCH_SIZE     = 128          # dataset chunks processed per outer loop
CHUNKS_PER_CALL = 10          # pack N chunks into one sample() call
MAX_CONCURRENT  = 20          # no RPM limit on Tinker — bound only by credits
MAX_TOKENS      = 4096        # reasoning model needs headroom for thinking + output
TEMPERATURE     = 0.0

PRINTOUT           = True
PRINT_RAW_RESPONSE = False
COUNT_ONLY         = False
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise RuntimeError("TINKER_API_KEY env var not set — add it to your .env file")

# ── few-shot message list (shared across all calls) ───────────────────────────
few_shot_messages: list[dict] = []

def _build_few_shots():
    examples = [
        (example_user_1, example_assistant_1, example_explanation_1, None),
        (example_user_2, example_assistant_2, example_explanation_2, None),
        (example_user_3, example_assistant_3, example_explanation_3, None),
        (example_user_negative_1, example_assistant_negative_1, None, example_explanation_negative_1),
    ]
    msgs = []
    msgs.append({"role": "user", "content": "I will provide you with examples"})
    msgs.append({"role": "assistant", "content": "Understood — send the sample and I'll output entities"})
    for user_q, asst_a, expl, neg_expl in examples:
        msgs.append({"role": "user", "content": user_q})
        msgs.append({"role": "assistant", "content": asst_a})
        if expl:
            msgs.append({"role": "user", "content": "Explanation of the previous output:"})
            msgs.append({"role": "assistant", "content": expl})
        if neg_expl:
            msgs.append({"role": "user", "content": "Explanation of what is wrong with the previous output:"})
            msgs.append({"role": "assistant", "content": neg_expl})
    msgs.append({"role": "user", "content": "**End of examples**.\nNow read the actual input:"})
    return msgs

few_shot_messages = _build_few_shots()

# ── parsing helpers ───────────────────────────────────────────────────────────

def extract_rightmost_list(response: str) -> list:
    response = re.sub(r'(?m)^```.*\n?', "", response)
    matches = re.findall(r"\[.*?\]", response, flags=re.DOTALL)
    if not matches:
        return []
    candidate = matches[-1]
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(candidate)
        except Exception:
            pass
    return []


def extract_batch_response(response: str, n: int) -> list[list]:
    response = re.sub(r'(?m)^```.*\n?', "", response)
    match = re.search(r"\[(\s*\[.*?\]\s*,?\s*)+\]", response, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(candidate)
                if isinstance(result, list) and all(isinstance(x, list) for x in result):
                    return (result + [[] for _ in range(n)])[:n]
            except Exception:
                pass
    sections = re.split(r'Input\s+\d+\s*:', response, flags=re.IGNORECASE)
    sections = [s for s in sections if s.strip()]
    results = [extract_rightmost_list(s) for s in sections]
    return (results + [[] for _ in range(n)])[:n]

# ── Tinker inference ──────────────────────────────────────────────────────────

_semaphore: asyncio.Semaphore | None = None

def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


async def tinker_call(sampling_client, tokenizer, user_query: str, max_retries: int = 3) -> str:
    messages = (
        [{"role": "system", "content": SYSTEM_CONTEXT}]
        + few_shot_messages
        + [{"role": "user", "content": user_query}]
    )

    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(prompt_text)
    model_input = tinker.types.ModelInput.from_ints(input_ids)
    sampling_params = tinker.types.SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    loop = asyncio.get_event_loop()

    async with get_semaphore():
        for attempt in range(1, max_retries + 1):
            try:
                # sample() returns a Future; wrap in executor to keep async-friendly
                response = await loop.run_in_executor(
                    None,
                    lambda: sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ).result(),
                )
                generated_ids = response.sequences[0].tokens
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                return text
            except Exception as e:
                logger.warning(f"Tinker attempt {attempt}/{max_retries} failed: {e!r}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                else:
                    logger.error(f"All {max_retries} attempts failed for: {user_query[:50]!r}")
                    return ""


async def process_batch(sampling_client, tokenizer, bert_tokenizer, examples, count_only: bool) -> list:
    seqs = [bert_tokenizer.decode(ex["input_ids"], skip_special_tokens=True) for ex in examples]
    results = []

    tasks = []
    sub_sizes = []
    for i in range(0, len(seqs), CHUNKS_PER_CALL):
        sub = seqs[i : i + CHUNKS_PER_CALL]
        n = len(sub)
        sub_sizes.append(n)

        if n == 1:
            query = f"Input:\n{sub[0]}"
        else:
            numbered = "\n\n".join(f"Input {j+1}:\n{s}" for j, s in enumerate(sub))
            query = (
                f"Process each of the following {n} inputs independently and return "
                f"a Python list of exactly {n} lists — one per input, in order.\n\n"
                f"{numbered}\n\n"
                f"Output format: [['entities for input 1'], ['entities for input 2'], ...]"
            )

        if count_only:
            tasks.append(asyncio.coroutine(lambda: "")())
        else:
            tasks.append(tinker_call(sampling_client, tokenizer, query))

    raw_resps = await asyncio.gather(*tasks)

    if PRINT_RAW_RESPONSE:
        for r in raw_resps:
            print(r)

    for raw, n in zip(raw_resps, sub_sizes):
        if n == 1:
            results.append(extract_rightmost_list(raw))
        else:
            results.extend(extract_batch_response(raw, n))

    return results


async def main():
    service_client = tinker.ServiceClient(api_key=TINKER_API_KEY)
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer = sampling_client.get_tokenizer()          # Qwen3 tokenizer (for prompt formatting)
    bert_tokenizer = HFAutoTokenizer.from_pretrained("bert-base-uncased")  # for decoding dataset chunks

    ds_full = Dataset.load_from_disk(str(DATASET_DIR))
    overall_end = min(SUBSET_SIZE, len(ds_full)) if TAKE_SUBSET else len(ds_full)

    for chunk_start in range(0, overall_end, SAVE_CHUNK):
        chunk_end = min(chunk_start + SAVE_CHUNK, overall_end)
        ds_chunk = ds_full.select(range(chunk_start, chunk_end))
        all_resp = []

        for start in range(0, len(ds_chunk), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(ds_chunk))
            batch = ds_chunk.select(range(start, end))
            cleaned = await process_batch(sampling_client, tokenizer, bert_tokenizer, batch, COUNT_ONLY)
            all_resp.extend(cleaned)
            if PRINTOUT:
                logger.info(f"Processed {chunk_start + end}/{chunk_end}")

        if not COUNT_ONLY:
            fixed = [[str(x) for x in t] if isinstance(t, list) else [str(t)] for t in all_resp]
            ds_chunk = ds_chunk.add_column("response", fixed)
            suffix = f"subset_{SUBSET_SIZE}_{chunk_start}-{chunk_end}" if TAKE_SUBSET else f"{chunk_start}-{chunk_end}"
            out_dir = PATH_TO_SAVE / f"{MODEL_NAME}_{suffix}"
            out_dir.mkdir(parents=True, exist_ok=True)
            ds_chunk.save_to_disk(str(out_dir))
            logger.info(f"Saved {chunk_start}-{chunk_end} → {out_dir}")

    logger.info("=== finished ===")


def unite_output():
    ds_full = Dataset.load_from_disk(str(DATASET_DIR))
    total = min(SUBSET_SIZE, len(ds_full)) if TAKE_SUBSET else len(ds_full)
    datasets, start = [], 0
    while start < total:
        end = min(start + SAVE_CHUNK, total)
        suffix = f"subset_{SUBSET_SIZE}_{start}-{end}" if TAKE_SUBSET else f"{start}-{end}"
        path = PATH_TO_SAVE / f"{MODEL_NAME}_{suffix}"
        datasets.append(load_from_disk(str(path)))
        logger.info(f"Loaded {path}")
        start += SAVE_CHUNK
    united = concatenate_datasets(datasets)
    out = PATH_TO_SAVE / f"{MODEL_NAME}_all"
    united.save_to_disk(str(out))
    logger.info(f"United dataset saved to {out} ({len(united)} records)")
    return out


if __name__ == "__main__":
    asyncio.run(main())
    if not COUNT_ONLY:
        unite_output()
