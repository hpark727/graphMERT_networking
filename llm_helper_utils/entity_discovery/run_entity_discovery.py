"""
Entity discovery script — equivalent to entity_discovery_gemini.ipynb.
Run from the repo root:
    python3 llm_helper_utils/entity_discovery/run_entity_discovery.py
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
import pyarrow as pa

from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import AutoTokenizer
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
import entity_discovery_prompts as prompts_module
from entity_discovery_prompts import (
    SYSTEM_CONTEXT,
    example_user_1, example_assistant_1, example_explanation_1,
    example_user_2, example_assistant_2, example_explanation_2,
    example_user_3, example_assistant_3, example_explanation_3,
    example_user_negative_1, example_assistant_negative_1, example_explanation_negative_1,
)

# ── config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("json_data/tokenized/ch1_txt")
PATH_TO_SAVE = Path("json_data/entity_discovery_output")
TOKENIZER_NAME = "bert-base-uncased"

MODEL      = "gemini-2.5-flash-lite"
MODEL_NAME = "gemini-2.5-flash-lite"

TAKE_SUBSET  = False
SUBSET_SIZE  = 5
SAVE_CHUNK   = 1000
BATCH_SIZE   = 128
TIME_SLEEP   = 0.05
PRINTOUT     = True
PRINT_RAW_RESPONSE = False
COUNT_ONLY   = False   # set True for a free token-count dry run
MAX_CONCURRENT = 1    # serialize requests to stay within RPM limit
REQUEST_DELAY  = 6.5  # seconds between requests
CHUNKS_PER_CALL = 50  # 246 chunks → 5 API calls; each ~6400 tokens, well under 250k limit
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY env var not set — run: export GEMINI_API_KEY=your_key")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

few_shot_examples = [
    {"user_query": example_user_1, "model_answer": example_assistant_1, "model_explanation": example_explanation_1},
    {"user_query": example_user_2, "model_answer": example_assistant_2, "model_explanation": example_explanation_2},
    {"user_query": example_user_3, "model_answer": example_assistant_3, "model_explanation": example_explanation_3},
    {"user_query": example_user_negative_1, "model_answer": example_assistant_negative_1, "model_explanation_negative": example_explanation_negative_1},
]

STRUCTURED_CONTENT_FEW_SHOTS = [
    {"role": "user", "parts": [{"text": "I will provide you with examples"}]},
    {"role": "model", "parts": [{"text": "Understood — send the sample and I'll output entities"}]},
]
for ex in few_shot_examples:
    STRUCTURED_CONTENT_FEW_SHOTS.append({"role": "user", "parts": [{"text": ex["user_query"]}]})
    STRUCTURED_CONTENT_FEW_SHOTS.append({"role": "model", "parts": [{"text": ex["model_answer"]}]})
    if "model_explanation" in ex:
        STRUCTURED_CONTENT_FEW_SHOTS.append({"role": "user", "parts": [{"text": "Explanation of the previous output:"}]})
        STRUCTURED_CONTENT_FEW_SHOTS.append({"role": "model", "parts": [{"text": ex["model_explanation"]}]})
    if "model_explanation_negative" in ex:
        STRUCTURED_CONTENT_FEW_SHOTS.append({"role": "user", "parts": [{"text": "Explanation of what is wrong with the previous output:"}]})
        STRUCTURED_CONTENT_FEW_SHOTS.append({"role": "model", "parts": [{"text": ex["model_explanation_negative"]}]})
STRUCTURED_CONTENT_FEW_SHOTS.append({"role": "user", "parts": [{"text": "**End of examples**.\nNow read the actual input:"}]})

input_tokens = 0
output_tokens = 0


def extract_rightmost_list(response: str) -> list:
    response = re.sub(r'(?m)^```.*\n?', "", response)
    matches = re.findall(r"\[.*?\]", response, flags=re.DOTALL)
    if not matches:
        return []
    candidate = matches[-1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(candidate)
    except Exception:
        return []


def extract_batch_response(response: str, n: int) -> list[list]:
    """Parse a batched response into a list of n entity lists.

    The model is asked to return [['e1','e2'], [], ['e3'], ...].
    Falls back to splitting on numbered labels if the outer list parse fails.
    """
    response = re.sub(r'(?m)^```.*\n?', "", response)
    # Try to parse the outermost [[...]] structure
    match = re.search(r"\[(\s*\[.*?\]\s*,?\s*)+\]", response, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        for parser in (json.loads, ast.literal_eval):
            try:
                result = parser(candidate)
                if isinstance(result, list) and all(isinstance(x, list) for x in result):
                    # Pad or trim to exactly n entries
                    result = (result + [[] for _ in range(n)])[:n]
                    return result
            except Exception:
                pass
    # Fallback: split on "Input N:" labels and extract one list per section
    sections = re.split(r'Input\s+\d+\s*:', response, flags=re.IGNORECASE)
    sections = [s for s in sections if s.strip()]
    results = [extract_rightmost_list(s) for s in sections]
    return (results + [[] for _ in range(n)])[:n]


_semaphore: asyncio.Semaphore | None = None

def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


async def gemini_call(client, user_query: str, count_only: bool, max_retries: int = 3) -> str:
    global input_tokens, output_tokens
    call_fn = client.aio.models.count_tokens if count_only else client.aio.models.generate_content
    config = types.GenerateContentConfig(system_instruction=SYSTEM_CONTEXT)
    contents = list(STRUCTURED_CONTENT_FEW_SHOTS) + [{"role": "user", "parts": [{"text": user_query}]}]

    async with get_semaphore():
        await asyncio.sleep(REQUEST_DELAY)
        for attempt in range(1, max_retries + 1):
            try:
                resp = await call_fn(model=MODEL, contents=contents, config=config)
                if count_only:
                    input_tokens += resp.total_tokens
                    return ""
                metadata = resp.usage_metadata
                input_tokens += (metadata.prompt_token_count or 0) - (metadata.cached_content_token_count or 0)
                output_tokens += metadata.candidates_token_count or 0
                return getattr(resp, "text", "") or ""
            except Exception as e:
                logger.warning(f"Gemini attempt {attempt}/{max_retries} failed: {e!r}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * attempt)
                else:
                    logger.error(f"All {max_retries} attempts failed for: {user_query[:50]!r}")
                    return ""


async def process_batch(client, examples, count_only: bool) -> list:
    """Process examples in sub-batches of CHUNKS_PER_CALL to reduce API calls."""
    seqs = [tokenizer.decode(ex["input_ids"], skip_special_tokens=True) for ex in examples]
    results = []

    for i in range(0, len(seqs), CHUNKS_PER_CALL):
        sub = seqs[i : i + CHUNKS_PER_CALL]
        n = len(sub)

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

        raw = await gemini_call(client, query, count_only)

        if PRINT_RAW_RESPONSE:
            print(raw)

        if count_only:
            results.extend([[] for _ in range(n)])
        elif n == 1:
            results.append(extract_rightmost_list(raw))
        else:
            results.extend(extract_batch_response(raw, n))

        await asyncio.sleep(TIME_SLEEP)

    return results


async def main():
    ds_full = Dataset.load_from_disk(str(DATASET_DIR))
    overall_end = min(SUBSET_SIZE, len(ds_full)) if TAKE_SUBSET else len(ds_full)
    client = genai.Client(api_key=GEMINI_API_KEY)

    for chunk_start in range(0, overall_end, SAVE_CHUNK):
        chunk_end = min(chunk_start + SAVE_CHUNK, overall_end)
        ds_chunk = ds_full.select(range(chunk_start, chunk_end))
        all_resp = []

        for start in range(0, len(ds_chunk), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(ds_chunk))
            batch = ds_chunk.select(range(start, end))
            cleaned = await process_batch(client, batch, COUNT_ONLY)
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
    logger.info(f"input_tokens:  {input_tokens}")
    logger.info(f"output_tokens: {output_tokens}")


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
