"""
Reclassify over-generic "uses" triples in validated KG CSVs.

For each triple with relation_type == "uses", asks a single LLM to pick
the most specific relation from:
  runs_on   | implements | has_part | requires | uses

All other relations are kept unchanged.

Input:  <val-dir>/validated_ch{N}.csv
Output: <out-dir>/validated_ch{N}.csv  (defaults to val-dir, i.e. in-place)

Run from repo root:
    python3 graphrag/reclassify_uses.py --val-dir gen2_triplets/validated
    python3 graphrag/reclassify_uses.py --val-dir gen2_triplets/validated \
        --out-dir gen2_triplets/validated_remapped
"""

import os
import re
import csv
import asyncio
import logging
import argparse
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import tinker

_REPO = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise RuntimeError("TINKER_API_KEY not set in .env")

MODEL          = "openai/gpt-oss-120b"
TEMPERATURE    = 0.0
MAX_TOKENS     = 512
MAX_CONCURRENT = 30

VALID_REMAPS = {"runs_on", "implements", "has_part", "requires", "uses"}

def _parse_relation(raw: str) -> str:
    # 1. look for explicit "Answer: <rel>" line
    m = re.search(r"Answer:\s*([a-z_]+)", raw, re.IGNORECASE)
    if m:
        candidate = m.group(1).lower().strip()
        if candidate in VALID_REMAPS:
            return candidate
    # 2. fall back: find the last valid relation name anywhere in the text
    text = raw.lower()
    for rel in ("runs_on", "implements", "has_part", "requires"):
        # search from the end so we pick up the conclusion, not an example mention
        if re.search(r'\b' + rel + r'\b', text):
            return rel
    return "uses"

SYSTEM_PROMPT = """You are a computer networking expert. A knowledge graph triple was labeled with the generic relation "uses". Choose the single most specific correct relation from the list below.

Definitions:
  runs_on    – head protocol, service, or device operates on top of tail (a lower-layer protocol, network layer, or physical medium)
  implements – head applies or executes tail as an algorithm, mechanism, or technique
  has_part   – head structurally contains tail as a component (e.g. a table, header, buffer, cache, field)
  requires   – head depends on tail to function correctly
  uses       – none of the above; "uses" is genuinely the best fit

Think briefly, then end your response with exactly:
Answer: <relation_name>"""

_semaphore: asyncio.Semaphore | None = None

def _get_sem() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


async def _reclassify(sampling_client, tokenizer, head: str, tail: str,
                      max_retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{head} | uses | {tail}"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids   = tokenizer.encode(prompt_text)
    model_input = tinker.types.ModelInput.from_ints(input_ids)
    sp          = tinker.types.SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
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
                    timeout=60.0,
                )
                raw = tokenizer.decode(resp.sequences[0].tokens, skip_special_tokens=True)
                return _parse_relation(raw)
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e!r}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
                else:
                    return "uses"  # safe fallback


async def reclassify_chapter(chapter: int, val_dir: Path, out_dir: Path,
                              sampling_client, tokenizer) -> None:
    src = val_dir / f"validated_ch{chapter}.csv"
    if not src.exists():
        logger.warning(f"Skipping ch{chapter} — file not found: {src}")
        return

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    uses_rows  = [(i, r) for i, r in enumerate(rows) if r["relation_type"] == "uses"]
    other_count = len(rows) - len(uses_rows)
    logger.info(f"Ch{chapter}: {len(uses_rows)} 'uses' triples to reclassify, {other_count} kept as-is")

    if not uses_rows:
        if out_dir != val_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            _write_csv(out_dir / f"validated_ch{chapter}.csv", rows)
        return

    bar = tqdm(total=len(uses_rows), desc=f"Ch{chapter}", unit="triple", dynamic_ncols=True)
    remap_counts: dict[str, int] = {}

    async def _run(idx: int, row: dict) -> tuple[int, str]:
        new_rel = await _reclassify(sampling_client, tokenizer, row["head"], row["tail"])
        bar.update(1)
        remap_counts[new_rel] = remap_counts.get(new_rel, 0) + 1
        bar.set_postfix(**{k: v for k, v in sorted(remap_counts.items())})
        return idx, new_rel

    results = await asyncio.gather(*[_run(i, r) for i, r in uses_rows])
    bar.close()

    for idx, new_rel in results:
        rows[idx]["relation_type"] = new_rel

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / f"validated_ch{chapter}.csv", rows)

    stayed = remap_counts.get("uses", 0)
    remapped = len(uses_rows) - stayed
    logger.info(f"Ch{chapter}: {remapped}/{len(uses_rows)} 'uses' triples remapped → {out_dir / f'validated_ch{chapter}.csv'}")
    for rel, cnt in sorted(remap_counts.items()):
        logger.info(f"  {rel:<15} {cnt}")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


async def main(chapters: list[int], val_dir: Path, out_dir: Path) -> None:
    service_client  = tinker.ServiceClient(api_key=TINKER_API_KEY)
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer       = sampling_client.get_tokenizer()

    for ch in chapters:
        await reclassify_chapter(ch, val_dir, out_dir, sampling_client, tokenizer)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--chapter",  type=int,          help="Single chapter")
    grp.add_argument("--chapters", type=int, nargs="+", metavar="N")
    grp.add_argument("--all",      action="store_true", help="Chapters 1-8")
    ap.add_argument("--val-dir", type=Path, default=_REPO / "gen2_triplets/validated",
                    help="Directory with validated_ch{N}.csv files")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: same as --val-dir, overwrites in place)")
    args = ap.parse_args()

    if args.all:
        chapters = list(range(1, 9))
    elif args.chapters:
        chapters = args.chapters
    else:
        chapters = [args.chapter]

    out_dir = args.out_dir if args.out_dir else args.val_dir
    asyncio.run(main(chapters, args.val_dir, out_dir))
