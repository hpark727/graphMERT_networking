"""
Use the Tinker LLM to validate vocabulary bridges between unmatched seed KG
heads and candidate discovered entities.

For each unmatched seed head, presents the top candidates to the LLM and asks
which (if any) refers to the same networking concept. Confirmed bridges are
written directly to vocab_bridge.json — no manual CSV editing needed.

Reads:  gen4_triplets/vocab_bridge_candidates.csv
Writes: gen4_triplets/vocab_bridge.json

Run: python3 graphrag/llm_generate_vocab_bridge.py
"""

import asyncio, csv, json, logging, os, re, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

import tinker

REPO = Path(__file__).resolve().parents[1]
CANDIDATES_CSV = REPO / "gen4_triplets/vocab_bridge_candidates.csv"
BRIDGE_JSON    = REPO / "gen4_triplets/vocab_bridge.json"

MODEL           = "openai/gpt-oss-120b"
MAX_CONCURRENT  = 20
MAX_TOKENS      = 512
TEMPERATURE     = 0.0
PAIRS_PER_CALL  = 15   # seed heads evaluated per LLM call

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

TINKER_API_KEY = os.getenv("TINKER_API_KEY")
if not TINKER_API_KEY:
    raise RuntimeError("TINKER_API_KEY not set — add it to .env")

SYSTEM_PROMPT = """\
You are a networking domain expert. You will be given a list of seed entities
from a knowledge graph of computer networking concepts (from Kurose & Ross
"Computer Networking: A Top-Down Approach"), and for each seed entity a set of
candidate entities discovered in the textbook text.

Your task: for each seed entity, decide which candidate (if any) refers to the
SAME networking concept (possibly with different wording, abbreviation, or
phrasing). Only confirm a match if the two entities are genuinely the same
concept — not merely related or overlapping.

Output a JSON object mapping each seed entity to its confirmed match, or to
null if none of the candidates match. Example:
{
  "arq": "automatic repeat request",
  "rdt": null,
  "slow start": "slow-start algorithm"
}
Output ONLY valid JSON. No explanation."""


def build_query(pairs: list[tuple[str, list[str]]]) -> str:
    lines = []
    for seed, candidates in pairs:
        cands_str = ", ".join(f'"{c}"' for c in candidates) if candidates else "(none)"
        lines.append(f'Seed: "{seed}"  |  Candidates: [{cands_str}]')
    return "Evaluate these pairs:\n" + "\n".join(lines)


def parse_response(text: str, seeds: list[str]) -> dict[str, str | None]:
    text = re.sub(r"(?m)^```.*\n?", "", text).strip()
    # Find the JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {s: None for s in seeds}
    try:
        result = json.loads(m.group(0))
        out = {}
        for s in seeds:
            val = result.get(s)
            out[s] = val.strip().lower() if isinstance(val, str) and val.strip() else None
        return out
    except Exception:
        return {s: None for s in seeds}


_semaphore: asyncio.Semaphore | None = None

def get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    return _semaphore


async def tinker_call(sampling_client, tokenizer, query: str, max_retries: int = 3) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    input_ids   = tokenizer.encode(prompt_text)
    model_input = tinker.types.ModelInput.from_ints(input_ids)
    sampling_params = tinker.types.SamplingParams(
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS
    )
    loop = asyncio.get_event_loop()
    async with get_semaphore():
        for attempt in range(1, max_retries + 1):
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ).result(),
                )
                return tokenizer.decode(response.sequences[0].tokens,
                                        skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e!r}")
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
    return ""


async def process_batch(sampling_client, tokenizer,
                        pairs: list[tuple[str, list[str]]]) -> dict[str, str | None]:
    seeds = [p[0] for p in pairs]
    query = build_query(pairs)
    raw   = await tinker_call(sampling_client, tokenizer, query)
    return parse_response(raw, seeds)


async def main():
    # Load candidates
    rows = []
    with open(CANDIDATES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seed = row["seed_head"].strip().lower()
            candidates = [
                row[f"candidate_{i}"].strip().lower()
                for i in range(1, 6)
                if row.get(f"candidate_{i}", "").strip()
            ]
            rows.append((seed, candidates))

    logger.info(f"Loaded {len(rows)} unmatched seed heads from {CANDIDATES_CSV}")

    service_client  = tinker.ServiceClient(api_key=TINKER_API_KEY)
    sampling_client = service_client.create_sampling_client(base_model=MODEL)
    tokenizer       = sampling_client.get_tokenizer()

    # Batch into groups of PAIRS_PER_CALL
    bridge: dict[str, str] = {}
    tasks = []
    batches = []
    for i in range(0, len(rows), PAIRS_PER_CALL):
        batch = rows[i : i + PAIRS_PER_CALL]
        batches.append(batch)
        tasks.append(process_batch(sampling_client, tokenizer, batch))

    results = await asyncio.gather(*tasks)

    confirmed = rejected = 0
    for result in results:
        for seed, match in result.items():
            if match:
                bridge[seed] = match
                confirmed += 1
            else:
                rejected += 1

    logger.info(f"Confirmed: {confirmed}  |  No match: {rejected}")

    with open(BRIDGE_JSON, "w", encoding="utf-8") as f:
        json.dump(bridge, f, indent=2, ensure_ascii=False, sort_keys=True)
    logger.info(f"Wrote {len(bridge)} bridges → {BRIDGE_JSON}")


if __name__ == "__main__":
    asyncio.run(main())
