"""
Convert a textbook PDF to pipeline-ready JSON.

Two modes
---------
inspect   Print all (font, size) pairs found on a page so you can identify
          the body-text font/size before committing to an extraction run.

extract   Extract body-text paragraphs and write {"text": "..."} JSON records.

Usage
-----
# 1. Inspect a representative page (0-indexed) to find body font/size:
python3 utils/pdf_to_json.py inspect path/to/book.pdf --page 10

# 2. Extract chapter pages once you know the font name and size:
python3 utils/pdf_to_json.py extract path/to/book.pdf \
    --font "NimbusRomNo9L-Regu" --size 9.96 \
    --pages 30-80 \
    --out json_data/ch2_text_only_no_equations.json

--pages accepts: single int, range "30-80", or comma list "30,31,45"
--size-tol  float tolerance around --size (default 0.5)
--min-words minimum word count to keep a paragraph (default 8)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_page_spec(spec: str, max_pages: int) -> list[int]:
    """Parse '30-80', '5', or '1,3,5' into a sorted list of 0-indexed page numbers."""
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            pages.update(range(int(lo), int(hi) + 1))
        else:
            pages.add(int(part))
    # clamp to valid range and convert to 0-indexed
    return sorted(p - 1 for p in pages if 1 <= p <= max_pages)


def blocks_on_page(page: fitz.Page) -> list[dict]:
    return page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]


def is_body_span(span: dict, target_font: str, target_size: float, tol: float) -> bool:
    # Match the base font family (e.g. "TimesLTPro" covers Roman, Bold, Italic variants)
    family = target_font.rsplit("-", 1)[0]
    return span["font"].startswith(family) and abs(span["size"] - target_size) < tol


def spans_to_paragraph(spans: list[str]) -> str:
    """Join span texts, collapse whitespace, fix hyphenation across lines."""
    raw = " ".join(spans)
    # re-join soft-hyphenated words (e.g. "com- puter" → "computer")
    raw = re.sub(r"-\s+", "", raw)
    # collapse multiple spaces / newlines
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


# ── inspect mode ──────────────────────────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace) -> None:
    doc = fitz.open(args.pdf)
    page_idx = args.page - 1  # user passes 1-indexed
    if page_idx < 0 or page_idx >= len(doc):
        sys.exit(f"Page {args.page} out of range (doc has {len(doc)} pages).")

    page = doc[page_idx]
    counts: dict[tuple[str, float], int] = {}
    samples: dict[tuple[str, float], str] = {}

    for block in blocks_on_page(page):
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                key = (span["font"], round(span["size"], 2))
                counts[key] = counts.get(key, 0) + len(span["text"].split())
                if key not in samples:
                    samples[key] = span["text"][:80]

    print(f"\n=== Page {args.page} font inventory ===")
    print(f"{'FONT':<45} {'SIZE':>6}  {'WORDS':>7}  SAMPLE")
    print("-" * 90)
    for (font, size), words in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{font:<45} {size:>6.2f}  {words:>7}  {samples[(font, size)]!r}")
    print("\nThe body font is almost always the row with the most words.")


# ── extract mode ──────────────────────────────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> None:
    doc = fitz.open(args.pdf)
    page_indices = parse_page_spec(args.pages, len(doc)) if args.pages else list(range(len(doc)))

    records: list[dict] = []

    caption_prefixes = ("FuturaLTPro", "FrutigerLTPro", "BerkeleyPro")

    for idx in page_indices:
        page = doc[idx]
        for block in blocks_on_page(page):
            if block["type"] != 0:
                continue
            # Skip caption/label blocks: blocks whose first non-empty span uses a
            # caption font (FuturaLTPro, FrutigerLTPro, BerkeleyPro) are figure
            # captions, section headers, or in-figure labels — not body text.
            first_font = next(
                (span["font"]
                 for line in block["lines"]
                 for span in line["spans"]
                 if span["text"].strip()),
                None,
            )
            if first_font and any(first_font.startswith(p) for p in caption_prefixes):
                continue

            body_spans: list[str] = []
            for line in block["lines"]:
                for span in line["spans"]:
                    if is_body_span(span, args.font, args.size, args.size_tol):
                        body_spans.append(span["text"])
            if not body_spans:
                continue
            text = spans_to_paragraph(body_spans)
            if len(text.split()) >= args.min_words:
                records.append({"text": text})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".txt":
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(" ".join(r["text"] for r in records))
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(records)} paragraphs → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract body-text paragraphs from a PDF textbook.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # inspect
    pi = sub.add_parser("inspect", help="Print font inventory for one page.")
    pi.add_argument("pdf", help="Path to PDF file.")
    pi.add_argument("--page", type=int, default=10, help="1-indexed page to inspect (default: 10).")

    # extract
    pe = sub.add_parser("extract", help="Extract body text to JSON.")
    pe.add_argument("pdf", help="Path to PDF file.")
    pe.add_argument("--font", required=True, help="Exact font name from inspect output.")
    pe.add_argument("--size", type=float, required=True, help="Body font size from inspect output.")
    pe.add_argument("--size-tol", type=float, default=0.5, dest="size_tol",
                    help="Tolerance around --size (default: 0.5).")
    pe.add_argument("--pages", default=None,
                    help="Pages to extract, e.g. '30-80' or '1,3,5' (1-indexed). Omit for all pages.")
    pe.add_argument("--out", required=True, help="Output JSON path.")
    pe.add_argument("--min-words", type=int, default=8, dest="min_words",
                    help="Minimum word count per paragraph (default: 8).")

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "inspect":
        cmd_inspect(args)
    elif args.cmd == "extract":
        cmd_extract(args)
