#!/usr/bin/env bash
# Run the full preprocessing pipeline for a single chapter.
# Usage: ./run_chapter.sh <chapter_number>
#   e.g. ./run_chapter.sh 2

set -euo pipefail

CH=${1:?"Usage: $0 <chapter_number>"}
PDF="/Users/haelpark/Desktop/graphMERT/split_textbook/Ch${CH}.pdf"
TXT="json_data/ch${CH}_text_only_no_equations.txt"
YAML="launch_configs/args_ch${CH}.yaml"
FONT="TimesLTPro-Roman"
SIZE="10.0"

echo "=== Chapter ${CH} pipeline ==="

# ── Step 1: PDF → TXT ─────────────────────────────────────────────────────────
echo "[1/4] Extracting text from ${PDF} → ${TXT}"
python3 utils/pdf_to_json.py extract "${PDF}" \
    --font "${FONT}" --size "${SIZE}" \
    --out "${TXT}"

# ── Step 2: Tokenize ──────────────────────────────────────────────────────────
echo "[2/4] Tokenizing → json_data/tokenized/ch${CH}_txt_train_tokenized"
python3 run_tokenization.py --yaml_file "${YAML}"

# ── Step 3: Entity discovery (Tinker) ─────────────────────────────────────────
echo "[3/4] Running entity discovery (Tinker) for chapter ${CH}"
python3 llm_helper_utils/entity_discovery/run_entity_discovery_tinker.py --chapter "${CH}"

# ── Step 4: Find head positions ───────────────────────────────────────────────
echo "[4/4] Finding head positions for chapter ${CH}"
python3 llm_helper_utils/entity_discovery/find_heads_positions.py --chapter "${CH}"

echo "=== Chapter ${CH} done ==="
echo "Output: json_data/entity_discovery_output/ch${CH}_${MODEL_NAME:-gpt-oss-120b}_all"
