#!/bin/bash
#SBATCH --job-name=graphmert_predict_tails_bert
#SBATCH --output=slurm_logs/graphmert_predict_tails_bert_%j.out
#SBATCH --error=slurm_logs/graphmert_predict_tails_bert_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=4:00:00
#SBATCH --mail-user=${USER}@princeton.edu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/scratch/gpfs/JHA/hp9084/graphMERT_cnets
YAML_TEMPLATE=${REPO}/launch_configs/args_predict_tails_bert_init.yaml
YAML_LIVE=${REPO}/launch_configs/args_predict_tails_bert_init_run.yaml

cd ${REPO}

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate graphmert

export PYTHONPATH=${REPO}:${REPO}/utils:${PYTHONPATH}
export HF_DATASETS_CACHE=${REPO}/cache/hf_datasets
export HF_HUB_CACHE=${REPO}/cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TQDM_DISABLE=0

# ── Resolve bert-base-uncased snapshot path from local HF cache ──────────────
BERT_SNAPSHOT=$(ls -d ${HF_HUB_CACHE}/models--bert-base-uncased/snapshots/*/ 2>/dev/null | head -1 | sed 's|/$||')
if [ -z "${BERT_SNAPSHOT}" ]; then
    echo "ERROR: bert-base-uncased not found in ${HF_HUB_CACHE}" >&2
    exit 1
fi
echo "Using bert tokenizer from: ${BERT_SNAPSHOT}"

# ── Substitute placeholders in yaml ──────────────────────────────────────────
sed "s|DELLA_ROOT|${REPO}|g; s|BERT_TOKENIZER_PATH|${BERT_SNAPSHOT}|g" ${YAML_TEMPLATE} > ${YAML_LIVE}

# ── Create required directories ───────────────────────────────────────────────
mkdir -p ${REPO}/outputs/kg_expansion_bert_init/train
mkdir -p slurm_logs

# ── Step 1: build predict_tails input dataset (skip if already exists) ────────
INPUT_DATASET=${REPO}/gen4_triplets/predict_tails_input/train
if [ -d "${INPUT_DATASET}" ]; then
    echo "=== predict_tails input dataset already exists at ${INPUT_DATASET}, skipping build ==="
else
    echo "=== Building predict_tails input dataset ==="
    conda run -n graphmert python3 graphrag/prepare_predict_tails_dataset.py \
        --heads_dataset ${REPO}/gen4_triplets/training_data/heads_train \
        --injections_csv ${REPO}/gen4_triplets/training_data/injections_train.csv \
        --output_path ${INPUT_DATASET}
fi

# ── Step 2: run tail prediction (GPU) ────────────────────────────────────────
echo "=== Predicting tails (BERT-init GraphMERT) ==="
conda run -n graphmert python3 predict_tails.py --yaml_file ${YAML_LIVE}
