#!/bin/bash
#SBATCH --job-name=graphmert_bert_init
#SBATCH --output=slurm_logs/graphmert_bert_init_%j.out
#SBATCH --error=slurm_logs/graphmert_bert_init_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=gpu
#SBATCH --constraint="intel&gpu40"
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=24:00:00
#SBATCH --mail-user=${USER}@princeton.edu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/scratch/gpfs/JHA/hp9084/graphMERT_cnets
YAML_TEMPLATE=${REPO}/launch_configs/args_mlm_networking_bert_init.yaml
YAML_LIVE=${REPO}/launch_configs/args_mlm_networking_bert_init_run.yaml
BERT_INIT_WEIGHTS=${REPO}/outputs/networking_graphmert_bert_init_weights

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
export WANDB_MODE=offline

# ── Resolve bert-base-uncased snapshot path ───────────────────────────────────
BERT_SNAPSHOT=$(ls -d ${HF_HUB_CACHE}/models--bert-base-uncased/snapshots/*/ 2>/dev/null | head -1 | sed 's|/$||')
if [ -z "${BERT_SNAPSHOT}" ]; then
    echo "ERROR: bert-base-uncased not found in ${HF_HUB_CACHE}" >&2
    exit 1
fi
echo "Using BERT snapshot: ${BERT_SNAPSHOT}"

# ── Substitute DELLA_ROOT placeholder in yaml ────────────────────────────────
sed "s|DELLA_ROOT|${REPO}|g" ${YAML_TEMPLATE} > ${YAML_LIVE}

# ── Create required directories ───────────────────────────────────────────────
mkdir -p ${REPO}/outputs/networking_graphmert_bert_init
mkdir -p slurm_logs

# ── Step 1: initialize GraphMERT weights from BERT (CPU) ─────────────────────
echo "=== Initializing GraphMERT from BERT ==="
conda run -n graphmert python3 scripts/init_from_bert.py \
    --bert_path ${BERT_SNAPSHOT} \
    --output_path ${BERT_INIT_WEIGHTS} \
    --config_overrides 'max_nodes=512,vocab_size=30522,num_hidden_layers=6,num_attention_heads=12,hidden_size=768,intermediate_size=3072,pretrained_emb_dim=0,num_relationships=22'

# ── Step 2: dataset preprocessing (CPU) ──────────────────────────────────────
echo "=== Dataset preprocessing ==="
conda run -n graphmert python3 run_dataset_preprocessing.py --yaml_file ${YAML_LIVE}

# ── Step 3: training ──────────────────────────────────────────────────────────
echo "=== Training ==="
conda run -n graphmert python3 run_mlm.py --yaml_file ${YAML_LIVE}
