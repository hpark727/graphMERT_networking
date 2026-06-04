#!/bin/bash
#SBATCH --job-name=graphmert_networking
#SBATCH --output=slurm_logs/graphmert_networking_%j.out
#SBATCH --error=slurm_logs/graphmert_networking_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=01:00:01
#SBATCH --mail-user=${USER}@princeton.edu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/scratch/gpfs/JHA/hp9084/graphMERT_cnets
YAML_TEMPLATE=${REPO}/launch_configs/args_mlm_networking.yaml
YAML_LIVE=${REPO}/launch_configs/args_mlm_networking_run.yaml

cd ${REPO}

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate graphmert

export PYTHONPATH=${REPO}:${REPO}/utils:${PYTHONPATH}
export HF_DATASETS_CACHE=${REPO}/cache/hf_datasets
export TRANSFORMERS_CACHE=${REPO}/cache/transformers
export TQDM_DISABLE=0

# ── Substitute DELLA_ROOT placeholder in yaml ────────────────────────────────
sed "s|DELLA_ROOT|${REPO}|g" ${YAML_TEMPLATE} > ${YAML_LIVE}

# ── Create required directories ───────────────────────────────────────────────
mkdir -p ${REPO}/cache
mkdir -p ${REPO}/outputs/networking_graphmert
mkdir -p slurm_logs

# ── Step 1: dataset preprocessing (CPU) ──────────────────────────────────────
echo "=== Dataset preprocessing ==="
conda run -n graphmert python3 run_dataset_preprocessing.py --yaml_file ${YAML_LIVE}

# ── Step 2: training ──────────────────────────────────────────────────────────
echo "=== Training ==="
conda run -n graphmert python3 run_mlm.py --yaml_file ${YAML_LIVE}
