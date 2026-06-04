#!/bin/bash
#SBATCH --job-name=graphmert_networking
#SBATCH --output=slurm_logs/graphmert_networking_%j.out
#SBATCH --error=slurm_logs/graphmert_networking_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${USER}@princeton.edu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/scratch/gpfs/JHA/hp9084/graphMERT_cnets
YAML_TEMPLATE=${REPO}/launch_configs/args_mlm_networking.yaml
YAML_LIVE=${REPO}/launch_configs/args_mlm_networking_run.yaml

cd ${REPO}

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.2
conda activate graphmert

export PYTHONPATH=${REPO}:${REPO}/utils:${PYTHONPATH}
export HF_DATASETS_CACHE=${REPO}/cache/hf_datasets
export TRANSFORMERS_CACHE=${REPO}/cache/transformers

# ── Substitute DELLA_ROOT placeholder in yaml ────────────────────────────────
sed "s|DELLA_ROOT|${REPO}|g" ${YAML_TEMPLATE} > ${YAML_LIVE}

# ── Create required directories ───────────────────────────────────────────────
mkdir -p ${REPO}/cache
mkdir -p ${REPO}/outputs/networking_graphmert
mkdir -p slurm_logs

# ── Step 1: dataset preprocessing (CPU) ──────────────────────────────────────
echo "=== Dataset preprocessing ==="
python3 run_dataset_preprocessing.py --yaml_file ${YAML_LIVE}

# ── Step 2: training ──────────────────────────────────────────────────────────
echo "=== Training ==="
python3 run_mlm.py --yaml_file ${YAML_LIVE}
