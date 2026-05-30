#!/bin/bash
#SBATCH --job-name=graphmert_networking
#SBATCH --output=logs/graphmert_networking_%j.out
#SBATCH --error=logs/graphmert_networking_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@princeton.edu

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.2
conda activate graphmert       # update to your conda env name

REPO=/scratch/gpfs/${USER}/graphmert_networking
cd ${REPO}

export PYTHONPATH=${REPO}/utils:${PYTHONPATH}
export HF_DATASETS_CACHE=${REPO}/cache/hf_datasets
export TRANSFORMERS_CACHE=${REPO}/cache/transformers

# ── Step 1: dataset preprocessing (CPU, single process) ──────────────────────
echo "=== Dataset preprocessing ==="
PYTHONPATH=${REPO}/utils python3 run_dataset_preprocessing.py \
    --yaml_file launch_configs/args_mlm_networking.yaml

# ── Step 2: training (multi-GPU via torchrun) ─────────────────────────────────
echo "=== Training ==="
NPROC=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

torchrun \
    --nproc_per_node=${NPROC} \
    --master_port=29500 \
    run_mlm.py \
    --yaml_file launch_configs/args_mlm_networking.yaml
