#!/bin/bash
#SBATCH --job-name=graphmert_net_s3
#SBATCH --output=slurm_logs/graphmert_networking_stage3_%j.out
#SBATCH --error=slurm_logs/graphmert_networking_stage3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --time=24:00:00
#SBATCH --mail-user=${USER}@princeton.edu

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO=/scratch/gpfs/JHA/hp9084/graphMERT_cnets
YAML_TEMPLATE=${REPO}/launch_configs/args_mlm_networking_stage3.yaml
YAML_LIVE=${REPO}/launch_configs/args_mlm_networking_stage3_run.yaml

cd ${REPO}

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate graphmert

export PYTHONPATH=${REPO}:${REPO}/utils:${PYTHONPATH}
export HF_DATASETS_CACHE=${REPO}/cache/hf_datasets
export HF_HUB_CACHE=${REPO}/cache/transformers
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TQDM_DISABLE=0
export WANDB_MODE=offline

# ── Substitute DELLA_ROOT placeholder in yaml ────────────────────────────────
sed "s|DELLA_ROOT|${REPO}|g" ${YAML_TEMPLATE} > ${YAML_LIVE}

# ── Create required directories ───────────────────────────────────────────────
mkdir -p ${REPO}/cache
mkdir -p ${REPO}/outputs/networking_graphmert_stage3
mkdir -p slurm_logs

# ── Build fresh init checkpoint from stage-2 weights (epoch/step reset to 0) ─
STAGE2_CKPT=${REPO}/outputs/networking_graphmert_stage2/span/rel22/bs32_lr_1e-05
STAGE3_INIT=${REPO}/outputs/networking_graphmert_stage3_init

mkdir -p ${STAGE3_INIT}
cp ${STAGE2_CKPT}/model.safetensors ${STAGE3_INIT}/
cp ${STAGE2_CKPT}/config.json ${STAGE3_INIT}/
python3 -c "
import json
state = {
    'epoch': 0.0,
    'global_step': 0,
    'log_history': [],
    'best_metric': None,
    'best_model_checkpoint': None,
    'is_world_process_zero': True,
    'stateful_callbacks': {},
    'total_flos': 0.0,
    'train_batch_size': 8
}
out = '${STAGE3_INIT}/trainer_state.json'
with open(out, 'w') as f:
    json.dump(state, f, indent=2)
print('Wrote fresh trainer_state.json ->', out)
"

# ── Step 1: dataset preprocessing (CPU) ──────────────────────────────────────
echo "=== Dataset preprocessing ==="
conda run -n graphmert python3 run_dataset_preprocessing.py --yaml_file ${YAML_LIVE}

# ── Step 2: training ──────────────────────────────────────────────────────────
echo "=== Training ==="
conda run -n graphmert python3 run_mlm.py --yaml_file ${YAML_LIVE}
