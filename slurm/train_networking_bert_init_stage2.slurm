#!/bin/bash
#SBATCH --job-name=graphmert_bert_s2
#SBATCH --output=slurm_logs/graphmert_bert_s2_%j.out
#SBATCH --error=slurm_logs/graphmert_bert_s2_%j.err
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
YAML_TEMPLATE=${REPO}/launch_configs/args_mlm_networking_bert_init_stage2.yaml
YAML_LIVE=${REPO}/launch_configs/args_mlm_networking_bert_init_stage2_run.yaml

# Stage-1 trained model (suffix added by mlm_utils.py at training time)
STAGE1_MODEL=${REPO}/outputs/networking_graphmert_bert_init/span/rel22/bs32_lr_2e-05

# Fresh init dir for stage-2 (global_step=0 so mlm_utils loads via from_pretrained)
STAGE2_INIT=${REPO}/outputs/networking_graphmert_bert_init_stage2_weights

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

# ── Substitute DELLA_ROOT placeholder in yaml ────────────────────────────────
sed "s|DELLA_ROOT|${REPO}|g" ${YAML_TEMPLATE} > ${YAML_LIVE}

# ── Create required directories ───────────────────────────────────────────────
mkdir -p ${REPO}/outputs/networking_graphmert_bert_init_stage2
mkdir -p slurm_logs

# ── Step 1: stage stage-1 weights as a global_step=0 init checkpoint ─────────
echo "=== Staging stage-1 weights for stage-2 init ==="
mkdir -p ${STAGE2_INIT}
cp ${STAGE1_MODEL}/model.safetensors ${STAGE2_INIT}/
cp ${STAGE1_MODEL}/config.json       ${STAGE2_INIT}/
cat > ${STAGE2_INIT}/trainer_state.json << 'JSONEOF'
{
  "epoch": 0.0,
  "global_step": 0,
  "log_history": [],
  "best_metric": null,
  "best_model_checkpoint": null,
  "is_world_process_zero": true,
  "stateful_callbacks": {},
  "total_flos": 0.0,
  "train_batch_size": 8
}
JSONEOF
echo "Wrote trainer_state.json to ${STAGE2_INIT}"

# ── Step 2: dataset preprocessing (uses cache — should be instant) ────────────
echo "=== Dataset preprocessing ==="
conda run -n graphmert python3 run_dataset_preprocessing.py --yaml_file ${YAML_LIVE}

# ── Step 3: training ──────────────────────────────────────────────────────────
echo "=== Training (stage 2) ==="
conda run -n graphmert python3 run_mlm.py --yaml_file ${YAML_LIVE}
