#!/usr/bin/env bash
# =============================================================================
# E3d — 2 estágios v4: ConvNeXt-S (backbone maior, ~50M params) p/ morfologia
#        fina (pares MD<->MV, MF<->NOISE). pad 0.75, img 384, focal + cw + early stop.
# Submeter:   sbatch fly-det/slurm/E3d_convnext_s.sh
# =============================================================================
#SBATCH --job-name=E3d-convnext-s
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail
USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
CROPS="${USER_ROOT}/datasets/DS-F2_crops_pad75"
OUT="${USER_ROOT}/flydet_runs/E3d_convnext_s"
SECRETS="${USER_ROOT}/secrets/wandb.env"
mkdir -p "${OUT}" "${REPO}/slurm_logs"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"
export WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-E3d_convnext_s}"

echo "==== E3d ConvNeXt-S + focal + cw + early-stop | $(date) ===="

apptainer exec --nv \
  --bind "${USER_ROOT}:${USER_ROOT}" \
  --pwd /workspace \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --env WANDB_ENTITY="${WANDB_ENTITY}" \
  --env WANDB_PROJECT="${WANDB_PROJECT}" \
  --env WANDB_RUN_NAME="${WANDB_RUN_NAME}" \
  --env WANDB_DIR="${WANDB_DIR}" \
  --env HF_HOME="${HF_HOME}" \
  --env TORCH_HOME="${TORCH_HOME}" \
  --env YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR}" \
  "${IMAGE}" \
  /opt/venv/bin/python "${REPO}/fly-det/scripts/train_classifier.py" \
    --data "${CROPS}" \
    --out "${OUT}" \
    --model convnext_s \
    --img-size 384 \
    --epochs 60 \
    --batch 32 \
    --lr 5e-5 \
    --workers 8 \
    --patience 10 \
    --focal \
    --class-weights

echo "Fim: $(date)"
