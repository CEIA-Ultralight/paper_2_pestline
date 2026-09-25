#!/usr/bin/env bash
# =============================================================================
# E5 — 2 estágios v6: ConvNeXt-T em img-size 512 (menos upscale de moscas
#        pequenas MF=42px/MD=111px). pad 0.75, focal + cw + early stop.
# Hipótese: o gargalo e' o upscale excessivo (384 de um crop de 42px = 9x).
# Submeter:   sbatch fly-det/slurm/E5_convnext_512.sh
# =============================================================================
#SBATCH --job-name=E5-convnext-512
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
OUT="${USER_ROOT}/flydet_runs/E5_convnext_512"
SECRETS="${USER_ROOT}/secrets/wandb.env"
mkdir -p "${OUT}" "${REPO}/slurm_logs"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"
export WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-E5_convnext_512}"

echo "==== E5 ConvNeXt-T img512 + focal + cw + early-stop | $(date) ===="

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
    --model convnext_t \
    --img-size 512 \
    --epochs 60 \
    --batch 32 \
    --lr 8e-5 \
    --workers 8 \
    --patience 10 \
    --focal \
    --class-weights

echo "Fim: $(date)"
