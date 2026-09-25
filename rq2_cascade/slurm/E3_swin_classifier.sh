#!/usr/bin/env bash
# =============================================================================
# E3 — 2 estágios: classificador de espécies Swin-T sobre crops das detecções.
# Treina em DS-F2_crops (ImageFolder), mede acurácia geral + por espécie (alvo 0.95).
# 1 GPU, via Slurm, partição h100n2 (mesma máquina dos dados).
# Submeter:   sbatch fly-det/slurm/E3_swin_classifier.sh
# =============================================================================
#SBATCH --job-name=E3-swin-classifier
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail

USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
CROPS="${USER_ROOT}/datasets/DS-F2_crops"
OUT="${USER_ROOT}/flydet_runs/E3_swin"
SECRETS="${USER_ROOT}/secrets/wandb.env"

mkdir -p "${OUT}" "${REPO}/slurm_logs"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"
export WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-E3_swin_t}"

echo "==== E3 Swin-T | crops=${CROPS} | out=${OUT} | $(date) ===="

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
    --model swin_t \
    --img-size 224 \
    --epochs 40 \
    --batch 64 \
    --lr 1e-4 \
    --workers 8

echo "Fim: $(date)"
