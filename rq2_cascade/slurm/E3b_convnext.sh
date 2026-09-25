#!/usr/bin/env bash
# =============================================================================
# E3b — 2 estágios v2: ConvNeXt-T, crops com MAIS CONTEXTO (pad 0.75),
#        img-size 384 (reduz upscale das moscas pequenas, esp. MF/MD).
# Alvo: acurácia de espécie. Roda após E3 (7 classes mantidas).
# Submeter:   sbatch fly-det/slurm/E3b_convnext.sh
# =============================================================================
#SBATCH --job-name=E3b-convnext
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
CROPS="${USER_ROOT}/datasets/DS-F2_crops_pad75"
OUT="${USER_ROOT}/flydet_runs/E3b_convnext"
SECRETS="${USER_ROOT}/secrets/wandb.env"
mkdir -p "${OUT}" "${REPO}/slurm_logs"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"
export WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-E3b_convnext_t_pad75_384}"

echo "==== E3b ConvNeXt-T | crops=${CROPS} | img=384 | $(date) ===="

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
    --img-size 384 \
    --epochs 50 \
    --batch 48 \
    --lr 8e-5 \
    --workers 8

echo "Fim: $(date)"
