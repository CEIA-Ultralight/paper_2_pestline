#!/usr/bin/env bash
# =============================================================================
# E3-ensemble — Ensemble (E3+E3b+E3c) + TTA no test. Sem treino, só inferência.
# Submeter:   sbatch fly-det/slurm/E3_ensemble_tta.sh
# =============================================================================
#SBATCH --job-name=E3-ensemble-tta
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail
USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
CROPS="${USER_ROOT}/datasets/DS-F2_crops_pad75"
RUNS="${USER_ROOT}/flydet_runs"
OUT="${RUNS}/E3_ensemble_tta"
SECRETS="${USER_ROOT}/secrets/wandb.env"
mkdir -p "${OUT}" "${REPO}/slurm_logs"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"
export WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-E3_ensemble_tta}"

echo "==== E3 Ensemble + TTA | $(date) ===="

# Nota: E3 (swin) foi treinado em 224; E3b/E3c (convnext) em 384.
# O ensemble le o img_size de cada checkpoint e redimensiona a entrada por modelo.
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
  /opt/venv/bin/python "${REPO}/fly-det/scripts/ensemble_tta.py" \
    --data "${CROPS}" \
    --ckpts "${RUNS}/E3_swin/best.pt" "${RUNS}/E3b_convnext/best.pt" "${RUNS}/E3c_focal/best.pt" \
    --img-size 384 \
    --batch 64 \
    --workers 8 \
    --tta \
    --out "${OUT}"

echo "Fim: $(date)"
