#!/usr/bin/env bash
# =============================================================================
# Loga matriz de confusão (test) de cada run do classificador na run W&B
# correspondente, para comparação lado a lado no dashboard.
# Submeter:   sbatch fly-det/slurm/log_confusion.sh
# =============================================================================
#SBATCH --job-name=log-confusion
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail
USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
SECRETS="${USER_ROOT}/secrets/wandb.env"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi

echo "==== Log confusion matrices -> W&B | $(date) ===="

apptainer exec --nv \
  --bind "${USER_ROOT}:${USER_ROOT}" \
  --pwd /workspace \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --env WANDB_DIR="${WANDB_DIR}" \
  --env HF_HOME="${HF_HOME}" \
  --env TORCH_HOME="${TORCH_HOME}" \
  --env YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR}" \
  "${IMAGE}" \
  /opt/venv/bin/python "${REPO}/fly-det/scripts/log_confusion_wandb.py" \
    --crops "${USER_ROOT}/datasets/DS-F2_crops_pad75"

echo "Fim: $(date)"
