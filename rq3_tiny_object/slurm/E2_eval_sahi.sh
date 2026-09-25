#!/usr/bin/env bash
# =============================================================================
# E2-eval — Avaliação JUSTA do E2 (SAHI) no test de imagens inteiras.
# Compara plain vs SAHI usando o best.pt treinado em tiles (E2).
# 1 GPU leve, via Slurm (regra do cluster). Mesma máquina dos dados (h100n2).
# Submeter:   sbatch fly-det/slurm/E2_eval_sahi.sh
# =============================================================================
#SBATCH --job-name=E2-eval-sahi
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail

USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
MODEL="${USER_ROOT}/flydet_runs/E2_yolo26m_sahi/runs/yolo26m.pt/weights/best.pt"
DATA_ROOT="${USER_ROOT}/datasets/DS-F2_v8.1.1"          # test INTEIRO (não fatiado)
OUT="${USER_ROOT}/flydet_runs/E2_eval_sahi"
SECRETS="${USER_ROOT}/secrets/wandb.env"

mkdir -p "${OUT}" "${REPO}/slurm_logs"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"

echo "==== E2-eval | model=${MODEL} | data=${DATA_ROOT} split=test | $(date) ===="

# tiles 512 (mesmos do treino E2), overlap 0.25, NMS p/ juntar detecções.
apptainer exec --nv \
  --bind "${USER_ROOT}:${USER_ROOT}" \
  --pwd /workspace \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --env WANDB_ENTITY="${WANDB_ENTITY}" \
  --env WANDB_DIR="${WANDB_DIR}" \
  --env HF_HOME="${HF_HOME}" \
  --env TORCH_HOME="${TORCH_HOME}" \
  --env YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR}" \
  --env PYTHONPATH="${REPO}/fly-det" \
  "${IMAGE}" \
  /opt/venv/bin/python -m fly_det.sahi_inference \
    --model "${MODEL}" \
    --data-root "${DATA_ROOT}" \
    --split test \
    --out-dir "${OUT}" \
    --device cuda:0 \
    --imgsz 512 \
    --tile-wh 512 512 \
    --overlap-ratio 0.25 0.25 \
    --conf 0.001 \
    --eval-iou 0.5 \
    --draw-gt --embed-legend --cm-normalize \
    --wandb --wandb-project fly-species --wandb-run-name E2_eval_sahi

echo "Fim: $(date)"
