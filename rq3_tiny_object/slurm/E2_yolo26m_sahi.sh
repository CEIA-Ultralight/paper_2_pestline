#!/usr/bin/env bash
# =============================================================================
# E2 — YOLO26m + SAHI fine-tune (treino em tiles 512) | DS-F2_v8.1.1
# Dataset pré-fatiado: DS-F2_v8.1.1_sliced512 (scripts/slice_dataset.py).
# Paralelo ao E0/E1 na partição h100n2 (mesma máquina dos dados).
# Submeter:   sbatch fly-det/slurm/E2_yolo26m_sahi.sh
# =============================================================================
#SBATCH --job-name=E2-yolo26m-sahi
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail

USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
DATA_YAML="${USER_ROOT}/datasets/DS-F2_v8.1.1_sliced512/data.yaml"
WORK="${USER_ROOT}/flydet_runs/E2_yolo26m_sahi"
SECRETS="${USER_ROOT}/secrets/wandb.env"

mkdir -p "${WORK}" "${REPO}/slurm_logs"

export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${USER_ROOT}/.cache/pip"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
mkdir -p "${WANDB_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${YOLO_CONFIG_DIR}"

if [[ -f "${SECRETS}" ]]; then
  set -a; source "${SECRETS}"; set +a
fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"

# E2: SAHI fine-tune. Tiles de 512 px (resolução nativa do recorte). Treina com
# imgsz=512 (tamanho do tile, sem downscale). Menos épocas: dataset ~15x maior.
MODEL="${MODEL:-yolo26m.pt}"
EPOCHS="${EPOCHS:-60}"
BATCH="${BATCH:-32}"
IMGSZ="${IMGSZ:-512}"
PATIENCE="${PATIENCE:-20}"
WORKERS="${WORKERS:-12}"
WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-E2_yolo26m_sahi}"

echo "============================== E2 SAHI fine-tune ========================="
echo "Job: ${SLURM_JOB_ID:-N/A} @ ${SLURM_NODELIST:-N/A} | Modelo: ${MODEL} | imgsz=${IMGSZ} batch=${BATCH} epochs=${EPOCHS}"
echo "Data: ${DATA_YAML} | W&B: ${WANDB_ENTITY}/${WANDB_PROJECT} run=${WANDB_RUN_NAME} | Início: $(date)"
echo "========================================================================="

apptainer exec --nv \
  --bind "${USER_ROOT}:${USER_ROOT}" \
  --bind "${WORK}:/workspace" \
  --pwd /workspace \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --env WANDB_ENTITY="${WANDB_ENTITY}" \
  --env WANDB_DIR="${WANDB_DIR}" \
  --env HF_HOME="${HF_HOME}" \
  --env TORCH_HOME="${TORCH_HOME}" \
  --env YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR}" \
  --env FLYDET_DATASET_ROOT="${USER_ROOT}/datasets/DS-F2_v8.1.1_sliced512" \
  "${IMAGE}" \
  /opt/venv/bin/fly-det-train \
    --backend yolo \
    --model "${MODEL}" \
    --data "${DATA_YAML}" \
    --project "${WORK}/runs" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH}" \
    --imgsz "${IMGSZ}" \
    --patience "${PATIENCE}" \
    --num-workers "${WORKERS}" \
    --gpus 1 \
    --topk 5 \
    --wandb \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "${WANDB_RUN_NAME}"

echo "Fim: $(date)"
