#!/usr/bin/env bash
# =============================================================================
# E1 — YOLO26m + head P2 (small objects) | DS-F2_v8.1.1
# Paralelo ao E0 na partição h100n2 (mesma máquina dos dados).
# Submeter:   sbatch fly-det/slurm/E1_yolo26m_p2.sh
# =============================================================================
#SBATCH --job-name=E1-yolo26m-p2
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
DATA_YAML="${USER_ROOT}/datasets/DS-F2_v8.1.1/data.yaml"
WORK="${USER_ROOT}/flydet_runs/E1_yolo26m_p2"
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

# E1: variante P2 (head extra p/ small objects). Sem pesos pré-treinados P2 ->
# treina a partir do YAML; backbone herda init padrão. Mais épocas p/ convergir.
MODEL="${MODEL:-yolo26m-p2.yaml}"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-1920}"
PATIENCE="${PATIENCE:-40}"
WORKERS="${WORKERS:-12}"
WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-E1_yolo26m_p2}"

echo "============================== E1 head P2 ==============================="
echo "Job: ${SLURM_JOB_ID:-N/A} @ ${SLURM_NODELIST:-N/A} | Modelo: ${MODEL} | imgsz=${IMGSZ} batch=${BATCH} epochs=${EPOCHS}"
echo "W&B: ${WANDB_ENTITY}/${WANDB_PROJECT} run=${WANDB_RUN_NAME} | Início: $(date)"
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
  --env FLYDET_DATASET_ROOT="${USER_ROOT}/datasets/DS-F2_v8.1.1" \
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
