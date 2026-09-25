#!/usr/bin/env bash
# =============================================================================
# E0 — Baseline YOLO26m | DS-F2_v8.1.1 (classificação de espécies de moscas)
# Cluster DGX / Slurm — regras: GPU só via Slurm, dados/caches no /raid,
# partição = h100n2 (onde os dados estão), checkpoints periódicos.
#
# Submeter:   sbatch fly-det/slurm/E0_yolo26m_baseline.sh
# Monitorar:  squeue -u $USER ; scontrol show job <job_id>
# =============================================================================
#SBATCH --job-name=E0-yolo26m-baseline
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

# ------------------------- caminhos (somente /raid) -------------------------
USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
PKG="${REPO}/fly-det"                       # pacote fly-det (fly_det/)
DATA_YAML="${USER_ROOT}/datasets/DS-F2_v8.1.1/data.yaml"
WORK="${USER_ROOT}/flydet_runs/E0_yolo26m_baseline"
SECRETS="${USER_ROOT}/secrets/wandb.env"

mkdir -p "${WORK}" "${WORK}/models" "${REPO}/slurm_logs"
# Pesos base (yolo26m.pt, yolo26l.pt) ficam no RAID e são ligados em /workspace/models.
if [[ -d "${USER_ROOT}/models" ]]; then
  cp -n "${USER_ROOT}/models"/*.pt "${WORK}/models/" 2>/dev/null || true
fi

# ------------------------- caches -> /raid (regra do cluster) ----------------
export WANDB_DIR="${USER_ROOT}/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${USER_ROOT}/.cache/pip"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
mkdir -p "${WANDB_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${YOLO_CONFIG_DIR}"

# ------------------------- credenciais W&B (fora do git) ---------------------
if [[ -f "${SECRETS}" ]]; then
  set -a; source "${SECRETS}"; set +a
fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"

# ------------------------- hiperparâmetros E0 --------------------------------
# Receita YOLO26 p/ dataset pequeno + tiny objects (ver PLANNING.md):
#  - pesos Objects365 (objv1-150), imgsz alto (1920 = resolução NATIVA, sem
#    reduzir o dataset), AdamW lr0=0.001, aug moderado, patience 30.
MODEL="${MODEL:-yolo26m.pt}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-1920}"
PATIENCE="${PATIENCE:-30}"
WORKERS="${WORKERS:-12}"
WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-E0_yolo26m_baseline}"

echo "============================== E0 baseline =============================="
echo "Job:        ${SLURM_JOB_ID:-N/A} @ ${SLURM_NODELIST:-N/A}"
echo "Imagem:     ${IMAGE}"
echo "Data:       ${DATA_YAML}"
echo "Work:       ${WORK}"
echo "Modelo:     ${MODEL}  imgsz=${IMGSZ}  batch=${BATCH}  epochs=${EPOCHS}"
echo "W&B:        ${WANDB_ENTITY}/${WANDB_PROJECT}  run=${WANDB_RUN_NAME}"
echo "Início:     $(date)"
echo "========================================================================="

# ------------------------- treino (GPU DENTRO do Slurm) ----------------------
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
