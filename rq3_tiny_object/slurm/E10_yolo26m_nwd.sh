#!/usr/bin/env bash
# =============================================================================
# E10 — YOLO26m + NWD (Normalized Wasserstein Distance) | DS-F2_v8.1.1
# Experimento: mudanca UNICA vs E0 — metrica de similaridade hibrida
#   CIoU+NWD (w=0.5, C=44px) na loss de regressao (BboxLoss) e no label
#   assignment (TaskAlignedAssigner). Cobre os 2 heads do E2EDetectLoss.
# Hipotesis (literatura tiny objects): IoU e hiper-sensivel a deslocamento
#   de 1-2 px em caixas ~40 px (mediana do dataset = 39 px); NWD suaviza.
#   Ref: NWD-RKA +4.3 AP em AI-TOD-v2 (ISPRS J. 2022, arXiv:2206.13996).
# Protocolo: mesma receita do E0 (modelo, epocas, batch, imgsz, seed do
#   config). Decisao em VAL; teste fica intocado.
#
# Submeter:   sbatch fly-det/slurm/E10_yolo26m_nwd.sh
# Monitorar:  squeue -u $USER ; scontrol show job <job_id>
# =============================================================================
#SBATCH --job-name=E10-yolo26m-nwd
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
DATA_YAML="${USER_ROOT}/datasets/DS-F2_v8.1.1/data.yaml"
WORK="${FLYDET_WORK_DIR:-${USER_ROOT}/flydet_runs/E10_yolo26m_nwd}"
SECRETS="${USER_ROOT}/secrets/wandb.env"

mkdir -p "${WORK}" "${WORK}/models" "${REPO}/slurm_logs"
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

# ------------------------- hiperparâmetros E10 -------------------------------
# IDENTICOS ao E0 (mudanca unica = patch NWD). Nao tocar.
MODEL="${MODEL:-yolo26m.pt}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-1920}"
PATIENCE="${PATIENCE:-30}"
WORKERS="${WORKERS:-12}"
WANDB_PROJECT="${WANDB_PROJECT:-fly-species}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-E10_yolo26m_nwd}"

# NWD: peso da mistura com CIoU e constante de escala (calibrada no dataset:
# media sqrt(w*h) = 44.4 px nos 59.346 boxes de treino).
export FLYDET_NWD_W="${FLYDET_NWD_W:-0.5}"
export FLYDET_NWD_C="${FLYDET_NWD_C:-44.0}"
# FLYDET_NWD_MODE (nwd|gcd) e FLYDET_RFLA (0|1) sao propagados se setados
# (ex.: E10b usa gcd, E14 usa rfla). E0 puro: FLYDET_NWD_MODE=off desliga o patch.
export FLYDET_NWD_MODE="${FLYDET_NWD_MODE:-nwd}"
export FLYDET_RFLA="${FLYDET_RFLA:-0}"
export FLYDET_SEED="${FLYDET_SEED:-}"

echo "============================== E10 NWD =================================="
echo "Job:        ${SLURM_JOB_ID:-N/A} @ ${SLURM_NODELIST:-N/A}"
echo "Modelo:     ${MODEL}  imgsz=${IMGSZ}  batch=${BATCH}  epochs=${EPOCHS}"
echo "NWD:        w=${FLYDET_NWD_W}  C=${FLYDET_NWD_C}px"
echo "W&B:        ${WANDB_ENTITY}/${WANDB_PROJECT}  run=${WANDB_RUN_NAME}"
echo "Início:     $(date)"
echo "========================================================================="

# ---------------- pre-check (fail-fast): patch aplica e loss roda ------------
apptainer exec \
  --bind "${USER_ROOT}:${USER_ROOT}" \
  --env FLYDET_NWD_W="${FLYDET_NWD_W}" \
  --env FLYDET_NWD_C="${FLYDET_NWD_C}" \
  --env FLYDET_NWD_MODE="${FLYDET_NWD_MODE}" \
  --env FLYDET_RFLA="${FLYDET_RFLA}" \
  --env FLYDET_SEED="${FLYDET_SEED}" \
  "${IMAGE}" \
  /opt/venv/bin/python - <<'PRECHECK'
import sys
sys.path.insert(0, "/raid/user_marcospaulo/fly-det/fly-det/scripts")
import torch
from nwd_patch import apply_nwd_patch
apply_nwd_patch()
from ultralytics.utils.loss import BboxLoss
bl = BboxLoss(reg_max=16)
B, N, R = 1, 8, 16
li, ld = bl(
    torch.rand(B, N, 4 * R),
    torch.tensor([[[0., 0., 40., 40.]] * N]).float(),
    torch.rand(N, 2) * 100,
    torch.tensor([[[1., 1., 41., 41.]] * N]).float(),
    torch.rand(B, N, 7),
    torch.tensor(5.0),
    torch.ones(B, N, dtype=torch.bool),
    torch.tensor([1920., 1920.]),
    torch.tensor([8.0]),
)
assert torch.isfinite(li) and torch.isfinite(ld), "NWD loss nao finita"
print("PRECHECK_NWD_OK")
PRECHECK

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
  --env FLYDET_NWD_W="${FLYDET_NWD_W}" \
  --env FLYDET_NWD_C="${FLYDET_NWD_C}" \
  --env FLYDET_NWD_MODE="${FLYDET_NWD_MODE}" \
  --env FLYDET_RFLA="${FLYDET_RFLA}" \
  --env FLYDET_SEED="${FLYDET_SEED}" \
  "${IMAGE}" \
  /opt/venv/bin/python "${REPO}/fly-det/scripts/train_nwd.py" \
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
