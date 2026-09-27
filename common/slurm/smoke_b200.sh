#!/usr/bin/env bash
# =============================================================================
# smoke_b200.sh — smoke test da particao b200n1 (~20 min GPU). RODAR ANTES das cadeias.
#
#   0) versoes torch/cuda/ultralytics, nome da GPU, capability (espera sm_100)
#   1) detector E0 com EPOCHS=1 (receita real: yolo26m, batch 8, imgsz 1920) + TEST eval
#   2) classificador ConvNeXt-T baseline 1 epoca em subconjunto (sem W&B)
#   3) avaliacao da cascata no VAL (5 imagens) com os dois checkpoints acima
# Imprime SMOKE_OK ao final. Saidas em $RUNS_ROOT/smoke_*/ (podem ser apagadas depois).
# Submissao: sbatch common/slurm/smoke_b200.sh
# =============================================================================
#SBATCH --job-name=p2-smoke-b200
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
# sbatch copia o script para /var/spool/slurmd/<job>/ -> nao usar dirname(BASH_SOURCE)
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
export B200_REQUEUE=0                       # smoke nunca re-enfileira
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
# Runs do smoke vao para um projeto W&B separado (nao poluir paper2-rq2/rq3)
export WANDB_PROJECT="${WANDB_PROJECT:-paper2-smoke}"
SMOKE_ROOT="${SMOKE_ROOT:-${RUNS_ROOT}/smoke_b200}"
mkdir -p "${SMOKE_ROOT}"
echo "SMOKE inicio $(date) job=${SLURM_JOB_ID} node=${SLURM_NODELIST:-?} image=${IMAGE} wandb=${WANDB_ENTITY}/${WANDB_PROJECT}"

# 0) ambiente (direto, sem chain_run: heredoc como stdin)
b200_apptainer /opt/venv/bin/python - <<'PY'
import torch, torchvision, ultralytics, wandb, pandas, albumentations
print(f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()}")
print(f"torchvision={torchvision.__version__} ultralytics={ultralytics.__version__} wandb={wandb.__version__} pandas={pandas.__version__} albumentations={albumentations.__version__}")
assert torch.cuda.is_available(), "CUDA indisponivel no job"
cap = torch.cuda.get_device_capability(0)
print(f"GPU={torch.cuda.get_device_name(0)} capability=sm_{cap[0]}{cap[1]} mem={torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GiB")
x = torch.randn(1024, 1024, device="cuda"); y = (x @ x).sum().item(); assert y == y
import fly_det.trainer, fly_det.sahi_inference, nwd_patch, architecture_lab.train, eval_cascade_matrix, stats  # noqa
print("imports OK (fly_det do container; rq*/common via PYTHONPATH)")
PY

# 1) detector 1 epoca (mesma receita, EPOCHS=1) + TEST eval
DET_RUN="smoke_E0"; DET_WORK="${SMOKE_ROOT}/${DET_RUN}"
if ! step_done "${DET_WORK}/DONE"; then
    chain_run env RUN_NAME="${DET_RUN}" FLYDET_WORK_DIR="${DET_WORK}" FLYDET_SEED=0 FLYDET_NWD_MODE=off \
        EPOCHS=1 PATIENCE=1 FLYDET_SAVE_PERIOD=1 WANDB_RUN_NAME="${DET_RUN}" \
        bash "${REPO}/common/slurm/train_detector_b200.sh"
fi
DET_BEST="${DET_WORK}/best.pt"; [[ -f "${DET_BEST}" ]] || { echo "SMOKE_FAIL: ${DET_BEST} ausente"; exit 1; }

# 2) classificador 1 epoca, subconjunto, sem W&B (contrato do architecture_lab: ambos os limites)
CLF_RUN="smoke_clf"; CLF_OUT="${SMOKE_ROOT}/${CLF_RUN}"
if [[ ! -f "${CLF_OUT}/best.pt" ]]; then
    chain_run env VARIANT=baseline SEED=42 RUN_NAME="${CLF_RUN}" OUT="${CLF_OUT}" \
        EPOCHS=1 PATIENCE=1 MAX_HOURS=0.3 LIMIT_TRAIN=256 LIMIT_VAL=128 \
        bash "${REPO}/rq2_cascade/slurm/train_classifier_b200.sh"
fi
CLF_BEST="${CLF_OUT}/best.pt"; [[ -f "${CLF_BEST}" ]] || { echo "SMOKE_FAIL: ${CLF_BEST} ausente"; exit 1; }

# 3) cascata no VAL, 5 imagens, sem W&B
CAS_OUT="${SMOKE_ROOT}/cascade_val"
chain_run b200_apptainer /opt/venv/bin/python "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
    --detector "${DET_BEST}" --detector-names smoke_E0 \
    --classifier "${CLF_BEST}" --classifier-names smoke_clf \
    --policies label_only product --data "${DATA_ROOT}" --split val \
    --out "${CAS_OUT}" --limit-images 5
[[ -f "${CAS_OUT}/summary.json" ]] || { echo "SMOKE_FAIL: ${CAS_OUT}/summary.json ausente"; exit 1; }
find "${CAS_OUT}" -name per_instance.csv | head -5

echo "SMOKE_OK $(date)"
