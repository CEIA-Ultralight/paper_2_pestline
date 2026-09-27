#!/usr/bin/env bash
# =============================================================================
# build_container_b200.sh — build do flydet-train-b200.sif em job Slurm CPU-only
# (sem --gres) na b200n1. Contexto de build = baseline/fly-det (%files relativo).
# Caches/tmp do Apptainer em /raid (regra do cluster). Apos o build, verificacao
# CPU (sem --nv): versoes torch/ultralytics + import fly_det.
#
# Submissao: sbatch common/slurm/build_container_b200.sh
# Fallback: se --fakeroot falhar no job, rodar o mesmo comando no login node (CPU).
# =============================================================================
#SBATCH --job-name=p2-build-b200
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
USER_ROOT="${USER_ROOT:-/raid/user_marcospaulo}"
REPO="${REPO:-${USER_ROOT}/paper_2_pestline}"
DEF="${REPO}/common/containers/flydet-train-b200.def"
SIF="${SIF:-${USER_ROOT}/containers/flydet-train-b200.sif}"
export APPTAINER_CACHEDIR="${USER_ROOT}/.cache/apptainer"
export APPTAINER_TMPDIR="${APPTAINER_CACHEDIR}/tmp"
export TMPDIR="${USER_ROOT}/tmp"
mkdir -p "${APPTAINER_TMPDIR}" "${TMPDIR}" "$(dirname "${SIF}")"

echo "BUILD inicio $(date) job=${SLURM_JOB_ID:-login} node=$(hostname) def=${DEF} sif=${SIF}"
echo "commit repo=$(git -C "${REPO}" rev-parse --short HEAD) baseline=$(git -C "${REPO}/baseline" rev-parse --short HEAD 2>/dev/null || echo '?')"
apptainer --version

cd "${REPO}/baseline/fly-det"
[[ -d fly_det && -f pyproject.toml ]] || { echo "ABORT: contexto de build invalido em $(pwd)"; exit 2; }

BUILD_OK=0
if apptainer build --force --fakeroot "${SIF}.tmp" "${DEF}"; then
    BUILD_OK=1
else
    echo "AVISO: build --fakeroot falhou (rc=$?); tentando build sem --fakeroot"
    if apptainer build --force "${SIF}.tmp" "${DEF}"; then BUILD_OK=1; fi
fi
[[ "${BUILD_OK}" == 1 ]] || { echo "BUILD_FAIL $(date)"; rm -f "${SIF}.tmp"; exit 1; }
mv -f "${SIF}.tmp" "${SIF}"
ls -la "${SIF}"

# Verificacao CPU (sem --nv; GPU so nos jobs de treino)
apptainer exec --bind "${USER_ROOT}:${USER_ROOT}" "${SIF}" /opt/venv/bin/python - <<'PY'
import torch, torchvision, ultralytics, albumentations, wandb, pandas, supervision, sklearn
import fly_det, fly_det.trainer, fly_det.sahi_inference
print(f"torch={torch.__version__} cuda_build={torch.version.cuda} torchvision={torchvision.__version__}")
print(f"ultralytics={ultralytics.__version__} albumentations={albumentations.__version__} wandb={wandb.__version__} pandas={pandas.__version__}")
# sem --nv nao ha driver: get_arch_list() vem vazio aqui; sm_100 e conferido no smoke (GPU)
assert ultralytics.__version__ == "8.4.142"
assert torch.version.cuda == "12.8", torch.version.cuda
print("BUILD_VERIFY_OK")
PY
echo "BUILD_OK $(date) ${SIF}"
