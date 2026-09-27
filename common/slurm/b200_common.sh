#!/usr/bin/env bash
# =============================================================================
# b200_common.sh — ambiente compartilhado dos jobs da particao b200n1 (paper 2).
# Uso: `source "$REPO/common/slurm/b200_common.sh"` no inicio de cada script.
#
# Define: USER_ROOT, REPO, IMAGE, DATA_ROOT, DATA_YAML, CROPS, RUNS_ROOT, LOGS,
#         SECRETS, caches (/raid), W&B (entity pestline; projects por RQ),
#         PY_PATH (PYTHONPATH dentro do container), funcoes:
#           b200_guard          — exige sbatch + particao b200n1
#           b200_apptainer ...  — apptainer exec --nv com binds/env padrao
#           chain_run CMD...    — executa etapa em background, encaminha
#                                 SIGUSR1/TERM/INT ao filho e sai com rc 75
#           step_done MARK      — true se o marcador de conclusao existe
#
# Decisao de layout: fly_det vem do CONTAINER (instalado no build a partir do
# submodule baseline/fly-det); o codigo do paper (rq*/, common/) vem do REPO via
# --bind $USER_ROOT + PYTHONPATH. Assim o baseline fica pinado no .sif e os
# scripts do paper podem evoluir sem rebuild.
# =============================================================================

USER_ROOT="${USER_ROOT:-/raid/user_marcospaulo}"
REPO="${REPO:-${USER_ROOT}/paper_2_pestline}"
IMAGE="${IMAGE:-${USER_ROOT}/containers/flydet-train-b200.sif}"
DATA_ROOT="${DATA_ROOT:-${USER_ROOT}/datasets/DS-F2_v8.1.1}"
DATA_YAML="${DATA_YAML:-${DATA_ROOT}/data.yaml}"
CROPS="${CROPS:-${USER_ROOT}/datasets/DS-F2_crops_pad75}"   # ImageFolder train/val/test, 7 classes
RUNS_ROOT="${RUNS_ROOT:-${USER_ROOT}/flydet_runs}"
LOGS="${LOGS:-${USER_ROOT}/slurm_logs}"
SECRETS="${SECRETS:-${USER_ROOT}/secrets/wandb.env}"
MODELS_DIR="${MODELS_DIR:-${USER_ROOT}/models}"

mkdir -p "${RUNS_ROOT}" "${LOGS}"

# ------------------------- caches -> /raid (regra do cluster) ----------------
export WANDB_DIR="${USER_ROOT}/wandb"
export WANDB_CACHE_DIR="${USER_ROOT}/.cache/wandb"
export WANDB_CONFIG_DIR="${USER_ROOT}/.config/wandb"
export HF_HOME="${USER_ROOT}/.cache/huggingface"
export TORCH_HOME="${USER_ROOT}/.cache/torch"
export PIP_CACHE_DIR="${USER_ROOT}/.cache/pip"
export YOLO_CONFIG_DIR="${USER_ROOT}/.config/Ultralytics"
export MPLCONFIGDIR="${USER_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${USER_ROOT}/.cache"
export TMPDIR="${USER_ROOT}/tmp"
export APPTAINER_CACHEDIR="${USER_ROOT}/cache/apptainer"
export APPTAINER_TMPDIR="${USER_ROOT}/cache/apptainer/tmp"
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${HF_HOME}" "${TORCH_HOME}" \
         "${PIP_CACHE_DIR}" "${YOLO_CONFIG_DIR}" "${MPLCONFIGDIR}" "${TMPDIR}" "${APPTAINER_TMPDIR}"

# ------------------------- credenciais W&B (fora do git) ---------------------
if [[ -f "${SECRETS}" ]]; then set -a; source "${SECRETS}"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-pestline}"
export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT_RQ2="${WANDB_PROJECT_RQ2:-paper2-rq2-cascade}"
WANDB_PROJECT_RQ3="${WANDB_PROJECT_RQ3:-paper2-rq3-tiny-object}"

# PYTHONPATH dentro do container: codigo do paper (fly_det ja esta no venv do .sif)
PY_PATH="${REPO}/rq3_tiny_object:${REPO}/rq3_tiny_object/patches:${REPO}/rq2_cascade:${REPO}/common"

b200_guard() {
    : "${SLURM_JOB_ID:?Requer sbatch (GPU so via Slurm)}"
    if [[ "${SLURM_JOB_PARTITION:-}" != "b200n1" ]]; then
        echo "ABORT: particao ${SLURM_JOB_PARTITION:-?} != b200n1" >&2; exit 2
    fi
    [[ -f "${IMAGE}" ]] || { echo "ABORT: container ausente: ${IMAGE} (ver common/containers/flydet-train-b200.def)" >&2; exit 2; }
    [[ -f "${DATA_YAML}" ]] || { echo "ABORT: data.yaml ausente: ${DATA_YAML}" >&2; exit 2; }
    export FLYDET_GPU_PARTITIONS="h100n2,b200n1"
}

# b200_apptainer [--pwd DIR] CMD...  -> apptainer exec --nv com binds/env padrao.
# Variaveis FLYDET_*/WANDB_* exportadas no host sao propagadas via --env.
# Segredos (WANDB_API_KEY etc.) NAO vao em --env (apareceriam em `ps`); o apptainer
# ja herda o ambiente do host, entao a chave chega ao container mesmo assim.
b200_apptainer() {
    local pwd_dir="${REPO}"
    if [[ "${1:-}" == "--pwd" ]]; then pwd_dir="$2"; shift 2; fi
    local envs=()
    local v
    for v in $(compgen -e | grep -E '^(FLYDET_|WANDB_|NWD_|CUDA_VISIBLE_DEVICES$|OMP_NUM_THREADS$)' \
                | grep -vE '(API_KEY|TOKEN|SECRET|PASSWORD)' || true); do
        envs+=(--env "${v}=${!v}")
    done
    apptainer exec --nv \
        --bind "${USER_ROOT}:${USER_ROOT}" \
        --pwd "${pwd_dir}" \
        --env PYTHONPATH="${PY_PATH}" \
        --env PYTHONUNBUFFERED=1 \
        --env HF_HOME="${HF_HOME}" --env TORCH_HOME="${TORCH_HOME}" \
        --env YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR}" --env MPLCONFIGDIR="${MPLCONFIGDIR}" \
        --env XDG_CACHE_HOME="${XDG_CACHE_HOME}" --env TMPDIR="${TMPDIR}" \
        "${envs[@]}" \
        "${IMAGE}" "$@"
}

# ------------------------- encadeamento com sinal ----------------------------
# Slurm envia SIGUSR1 300 s antes do fim (--signal=B:SIGUSR1@300). O trap
# encaminha ao filho (que salva last.pt e sai 75) e a cadeia sai 75 para ser
# ressubmetida (reentrante: cada etapa pula se ja concluida).
CHAIN_CHILD=''
CHAIN_INTERRUPTED=0
_chain_forward() { CHAIN_INTERRUPTED=1; [[ -z "${CHAIN_CHILD}" ]] || kill -USR1 "${CHAIN_CHILD}" 2>/dev/null || true; }
trap _chain_forward USR1 TERM INT

chain_run() {
    echo ""
    echo "########## $(date '+%F %T') — $* ##########"
    "$@" &
    CHAIN_CHILD=$!
    local rc=0
    wait "${CHAIN_CHILD}" || rc=$?
    if [[ "${CHAIN_INTERRUPTED}" == 1 ]]; then wait "${CHAIN_CHILD}" 2>/dev/null || true; echo "CHAIN_INTERRUPTED rc=75"; exit 75; fi
    CHAIN_CHILD=''
    if [[ "${rc}" == 75 ]]; then echo "CHAIN_STEP_PREEMPTED rc=75 — ressubmeter"; exit 75; fi
    if [[ "${rc}" != 0 ]]; then echo "CHAIN_ABORT: etapa falhou rc=${rc} — $*"; exit "${rc}"; fi
}

step_done() { [[ -f "$1" ]]; }
