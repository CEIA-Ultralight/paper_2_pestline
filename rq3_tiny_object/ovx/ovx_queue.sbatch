#!/usr/bin/env bash
# =============================================================================
# ovx_queue.sbatch — fila de treinos da RQ3 no ovx-l40s-01 (partição ovx01, L40S 46 GB), via Slurm + Apptainer.
# Uso (RUNS = nomes de run_queue.sh, separados por espaço):
#   sbatch --job-name=rq3-A --export=ALL,RUNS="E11_gcd_w0.5_s0 E11_gcd_w0.5_s1" rq3_tiny_object/ovx/ovx_queue.sbatch
# Reentrante: run com DONE e pulado; interrompido retoma de last.pt (train_one.sh). SIGUSR1@300 -> rc 75 -> requeue.
# Sem /raid neste usuário: tudo sob $ROOT (home), fora do repo. Receita identica a dgx1/train_one.sh.
# =============================================================================
#SBATCH --partition=ovx01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/user_marcospaulo/pestline_rq3/logs/slurm_%x_%j.out
set -euo pipefail
: "${SLURM_JOB_ID:?Requer sbatch (GPU so via Slurm)}"
: "${RUNS:?Defina RUNS=\"RUN1 RUN2 ...\"}"
ROOT="${ROOT:-${HOME}/pestline_rq3}"
REPO="${REPO:-${HOME}/paper_2_pestline}"
IMAGE="${IMAGE:-${ROOT}/containers/pestline-rq3.sif}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets/DS-F2_v9.0.5}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
LOGS="${ROOT}/logs"
SECRETS="${SECRETS:-${HOME}/lab/secrets/wandb.env}"
CACHE="${ROOT}/.cache"
WANDB_PROJECT="${WANDB_PROJECT:-paper2_rq3}"
mkdir -p "${RUNS_ROOT}" "${MODELS_DIR}" "${LOGS}" "${CACHE}"/{home,ultralytics,wandb,mpl,torch,xdg}
[[ -f "${SECRETS}" ]] || { echo "ABORT: ${SECRETS} ausente"; exit 2; }
set -a; source "${SECRETS}"; set +a      # chave só no ambiente (herdada pelo apptainer), nunca em --env nem em log

spec() {  # idem run_queue.sh
    local run="$1" s="${1##*_s}"
    case "${run}" in
        E0_s[0-9])                 printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=off ;;
        E10_nwd_w0.5_s[0-9])       printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=both ;;
        E11_gcd_w0.5_s[0-9])       printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=gcd FLYDET_NWD_SCOPE=both ;;
        E14_nwd_rfla_s[0-9])       printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=both FLYDET_RFLA=1 ;;
        E15_nwd_loss_s[0-9])       printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=loss ;;
        E16_nwd_assigner_s[0-9])   printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=assigner ;;
        E20_wiou_s[0-9])           printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=wiou FLYDET_NWD_SCOPE=loss ;;
        E21_imgsz2560_s[0-9])      printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=off IMGSZ=2560 BATCH=4 ;;
        E22_yolo11m_s0)            printf '%s\n' FLYDET_SEED=0 FLYDET_NWD_MODE=off MODEL=yolo11m.pt ;;
        E1_p2_s[0-9])              printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=off MODEL=yolo26m.pt MODEL_CFG=yolo26m-p2.yaml ;;
        E7_yolo26l_s[0-9])         printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=off MODEL=yolo26l.pt ;;
        smoke_E0)                  printf '%s\n' FLYDET_SEED=0 FLYDET_NWD_MODE=off EPOCHS=1 PATIENCE=1 FLYDET_SAVE_PERIOD=1 ;;
        *) return 1 ;;
    esac
}
for RUN in ${RUNS}; do spec "${RUN}" >/dev/null || { echo "ABORT: run desconhecido: ${RUN}"; exit 2; }; done

CHILD=""
on_signal() { echo "SINAL $(date): encaminhando ao treino; rc 75 (requeue)"; [[ -z "${CHILD}" ]] || kill -TERM "${CHILD}" 2>/dev/null || true; wait "${CHILD}" 2>/dev/null || true; exit 75; }
trap on_signal USR1 TERM INT

echo "OVX QUEUE inicio $(date) job=${SLURM_JOB_ID} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} RUNS=${RUNS}"
for RUN in ${RUNS}; do
    if [[ -f "${RUNS_ROOT}/${RUN}/DONE" ]]; then echo "SKIP ${RUN}: DONE"; continue; fi
    envs=()
    while IFS= read -r kv; do envs+=(--env "${kv}"); done < <(spec "${RUN}")
    for v in BATCH IMGSZ EPOCHS PATIENCE WORKERS; do [[ -z "${!v:-}" ]] || envs+=(--env "${v}=${!v}"); done
    echo "########## $(date '+%F %T') ${RUN} ##########"
    apptainer exec --nv --cleanenv --no-home \
        --bind "${REPO}:${REPO}:ro" --bind "${DATA_ROOT}:${DATA_ROOT}" \
        --bind "${RUNS_ROOT}:${RUNS_ROOT}" --bind "${MODELS_DIR}:${MODELS_DIR}" --bind "${CACHE}:${CACHE}" \
        --env WANDB_API_KEY="${WANDB_API_KEY}" \
        --env HOME="${CACHE}/home" --env YOLO_CONFIG_DIR="${CACHE}/ultralytics" \
        --env WANDB_DIR="${CACHE}/wandb" --env WANDB_CACHE_DIR="${CACHE}/wandb" --env WANDB_CONFIG_DIR="${CACHE}/wandb" \
        --env MPLCONFIGDIR="${CACHE}/mpl" --env TORCH_HOME="${CACHE}/torch" --env XDG_CACHE_HOME="${CACHE}/xdg" \
        --env WANDB_ENTITY=pestline --env WANDB_PROJECT="${WANDB_PROJECT}" --env WANDB_MODE=online \
        --env WANDB_RUN_NAME="${RUN}" --env WANDB_TAGS="DS-F2_v9.0.5,paper2,ovx-l40s" \
        --env REPO="${REPO}" --env DATA_ROOT="${DATA_ROOT}" --env RUNS_ROOT="${RUNS_ROOT}" --env MODELS_DIR="${MODELS_DIR}" \
        --env RUN_NAME="${RUN}" --env FLYDET_GPU_PARTITIONS=ovx01 --env FLYDET_RAID="${ROOT}" \
        --env SLURM_JOB_ID="${SLURM_JOB_ID}" --env SLURM_JOB_PARTITION=ovx01 \
        "${envs[@]}" \
        "${IMAGE}" bash "${REPO}/rq3_tiny_object/dgx1/train_one.sh" >> "${LOGS}/${RUN}.log" 2>&1 &
    CHILD=$!
    wait "${CHILD}" || { rc=$?; echo "FALHA ${RUN} rc=${rc} (ver ${LOGS}/${RUN}.log)"; exit "${rc}"; }
    CHILD=""
    echo "########## $(date '+%F %T') FIM ${RUN} ##########"
done
echo "FILA_OK job=${SLURM_JOB_ID} $(date)"
