#!/usr/bin/env bash
# =============================================================================
# run_queue.sh — fila de treinos da RQ3 na dgx1 (Docker, 1 GPU por fila).
#
# Uso:
#   GPU=0 bash rq3_tiny_object/dgx1/run_queue.sh E0_s0 E10_nwd_w0.5_s0 ...
# Em background (sobrevive ao logout):
#   GPU=0 nohup bash rq3_tiny_object/dgx1/run_queue.sh E0_s0 E10_nwd_w0.5_s0 > ~/pestline/logs/fila_gpu0.log 2>&1 &
#
# Runs conhecidos (mesmos nomes das cadeias A/C da B200):
#   E0_s{0,1,2}                 baseline yolo26m (sem patch)
#   E10_nwd_w0.5_s{0,1,2}       NWD w=0,5 C=44 scope=both
#   E11_gcd_w0.5_s{0,1,2}       GCD w=0,5
#   E10_nwd_w0.5_loss_s0        NWD so na loss
#   E10_nwd_w0.5_assigner_s0    NWD so no assigner
#   E14_nwd_rfla_s0             NWD + RFLA
#   E1_p2_s{0,1,2}              head P2 (yolo26m-p2.yaml, pretrained yolo26m.pt)
#   E7_yolo26l_s0               yolo26l baseline
#   smoke_E0                    1 epoca (confere GPU, memoria, W&B e o TEST eval)
#
# Reentrante: run com $RUNS_ROOT/<RUN>/DONE e pulado; run interrompido retoma de last.pt.
# Para no primeiro run que falhar (ex.: OOM -> rodar de novo com BATCH=4 em TODOS os runs).
# =============================================================================
set -euo pipefail

GPU="${GPU:?Defina GPU (indice da V100 livre, ver nvidia-smi)}"
ROOT="${ROOT:-${HOME}/pestline}"
REPO="${REPO:-${ROOT}/paper_2_pestline}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets/DS-F2_v9.0.5}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/flydet_runs_rq3_v9_dgx1}"
MODELS_DIR="${MODELS_DIR:-${ROOT}/models}"
LOGS="${LOGS:-${ROOT}/logs}"
SECRETS="${SECRETS:-${ROOT}/secrets/wandb.env}"     # uma linha: WANDB_API_KEY=...
IMAGE="${IMAGE:-pestline-rq3:cu124}"
WANDB_PROJECT="${WANDB_PROJECT:-paper2_rq3}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"                 # opcional: nome do run no W&B (so com 1 run na fila)
CACHE="${RUNS_ROOT}/.cache"

[[ $# -gt 0 ]] || { echo "Uso: GPU=<n> bash $0 RUN [RUN ...]"; exit 2; }
[[ -f "${SECRETS}" ]] || { echo "ABORT: ${SECRETS} ausente (WANDB_API_KEY=...)"; exit 2; }
[[ -f "${DATA_ROOT}/data.yaml" ]] || { echo "ABORT: ${DATA_ROOT}/data.yaml ausente"; exit 2; }
mkdir -p "${RUNS_ROOT}" "${MODELS_DIR}" "${LOGS}" "${CACHE}"/{home,ultralytics,wandb,mpl,torch,xdg}

# spec RUN -> variaveis do train_one.sh (uma por linha)
spec() {
    local run="$1" s="${1##*_s}"
    case "${run}" in
        E0_s[0-9])                 printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=off ;;
        E10_nwd_w0.5_s[0-9])       printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=both ;;
        E11_gcd_w0.5_s[0-9])       printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=gcd FLYDET_NWD_SCOPE=both ;;
        E10_nwd_w0.5_loss_s0)      printf '%s\n' FLYDET_SEED=0 FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=loss ;;
        E10_nwd_w0.5_assigner_s0)  printf '%s\n' FLYDET_SEED=0 FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=assigner ;;
        E14_nwd_rfla_s0)           printf '%s\n' FLYDET_SEED=0 FLYDET_NWD_MODE=nwd FLYDET_NWD_SCOPE=both FLYDET_RFLA=1 ;;
        E1_p2_s[0-9])              printf '%s\n' "FLYDET_SEED=${s}" FLYDET_NWD_MODE=off MODEL=yolo26m.pt MODEL_CFG=yolo26m-p2.yaml ;;
        E7_yolo26l_s0)             printf '%s\n' FLYDET_SEED=0 FLYDET_NWD_MODE=off MODEL=yolo26l.pt ;;
        smoke_E0)                  printf '%s\n' FLYDET_SEED=0 FLYDET_NWD_MODE=off EPOCHS=1 PATIENCE=1 FLYDET_SAVE_PERIOD=1 ;;
        *) return 1 ;;
    esac
}

for RUN in "$@"; do spec "${RUN}" >/dev/null || { echo "ABORT: run desconhecido: ${RUN}"; exit 2; }; done
[[ -z "${WANDB_RUN_NAME}" || $# -eq 1 ]] || { echo "ABORT: WANDB_RUN_NAME so com 1 run na fila"; exit 2; }

for RUN in "$@"; do
    if [[ -f "${RUNS_ROOT}/${RUN}/DONE" ]]; then echo "SKIP ${RUN}: DONE"; continue; fi
    envs=()
    while IFS= read -r kv; do envs+=(-e "${kv}"); done < <(spec "${RUN}")
    # Overrides opcionais vindos do host (ex.: BATCH=4 se der OOM; vale para a fila inteira)
    for v in BATCH EPOCHS PATIENCE WORKERS; do [[ -z "${!v:-}" ]] || envs+=(-e "${v}=${!v}"); done

    echo "########## $(date '+%F %T') GPU=${GPU} ${RUN} ##########"
    # rq2_cascade/eval_yolo_cascade.allocation_guard foi escrito para a B200 (exige Slurm + Apptainer);
    # na dgx1 a GPU e usada via Docker, entao as variaveis abaixo identificam esse ambiente para o guard.
    docker run --rm --name "rq3-${RUN}-gpu${GPU}" \
        --gpus "\"device=${GPU}\"" --ipc=host \
        --user "$(id -u):$(id -g)" \
        -v "${REPO}:${REPO}:ro" \
        -v "${DATA_ROOT}:${DATA_ROOT}:ro" \
        -v "${RUNS_ROOT}:${RUNS_ROOT}" \
        -v "${MODELS_DIR}:${MODELS_DIR}" \
        --env-file "${SECRETS}" \
        -e HOME="${CACHE}/home" \
        -e YOLO_CONFIG_DIR="${CACHE}/ultralytics" \
        -e WANDB_DIR="${CACHE}/wandb" -e WANDB_CACHE_DIR="${CACHE}/wandb" -e WANDB_CONFIG_DIR="${CACHE}/wandb" \
        -e MPLCONFIGDIR="${CACHE}/mpl" -e TORCH_HOME="${CACHE}/torch" -e XDG_CACHE_HOME="${CACHE}/xdg" \
        -e WANDB_ENTITY=pestline -e WANDB_PROJECT="${WANDB_PROJECT}" -e WANDB_MODE=online \
        -e WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN}}" \
        -e WANDB_TAGS="DS-F2_v9.0.5,paper2,dgx1-v100" \
        -e REPO="${REPO}" -e DATA_ROOT="${DATA_ROOT}" -e RUNS_ROOT="${RUNS_ROOT}" -e MODELS_DIR="${MODELS_DIR}" \
        -e RUN_NAME="${RUN}" \
        -e SLURM_JOB_ID="dgx1-docker-${RUN}" -e SLURM_JOB_PARTITION=dgx1-docker \
        -e FLYDET_GPU_PARTITIONS=dgx1-docker -e APPTAINER_NAME="docker:${IMAGE}" \
        "${envs[@]}" \
        "${IMAGE}" bash "${REPO}/rq3_tiny_object/dgx1/train_one.sh" 2>&1 | tee -a "${LOGS}/${RUN}.log"
    echo "########## $(date '+%F %T') FIM ${RUN} ##########"
done
echo "FILA_OK GPU=${GPU} $(date)"
