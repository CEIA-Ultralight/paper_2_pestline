#!/usr/bin/env bash
# =============================================================================
# ovx_infer.sbatch — SAHI + TTA/WBF nos 3 E0 (split TEST) da campanha ovx (partição ovx01, L40S), via Slurm + Apptainer.
# Equivalente aos passos SAHI/TTA/WBF de rq3_tiny_object/slurm/chain_E_rq3_v9.sh. Idempotente (marcador DONE por passo).
# Uso: sbatch rq3_tiny_object/ovx/ovx_infer.sbatch    (pré-requisito: E0_s{0,1,2}/best.pt em $RUNS_ROOT)
# ensemble-boxes (WBF) não está no .sif: instalado em $CACHE/pylibs (pip --target) e exposto por PYTHONPATH.
# =============================================================================
#SBATCH --job-name=rq3-infer
#SBATCH --partition=ovx01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/user_marcospaulo/pestline_rq3/logs/slurm_%x_%j.out
set -euo pipefail
: "${SLURM_JOB_ID:?Requer sbatch (GPU so via Slurm)}"
ROOT="${ROOT:-${HOME}/pestline_rq3}"
REPO="${REPO:-${HOME}/paper_2_pestline}"
IMAGE="${IMAGE:-${ROOT}/containers/pestline-rq3.sif}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/datasets/DS-F2_v9.0.5}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
LOGS="${ROOT}/logs"
SECRETS="${SECRETS:-${HOME}/lab/secrets/wandb.env}"
CACHE="${ROOT}/.cache"
WANDB_PROJECT="${WANDB_PROJECT:-paper2_rq3}"
SAHI_TILE="${SAHI_TILE:-512}"; SAHI_IMGSZ="${SAHI_IMGSZ:-512}"; SAHI_OVERLAP="${SAHI_OVERLAP:-0.25}"
TTA_IMGSZ="${TTA_IMGSZ:-1920}"; TTA_CONF="${TTA_CONF:-0.001}"
DATA_YAML="${DATA_ROOT}/data.yaml"
mkdir -p "${LOGS}" "${CACHE}"/{home,ultralytics,wandb,mpl,torch,xdg}
[[ -f "${SECRETS}" ]] || { echo "ABORT: ${SECRETS} ausente"; exit 2; }
set -a; source "${SECRETS}"; set +a
for s in 0 1 2; do [[ -f "${RUNS_ROOT}/E0_s${s}/best.pt" ]] || { echo "ABORT: E0_s${s}/best.pt ausente"; exit 2; }; done

CHILD=""
on_signal() { echo "SINAL $(date): rc 75 (requeue)"; [[ -z "${CHILD}" ]] || kill -TERM "${CHILD}" 2>/dev/null || true; wait "${CHILD}" 2>/dev/null || true; exit 75; }
trap on_signal USR1 TERM INT

ctr() {  # ctr WORKDIR cmd...  — roda dentro do container, cwd = WORKDIR
    local wd="$1"; shift
    apptainer exec --nv --cleanenv --no-home --pwd "${wd}" \
        --bind "${REPO}:${REPO}:ro" --bind "${DATA_ROOT}:${DATA_ROOT}:ro" \
        --bind "${RUNS_ROOT}:${RUNS_ROOT}" --bind "${CACHE}:${CACHE}" \
        --env WANDB_API_KEY="${WANDB_API_KEY}" --env HOME="${CACHE}/home" --env YOLO_CONFIG_DIR="${CACHE}/ultralytics" \
        --env WANDB_DIR="${CACHE}/wandb" --env WANDB_CACHE_DIR="${CACHE}/wandb" --env WANDB_CONFIG_DIR="${CACHE}/wandb" \
        --env MPLCONFIGDIR="${CACHE}/mpl" --env TORCH_HOME="${CACHE}/torch" --env XDG_CACHE_HOME="${CACHE}/xdg" \
        --env WANDB_MODE=online --env PYTHONPATH="${CACHE}/pylibs:${REPO}/rq3_tiny_object" \
        "${IMAGE}" "$@" &
    CHILD=$!
    wait "${CHILD}" || { rc=$?; echo "FALHA rc=${rc}: $*"; exit "${rc}"; }
    CHILD=""
}

echo "OVX INFER inicio $(date) job=${SLURM_JOB_ID} node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"
TTA_SPECS=()
for s in 0 1 2; do
    BEST="${RUNS_ROOT}/E0_s${s}/best.pt"; TTA_SPECS+=("E0_s${s}=${BEST}")
    RUN="sahi_E0_s${s}"; OUT="${RUNS_ROOT}/${RUN}"
    if [[ -f "${OUT}/DONE" ]]; then echo "SKIP ${RUN}"; else
        echo "########## $(date '+%F %T') ${RUN} ##########"; mkdir -p "${OUT}"
        ctr "${OUT}" python -m fly_det.sahi_inference --model "${BEST}" --data-root "${DATA_ROOT}" --split test --out-dir "${OUT}" \
            --device cuda:0 --imgsz "${SAHI_IMGSZ}" --tile-wh "${SAHI_TILE}" "${SAHI_TILE}" \
            --overlap-ratio "${SAHI_OVERLAP}" "${SAHI_OVERLAP}" --conf 0.001 --eval-iou 0.5 --draw-gt --embed-legend --cm-normalize \
            --wandb --wandb-entity pestline --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}" >> "${LOGS}/${RUN}.log" 2>&1
        echo "{\"model\": \"${BEST}\", \"tile\": ${SAHI_TILE}, \"imgsz\": ${SAHI_IMGSZ}, \"overlap\": ${SAHI_OVERLAP}}" > "${OUT}/sahi_args.json"
        touch "${OUT}/DONE"; echo "########## $(date '+%F %T') FIM ${RUN} ##########"
    fi
    RUN="ttawbf_E0_s${s}"; OUT="${RUNS_ROOT}/${RUN}"
    if [[ -f "${OUT}/DONE" ]]; then echo "SKIP ${RUN}"; else
        echo "########## $(date '+%F %T') ${RUN} ##########"; mkdir -p "${OUT}"
        ctr "${OUT}" python "${REPO}/rq3_tiny_object/eval_wbf_tta.py" --data "${DATA_YAML}" --split test \
            --detectors "E0_s${s}=${BEST}" --out "${OUT}" --imgsz "${TTA_IMGSZ}" --conf "${TTA_CONF}" \
            --wandb --wandb-entity pestline --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}" >> "${LOGS}/${RUN}.log" 2>&1
        touch "${OUT}/DONE"; echo "########## $(date '+%F %T') FIM ${RUN} ##########"
    fi
done
RUN="ttawbf_E0_ensemble"; OUT="${RUNS_ROOT}/${RUN}"
if [[ -f "${OUT}/DONE" ]]; then echo "SKIP ${RUN}"; else
    echo "########## $(date '+%F %T') ${RUN} ##########"; mkdir -p "${OUT}"
    ctr "${OUT}" python "${REPO}/rq3_tiny_object/eval_wbf_tta.py" --data "${DATA_YAML}" --split test \
        --detectors "${TTA_SPECS[@]}" --out "${OUT}" --imgsz "${TTA_IMGSZ}" --conf "${TTA_CONF}" --wbf-only \
        --wandb --wandb-entity pestline --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}" >> "${LOGS}/${RUN}.log" 2>&1
    touch "${OUT}/DONE"; echo "########## $(date '+%F %T') FIM ${RUN} ##########"
fi
echo "FILA_OK job=${SLURM_JOB_ID} $(date)"
