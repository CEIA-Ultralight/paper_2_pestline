#!/usr/bin/env bash
# =============================================================================
# chain_E_rq3_v9.sh — RQ3 na v9.0.5 (b200n1), 3 seeds por variante, 1 job sequencial.
# Plano: docs/archive/PLANO_FINAL_2026-10-16.md (bloco A1/A2). Base: chain_C_extra.sh.
#   1) NWD E10 w0.5 s{0,1,2}  2) GCD E11 s{0,1,2}  3) NWD+RFLA E14 s{0,1,2}
#   4) SAHI nos 3 E0   5) TTA + WBF nos 3 E0 (TEST)   6) P2 E1 s{0,1,2} (por ultimo: maior risco de OOM)
# Receita identica a E0 (train_detector_b200.sh). Idempotente (DONE) e reentrante (rc 75 -> requeue).
# Pre-requisito: E0_s{0,1,2}/best.pt em $RUNS_ROOT (campanha v9).
# =============================================================================
#SBATCH --job-name=p2-chainE-rq3
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
# sbatch copia o script para /var/spool/slurmd/<job>/ -> nao usar dirname(BASH_SOURCE)
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
TRAIN="${REPO}/common/slurm/train_detector_b200.sh"
export WANDB_PROJECT="${WANDB_PROJECT_RQ3}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-rq3_tiny}"
SAHI_TILE="${SAHI_TILE:-512}"
SAHI_IMGSZ="${SAHI_IMGSZ:-512}"
SAHI_OVERLAP="${SAHI_OVERLAP:-0.25}"
P2_MODEL_CFG="${P2_MODEL_CFG:-yolo26m-p2.yaml}"
P2_BATCH="${P2_BATCH:-8}"
TTA_IMGSZ="${TTA_IMGSZ:-1920}"
TTA_CONF="${TTA_CONF:-0.001}"

for s in 0 1 2; do [[ -f "${RUNS_ROOT}/E0_s${s}/best.pt" ]] || { echo "ABORT: E0_s${s}/best.pt ausente"; exit 2; }; done
echo "CHAIN E inicio $(date) job=${SLURM_JOB_ID}"

train_step() {  # RUN SEED MODE SCOPE RFLA [ENV=VAL ...]  (extras: MODEL=, MODEL_CFG=, BATCH=)
    local RUN="$1" s="$2" mode="$3" scope="$4" rfla="$5"; shift 5
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; return 0; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE="${mode}" FLYDET_NWD_W=0.5 \
        FLYDET_NWD_C=44.0 FLYDET_NWD_SCOPE="${scope}" FLYDET_RFLA="${rfla}" WANDB_RUN_NAME="${RUN}" "$@" bash "${TRAIN}"
}

# 1) NWD w=0.5 (E10), 2) GCD (E11), 3) NWD+RFLA (E14): seeds 0,1,2
for s in ${SEEDS:-0 1 2}; do train_step "E10_nwd_w0.5_s${s}" "${s}" nwd both 0; done
for s in ${SEEDS:-0 1 2}; do train_step "E11_gcd_w0.5_s${s}" "${s}" gcd both 0; done
for s in ${SEEDS:-0 1 2}; do train_step "E14_nwd_rfla_s${s}" "${s}" nwd both 1; done

# 5) SAHI nos 3 E0 (TEST)
for s in ${SEEDS:-0 1 2}; do
    RUN="sahi_E0_s${s}"
    OUT="${RUNS_ROOT}/${RUN}"
    BEST="${RUNS_ROOT}/E0_s${s}/best.pt"
    if step_done "${OUT}/DONE"; then echo "SKIP ${RUN}"; continue; fi
    [[ -f "${BEST}" ]] || { echo "ABORT: ${BEST} ausente (cadeia A)"; exit 2; }
    mkdir -p "${OUT}"
    chain_run b200_apptainer --pwd "${OUT}" /opt/venv/bin/python -m fly_det.sahi_inference \
        --model "${BEST}" --data-root "${DATA_ROOT}" --split test --out-dir "${OUT}" \
        --device cuda:0 --imgsz "${SAHI_IMGSZ}" --tile-wh "${SAHI_TILE}" "${SAHI_TILE}" \
        --overlap-ratio "${SAHI_OVERLAP}" "${SAHI_OVERLAP}" \
        --conf 0.001 --eval-iou 0.5 --draw-gt --embed-legend --cm-normalize \
        --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}"
    echo "JOB→RUN ${RUN}"
    echo "{\"model\": \"${BEST}\", \"tile\": ${SAHI_TILE}, \"imgsz\": ${SAHI_IMGSZ}, \"overlap\": ${SAHI_OVERLAP}}" > "${OUT}/sahi_args.json"
    touch "${OUT}/DONE"
done

# 7) TTA + WBF nos 3 E0 (TEST): por seed vanilla+TTA; ensemble WBF dos 3 (exige >= 2 detectores)
TTA_SPECS=()
for s in ${SEEDS:-0 1 2}; do
    RUN="ttawbf_E0_s${s}"
    OUT="${RUNS_ROOT}/${RUN}"
    BEST="${RUNS_ROOT}/E0_s${s}/best.pt"
    [[ -f "${BEST}" ]] || { echo "ABORT: ${BEST} ausente (cadeia A)"; exit 2; }
    TTA_SPECS+=("E0_s${s}=${BEST}")
    if step_done "${OUT}/DONE"; then echo "SKIP ${RUN}"; continue; fi
    mkdir -p "${OUT}"
    chain_run b200_apptainer --pwd "${OUT}" /opt/venv/bin/python "${REPO}/rq3_tiny_object/eval_wbf_tta.py" \
        --data "${DATA_YAML}" --split test --detectors "E0_s${s}=${BEST}" --out "${OUT}" \
        --imgsz "${TTA_IMGSZ}" --conf "${TTA_CONF}" \
        --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}"
    echo "JOB→RUN ${RUN}"
    touch "${OUT}/DONE"
done
if (( ${#TTA_SPECS[@]} >= 2 )); then
    RUN="ttawbf_E0_ensemble"
    OUT="${RUNS_ROOT}/${RUN}"
    if step_done "${OUT}/DONE"; then echo "SKIP ${RUN}"; else
        mkdir -p "${OUT}"
        chain_run b200_apptainer --pwd "${OUT}" /opt/venv/bin/python "${REPO}/rq3_tiny_object/eval_wbf_tta.py" \
            --data "${DATA_YAML}" --split test --detectors "${TTA_SPECS[@]}" --out "${OUT}" \
            --imgsz "${TTA_IMGSZ}" --conf "${TTA_CONF}" --wbf-only \
            --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}"
        echo "JOB→RUN ${RUN}"
        touch "${OUT}/DONE"
    fi
fi

# 4) Head P2 (baseline sem patch NWD), seeds 0,1,2
for s in ${P2_SEEDS:-0 1 2}; do
    train_step "E1_p2_s${s}" "${s}" off both 0 MODEL=yolo26m.pt MODEL_CFG="${P2_MODEL_CFG}" BATCH="${P2_BATCH}"
done

touch "${RUNS_ROOT}/CHAIN_E_DONE"
echo "CHAIN_E_DONE $(date)"
