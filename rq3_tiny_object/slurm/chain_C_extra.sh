#!/usr/bin/env bash
# =============================================================================
# chain_C_extra.sh — Cadeia C (b200n1): ablacoes extras da RQ3. DEPENDE da cadeia A
# (exige $RUNS_ROOT/CHAIN_A_DONE; para o SAHI precisa dos best.pt de E0_s0..s2).
#
#   1) GCD w=0.5          seeds 1,2   (FLYDET_NWD_MODE=gcd)          -> E11_gcd_w0.5_s{1,2}
#   2) NWD scope=loss     seed 0      (FLYDET_NWD_SCOPE=loss)        -> E10_nwd_w0.5_loss_s0
#   3) NWD scope=assigner seed 0      (FLYDET_NWD_SCOPE=assigner)    -> E10_nwd_w0.5_assigner_s0
#   4) NWD + RFLA         seed 0      (FLYDET_RFLA=1)                -> E14_nwd_rfla_s0
#   5) SAHI (fly_det.sahi_inference) nos 3 E0 no TEST               -> $RUNS_ROOT/sahi_E0_s{0,1,2}
# Receita de treino identica a cadeia A (train_detector_b200.sh; cada treino ja
# roda o TEST eval por instancia).
#
# SAHI: mesmo protocolo de rq3_tiny_object/slurm/E2_eval_sahi.sh (tile 512x512,
# --imgsz 512, overlap 0.25, conf 0.001, eval-iou 0.5). Como E0 treina a imgsz 1920
# em frames 1920x1080 (escala nativa), um tile 512 inferido a 512 preserva a escala
# de pixels dos objetos — comparacao justa plain vs SAHI. Ajustavel via SAHI_TILE /
# SAHI_IMGSZ / SAHI_OVERLAP; o valor efetivo fica em $OUT/sahi_args.json.
#
# Submissao: sbatch rq3_tiny_object/slurm/chain_C_extra.sh   (apos CHAIN_A_DONE)
# Custo estimado: 5 treinos x ~0.5-0.7 GPU-h + 3 SAHI (~10-20 min cada) ~= 3-4.5 GPU-h.
# W&B: pestline/paper2-rq3-tiny-object runs E11_*, E10_*_loss_s0, E10_*_assigner_s0, E14_*, sahi_E0_s*.
# =============================================================================
#SBATCH --job-name=p2-chainC-extra
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../common/slurm/b200_common.sh"
b200_guard
TRAIN="${REPO}/common/slurm/train_detector_b200.sh"
export WANDB_PROJECT="${WANDB_PROJECT_RQ3}"
SAHI_TILE="${SAHI_TILE:-512}"
SAHI_IMGSZ="${SAHI_IMGSZ:-512}"
SAHI_OVERLAP="${SAHI_OVERLAP:-0.25}"

step_done "${RUNS_ROOT}/CHAIN_A_DONE" || { echo "ABORT: cadeia A nao concluida (${RUNS_ROOT}/CHAIN_A_DONE ausente)"; exit 2; }
echo "CHAIN C inicio $(date) job=${SLURM_JOB_ID}"

train_step() {  # RUN SEED MODE SCOPE RFLA
    local RUN="$1" s="$2" mode="$3" scope="$4" rfla="$5"
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; return 0; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE="${mode}" FLYDET_NWD_W=0.5 \
        FLYDET_NWD_C=44.0 FLYDET_NWD_SCOPE="${scope}" FLYDET_RFLA="${rfla}" WANDB_RUN_NAME="${RUN}" bash "${TRAIN}"
}

# 1) GCD seeds 1,2 (seed 0 existe da campanha anterior; se nao, adicionar "0" em GCD_SEEDS)
for s in ${GCD_SEEDS:-1 2}; do train_step "E11_gcd_w0.5_s${s}" "${s}" gcd both 0; done
# 2-3) escopo NWD (so loss / so assigner), seed 0
train_step "E10_nwd_w0.5_loss_s0"     0 nwd loss     0
train_step "E10_nwd_w0.5_assigner_s0" 0 nwd assigner 0
# 4) NWD + RFLA, seed 0
train_step "E14_nwd_rfla_s0"          0 nwd both     1

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

touch "${RUNS_ROOT}/CHAIN_C_DONE"
echo "CHAIN_C_DONE $(date)"
