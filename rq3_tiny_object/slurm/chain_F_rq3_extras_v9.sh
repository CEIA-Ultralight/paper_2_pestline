#!/usr/bin/env bash
# =============================================================================
# chain_F_rq3_extras_v9.sh — RQ3 extras na v9.0.5 (b200n1), 3 seeds por variante, 1 job sequencial.
# Plano: docs/archive/PLANO_FINAL_2026-10-16.md (bloco A1/A2). Base: chain_C_extra.sh.
#   B2 Wise-IoU v3 s{0,1,2} | B1 E0 @2560 batch 4 s{0,1,2} | B3 YOLO11m s0 | C1 NWD scope loss/assigner s{0,1,2} | C2 yolo26l s{1,2}
# Receita identica a E0 (train_detector_b200.sh). Idempotente (DONE) e reentrante (rc 75 -> requeue).
# Pre-requisito: E0_s{0,1,2}/best.pt em $RUNS_ROOT (campanha v9).
# =============================================================================
#SBATCH --job-name=p2-chainF-rq3
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

for s in 0 1 2; do [[ -f "${RUNS_ROOT}/E0_s${s}/best.pt" ]] || { echo "ABORT: E0_s${s}/best.pt ausente"; exit 2; }; done
echo "CHAIN E inicio $(date) job=${SLURM_JOB_ID}"

train_step() {  # RUN SEED MODE SCOPE RFLA [ENV=VAL ...]  (extras: MODEL=, MODEL_CFG=, BATCH=)
    local RUN="$1" s="$2" mode="$3" scope="$4" rfla="$5"; shift 5
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; return 0; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE="${mode}" FLYDET_NWD_W=0.5 \
        FLYDET_NWD_C=44.0 FLYDET_NWD_SCOPE="${scope}" FLYDET_RFLA="${rfla}" WANDB_RUN_NAME="${RUN}" "$@" bash "${TRAIN}"
}

# B2) Wise-IoU v3 (so loss de caixa), seeds 0,1,2
for s in ${SEEDS:-0 1 2}; do train_step "E20_wiou_s${s}" "${s}" wiou loss 0; done
# B1) E0 em 2560 (batch 4), seeds 0,1,2
for s in ${SEEDS:-0 1 2}; do train_step "E21_imgsz2560_s${s}" "${s}" off both 0 IMGSZ=2560 BATCH=4; done
# B3) detector de outra familia (YOLO11m), seed 0
train_step "E22_yolo11m_s0" 0 off both 0 MODEL=yolo11m.pt
# C2) yolo26l baseline seeds 1,2 (s0 ja existe)
for s in 1 2; do train_step "E7_yolo26l_s${s}" "${s}" off both 0 MODEL=yolo26l.pt; done
# C1) NWD w=0.5 so loss / so assigner, seeds 0,1,2
for s in ${SEEDS:-0 1 2}; do train_step "E15_nwd_loss_s${s}" "${s}" nwd loss 0; done
for s in ${SEEDS:-0 1 2}; do train_step "E16_nwd_assigner_s${s}" "${s}" nwd assigner 0; done

touch "${RUNS_ROOT}/CHAIN_F_DONE"
echo "CHAIN_F_DONE $(date)"
