#!/usr/bin/env bash
# =============================================================================
# chain_G_rq3_tuned_h100.sh — RQ3: bracos NWD ajustados + controle E0 na MESMA GPU (h100n2), 3 seeds.
# Responde ao major "receita unica sem ajuste": NWD (loss+assigner) com C e peso variados, contra um E0
# treinado na h100n2 (mesmo hardware; B200 e H100 diferem ~0.6 pp no mesmo checkpoint, entao so se compara dentro da H100).
#   G0 E0 s{0,1,2} | G1 NWD w0.5 C22 | G2 NWD w1.0 C44 | G3 NWD w0.5 C88   (todos scope=both, receita E0: 150 ep, batch 8, 1920)
# Idempotente (DONE) e reentrante (rc 75 -> requeue). Submissao: hold, jail, gpu_watch, Monitor, release.
# =============================================================================
#SBATCH --job-name=p2-chainG-rq3
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
export GPU_AUDIT_LOG="/raid/user_marcospaulo/slurm_logs/gpu_audit_${SLURM_JOB_ID:-x}.log"
source "${REPO}/common/gpu/slurm/gpu_env.sh"
set -euo pipefail
export FLYDET_PARTITION=h100n2
export IMAGE=/raid/user_marcospaulo/containers/flydet-train.sif
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
TRAIN="${REPO}/common/slurm/train_detector_b200.sh"
export WANDB_PROJECT="${WANDB_PROJECT_RQ3}"
export WANDB_RUN_GROUP="rq3_tuned_h100"
export WANDB_TAGS="rq3,tuned,h100,DS-F2_v9.0.5"
echo "CHAIN G inicio $(date) job=${SLURM_JOB_ID}"

train_step() {  # RUN SEED MODE W C
    local RUN="$1" s="$2" mode="$3" w="$4" c="$5"
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; return 0; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE="${mode}" FLYDET_NWD_W="${w}" \
        FLYDET_NWD_C="${c}" FLYDET_NWD_SCOPE=both FLYDET_RFLA=0 WANDB_RUN_NAME="${RUN}" bash "${TRAIN}"
}

for s in 0 1 2; do train_step "G0_e0_h100_s${s}" "${s}" off 0.5 44.0; done
for s in 0 1 2; do train_step "G1_nwd_w0.5_C22_s${s}" "${s}" nwd 0.5 22.0; done
for s in 0 1 2; do train_step "G2_nwd_w1.0_C44_s${s}" "${s}" nwd 1.0 44.0; done
for s in 0 1 2; do train_step "G3_nwd_w0.5_C88_s${s}" "${s}" nwd 0.5 88.0; done

touch "${RUNS_ROOT}/CHAIN_G_DONE"
echo "CHAIN_G_DONE $(date)"
