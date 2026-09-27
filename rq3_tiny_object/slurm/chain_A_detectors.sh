#!/usr/bin/env bash
# =============================================================================
# chain_A_detectors.sh — Cadeia A (b200n1): detectores principais, multi-seed.
#
#   E0  (yolo26m baseline, FLYDET_NWD_MODE=off)      seeds 0,1,2
#   E10 (yolo26m + NWD w=0.5, C=44, scope=both)      seeds 0,1,2
# Receita identica (common/slurm/train_detector_b200.sh). Apos cada treino:
# avaliacao no TEST -> $RUNS_ROOT/<RUN>/test_eval/ (per_instance.csv, metrics.json).
#
# Reentrante: cada etapa pula se $RUNS_ROOT/<RUN>/DONE existe; retoma de last.pt.
# Preempcao/manutencao: SIGUSR1 -> rc 75 -> ressubmeter o MESMO comando.
#
# Submissao (slurm-runner):  sbatch rq3_tiny_object/slurm/chain_A_detectors.sh
# Custo estimado: 6 treinos x ~0.5-0.7 GPU-h (B200) + 6 evals TEST (~minutos) ~= 3-4.5 GPU-h.
# Saidas: $RUNS_ROOT/{E0_s0,E0_s1,E0_s2,E10_nwd_w0.5_s0,_s1,_s2}/{best.pt,runs/,test_eval/,DONE}
# W&B: pestline/paper2-rq3-tiny-object runs E0_s*, E10_nwd_w0.5_s*, *_test_eval.
# =============================================================================
#SBATCH --job-name=p2-chainA-detectors
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
SEEDS="${SEEDS:-0 1 2}"

echo "CHAIN A inicio $(date) job=${SLURM_JOB_ID} seeds=${SEEDS}"

for s in ${SEEDS}; do
    RUN=E0_s${s}
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; continue; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE=off FLYDET_RFLA=0 \
        WANDB_RUN_NAME="${RUN}" bash "${TRAIN}"
done

for s in ${SEEDS}; do
    RUN=E10_nwd_w0.5_s${s}
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; continue; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE=nwd FLYDET_NWD_W=0.5 \
        FLYDET_NWD_C=44.0 FLYDET_NWD_SCOPE=both FLYDET_RFLA=0 WANDB_RUN_NAME="${RUN}" bash "${TRAIN}"
done

touch "${RUNS_ROOT}/CHAIN_A_DONE"
echo "CHAIN_A_DONE $(date)"
