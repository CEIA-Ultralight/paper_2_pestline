#!/usr/bin/env bash
# =============================================================================
# chain_rq1_ladder.sh — RQ1 (b200n1), UM job, 1 GPU: escada de dados no TEST fixo do DS-F2_v9.0.5.
#   Bracos: L0 (v5.1) | L1 (+v6.0,v7) | L2 (+v8,v9; sem copy-paste) | L2ORG (L2 + DS-ORG)
#   (L3 = v9.0.5 completo = E0_s{0,1,2} da RQ2, ja treinado em flydet_runs_v9; nao se retreina)
# Receita identica a E0 (common/slurm/train_detector_b200.sh): yolo26m, 150 ep., batch 8, imgsz 1920, patience 30.
# Ordem seed-major (seed 0 de todos os bracos primeiro) para que resultados parciais ja sejam comparaveis.
# Reentrante: cada run pula se $RUNS_ROOT/<RUN>/DONE existe; retoma de last.pt; SIGUSR1 -> rc 75 -> requeue.
# Subconjuntos: python rq1_data/make_ladder.py (hardlinks em /raid/user_marcospaulo/datasets/rq1_ladder/<ARM>).
# W&B: entity pestline, projeto paper2-rq1-data (so RQ1).
# Submeter:  sbatch rq1_data/slurm/chain_rq1_ladder.sh      Custo: 12 treinos x ~0,5-0,7 GPU-h
# =============================================================================
#SBATCH --job-name=p2-rq1-ladder
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
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
export RUNS_ROOT="${RUNS_ROOT:-/raid/user_marcospaulo/flydet_runs_rq1}"
LADDER="${LADDER:-/raid/user_marcospaulo/datasets/rq1_ladder}"
export DATA_ROOT="${LADDER}/L2"; export DATA_YAML="${DATA_ROOT}/data.yaml"   # so p/ o b200_guard; cada run troca abaixo
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
TRAIN="${REPO}/common/slurm/train_detector_b200.sh"
export WANDB_PROJECT="paper2-rq1-data"
export WANDB_RUN_GROUP="rq1-ladder"
SEEDS="${SEEDS:-0 1 2}"; ARMS="${ARMS:-L0 L1 L2 L2ORG}"
echo "RQ1 LADDER inicio $(date) job=${SLURM_JOB_ID} seeds=${SEEDS} arms=${ARMS} runs=${RUNS_ROOT}"

for s in ${SEEDS}; do
  for arm in ${ARMS}; do
    RUN="RQ1_${arm}_s${s}"
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; continue; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE=off FLYDET_RFLA=0 \
        DATA_ROOT="${LADDER}/${arm}" DATA_YAML="${LADDER}/${arm}/data.yaml" \
        WANDB_PROJECT="${WANDB_PROJECT}" WANDB_RUN_GROUP="${WANDB_RUN_GROUP}" WANDB_RUN_NAME="${RUN}" \
        WANDB_TAGS="rq1,ladder,${arm},DS-F2_v9.0.5-test" bash "${TRAIN}"
  done
done

touch "${RUNS_ROOT}/RQ1_CHAIN_DONE"
echo "RQ1_CHAIN_DONE $(date)"
