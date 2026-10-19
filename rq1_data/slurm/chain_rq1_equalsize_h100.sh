#!/usr/bin/env bash
# =============================================================================
# chain_rq1_equalsize_h100.sh — RQ1 controle de mesmo tamanho na h100n2 (dgx-H100-02), UM job, 1 GPU.
#   Faltantes: EQ869_s2, EQ103_s0..2 (subamostras aleatorias do TRAIN do v9.0.5; make_equalsize.py).
# Receita E0 (train_detector_b200.sh): yolo26m, 150 ep., batch 8, imgsz 1920, patience 30.
# Container cu124 (flydet-train.sif; ultralytics 8.4.142). GPU isolada por jail (common/gpu/slurm/gpu_env.sh).
# Reentrante (DONE por run, last.pt, SIGUSR1 -> rc 75 -> requeue).
# Submeter (protocolo): sbatch --hold rq1_data/slurm/chain_rq1_equalsize_h100.sh ;
#   nohup python common/gpu/gpu_watch.py --jobs <id> & (no node de login) ; Monitor ; scontrol release <id>
# =============================================================================
#SBATCH --job-name=p2-rq1-eq
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
export GPU_AUDIT_LOG="/raid/user_marcospaulo/slurm_logs/gpu_audit_${SLURM_JOB_ID:-x}.log"
source "${REPO}/common/gpu/slurm/gpu_env.sh"   # re-executa o script dentro do jail de GPU
set -euo pipefail
export FLYDET_PARTITION=h100n2
export IMAGE=/raid/user_marcospaulo/containers/flydet-train.sif
export RUNS_ROOT="${RUNS_ROOT:-/raid/user_marcospaulo/flydet_runs_rq1_h100}"
EQ="${EQ:-/raid/user_marcospaulo/datasets/rq1_equalsize}"
export DATA_ROOT="${EQ}/EQ103_s0"; export DATA_YAML="${DATA_ROOT}/data.yaml"
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
TRAIN="${REPO}/common/slurm/train_detector_b200.sh"
export WANDB_PROJECT="paper2-rq1-data"; export WANDB_RUN_GROUP="rq1-equalsize"
RUNLIST="${RUNLIST:-EQ869_s2 EQ103_s0 EQ103_s1 EQ103_s2}"
echo "RQ1 EQUALSIZE H100 inicio $(date) job=${SLURM_JOB_ID} runs=${RUNLIST} root=${RUNS_ROOT}"
for RUN in ${RUNLIST}; do
  if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; continue; fi
  s="${RUN##*_s}"
  chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE=off FLYDET_RFLA=0 \
      DATA_ROOT="${EQ}/${RUN}" DATA_YAML="${EQ}/${RUN}/data.yaml" \
      WANDB_PROJECT="${WANDB_PROJECT}" WANDB_RUN_GROUP="${WANDB_RUN_GROUP}" WANDB_RUN_NAME="${RUN}-h100" \
      WANDB_TAGS="rq1,equalsize,${RUN},h100,DS-F2_v9.0.5-test" bash "${TRAIN}"
done
touch "${RUNS_ROOT}/RQ1_EQ_CHAIN_DONE"; echo "RQ1_EQ_CHAIN_DONE $(date)"
