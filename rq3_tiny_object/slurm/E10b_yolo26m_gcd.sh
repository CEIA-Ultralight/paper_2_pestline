#!/usr/bin/env bash
# =============================================================================
# E10b — YOLO26m + GCD (Gaussian Combined Distance) | DS-F2_v8.1.1
# Variante do E10: mesma infra do patch NWD, mas metrica com invariancia de
#   escala (C local por caixa em vez de C=44px fixo). Ref: GRSL 2025
#   (arXiv:2510.27649), SOTA em AI-TOD-v2.
# Protocolo: identico ao E10 (mudanca unica = FLYDET_NWD_MODE=gcd).
# Submeter:   sbatch fly-det/slurm/E10b_yolo26m_gcd.sh
# =============================================================================
#SBATCH --job-name=E10b-yolo26m-gcd
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail

USER_ROOT="/raid/user_marcospaulo"
export FLYDET_NWD_MODE="${FLYDET_NWD_MODE:-gcd}"
export FLYDET_NWD_W="${FLYDET_NWD_W:-0.5}"
export FLYDET_NWD_C="${FLYDET_NWD_C:-44.0}"   # ignorado no modo gcd, mantido p/ log
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-E10b_yolo26m_gcd}"
export FLYDET_WORK_DIR="${FLYDET_WORK_DIR:-${USER_ROOT}/flydet_runs/E10b_yolo26m_gcd}"

exec bash "${USER_ROOT}/fly-det/fly-det/slurm/E10_yolo26m_nwd.sh"
