#!/usr/bin/env bash
# CPU-only unit tests in Slurm, not a second training instance.
#SBATCH --job-name=flydet-followup-tests-cpu
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$REPO/fly-det/scripts"
export ARCHITECTURE_ROI_TORCH_TESTS=1
export TORCH_HOME="$ROOT/.cache/torch" XDG_CACHE_HOME="$ROOT/cache/architecture_lab/xdg"
export TMPDIR="$ROOT/cache/architecture_lab/tmp"
mkdir -p "$TMPDIR" "$ROOT/cache/architecture_lab/home"
apptainer exec --bind "$ROOT:$ROOT" --home "$ROOT/cache/architecture_lab/home" \
    --pwd "$REPO" "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python \
    -m unittest discover -s "$REPO/tests" -p 'test_architecture_*.py'