#!/usr/bin/env bash
#SBATCH --job-name=flydet-arch-tracking-probe
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
export PYTHONPATH="$REPO/fly-det/scripts"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/architecture_lab/xdg"
export TMPDIR="$ROOT/cache/architecture_lab/tmp"
export MPLCONFIGDIR="$ROOT/cache/architecture_lab/matplotlib"
export CUDA_CACHE_PATH="$ROOT/cache/architecture_lab/cuda"
export YOLO_CONFIG_DIR="$ROOT/.config/Ultralytics"
export WANDB_DIR="$ROOT/wandb" WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/cache/architecture_lab/wandb/cache"
export WANDB_CONFIG_DIR="$ROOT/cache/architecture_lab/wandb/config"
export WANDB_DATA_DIR="$ROOT/cache/architecture_lab/wandb/data"
mkdir -p "$TMPDIR" "$ROOT/cache/architecture_lab/home"
set -a
source "$ROOT/secrets/wandb.env"
set +a
OUT="$ROOT/flydet_runs/architecture-tracking-probe-$SLURM_JOB_ID"
run() { apptainer exec --nv --bind "$ROOT:$ROOT" --home "$ROOT/cache/architecture_lab/home" \
    --pwd "$REPO" "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python "$@"; }
export ARCHITECTURE_ROI_TORCH_TESTS=1
run -m unittest discover -s "$REPO/tests" -p 'test_architecture_*.py'
run -m architecture_lab.train --data "$ROOT/datasets/DS-F2_crops_pad75" \
    --out "$OUT/baseline" --variant baseline --epochs 1 --batch 4 --img-size 384 \
    --workers 0 --limit-train 14 --limit-val 28 --max-hours .1 --wandb \
    --wandb-entity pestline --wandb-project fly-species --wandb-run-name "SMOKE_architecture_tracking_$SLURM_JOB_ID"
run -m architecture_lab.diagnose --run "$OUT/baseline" --samples 8 --batch 32
# All existing train/val boxes and image dimensions are checked before the night.
run -m architecture_lab.roi cache --data "$ROOT/datasets/DS-F2_v8.1.1" \
    --detector "$ROOT/flydet_runs/E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt" \
    --out "$OUT/roi_cache"
echo "TRACKING_AND_FULL_ROI_OK $OUT"