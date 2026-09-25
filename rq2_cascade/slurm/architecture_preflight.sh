#!/usr/bin/env bash
#SBATCH --job-name=flydet-arch-preflight
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
export ARCHITECTURE_ROI_TORCH_TESTS=1
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/cache/architecture_lab/xdg"
export TMPDIR="$ROOT/cache/architecture_lab/tmp"
export MPLCONFIGDIR="$ROOT/cache/architecture_lab/matplotlib"
export CUDA_CACHE_PATH="$ROOT/cache/architecture_lab/cuda"
export YOLO_CONFIG_DIR="$ROOT/.config/Ultralytics"
mkdir -p "$TMPDIR" "$ROOT/cache/architecture_lab/home"
OUT="$ROOT/flydet_runs/architecture-preflight-$SLURM_JOB_ID"
run() { apptainer exec --nv --bind "$ROOT:$ROOT" --home "$ROOT/cache/architecture_lab/home" \
    --pwd "$REPO" "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python "$@"; }
run -m unittest discover -s "$REPO/tests" -p 'test_architecture_*.py' -v
for variant in baseline supcon arcface parts hires; do
    run -m architecture_lab.train --data "$ROOT/datasets/DS-F2_crops_pad75" \
        --out "$OUT/$variant" --variant "$variant" --epochs 1 --batch 4 --img-size 384 \
        --workers 0 --limit-train 8 --limit-val 8 --no-pretrained --max-hours .1
    run -m architecture_lab.train --data "$ROOT/datasets/DS-F2_crops_pad75" \
        --out "$OUT/$variant" --variant "$variant" --epochs 1 --batch 4 --img-size 384 \
        --workers 0 --limit-train 8 --limit-val 8 --no-pretrained --max-hours .1 --resume
done
run -m architecture_lab.roi cache --data "$ROOT/datasets/DS-F2_v8.1.1" \
    --detector "$ROOT/flydet_runs/E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt" \
    --out "$OUT/roi_cache" --limit-images 1
run -m architecture_lab.roi train --cache "$OUT/roi_cache" --out "$OUT/roi_head" \
    --epochs 1 --max-hours .1
echo "PREFLIGHT_OK $OUT"