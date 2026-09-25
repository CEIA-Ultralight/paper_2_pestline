#!/usr/bin/env bash
#SBATCH --job-name=flydet-dinov2
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
# Treina o classificador DINOv2 (seed 42 e 84) com o trainer do architecture_lab.
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
OUT="${FLYDET_DINOV2_ROOT:-$ROOT/flydet_runs/dinov2-20260909}"
: "${SLURM_JOB_ID:?Requires sbatch}"
: "${FLYDET_SOURCE_COMMIT:?Submit a committed source hash}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
mkdir -p "$OUT" "$ROOT/cache/architecture_lab/tmp" "$ROOT/cache/architecture_lab/home"
exec 9>"$ROOT/flydet_runs/architecture-campaign.lock"
flock -n 9 || { echo 'Another fly-det campaign holds the single-GPU lock'; exit 1; }
SOURCE="$OUT/source"
if [[ ! -d "$SOURCE" ]]; then
    mkdir -p "$SOURCE"
    git -C "$REPO" archive "$FLYDET_SOURCE_COMMIT" | tar -x -C "$SOURCE"
    printf '%s\n' "$FLYDET_SOURCE_COMMIT" > "$OUT/source_commit.txt"
fi
[[ "$(cat "$OUT/source_commit.txt")" == "$FLYDET_SOURCE_COMMIT" ]]
export PYTHONPATH="$SOURCE/fly-det/scripts"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/cache/architecture_lab/tmp"
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer" APPTAINER_TMPDIR="$ROOT/cache/apptainer/tmp"
export XDG_CACHE_HOME="$ROOT/cache/architecture_lab/xdg"
export TORCH_HOME="$ROOT/.cache/torch" HF_HOME="$ROOT/.cache/huggingface"
export YOLO_CONFIG_DIR="$ROOT/.config/Ultralytics" MPLCONFIGDIR="$ROOT/cache/architecture_lab/matplotlib"
export CUDA_CACHE_PATH="$ROOT/cache/architecture_lab/cuda"
export WANDB_DIR="$ROOT/wandb" WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/cache/architecture_lab/wandb/cache"
export WANDB_CONFIG_DIR="$ROOT/cache/architecture_lab/wandb/config"
export WANDB_DATA_DIR="$ROOT/cache/architecture_lab/wandb/data"
set -a
source "$ROOT/secrets/wandb.env"
set +a
child=''
interrupted=0
forward_signal() { interrupted=1; [[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true; }
trap forward_signal USR1 TERM INT
run() {
    apptainer exec --nv --bind "$ROOT:$ROOT" \
        --home "$ROOT/cache/architecture_lab/home" --pwd "$SOURCE" \
        "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python "$@" &
    child=$!
    local rc=0
    wait "$child" || rc=$?
    if [[ "$interrupted" == 1 ]]; then wait "$child" || true; exit 75; fi
    child=''
    return "$rc"
}
DATA="$ROOT/datasets/DS-F2_crops_pad75"
COMMON=(--data "$DATA" --variant dinov2 --epochs 24 --patience 5 --batch 32
        --img-size 384 --workers 8 --lr 8e-5 --max-hours 1.25)
# Smoke técnico: pesos aleatórios, 8 exemplos, sem W&B.
run -m architecture_lab.train --data "$DATA" --variant dinov2 --epochs 1 --patience 5 \
    --batch 4 --img-size 384 --workers 0 --lr 8e-5 --max-hours .1 \
    --limit-train 8 --limit-val 8 --no-pretrained --out "$OUT/smoke"
for seed in 42 84; do
    RDIR="$OUT/seed$seed"
    EXTRA=()
    [[ ! -f "$RDIR/last.pt" ]] || EXTRA=(--resume)
    run -m architecture_lab.train "${COMMON[@]}" --seed "$seed" --out "$RDIR" "${EXTRA[@]}" \
        --wandb --wandb-entity pestline --wandb-project fly-species --wandb-run-name "dinov2_seed$seed"
done
echo "DINOV2_COMPLETE $OUT"
