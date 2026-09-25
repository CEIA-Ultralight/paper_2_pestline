#!/usr/bin/env bash
#SBATCH --job-name=flydet-eval-yolo-cascade
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --signal=B:SIGUSR1@180
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
: "${SLURM_JOB_ID:?Requires Slurm}"
: "${FLYDET_SOURCE_COMMIT:?Pin committed evaluator source}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
CAMPAIGN="${FLYDET_CASCADE_ROOT:-$ROOT/flydet_runs/E0_cascade_val_20260909}"
mkdir -p "$CAMPAIGN" "$ROOT/cache/yolo_cascade/home" "$ROOT/cache/yolo_cascade/tmp"
exec 9>"$ROOT/flydet_runs/architecture-campaign.lock"
flock -n 9 || { echo 'Another fly-det campaign is active'; exit 1; }
SOURCE="$CAMPAIGN/source"
if [[ ! -d "$SOURCE" ]]; then
    mkdir -p "$SOURCE"
    git -C "$REPO" archive "$FLYDET_SOURCE_COMMIT" | tar -x -C "$SOURCE"
    printf '%s\n' "$FLYDET_SOURCE_COMMIT" > "$CAMPAIGN/source_commit.txt"
fi
[[ "$(cat "$CAMPAIGN/source_commit.txt")" == "$FLYDET_SOURCE_COMMIT" ]]
export FLYDET_SOURCE_COMMIT PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FLYDET_CASCADE_METRIC_TESTS=1
export XDG_CACHE_HOME="$ROOT/cache/yolo_cascade/xdg" TORCH_HOME="$ROOT/.cache/torch"
export YOLO_CONFIG_DIR="$ROOT/cache/yolo_cascade/ultralytics" MPLCONFIGDIR="$ROOT/cache/yolo_cascade/matplotlib"
export TMPDIR="$ROOT/cache/yolo_cascade/tmp" CUDA_CACHE_PATH="$ROOT/cache/yolo_cascade/cuda"
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer" APPTAINER_TMPDIR="$ROOT/cache/apptainer/tmp"
export WANDB_DIR="$ROOT/wandb" WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/cache/yolo_cascade/wandb/cache"
export WANDB_CONFIG_DIR="$ROOT/cache/yolo_cascade/wandb/config"
export WANDB_DATA_DIR="$ROOT/cache/yolo_cascade/wandb/data"
set -a
source "$ROOT/secrets/wandb.env"
set +a
child=''
interrupted=0
forward_signal() { interrupted=1; [[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true; }
trap forward_signal USR1 TERM INT
run() {
    apptainer exec --nv --bind "$ROOT:$ROOT" --home "$ROOT/cache/yolo_cascade/home" \
        --pwd "$SOURCE" "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python "$@" &
    child=$!
    local rc=0
    wait "$child" || rc=$?
    if [[ "$interrupted" == 1 ]]; then wait "$child" || true; exit 75; fi
    child=''
    return "$rc"
}
run -m unittest discover -s "$SOURCE/fly-det/tests" -p test_yolo_cascade.py -v
SMOKE_ARGS=()
[[ ! -f "$CAMPAIGN/smoke/state.json" ]] || SMOKE_ARGS+=(--resume)
run "$SOURCE/fly-det/scripts/eval_yolo_cascade.py" --out "$CAMPAIGN/smoke" \
    --limit-images 2 --no-wandb "${SMOKE_ARGS[@]}"
# Exercise artifact validation/resume without rerunning the two images.
run "$SOURCE/fly-det/scripts/eval_yolo_cascade.py" --out "$CAMPAIGN/smoke" \
    --limit-images 2 --no-wandb --resume
FULL_ARGS=()
[[ ! -f "$CAMPAIGN/full/state.json" ]] || FULL_ARGS+=(--resume)
run "$SOURCE/fly-det/scripts/eval_yolo_cascade.py" --out "$CAMPAIGN/full" "${FULL_ARGS[@]}"
echo "CASCADE_VAL_COMPLETE $CAMPAIGN/full"