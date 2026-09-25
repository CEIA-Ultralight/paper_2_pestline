#!/usr/bin/env bash
#SBATCH --job-name=flydet-test-campaign
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
: "${SLURM_JOB_ID:?Requires Slurm}"
: "${FLYDET_SOURCE_COMMIT:?Pin committed source}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
CAMPAIGN="${FLYDET_TEST_ROOT:-$ROOT/flydet_runs/test-campaign-20260909}"
mkdir -p "$CAMPAIGN" "$ROOT/cache/test_campaign/home" "$ROOT/cache/test_campaign/tmp"
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
export XDG_CACHE_HOME="$ROOT/cache/test_campaign/xdg" XDG_CONFIG_HOME="$ROOT/cache/test_campaign/config"
export TORCH_HOME="$ROOT/.cache/torch" HF_HOME="$ROOT/cache/test_campaign/hf"
export YOLO_CONFIG_DIR="$ROOT/cache/test_campaign/ultralytics" MPLCONFIGDIR="$ROOT/cache/test_campaign/matplotlib"
export TMPDIR="$ROOT/cache/test_campaign/tmp" CUDA_CACHE_PATH="$ROOT/cache/test_campaign/cuda"
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer" APPTAINER_TMPDIR="$ROOT/cache/apptainer/tmp"
export WANDB_DIR="$ROOT/wandb" WANDB_MODE=online
export WANDB_CACHE_DIR="$ROOT/cache/test_campaign/wandb/cache"
export WANDB_CONFIG_DIR="$ROOT/cache/test_campaign/wandb/config"
export WANDB_DATA_DIR="$ROOT/cache/test_campaign/wandb/data"
set -a
source "$ROOT/secrets/wandb.env"
set +a
child=''
interrupted=0
forward_signal() { interrupted=1; [[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true; }
trap forward_signal USR1 TERM INT
run() {
    apptainer exec --nv --bind "$ROOT:$ROOT" --home "$ROOT/cache/test_campaign/home" \
        --pwd "$SOURCE" "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python "$@" &
    child=$!
    local rc=0
    wait "$child" || rc=$?
    if [[ "$interrupted" == 1 ]]; then wait "$child" || true; exit 75; fi
    child=''
    return "$rc"
}
for pattern in test_yolo_cascade.py test_test_campaign_heads.py test_eval_test_campaign.py; do
    run -m unittest discover -s "$SOURCE/fly-det/tests" -p "$pattern" -v
done
SMOKE_ARGS=()
[[ ! -f "$CAMPAIGN/smoke/state.json" ]] || SMOKE_ARGS+=(--resume)
run "$SOURCE/fly-det/scripts/eval_test_campaign.py" --out "$CAMPAIGN/smoke" \
    --limit-images 1 --no-wandb "${SMOKE_ARGS[@]}"
run "$SOURCE/fly-det/scripts/eval_test_campaign.py" --out "$CAMPAIGN/smoke" \
    --limit-images 1 --no-wandb --resume
FULL_ARGS=()
[[ ! -f "$CAMPAIGN/full/state.json" ]] || FULL_ARGS+=(--resume)
run "$SOURCE/fly-det/scripts/eval_test_campaign.py" --out "$CAMPAIGN/full" "${FULL_ARGS[@]}"
echo "TEST_CAMPAIGN_COMPLETE $CAMPAIGN/full/report.md"