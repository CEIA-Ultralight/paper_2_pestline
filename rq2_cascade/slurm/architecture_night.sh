#!/usr/bin/env bash
# Submit with --export=ALL,FLYDET_SOURCE_COMMIT=<committed hash>
#SBATCH --job-name=flydet-architecture-night
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
CAMPAIGN="${FLYDET_CAMPAIGN:-$ROOT/flydet_runs/architecture-night-v1-20260907}"
: "${SLURM_JOB_ID:?Requires sbatch}"
: "${FLYDET_SOURCE_COMMIT:?Submit a committed source hash}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
mkdir -p "$CAMPAIGN" "$ROOT/cache/architecture_lab/tmp" "$ROOT/cache/architecture_lab/home"
exec 9>"$ROOT/flydet_runs/architecture-campaign.lock"
flock -n 9 || { echo 'Another architecture campaign holds the single-training lock'; exit 1; }
SOURCE="$CAMPAIGN/source"
if [[ ! -d "$SOURCE" ]]; then
    mkdir -p "$SOURCE"
    git -C "$REPO" archive "$FLYDET_SOURCE_COMMIT" | tar -x -C "$SOURCE"
    printf '%s\n' "$FLYDET_SOURCE_COMMIT" > "$CAMPAIGN/source_commit.txt"
fi
[[ "$(cat "$CAMPAIGN/source_commit.txt")" == "$FLYDET_SOURCE_COMMIT" ]]
export FLYDET_SOURCE_COMMIT
export PYTHONPATH="$SOURCE/fly-det/scripts"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="$ROOT/cache/architecture_lab/tmp"
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer"
export APPTAINER_TMPDIR="$ROOT/cache/apptainer/tmp"
export XDG_CACHE_HOME="$ROOT/cache/architecture_lab/xdg"
export TORCH_HOME="$ROOT/.cache/torch"
export HF_HOME="$ROOT/.cache/huggingface"
export YOLO_CONFIG_DIR="$ROOT/.config/Ultralytics"
export MPLCONFIGDIR="$ROOT/cache/architecture_lab/matplotlib"
export CUDA_CACHE_PATH="$ROOT/cache/architecture_lab/cuda"
export WANDB_DIR="$ROOT/wandb"
export WANDB_CACHE_DIR="$ROOT/cache/architecture_lab/wandb/cache"
export WANDB_CONFIG_DIR="$ROOT/cache/architecture_lab/wandb/config"
export WANDB_DATA_DIR="$ROOT/cache/architecture_lab/wandb/data"
export WANDB_MODE=online
set -a
source "$ROOT/secrets/wandb.env"
set +a
# Environment inheritance carries credentials, never --env KEY=secret arguments.
child=''
interrupted=0
forward_signal() { interrupted=1; [[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true; }
trap forward_signal USR1 TERM INT
apptainer exec --nv --bind "$ROOT:$ROOT" \
    --home "$ROOT/cache/architecture_lab/home" --pwd "$SOURCE" \
    "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python \
    -m architecture_lab.campaign --out "$CAMPAIGN" --source "$SOURCE" &
child=$!
set +e
wait "$child"
rc=$?
if [[ "$interrupted" == 1 ]]; then
    wait "$child"
    rc=75
fi
exit "$rc"