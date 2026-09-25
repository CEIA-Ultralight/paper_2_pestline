#!/usr/bin/env bash
#SBATCH --job-name=flydet-policy-study
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
# Estudo de políticas de fusão/seleção sobre predições VAL já salvas (CPU, sem GPU).
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
OUT="${FLYDET_POLICY_ROOT:-$ROOT/flydet_runs/policy_study_val}"
: "${SLURM_JOB_ID:?Requires sbatch}"
: "${FLYDET_SOURCE_COMMIT:?Submit a committed source hash}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
mkdir -p "$OUT" "$ROOT/cache/architecture_lab/tmp" "$ROOT/cache/architecture_lab/home"
exec 9>"$ROOT/flydet_runs/architecture-campaign.lock"
flock -n 9 || { echo 'Another fly-det campaign holds the lock'; exit 1; }
SOURCE="$OUT/source"
if [[ ! -d "$SOURCE" ]]; then
    mkdir -p "$SOURCE"
    git -C "$REPO" archive "$FLYDET_SOURCE_COMMIT" | tar -x -C "$SOURCE"
    printf '%s\n' "$FLYDET_SOURCE_COMMIT" > "$OUT/source_commit.txt"
fi
[[ "$(cat "$OUT/source_commit.txt")" == "$FLYDET_SOURCE_COMMIT" ]]
export PYTHONPATH="$SOURCE/fly-det/scripts"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export TMPDIR="$ROOT/cache/architecture_lab/tmp"
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer" APPTAINER_TMPDIR="$ROOT/cache/apptainer/tmp"
export XDG_CACHE_HOME="$ROOT/cache/architecture_lab/xdg"
export YOLO_CONFIG_DIR="$ROOT/.config/Ultralytics"
export WANDB_DIR="$ROOT/wandb" WANDB_MODE=online
set -a
source "$ROOT/secrets/wandb.env"
set +a
apptainer exec --bind "$ROOT:$ROOT" \
    --home "$ROOT/cache/architecture_lab/home" --pwd "$SOURCE" \
    "$ROOT/containers/flydet-train.sif" /opt/venv/bin/python \
    "$SOURCE/fly-det/scripts/policy_study.py" --out "$OUT"
