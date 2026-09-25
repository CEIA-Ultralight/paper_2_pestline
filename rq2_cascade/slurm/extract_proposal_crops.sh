#!/usr/bin/env bash
#SBATCH --job-name=flydet-prop-crops
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
# Extrai crops das PROPOSTAS do detector E0 (train+val de DS-F2_v8.1.1) com a 8a classe BG.
# Sem W&B, sem acesso ao split test; resume por marcador por imagem; sinal -> exit 75.
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
RUN="$ROOT/flydet_runs/proposal-crops"
OUT="${FLYDET_PROP_CROPS_ROOT:-$ROOT/datasets/DS-F2_crops_proposals}"
: "${SLURM_JOB_ID:?Requires sbatch}"
: "${FLYDET_SOURCE_COMMIT:?Submit a committed source hash}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
mkdir -p "$OUT" "$RUN" "$ROOT/cache/architecture_lab/tmp" "$ROOT/cache/architecture_lab/home"
exec 9>"$ROOT/flydet_runs/architecture-campaign.lock"
flock -n 9 || { echo 'Another fly-det campaign holds the single-GPU lock'; exit 1; }
SOURCE="$RUN/source"
if [[ ! -d "$SOURCE" ]]; then
    mkdir -p "$SOURCE"
    git -C "$REPO" archive "$FLYDET_SOURCE_COMMIT" | tar -x -C "$SOURCE"
    printf '%s\n' "$FLYDET_SOURCE_COMMIT" > "$RUN/source_commit.txt"
fi
[[ "$(cat "$RUN/source_commit.txt")" == "$FLYDET_SOURCE_COMMIT" ]]
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$ROOT/cache/architecture_lab/tmp"
export APPTAINER_CACHEDIR="$ROOT/cache/apptainer" APPTAINER_TMPDIR="$ROOT/cache/apptainer/tmp"
export XDG_CACHE_HOME="$ROOT/cache/architecture_lab/xdg"
export TORCH_HOME="$ROOT/.cache/torch" HF_HOME="$ROOT/.cache/huggingface"
export YOLO_CONFIG_DIR="$ROOT/.config/Ultralytics" MPLCONFIGDIR="$ROOT/cache/architecture_lab/matplotlib"
export CUDA_CACHE_PATH="$ROOT/cache/architecture_lab/cuda"
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
run "$SOURCE/fly-det/scripts/extract_proposal_crops.py" --out "$OUT"
echo "PROP_CROPS_COMPLETE $OUT"
