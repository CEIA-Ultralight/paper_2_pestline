#!/usr/bin/env bash
#SBATCH --job-name=flydet-val-study
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
# Avalia em VAL: detectores novos (E6/E7) e cascatas com dinov2/parts/propcls.
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
OUT="${FLYDET_VALSTUDY_ROOT:-$ROOT/flydet_runs/val-study-20260909}"
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
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
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
# Detectores: E0 (referência), E6 (treino longo), E7 (yolo26l) — os novos só entram se existirem.
DET_E0="$ROOT/flydet_runs/E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt"
DET_E6="$ROOT/flydet_runs/E0_yolo26m_baseline/runs/E6_yolo26m_long/weights/best.pt"
DET_E7="$ROOT/flydet_runs/E0_yolo26m_baseline/runs/E7_yolo26l/weights/best.pt"
# Cascatas: parts (escolhido na validação anterior), dinov2, propcls — só as que existirem.
HEADS=""
for h in parts dinov2_42 dinov2_84 propcls_42 propcls_84; do :; done
run_one() {
    local det="$1" tag="$2"
    [[ -f "$det" ]] || { echo "SKIP $tag (sem checkpoint)"; return 0; }
    local heads=(parts)
    [[ -f "$ROOT/flydet_runs/dinov2-20260909/seed42/best.pt" ]] && heads+=(dinov2_42)
    [[ -f "$ROOT/flydet_runs/proposal-cls-20260909/seed42/best.pt" ]] && heads+=(propcls_42)
    run "$SOURCE/fly-det/scripts/val_cascade_study.py" \
        --out "$OUT/$tag" --detector "$det" --heads "${heads[@]}" \
        --campaign-roots architecture-night-v1-20260907 dinov2-20260909 proposal-cls-20260909
}
run_one "$DET_E0" e0
run_one "$DET_E6" e6
run_one "$DET_E7" e7
echo "VAL_STUDY_ALL_COMPLETE $OUT"
