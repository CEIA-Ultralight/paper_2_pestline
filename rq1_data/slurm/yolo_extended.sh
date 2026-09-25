#!/usr/bin/env bash
#SBATCH --job-name=flydet-yolo-extended
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err
# E6: E0 com treino estendido (subtreino?). E7: YOLO26l. Mesma receita do E0.
set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
: "${SLURM_JOB_ID:?Requires sbatch}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
mkdir -p "$ROOT/flydet_runs" "$ROOT/cache/architecture_lab/tmp" "$ROOT/cache/architecture_lab/home"
exec 9>"$ROOT/flydet_runs/architecture-campaign.lock"
flock -n 9 || { echo 'Another fly-det campaign holds the single-GPU lock'; exit 1; }
child=''
interrupted=0
forward_signal() { interrupted=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
trap forward_signal USR1 TERM INT
run() {
    "$@" &
    child=$!
    local rc=0
    wait "$child" || rc=$?
    if [[ "$interrupted" == 1 ]]; then wait "$child" || true; exit 75; fi
    child=''
    return "$rc"
}
# E6: mesmo modelo do E0, 400 épocas, patience 50 — testa subtreino.
run env MODEL=yolo26m.pt EPOCHS=400 PATIENCE=50 \
    WANDB_RUN_NAME=E6_yolo26m_long \
    bash "$REPO/fly-det/slurm/E0_yolo26m_baseline.sh"
# E7: backbone maior (l), mesma receita base do E0.
run env MODEL=yolo26l.pt EPOCHS=150 PATIENCE=30 \
    WANDB_RUN_NAME=E7_yolo26l \
    bash "$REPO/fly-det/slurm/E0_yolo26m_baseline.sh"
echo "YOLO_EXTENDED_COMPLETE"
