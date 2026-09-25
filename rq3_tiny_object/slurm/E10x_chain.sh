#!/usr/bin/env bash
# =============================================================================
# E10x — Orquestrador da cadeia pos-E10 (itens 1-6 do plano mAP50).
#
# Cadeia (cada etapa so roda se a anterior terminar OK — se uma falha,
#   aborta com log claro em vez de deixar jobs zumbis na fila):
#   1) E11: NWD w=0.75  (sweep do peso)
#   2) E12: NWD w=1.00  (sweep do peso)
#   3) E10b: GCD w=0.5  (variante com invariancia de escala)
#   4) E13: NWD w=0.5 + multi_scale/rect=false
#   5) E7 : yolo26l + NWD w=0.5
#   6) TTA + WBF: avaliacao no VAL de todos os detectores disponiveis
#
# Prerequisito: job partnr-f (32922) e os outros da fila devem liberar a
# GPU; esta cadeia entra na fila normalmente (QOS onejob).
#
# Submeter:   sbatch fly-det/slurm/E10x_chain.sh
# Monitorar:  squeue -u $USER ; tail -f slurm_logs/flydet-E10x-chain_<id>.out
# =============================================================================
#SBATCH --job-name=flydet-E10x-chain
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=120:00:00
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/raid/user_marcospaulo
REPO="$ROOT/fly-det"
: "${SLURM_JOB_ID:?Requires sbatch}"
[[ "$SLURM_JOB_PARTITION" == h100n2 ]]
mkdir -p "$ROOT/flydet_runs" "$REPO/slurm_logs"

child=''
interrupted=0
forward_signal() { interrupted=1; [[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true; }
trap forward_signal USR1 TERM INT
run() {
    echo ""
    echo "########## $(date '+%H:%M:%S') — $* ##########"
    "$@" &
    child=$!
    local rc=0
    wait "$child" || rc=$?
    if [[ "$interrupted" == 1 ]]; then wait "$child" || true; exit 75; fi
    child=''
    if [[ "$rc" != 0 ]]; then
        echo "E10X_CHAIN_ABORT: etapa falhou com rc=$rc — $*"
        exit "$rc"
    fi
    return 0
}

# 1-2) sweep do peso do NWD
run env FLYDET_NWD_W=0.75 FLYDET_WORK_DIR="$ROOT/flydet_runs/E10_yolo26m_nwd_w075" \
    WANDB_RUN_NAME=E10_w075 \
    bash "$REPO/fly-det/slurm/E10_yolo26m_nwd.sh"

run env FLYDET_NWD_W=1.0 FLYDET_WORK_DIR="$ROOT/flydet_runs/E10_yolo26m_nwd_w100" \
    WANDB_RUN_NAME=E10_w100 \
    bash "$REPO/fly-det/slurm/E10_yolo26m_nwd.sh"

# 3) GCD (variante com invariancia de escala)
run bash "$REPO/fly-det/slurm/E10b_yolo26m_gcd.sh"

# 4) multi_scale + rect=false sobre base E10 (NWD w=0.5)
run bash "$REPO/fly-det/slurm/E13_yolo26m_multiscale.sh"

# 5) E7: yolo26l + NWD (modelo maior)
run env MODEL=yolo26l.pt WANDB_RUN_NAME=E7_yolo26l_nwd \
    FLYDET_WORK_DIR="$ROOT/flydet_runs/E7_yolo26l_nwd" \
    bash "$REPO/fly-det/slurm/E10_yolo26m_nwd.sh"

# 6) TTA + WBF (avaliacao, sem treino)
run bash "$REPO/fly-det/slurm/tta_wbf_eval.sh"

echo "E10X_CHAIN_COMPLETE"
