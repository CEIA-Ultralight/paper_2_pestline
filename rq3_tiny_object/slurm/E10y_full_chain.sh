#!/usr/bin/env bash
# =============================================================================
# E10y — Orquestrador da cadeia completa pos-E10 (todos os experimentos restantes).
#
# Cadeia (cada etapa so roda se a anterior terminar OK; se falhar, aborta com
#   log claro — sem jobs zumbis):
#   1-3) E10 seeds 0/1/2 (multi-seed — valida o +1,4 p.p. contra variancia ~5)
#   4-6) E0  seeds 0/1/2 (baseline de comparacao, mesmo protocolo)
#   7)   E7  yolo26l + NWD w=0.5 (modelo maior)
#   8)   E14 NWD + RFLA (pool de ancoras tiny ampliado)
#   9)   TTA + WBF (avaliacao no VAL de todos os detectores, sem treino)
#
# Submeter:   sbatch fly-det/slurm/E10y_full_chain.sh
# Monitorar:  squeue -u $USER ; tail -f slurm_logs/flydet-E10y-full_<id>.out
# =============================================================================
#SBATCH --job-name=flydet-E10y-full
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=200:00:00
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
        echo "E10Y_CHAIN_ABORT: etapa falhou com rc=$rc — $*"
        exit "$rc"
    fi
    return 0
}

E10="$REPO/fly-det/slurm/E10_yolo26m_nwd.sh"

# 1-3) multi-seed do E10 (NWD w=0.5) — seeds 0,1,2
for s in 0 1 2; do
  run env FLYDET_SEED=$s FLYDET_NWD_MODE=nwd FLYDET_NWD_W=0.5 \
      FLYDET_WORK_DIR="$ROOT/flydet_runs/E10_nwd_seed$s" \
      WANDB_RUN_NAME=E10_nwd_seed$s \
      bash "$E10"
done

# 4-6) multi-seed do E0 (baseline, sem patch) — seeds 0,1,2
for s in 0 1 2; do
  run env FLYDET_SEED=$s FLYDET_NWD_MODE=off \
      FLYDET_WORK_DIR="$ROOT/flydet_runs/E0_baseline_seed$s" \
      WANDB_RUN_NAME=E0_baseline_seed$s \
      bash "$E10"
done

# 7) E7: yolo26l + NWD w=0.5
run env MODEL=yolo26l.pt FLYDET_NWD_W=0.5 FLYDET_NWD_MODE=nwd \
    FLYDET_WORK_DIR="$ROOT/flydet_runs/E7_yolo26l_nwd" \
    WANDB_RUN_NAME=E7_yolo26l_nwd \
    bash "$E10"

# 8) E14: NWD + RFLA (pool de ancoras tiny ampliado)
run env FLYDET_RFLA=1 FLYDET_NWD_W=0.5 FLYDET_NWD_MODE=nwd \
    FLYDET_WORK_DIR="$ROOT/flydet_runs/E14_nwd_rfla" \
    WANDB_RUN_NAME=E14_nwd_rfla \
    bash "$E10"

# 9) TTA + WBF (avaliacao, sem treino). E13 (multi_scale) ficou de fora:
#    OOM na H100 (imgsz 1920 x 1.5 = ~3776 px nao coube nem batch=2). Retomar
#    so com gradient accumulation ou imgsz menor — fora do escopo desta cadeia.
run bash "$REPO/fly-det/slurm/tta_wbf_eval.sh"

echo "E10Y_CHAIN_COMPLETE"
