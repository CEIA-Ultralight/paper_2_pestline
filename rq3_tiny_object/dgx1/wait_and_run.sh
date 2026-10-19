#!/usr/bin/env bash
# =============================================================================
# wait_and_run.sh — espera a GPU ficar livre e so entao chama run_queue.sh.
#
# "Livre" = nenhuma fila run_queue.sh rodando e nenhum container rq3-*-gpu<GPU> (so os desta campanha).
# Evita dois treinos na mesma V100 (cada um usa ~24 de 32 GB).
# Runs com DONE sao pulados pelo run_queue.sh, entao pode repetir runs ja feitos.
#
# Uso:
#   GPU=0 nohup bash rq3_tiny_object/dgx1/wait_and_run.sh RUN [RUN ...] > ~/pestline/logs/fila.log 2>&1 &
# =============================================================================
set -euo pipefail

GPU="${GPU:?Defina GPU}"
WAIT_SECS="${WAIT_SECS:-300}"
HERE="$(cd "$(dirname "$0")" && pwd)"
[[ $# -gt 0 ]] || { echo "Uso: GPU=<n> bash $0 RUN [RUN ...]"; exit 2; }

gpu_busy() {
    pgrep -f run_queue.sh >/dev/null && return 0
    # so containers desta campanha (rq3-<RUN>-gpuN); a dgx1 tem containers de outros usuarios com -gpuN
    docker ps --format '{{.Names}}' | grep -q -- "^rq3-.*-gpu${GPU}\$"
}

while gpu_busy; do
    echo "$(date '+%F %T') GPU ${GPU} ocupada; nova checagem em ${WAIT_SECS}s"
    sleep "${WAIT_SECS}"
done
echo "$(date '+%F %T') GPU ${GPU} livre; iniciando fila: $*"
exec bash "${HERE}/run_queue.sh" "$@"
