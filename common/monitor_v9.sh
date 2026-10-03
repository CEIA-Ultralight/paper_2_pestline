#!/usr/bin/env bash
# monitor_v9.sh — vigia os jobs da campanha v9 (CPU, leve; roda no login node, NAO usa GPU).
# Sai (rc 0) assim que ocorrer algo que exige atencao e imprime o motivo; sai com "NO_EVENT" ao fim de MAX_MIN minutos.
# Eventos: linha ERROR nova em $RUNS_ROOT/ALERTS.log; job principal em FAILED/TIMEOUT/NODE_FAIL/OUT_OF_MEMORY/CANCELLED;
# reserva comecando a rodar (= o principal falhou); log de job RUNNING parado ha > STALL_MIN min; CHAIN_EXTRAS_DONE (fim).
# Uso: JOBS_MAIN="33878 33881" JOBS_RESERVE="33879 33880 33882" bash common/monitor_v9.sh
RUNS_ROOT="${RUNS_ROOT:-/raid/user_marcospaulo/flydet_runs_v9}"
LOGS="${LOGS:-/raid/user_marcospaulo/slurm_logs}"
JOBS_MAIN="${JOBS_MAIN:?JOBS_MAIN}"; JOBS_RESERVE="${JOBS_RESERVE:-}"
MAX_MIN="${MAX_MIN:-110}"; STALL_MIN="${STALL_MIN:-40}"; POLL="${POLL:-60}"
ALERTS="${RUNS_ROOT}/ALERTS.log"; mkdir -p "${RUNS_ROOT}"; touch "${ALERTS}"
base_err=$(grep -c " ERROR " "${ALERTS}" || true)
end=$(( $(date +%s) + MAX_MIN*60 ))
state() { sacct -X -n -P -j "$1" -o State 2>/dev/null | head -1 | awk '{print $1}'; }
while (( $(date +%s) < end )); do
    [[ -f "${RUNS_ROOT}/CHAIN_EXTRAS_DONE" ]] && { echo "COMPLETE: CHAIN_EXTRAS_DONE"; exit 0; }
    n=$(grep -c " ERROR " "${ALERTS}" || true)
    if (( n > base_err )); then echo "ALERTA: novo ERROR em ALERTS.log"; grep " ERROR " "${ALERTS}" | tail -$((n-base_err)); exit 0; fi
    for j in ${JOBS_MAIN}; do
        s=$(state "$j")
        case "$s" in FAILED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*|CANCELLED*|BOOT_FAIL*) echo "ALERTA: job principal $j em $s"; exit 0;; esac
        if [[ "$s" == RUNNING ]]; then
            lg=$(ls -1t "${LOGS}"/p2v9-*_"$j".out 2>/dev/null | head -1)
            if [[ -n "$lg" ]] && (( $(date +%s) - $(stat -c %Y "$lg") > STALL_MIN*60 )); then echo "ALERTA: log de $j parado ha > ${STALL_MIN} min ($lg)"; tail -3 "$lg" | cut -c1-200; exit 0; fi
        fi
    done
    for j in ${JOBS_RESERVE}; do
        s=$(state "$j"); [[ "$s" == RUNNING ]] && { echo "ALERTA: reserva $j comecou a rodar (principal falhou)"; exit 0; }
    done
    sleep "${POLL}"
done
echo "NO_EVENT apos ${MAX_MIN} min"
