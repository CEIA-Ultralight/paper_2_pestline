#!/usr/bin/env bash
# =============================================================================
# chain_core_v9.sh — Campanha oficial RQ2 no DS-F2_v9.0.5 (b200n1), UM job. Com 2 GPUs (sbatch --gres=gpu:2) as faixas rodam em paralelo; com 1 GPU (padrao), em sequencia.
#   Faixa 1 (GPU 0): detectores E0 seeds 0/1/2 + yolo26l seed 0  (chain_A_detectors.sh, RUN_E10=0 RUN_E7=1)
#   Faixa 2 (GPU 1): classificadores ConvNeXt-T CE e partes x seeds 42/84/126 (chain_B_classifiers.sh)
#   Depois (GPU 0): matriz da cascata + oraculo em VAL e TEST (chain_D_cascade_cross.sh)
#     detectores {E0_s0..2, E7_yolo26l_s0} x classificadores {CE, partes} x politicas {label_only, product}
# Reentrante (marcadores DONE/summary.json por etapa); SIGUSR1 -> encaminha as duas faixas -> requeue.
# Dados: DS-F2_v9.0.5 (HEIC convertido sem perda para PNG), crops pad 0,75, RUNS_ROOT=flydet_runs_v9.
# Custo estimado: faixa 1 ~5-6 h, faixa 2 ~6-8 h (paralelas) + matriz ~1,5-2 h => ~9-10 h de relogio, ~25 GPU-h.
# =============================================================================
#SBATCH --job-name=p2v9-core
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
export MAX_HOURS="${MAX_HOURS:-2.5}"   # classificadores: o treino v9 tem ~2x mais crops que o v8

IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
G0="${GPUS[0]}"; G1="${GPUS[1]:-${GPUS[0]}}"
echo "CORE v9 inicio $(date) job=${SLURM_JOB_ID} gpus=${G0},${G1} data=${DATA_ROOT} runs=${RUNS_ROOT}"

PID_A=''; PID_B=''
fwd() { CHAIN_SIGNAL="${CHAIN_SIGNAL:-$1}"; CHAIN_INTERRUPTED=1
        [[ -z "${PID_A}" ]] || kill -USR1 "${PID_A}" 2>/dev/null || true
        [[ -z "${PID_B}" ]] || kill -USR1 "${PID_B}" 2>/dev/null || true; }
trap 'fwd USR1' USR1; trap 'fwd TERM' TERM; trap 'fwd INT' INT

# wait_rc PID -> rc em WRC (sem subshell: `wait` so enxerga filhos do shell atual)
wait_rc() { local p=$1; WRC=0; while :; do wait "$p" && WRC=0 || WRC=$?; kill -0 "$p" 2>/dev/null || break; done; }

if ! { step_done "${RUNS_ROOT}/CHAIN_A_DONE" && step_done "${RUNS_ROOT}/CHAIN_B_DONE"; }; then
    LANE_A=("${REPO}/rq3_tiny_object/slurm/chain_A_detectors.sh"); LANE_B=("${REPO}/rq2_cascade/slurm/chain_B_classifiers.sh")
    if [[ "${G0}" != "${G1}" ]]; then   # 2 GPUs: faixas em paralelo
        CUDA_VISIBLE_DEVICES="${G0}" RUN_E10=0 RUN_E7=1 bash "${LANE_A[@]}" & PID_A=$!
        CUDA_VISIBLE_DEVICES="${G1}" bash "${LANE_B[@]}" & PID_B=$!
        wait_rc "${PID_A}"; RC_A=${WRC}; wait_rc "${PID_B}"; RC_B=${WRC}
    else                                 # 1 GPU: detectores e depois classificadores
        CUDA_VISIBLE_DEVICES="${G0}" RUN_E10=0 RUN_E7=1 bash "${LANE_A[@]}" & PID_A=$!
        wait_rc "${PID_A}"; RC_A=${WRC}; PID_A=''; RC_B=0
        if [[ "${RC_A}" == 0 ]]; then
            CUDA_VISIBLE_DEVICES="${G0}" bash "${LANE_B[@]}" & PID_B=$!
            wait_rc "${PID_B}"; RC_B=${WRC}
        fi
    fi
    echo "faixa detectores rc=${RC_A}; faixa classificadores rc=${RC_B}"
    if [[ "${RC_A}" == 75 || "${RC_B}" == 75 || "${CHAIN_INTERRUPTED:-0}" == 1 ]]; then _chain_exit75; fi
    [[ "${RC_A}" == 0 && "${RC_B}" == 0 ]] || { echo "CORE ABORT rcA=${RC_A} rcB=${RC_B}"; exit 1; }
fi
PID_A=''; PID_B=''

export DET_RUNS="E0_s0 E0_s1 E0_s2 E7_yolo26l_s0"
chain_run env CUDA_VISIBLE_DEVICES="${G0}" bash "${REPO}/rq2_cascade/slurm/chain_D_cascade_cross.sh"
echo "CORE v9 FIM $(date)"
