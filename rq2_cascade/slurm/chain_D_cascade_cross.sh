#!/usr/bin/env bash
# =============================================================================
# chain_D_cascade_cross.sh — Cadeia D (b200n1): matriz cruzada da cascata no TEST.
# DEPENDE das cadeias A (E0_s0..s2/best.pt) e B (clf_convnext_t_{ce,parts}_s{42,84,126}/best.pt).
#
#   3 detectores E0 x (3 CE + 3 partes) x politicas {label_only, product}
#   -> rq2_cascade/eval_cascade_matrix.py escreve, por (detector, classificador, politica):
#      $OUT/<det>/<clf>/<policy>/{per_instance.csv, per_image_preds.jsonl, metrics.json}
#      $OUT/<det>/single/label_only/  (detector sozinho, mesmas propostas)
#      $OUT/{manifest.json, summary.json}; metrics.json inclui latencia por estagio
#      (detector, classifier_extra, end_to_end) e discordancias pareadas (McNemar via common/stats.py).
#
# Submissao: sbatch rq2_cascade/slurm/chain_D_cascade_cross.sh   (apos CHAIN_A_DONE e CHAIN_B_DONE)
# Custo estimado: < 1 GPU-h (TEST = 55 imagens; 3 det x 7 configs).
# Saida: $RUNS_ROOT/cascade_cross_test/ ; W&B pestline/paper2-rq2-cascade run cascade_cross_test.
# =============================================================================
#SBATCH --job-name=p2-chainD-cascade-cross
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../common/slurm/b200_common.sh"
b200_guard
export WANDB_PROJECT="${WANDB_PROJECT_RQ2}"
SPLIT="${SPLIT:-test}"
OUT="${OUT:-${RUNS_ROOT}/cascade_cross_${SPLIT}}"
RUN_NAME="${RUN_NAME:-cascade_cross_${SPLIT}}"
POLICIES="${POLICIES:-label_only product}"

step_done "${RUNS_ROOT}/CHAIN_A_DONE" || { echo "ABORT: CHAIN_A_DONE ausente"; exit 2; }
step_done "${RUNS_ROOT}/CHAIN_B_DONE" || { echo "ABORT: CHAIN_B_DONE ausente"; exit 2; }
if step_done "${OUT}/summary.json"; then echo "SKIP ${RUN_NAME}: ${OUT}/summary.json"; echo "JOB→RUN ${RUN_NAME}"; exit 0; fi

DETS=(); DET_NAMES=()
for s in ${DET_SEEDS:-0 1 2}; do
    f="${RUNS_ROOT}/E0_s${s}/best.pt"; [[ -f "${f}" ]] || { echo "ABORT: ${f} ausente"; exit 2; }
    DETS+=("${f}"); DET_NAMES+=("E0_s${s}")
done
CLFS=(); CLF_NAMES=()
for tag in ${CLF_TAGS:-ce parts}; do
    for s in ${CLF_SEEDS:-42 84 126}; do
        n="clf_convnext_t_${tag}_s${s}"; f="${RUNS_ROOT}/${n}/best.pt"
        [[ -f "${f}" ]] || { echo "ABORT: ${f} ausente"; exit 2; }
        CLFS+=("${f}"); CLF_NAMES+=("${n}")
    done
done

echo "CHAIN D inicio $(date) job=${SLURM_JOB_ID} split=${SPLIT} dets=${DET_NAMES[*]} clfs=${CLF_NAMES[*]} policies=${POLICIES}"
mkdir -p "${OUT}"

# shellcheck disable=SC2086
chain_run b200_apptainer /opt/venv/bin/python "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
    --detector "${DETS[@]}" --detector-names "${DET_NAMES[@]}" \
    --classifier "${CLFS[@]}" --classifier-names "${CLF_NAMES[@]}" \
    --policies ${POLICIES} \
    --data "${DATA_ROOT}" --split "${SPLIT}" --out "${OUT}" \
    --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN_NAME}"

echo "JOB→RUN ${RUN_NAME}"
touch "${RUNS_ROOT}/CHAIN_D_DONE"
echo "CHAIN_D_DONE $(date)  saida: ${OUT}"
