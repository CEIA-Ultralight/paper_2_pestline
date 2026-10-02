#!/usr/bin/env bash
# =============================================================================
# chain_D_cascade_cross.sh — Cadeia D (b200n1): matriz cruzada da cascata + oraculo de crops GT,
# em VAL e TEST. DEPENDE das cadeias A (E0_s{0,1,2}, E10_nwd_w0.5_s{0,1,2}: best.pt) e
# B (clf_convnext_t_{ce,parts}_s{42,84,126}/best.pt).
#
#   Para cada SPLIT em ${SPLITS:-val test}:
#   1) Matriz: 6 detectores (3 E0 + 3 E10 NWD) x (3 CE + 3 partes) x politicas {label_only, product}
#      -> rq2_cascade/eval_cascade_matrix.py escreve, por (detector, classificador, politica):
#         $OUT/<det>/<clf>/<policy>/{per_instance.csv, per_image_preds.jsonl, metrics.json}
#         $OUT/<det>/single/label_only/  (detector sozinho, mesmas propostas)
#         $OUT/{manifest.json, summary.json}; metrics.json inclui latencia por estagio
#         (detector, classifier_extra, end_to_end) e discordancias pareadas (McNemar via common/stats.py).
#      Saida $RUNS_ROOT/cascade_cross_${SPLIT}/ ; W&B run cascade_cross_${SPLIT}. SKIP se summary.json existe.
#   2) Oraculo: os 6 classificadores sobre crops das caixas GT (localizacao perfeita), mesmo
#      padding/pre-processamento da cascata -> rq2_cascade/eval_oracle_crops.py escreve
#         $OUT/<clf>/{per_instance.csv, metrics.json}, $OUT/{summary.json, manifest.json}
#      Saida $RUNS_ROOT/oracle_crops_${SPLIT}/ ; W&B run oracle_crops_${SPLIT}. DONE por split.
#
# VAL = 35 imagens / 617 instancias de especie (1649 GT no total); TEST = 55 / 918.
# VAL foi usado para selecionar os checkpoints (best.pt) -> reportar como VAL, nao held-out.
#
# Env: DET_RUNS (nomes de run em $RUNS_ROOT; default os 6), CLF_TAGS/CLF_SEEDS, POLICIES, SPLITS.
# Submissao: sbatch rq2_cascade/slurm/chain_D_cascade_cross.sh   (apos CHAIN_A_DONE e CHAIN_B_DONE)
# Custo estimado: matriz 6 det x (1 + 6 clf) configs x 35-55 imagens (~20-40 min por split no B200)
# + oraculo (~2 min por split) -> < 2 GPU-h no total.
# =============================================================================
#SBATCH --job-name=p2-chainD-cascade-cross
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
# sbatch copia o script para /var/spool/slurmd/<job>/ -> nao usar dirname(BASH_SOURCE)
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
export WANDB_PROJECT="${WANDB_PROJECT_RQ2}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-cascade}"
SPLITS="${SPLITS:-val test}"
POLICIES="${POLICIES:-label_only product}"
DET_RUNS="${DET_RUNS:-E0_s0 E0_s1 E0_s2 E10_nwd_w0.5_s0 E10_nwd_w0.5_s1 E10_nwd_w0.5_s2}"

step_done "${RUNS_ROOT}/CHAIN_A_DONE" || { echo "ABORT: CHAIN_A_DONE ausente"; exit 2; }
step_done "${RUNS_ROOT}/CHAIN_B_DONE" || { echo "ABORT: CHAIN_B_DONE ausente"; exit 2; }

DETS=(); DET_NAMES=()
for n in ${DET_RUNS}; do
    f="${RUNS_ROOT}/${n}/best.pt"; [[ -f "${f}" ]] || { echo "ABORT: ${f} ausente"; exit 2; }
    DETS+=("${f}"); DET_NAMES+=("${n}")
done
CLFS=(); CLF_NAMES=()
for tag in ${CLF_TAGS:-ce parts}; do
    for s in ${CLF_SEEDS:-42 84 126}; do
        n="clf_convnext_t_${tag}_s${s}"; f="${RUNS_ROOT}/${n}/best.pt"
        [[ -f "${f}" ]] || { echo "ABORT: ${f} ausente"; exit 2; }
        CLFS+=("${f}"); CLF_NAMES+=("${n}")
    done
done

echo "CHAIN D inicio $(date) job=${SLURM_JOB_ID} splits=${SPLITS} dets=${DET_NAMES[*]} clfs=${CLF_NAMES[*]} policies=${POLICIES}"

for SPLIT in ${SPLITS}; do
    # 1) matriz cruzada
    RUN_NAME="cascade_cross_${SPLIT}"
    OUT="${RUNS_ROOT}/${RUN_NAME}"
    if step_done "${OUT}/summary.json"; then
        echo "SKIP ${RUN_NAME}: ${OUT}/summary.json"
    else
        mkdir -p "${OUT}"
        # shellcheck disable=SC2086
        chain_run b200_apptainer /opt/venv/bin/python "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
            --detector "${DETS[@]}" --detector-names "${DET_NAMES[@]}" \
            --classifier "${CLFS[@]}" --classifier-names "${CLF_NAMES[@]}" \
            --policies ${POLICIES} \
            --data "${DATA_ROOT}" --split "${SPLIT}" --out "${OUT}" \
            --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN_NAME}"
    fi
    echo "JOB→RUN ${RUN_NAME}"

    # 2) oraculo de identificacao (crops das caixas GT, sem detector)
    RUN_NAME="oracle_crops_${SPLIT}"
    OUT="${RUNS_ROOT}/${RUN_NAME}"
    if step_done "${OUT}/DONE"; then
        echo "SKIP ${RUN_NAME}: ${OUT}/DONE"
    else
        mkdir -p "${OUT}"
        chain_run b200_apptainer /opt/venv/bin/python "${REPO}/rq2_cascade/eval_oracle_crops.py" \
            --classifier "${CLFS[@]}" --classifier-names "${CLF_NAMES[@]}" \
            --data "${DATA_ROOT}" --split "${SPLIT}" --out "${OUT}" \
            --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN_NAME}"
        touch "${OUT}/DONE"
    fi
    echo "JOB→RUN ${RUN_NAME}"
done

touch "${RUNS_ROOT}/CHAIN_D_DONE"
echo "CHAIN_D_DONE $(date)  saida: ${RUNS_ROOT}/{cascade_cross,oracle_crops}_{${SPLITS// /,}}"
