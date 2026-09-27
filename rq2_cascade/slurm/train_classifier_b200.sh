#!/usr/bin/env bash
# =============================================================================
# train_classifier_b200.sh — ConvNeXt-T (architecture_lab) na b200n1.
#
# Variantes usadas no paper 2:
#   VARIANT=baseline -> ConvNeXt-T + CE  (RUN_NAME padrao clf_convnext_t_ce_s<SEED>)
#   VARIANT=parts    -> ConvNeXt-T "partes" (RUN_NAME padrao clf_convnext_t_parts_s<SEED>)
# Seeds do paper: 42, 84, 126. Dados: crops pad75 ImageFolder (train/val/test,
# 7 classes INS MAR MC MD MF MV NOISE) em $CROPS — verificado em 2026 no no dgx-B200-1.
# Avaliacao: VAL (selecao species_macro_f1 dentro do trainer; best.pt + summary.json).
#
# Uso: VARIANT=baseline SEED=42 bash rq2_cascade/slurm/train_classifier_b200.sh
# Reentrante: OUT=$RUNS_ROOT/$RUN_NAME
#   - summary.json status=succeeded e export_complete=true -> SKIP
#   - last.pt existe -> --resume (mesmo contrato de args, obrigatorio)
#   - senao -> treino novo
# Log: 'JOB→RUN <wandb_run_name>'.
# =============================================================================
#SBATCH --job-name=p2-train-classifier
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
# sbatch copia o script para /var/spool/slurmd/<job>/ -> nao usar dirname(BASH_SOURCE)
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
source "${REPO}/common/slurm/b200_common.sh"
b200_guard

VARIANT="${VARIANT:?Defina VARIANT (baseline|parts)}"
SEED="${SEED:?Defina SEED (42|84|126)}"
case "${VARIANT}" in
    baseline) TAG="ce" ;;
    parts)    TAG="parts" ;;
    *)        TAG="${VARIANT}" ;;
esac
RUN_NAME="${RUN_NAME:-clf_convnext_t_${TAG}_s${SEED}}"
OUT="${OUT:-${RUNS_ROOT}/${RUN_NAME}}"
EPOCHS="${EPOCHS:-24}"
PATIENCE="${PATIENCE:-5}"
BATCH="${BATCH:-32}"
IMG="${IMG:-384}"
LR="${LR:-8e-5}"
WORKERS="${WORKERS:-8}"
MAX_HOURS="${MAX_HOURS:-1.25}"
export WANDB_PROJECT="${WANDB_PROJECT:-${WANDB_PROJECT_RQ2}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
# Smoke (sem W&B): LIMIT_TRAIN e LIMIT_VAL ambos definidos
EXTRA=()
if [[ -n "${LIMIT_TRAIN:-}" && -n "${LIMIT_VAL:-}" ]]; then
    EXTRA+=(--limit-train "${LIMIT_TRAIN}" --limit-val "${LIMIT_VAL}")
else
    EXTRA+=(--wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${WANDB_RUN_NAME}")
fi

echo "============================== ${RUN_NAME} =============================="
echo "Job: ${SLURM_JOB_ID} @ ${SLURM_NODELIST:-?}  variant=${VARIANT} seed=${SEED}"
echo "epochs=${EPOCHS} patience=${PATIENCE} batch=${BATCH} img=${IMG} lr=${LR} max_hours=${MAX_HOURS}"
echo "CROPS=${CROPS}  OUT=${OUT}  W&B ${WANDB_ENTITY}/${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "========================================================================="

for split in train val; do
    [[ -d "${CROPS}/${split}" ]] || { echo "ABORT: ${CROPS}/${split} ausente"; exit 2; }
done
mkdir -p "${OUT}"

# architecture_lab grava status completed|early_stop|budget|interrupted|failed (nunca "succeeded");
# concluido = exit_code 0 + export_complete (best.pt exportado apos avaliacao no VAL).
if [[ -f "${OUT}/summary.json" ]] && b200_apptainer /opt/venv/bin/python - "${OUT}/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"[summary] status={s.get('status')} exit_code={s.get('exit_code')} export_complete={s.get('export_complete')} epochs_ran={s.get('epochs_ran')}")
sys.exit(0 if s.get("exit_code") == 0 and s.get("export_complete") else 1)
PY
then
    echo "SKIP ${RUN_NAME}: summary.json exit_code=0 + export_complete"; echo "JOB→RUN ${WANDB_RUN_NAME}"; exit 0
fi

RESUME=()
if [[ -f "${OUT}/last.pt" ]]; then echo "RESUME: ${OUT}/last.pt"; RESUME+=(--resume); fi

chain_run b200_apptainer --pwd "${REPO}/rq2_cascade" \
    /opt/venv/bin/python -m architecture_lab.train \
        --data "${CROPS}" --out "${OUT}" --variant "${VARIANT}" \
        --epochs "${EPOCHS}" --patience "${PATIENCE}" --batch "${BATCH}" \
        --img-size "${IMG}" --lr "${LR}" --seed "${SEED}" --workers "${WORKERS}" \
        --max-hours "${MAX_HOURS}" "${RESUME[@]}" "${EXTRA[@]}"

echo "JOB→RUN ${WANDB_RUN_NAME}"
[[ -f "${OUT}/best.pt" ]] || { echo "ABORT: best.pt ausente em ${OUT}"; exit 1; }
echo "BEST ${OUT}/best.pt"
echo "DONE ${RUN_NAME} $(date)"
