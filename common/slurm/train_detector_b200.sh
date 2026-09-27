#!/usr/bin/env bash
# =============================================================================
# train_detector_b200.sh — receita UNICA de detector (E0 e E10*) na b200n1.
#
# A receita e IDENTICA para E0/E10: yolo26m.pt, 150 epocas, batch 8, imgsz 1920,
# patience 30, save_period 10, seed via FLYDET_SEED, checkpoint = best.pt.
# A UNICA diferenca entre experimentos sao as variaveis FLYDET_NWD_* / FLYDET_RFLA
# (patch monkeypatch em rq3_tiny_object/patches, aplicado por train_nwd.py).
#
# Uso (pelas cadeias, com env):
#   RUN_NAME=E0_s0 FLYDET_SEED=0 FLYDET_NWD_MODE=off bash common/slurm/train_detector_b200.sh
#   RUN_NAME=E10_nwd_w0.5_s1 FLYDET_SEED=1 FLYDET_NWD_MODE=nwd FLYDET_NWD_W=0.5 bash ...
# Ou direto: sbatch --export=ALL,RUN_NAME=E0_s0,FLYDET_SEED=0,FLYDET_NWD_MODE=off \
#   common/slurm/train_detector_b200.sh
#
# Reentrante: WORK=$RUNS_ROOT/$RUN_NAME
#   - $WORK/DONE existe                      -> nada a fazer
#   - $WORK/runs/*/training_summary.json     -> treino ok; so roda o TEST eval
#   - $WORK/runs/*/weights/last.pt           -> retoma (FLYDET_RESUME=1, --model last.pt)
#   - senao                                  -> treino novo
# Apos o treino: avaliacao no TEST (per_instance.csv) via rq2_cascade/eval_cascade_matrix.py
# (detector-only) em $WORK/test_eval/. NOTA: common/eval_test_campaign.py e o harness
# congelado da campanha 2026-09 (checkpoints/heads fixos) e NAO aceita novos detectores;
# o harness por instancia substitui essa etapa.
# Log: 'JOB→RUN <wandb_run_name>' (o slurm-runner copia para rq3_tiny_object/results/runs.md).
# =============================================================================
#SBATCH --job-name=p2-train-detector
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
source "${HERE}/b200_common.sh"
b200_guard

: "${RUN_NAME:?Defina RUN_NAME (ex.: E0_s0, E10_nwd_w0.5_s1)}"
export FLYDET_SEED="${FLYDET_SEED:?Defina FLYDET_SEED (0|1|2)}"
export FLYDET_NWD_MODE="${FLYDET_NWD_MODE:-off}"      # off = E0 (baseline sem patch)
export FLYDET_NWD_W="${FLYDET_NWD_W:-0.5}"
export FLYDET_NWD_C="${FLYDET_NWD_C:-44.0}"
export FLYDET_NWD_SCOPE="${FLYDET_NWD_SCOPE:-both}"    # loss|assigner|both
export FLYDET_RFLA="${FLYDET_RFLA:-0}"
export FLYDET_SAVE_PERIOD="${FLYDET_SAVE_PERIOD:-10}"
export FLYDET_DATASET_ROOT="${DATA_ROOT}"

MODEL="${MODEL:-yolo26m.pt}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-1920}"
PATIENCE="${PATIENCE:-30}"
WORKERS="${WORKERS:-12}"
export WANDB_PROJECT="${WANDB_PROJECT:-${WANDB_PROJECT_RQ3}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
WORK="${FLYDET_WORK_DIR:-${RUNS_ROOT}/${RUN_NAME}}"
mkdir -p "${WORK}/models" "${WORK}/runs"

echo "============================== ${RUN_NAME} =============================="
echo "Job:     ${SLURM_JOB_ID} @ ${SLURM_NODELIST:-?}  particao=${SLURM_JOB_PARTITION}"
echo "Receita: ${MODEL} epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} patience=${PATIENCE} save_period=${FLYDET_SAVE_PERIOD}"
echo "Seed:    FLYDET_SEED=${FLYDET_SEED}  NWD: mode=${FLYDET_NWD_MODE} w=${FLYDET_NWD_W} C=${FLYDET_NWD_C} scope=${FLYDET_NWD_SCOPE} rfla=${FLYDET_RFLA}"
echo "W&B:     ${WANDB_ENTITY}/${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "WORK:    ${WORK}   inicio=$(date)"
echo "========================================================================="

if step_done "${WORK}/DONE"; then echo "SKIP ${RUN_NAME}: DONE"; echo "JOB→RUN ${WANDB_RUN_NAME}"; exit 0; fi

# ------------------------- pesos pre-treinados (models/ no cwd do fly_det) ---
if [[ ! -f "${WORK}/models/${MODEL}" ]]; then
    if [[ -f "${MODELS_DIR}/${MODEL}" ]]; then
        cp -n "${MODELS_DIR}/${MODEL}" "${WORK}/models/"
    else
        echo "AVISO: ${MODELS_DIR}/${MODEL} ausente — tentando download (rede) para ${MODELS_DIR}"
        mkdir -p "${MODELS_DIR}"
        b200_apptainer --pwd "${MODELS_DIR}" /opt/venv/bin/python - <<PY
from ultralytics.utils.downloads import attempt_download_asset
print(attempt_download_asset("${MODEL}"))
PY
        [[ -f "${MODELS_DIR}/${MODEL}" ]] || { echo "ABORT: nao foi possivel obter ${MODEL}"; exit 2; }
        cp -n "${MODELS_DIR}/${MODEL}" "${WORK}/models/"
    fi
fi

# ------------------------- estado (reentrancia) -------------------------------
SUMMARY="$(ls -1t "${WORK}"/runs/*/training_summary.json 2>/dev/null | head -1 || true)"
LAST="$(ls -1t "${WORK}"/runs/*/weights/last.pt 2>/dev/null | head -1 || true)"

if [[ -z "${SUMMARY}" ]]; then
    MODEL_ARG="${MODEL}"
    unset FLYDET_RESUME
    if [[ -n "${LAST}" ]]; then
        echo "RESUME: ${LAST}"
        MODEL_ARG="${LAST}"
        export FLYDET_RESUME=1
    fi

    # ---------------- pre-check CPU (fail-fast): versoes + patch aplica -----------
    b200_apptainer /opt/venv/bin/python - <<'PRECHECK'
import torch, ultralytics
print(f"[precheck] torch={torch.__version__} cuda={torch.version.cuda} ultralytics={ultralytics.__version__}")
assert ultralytics.__version__ == "8.4.142", ultralytics.__version__
from nwd_patch import apply_nwd_patch
cfg = apply_nwd_patch()
print("[precheck] nwd cfg:", cfg)
if cfg.get("nwd"):
    from ultralytics.utils.loss import BboxLoss
    bl = BboxLoss(reg_max=16); B, N, R = 1, 8, 16
    li, ld = bl(torch.rand(B, N, 4*R), torch.tensor([[[0., 0., 40., 40.]]*N]), torch.rand(N, 2)*100,
                torch.tensor([[[1., 1., 41., 41.]]*N]), torch.rand(B, N, 7), torch.tensor(5.0),
                torch.ones(B, N, dtype=torch.bool), torch.tensor([1920., 1920.]), torch.tensor([8.0]))
    assert torch.isfinite(li) and torch.isfinite(ld)
print("PRECHECK_OK")
PRECHECK

    # ---------------- treino (GPU dentro do Slurm) ---------------------------------
    chain_run b200_apptainer --pwd "${WORK}" \
        /opt/venv/bin/python "${REPO}/rq3_tiny_object/train_nwd.py" \
            --backend yolo \
            --model "${MODEL_ARG}" \
            --data "${DATA_YAML}" \
            --project "${WORK}/runs" \
            --epochs "${EPOCHS}" \
            --batch "${BATCH}" \
            --imgsz "${IMGSZ}" \
            --patience "${PATIENCE}" \
            --num-workers "${WORKERS}" \
            --gpus 1 \
            --topk 5 \
            --wandb \
            --wandb-entity "${WANDB_ENTITY}" \
            --wandb-project "${WANDB_PROJECT}" \
            --wandb-run-name "${WANDB_RUN_NAME}"
    SUMMARY="$(ls -1t "${WORK}"/runs/*/training_summary.json 2>/dev/null | head -1 || true)"
    [[ -n "${SUMMARY}" ]] || { echo "ABORT: treino terminou sem training_summary.json"; exit 1; }
else
    echo "SKIP treino: ${SUMMARY}"
fi
echo "JOB→RUN ${WANDB_RUN_NAME}"

# ------------------------- checkpoint oficial + TEST eval ---------------------
RUN_DIR="$(dirname "${SUMMARY}")"
BEST="${RUN_DIR}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "ABORT: best.pt ausente em ${RUN_DIR}/weights"; exit 1; }
ln -sfn "${BEST}" "${WORK}/best.pt"
echo "BEST ${WORK}/best.pt -> ${BEST}"

if ! step_done "${WORK}/test_eval/summary.json"; then
    export WANDB_RUN_NAME_EVAL="${RUN_NAME}_test_eval"
    chain_run b200_apptainer /opt/venv/bin/python "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
        --detector "${BEST}" --detector-names "${RUN_NAME}" \
        --data "${DATA_ROOT}" --split test \
        --out "${WORK}/test_eval" \
        --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "${WANDB_RUN_NAME_EVAL}"
    echo "JOB→RUN ${WANDB_RUN_NAME_EVAL}"
else
    echo "SKIP test_eval: ${WORK}/test_eval/summary.json"
fi

touch "${WORK}/DONE"
echo "DONE ${RUN_NAME} $(date)"
