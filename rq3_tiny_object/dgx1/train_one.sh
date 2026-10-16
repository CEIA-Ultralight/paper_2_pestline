#!/usr/bin/env bash
# =============================================================================
# train_one.sh — um treino de detector da RQ3 DENTRO do container Docker da dgx1.
# Chamado por run_queue.sh (nao rodar direto no host).
#
# Receita identica a common/slurm/train_detector_b200.sh: yolo26m.pt, 150 epocas, batch 8,
# imgsz 1920, patience 30, seed via FLYDET_SEED, checkpoint = best.pt. A unica diferenca entre
# experimentos sao FLYDET_NWD_* / FLYDET_RFLA / MODEL / MODEL_CFG.
#
# Reentrante: WORK=$RUNS_ROOT/$RUN_NAME
#   - $WORK/DONE existe                      -> nada a fazer
#   - $WORK/runs/*/training_summary.json     -> treino ok; so roda o TEST eval
#   - $WORK/runs/*/weights/last.pt           -> retoma (FLYDET_RESUME=1)
#   - senao                                  -> treino novo
# Depois do treino: avaliacao no TEST (per_instance.csv, metrics.json) em $WORK/test_eval/.
# =============================================================================
set -euo pipefail

: "${RUN_NAME:?}" "${FLYDET_SEED:?}" "${REPO:?}" "${DATA_ROOT:?}" "${RUNS_ROOT:?}" "${MODELS_DIR:?}"
export FLYDET_NWD_MODE="${FLYDET_NWD_MODE:-off}"
export FLYDET_NWD_W="${FLYDET_NWD_W:-0.5}"
export FLYDET_NWD_C="${FLYDET_NWD_C:-44.0}"
export FLYDET_NWD_SCOPE="${FLYDET_NWD_SCOPE:-both}"
export FLYDET_RFLA="${FLYDET_RFLA:-0}"
export FLYDET_SAVE_PERIOD="${FLYDET_SAVE_PERIOD:-10}"
export FLYDET_DATASET_ROOT="${DATA_ROOT}"

MODEL="${MODEL:-yolo26m.pt}"
MODEL_CFG="${MODEL_CFG:-}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-1920}"
PATIENCE="${PATIENCE:-30}"
WORKERS="${WORKERS:-8}"
WANDB_ENTITY="${WANDB_ENTITY:-pestline}"
WANDB_PROJECT="${WANDB_PROJECT:?}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${RUN_NAME}}"
DATA_YAML="${DATA_ROOT}/data.yaml"
WORK="${RUNS_ROOT}/${RUN_NAME}"
export PYTHONPATH="${REPO}/rq3_tiny_object:${REPO}/rq3_tiny_object/patches:${REPO}/rq2_cascade:${REPO}/common"
mkdir -p "${WORK}/models" "${WORK}/runs"

echo "============================== ${RUN_NAME} =============================="
echo "Receita: ${MODEL} epochs=${EPOCHS} batch=${BATCH} imgsz=${IMGSZ} patience=${PATIENCE} save_period=${FLYDET_SAVE_PERIOD}"
[[ -z "${MODEL_CFG}" ]] || echo "Arquitetura: ${MODEL_CFG} (pretrained=${MODEL})"
echo "Seed:    FLYDET_SEED=${FLYDET_SEED}  NWD: mode=${FLYDET_NWD_MODE} w=${FLYDET_NWD_W} C=${FLYDET_NWD_C} scope=${FLYDET_NWD_SCOPE} rfla=${FLYDET_RFLA}"
echo "W&B:     ${WANDB_ENTITY}/${WANDB_PROJECT} run=${WANDB_RUN_NAME}"
echo "WORK:    ${WORK}   inicio=$(date)"
echo "========================================================================="

if [[ -f "${WORK}/DONE" ]]; then echo "SKIP ${RUN_NAME}: DONE"; exit 0; fi
[[ -f "${DATA_YAML}" ]] || { echo "ABORT: data.yaml ausente: ${DATA_YAML}"; exit 2; }

# ------------------------- pesos pre-treinados (models/ no cwd do fly_det) ---
if [[ ! -f "${WORK}/models/${MODEL}" ]]; then
    if [[ ! -f "${MODELS_DIR}/${MODEL}" ]]; then
        echo "AVISO: ${MODELS_DIR}/${MODEL} ausente — baixando"
        (cd "${MODELS_DIR}" && python -c "from ultralytics.utils.downloads import attempt_download_asset as d; print(d('${MODEL}'))")
        [[ -f "${MODELS_DIR}/${MODEL}" ]] || { echo "ABORT: nao foi possivel obter ${MODEL}"; exit 2; }
    fi
    cp -n "${MODELS_DIR}/${MODEL}" "${WORK}/models/"
fi

# ------------------------- estado (reentrancia) -------------------------------
SUMMARY="$(ls -1t "${WORK}"/runs/*/training_summary.json 2>/dev/null | head -1 || true)"
LAST="$(ls -1t "${WORK}"/runs/*/weights/last.pt 2>/dev/null | head -1 || true)"

if [[ -z "${SUMMARY}" ]]; then
    MODEL_ARG="${MODEL}"
    unset FLYDET_RESUME FLYDET_PRETRAINED
    if [[ -n "${LAST}" ]]; then
        echo "RESUME: ${LAST}"
        MODEL_ARG="${LAST}"
        export FLYDET_RESUME=1
    elif [[ -n "${MODEL_CFG}" ]]; then
        MODEL_ARG="${MODEL_CFG}"
        export FLYDET_PRETRAINED="${WORK}/models/${MODEL}"
    fi

    # pre-check (fail-fast): versoes + GPU + patch aplica
    python - <<'PRECHECK'
import torch, ultralytics
print(f"[precheck] torch={torch.__version__} cuda={torch.version.cuda} ultralytics={ultralytics.__version__}")
assert ultralytics.__version__ == "8.4.142", ultralytics.__version__
assert torch.cuda.is_available(), "CUDA indisponivel no container"
print(f"[precheck] GPU={torch.cuda.get_device_name(0)} mem={torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GiB")
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

    cd "${WORK}"
    python "${REPO}/rq3_tiny_object/train_nwd.py" \
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

# ------------------------- checkpoint oficial + TEST eval ---------------------
RUN_DIR="$(dirname "${SUMMARY}")"
BEST="${RUN_DIR}/weights/best.pt"
[[ -f "${BEST}" ]] || { echo "ABORT: best.pt ausente em ${RUN_DIR}/weights"; exit 1; }
ln -sfn "${BEST}" "${WORK}/best.pt"
echo "BEST ${WORK}/best.pt -> ${BEST}"

if [[ ! -f "${WORK}/test_eval/summary.json" ]]; then
    rm -rf "${WORK}/test_eval"
    python "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
        --detector "${BEST}" --detector-names "${RUN_NAME}" \
        --data "${DATA_ROOT}" --split test \
        --out "${WORK}/test_eval" \
        --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" \
        --wandb-run-name "${WANDB_RUN_NAME}_test_eval"
else
    echo "SKIP test_eval: ${WORK}/test_eval/summary.json"
fi

touch "${WORK}/DONE"
echo "DONE ${RUN_NAME} $(date)"
