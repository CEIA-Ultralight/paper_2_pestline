#!/usr/bin/env bash
# =============================================================================
# chain_extras_v9.sh — Extras da RQ2 no DS-F2_v9.0.5 (b200n1), UM job, 1 GPU, etapas OPCIONAIS (chain_try):
# a falha de uma etapa e registrada em $RUNS_ROOT/extras/<etapa>.FAIL e a cadeia segue; sucesso -> <etapa>.OK.
# Pre-requisito: cadeia core (CHAIN_A_DONE e CHAIN_B_DONE em $RUNS_ROOT). Submeter com
#   sbatch --dependency=afterok:<core>?afterok:<reserva1>?afterok:<reserva2> --kill-on-invalid-dep=yes <este script>
#
#   1  label_issues     rotulos suspeitos (confident learning) em VAL/TEST com os 6 classificadores CE/partes
#   2  prop_crops       crops das propostas do E0_s0 (train/val) com 8a classe BG
#   3  clf_bg           ConvNeXt-T CE com BG x seeds 42/84/126
#   4  matrix_bg        cascata E0 x {BG} em VAL/TEST (+ oraculo)
#   5  rtdetr_smoke     RT-DETR-l, 1 epoca (+ avaliacao TEST); define o batch
#   6  rtdetr           RT-DETR-l x seeds 0/1/2 (receita E0)            [so se o smoke passou]
#   7  clf_dinov2       DINOv2-B/14 x 3 seeds (fine-tune completo, lr 2e-5)
#   8  clf_dinov3       DINOv3-B/16 x 3 seeds (idem)
#   9  clf_bioclip2     BioCLIP 2 (ViT-L/14, img 224) x 3 seeds (idem)
#  10  matrix_found     cascata {E0 x3, yolo26l} x {DINOv2, DINOv3, BioCLIP 2} em VAL/TEST (+ oraculo)
#  11  matrix_rtdetr    cascata RT-DETR x CE em VAL/TEST                 [so se o treino passou]
# Fine-tune completo e nao LP-FT (o trainer nao tem fase de congelamento); declarar no paper.
# Custo estimado ~16-22 GPU-h. Reentrante (marcadores .OK, DONE, summary.json) e com requeue.
# =============================================================================
#SBATCH --job-name=p2v9-extras
#SBATCH --partition=b200n1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

set -euo pipefail
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
export MAX_HOURS="${MAX_HOURS:-2.5}"
export FLYDET_SOURCE_COMMIT="${FLYDET_SOURCE_COMMIT:-$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo unknown)}"
export HF_HUB_OFFLINE=1
step_done "${RUNS_ROOT}/CHAIN_A_DONE" && step_done "${RUNS_ROOT}/CHAIN_B_DONE" || { echo "ABORT: core v9 incompleto"; exit 2; }

X="${RUNS_ROOT}/extras"; mkdir -p "${X}"
PROP="${USER_ROOT}/datasets/DS-F2_v9.0.5_crops_proposals"
TRAIN_DET="${REPO}/common/slurm/train_detector_b200.sh"
TRAIN_CLF="${REPO}/rq2_cascade/slurm/train_classifier_b200.sh"
export WANDB_PROJECT="${WANDB_PROJECT_RQ2}"
SEEDS_C="42 84 126"; SEEDS_D="0 1 2"

# step NAME CMD... : pula se .OK; roda como etapa opcional; grava .OK ou .FAIL
step() {
    local name="$1"; shift
    if [[ -f "${X}/${name}.OK" ]]; then echo "SKIP ${name} (.OK)"; return 0; fi
    rm -f "${X}/${name}.FAIL"
    chain_try "$@"
    if [[ "${CHAIN_RC}" == 0 ]]; then touch "${X}/${name}.OK"; echo "STEP_OK ${name} $(date)"
    else echo "rc=${CHAIN_RC} $(date)" > "${X}/${name}.FAIL"; echo "STEP_FAIL ${name} rc=${CHAIN_RC}"; fi
}
ok() { [[ -f "${X}/$1.OK" ]]; }
clf_ok() { [[ -f "${RUNS_ROOT}/$1/best.pt" ]] && grep -q '"exit_code": *0[,}]' "${RUNS_ROOT}/$1/summary.json" 2>/dev/null; }

train_clf() {  # NAME(sem seed) VARIANT [VAR=VAL ...]  -> 3 seeds
    local tag="$1" variant="$2"; shift 2
    for s in ${SEEDS_C}; do
        local run="clf_${tag}_s${s}"
        step "${run}" env VARIANT="${variant}" SEED="${s}" RUN_NAME="${run}" WANDB_RUN_NAME="${run}" "$@" bash "${TRAIN_CLF}"
    done
}

matrix() {  # TAG DETS(sep. espaço) CLFS(sep. espaço) -> VAL e TEST: matriz + oraculo
    local tag="$1" dets="$2" clfs="$3" split
    local DETS=() DNAMES=() CLFP=() CNAMES=() n
    for n in ${dets}; do [[ -f "${RUNS_ROOT}/${n}/best.pt" ]] && { DETS+=("${RUNS_ROOT}/${n}/best.pt"); DNAMES+=("${n}"); }; done
    for n in ${clfs}; do clf_ok "${n}" && { CLFP+=("${RUNS_ROOT}/${n}/best.pt"); CNAMES+=("${n}"); }; done
    if (( ${#DETS[@]} == 0 || ${#CLFP[@]} == 0 )); then echo "matrix ${tag}: sem detectores/classificadores prontos"; return 1; fi
    for split in val test; do
        local out="${RUNS_ROOT}/cascade_${tag}_${split}" oo="${RUNS_ROOT}/oracle_${tag}_${split}"
        step "matrix_${tag}_${split}" b200_apptainer /opt/venv/bin/python "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
            --detector "${DETS[@]}" --detector-names "${DNAMES[@]}" --classifier "${CLFP[@]}" --classifier-names "${CNAMES[@]}" \
            --policies label_only product --data "${DATA_ROOT}" --split "${split}" --out "${out}" \
            --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "cascade_${tag}_${split}"
        step "oracle_${tag}_${split}" b200_apptainer /opt/venv/bin/python "${REPO}/rq2_cascade/eval_oracle_crops.py" \
            --classifier "${CLFP[@]}" --classifier-names "${CNAMES[@]}" --data "${DATA_ROOT}" --split "${split}" --out "${oo}" \
            --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "oracle_${tag}_${split}"
    done
}

echo "EXTRAS v9 inicio $(date) job=${SLURM_JOB_ID} commit=${FLYDET_SOURCE_COMMIT}"

# 1) rotulos suspeitos (so inferencia, minutos)
BASE_CK=(); for tag in ce parts; do for s in ${SEEDS_C}; do BASE_CK+=("${RUNS_ROOT}/clf_convnext_t_${tag}_s${s}/best.pt"); done; done
for split in val test; do
    step "label_issues_${split}" b200_apptainer --pwd "${REPO}/rq2_cascade" /opt/venv/bin/python label_issues.py \
        --crops "${CROPS}" --split "${split}" --classifier "${BASE_CK[@]}" --out "${RUNS_ROOT}/label_issues"
done

# 2-4) classe de fundo
step prop_crops b200_apptainer --pwd "${REPO}/rq2_cascade" /opt/venv/bin/python extract_proposal_crops.py \
    --detector "${RUNS_ROOT}/E0_s0/best.pt" --data "${DATA_ROOT}" --out "${PROP}"
if ok prop_crops; then
    train_clf convnext_t_bg baseline WITH_BG=1 CROPS="${PROP}" LR=8e-5 IMG=384
    matrix bg "E0_s0 E0_s1 E0_s2" "clf_convnext_t_bg_s42 clf_convnext_t_bg_s84 clf_convnext_t_bg_s126" || true
fi

# 5-6) RT-DETR
RT_BATCH=""
for b in 8 4; do
    if [[ -z "${RT_BATCH}" ]]; then
        rm -rf "${RUNS_ROOT}/smoke_rtdetr"
        step "rtdetr_smoke_b${b}" env RUN_NAME=smoke_rtdetr FLYDET_SEED=0 MODEL=rtdetr-l.pt EPOCHS=1 BATCH="${b}" \
            WANDB_PROJECT=paper2-smoke WANDB_RUN_NAME=smoke_rtdetr bash "${TRAIN_DET}"
        { ok "rtdetr_smoke_b${b}" && RT_BATCH="${b}"; } || true
    fi
done
if [[ -n "${RT_BATCH}" ]]; then
    for s in ${SEEDS_D}; do
        run="E8_rtdetr_s${s}"
        step "${run}" env RUN_NAME="${run}" FLYDET_SEED="${s}" MODEL=rtdetr-l.pt BATCH="${RT_BATCH}" \
            WANDB_PROJECT="${WANDB_PROJECT_RQ3}" WANDB_RUN_NAME="${run}" bash "${TRAIN_DET}"
    done
else
    echo "RT-DETR: smoke falhou com batch 8 e 4; etapas 6 e 11 puladas"
fi

# 7-9) foundation models
train_clf dinov2 dinov2 LR=2e-5 IMG=384
train_clf dinov3 dinov3 LR=2e-5 IMG=384
train_clf bioclip2 bioclip2 LR=2e-5 IMG=224

# 10-11) matrizes
matrix found "E0_s0 E0_s1 E0_s2 E7_yolo26l_s0" \
    "clf_dinov2_s42 clf_dinov2_s84 clf_dinov2_s126 clf_dinov3_s42 clf_dinov3_s84 clf_dinov3_s126 clf_bioclip2_s42 clf_bioclip2_s84 clf_bioclip2_s126" || true
matrix rtdetr "E8_rtdetr_s0 E8_rtdetr_s1 E8_rtdetr_s2" \
    "clf_convnext_t_ce_s42 clf_convnext_t_ce_s84 clf_convnext_t_ce_s126" || true

touch "${RUNS_ROOT}/CHAIN_EXTRAS_DONE"
echo "EXTRAS v9 FIM $(date); resumo: $(ls "${X}"/*.OK 2>/dev/null | wc -l) etapas OK, $(ls "${X}"/*.FAIL 2>/dev/null | wc -l) falhas (ver ${X}/*.FAIL)"
