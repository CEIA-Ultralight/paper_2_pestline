#!/usr/bin/env bash
# =============================================================================
# chain_pad_ablation_h100.sh — ablacao da margem de contexto do crop (RQ2, M2) na h100n2, UM job, 1 GPU.
#   pad (por lado, fracao da caixa) em ${PADS:-"00 25"} -> 0.00 e 0.25 (referencia do paper: 0.75, ja treinado).
#   1) extrai crops GT do v9.0.5 (CPU, container) -> datasets/DS-F2_v9.0.5_crops_pad<P>
#   2) treina ConvNeXt-T CE (receita do paper: 24 ep., batch 32, 384, lr 8e-5, patience 5) seeds 42/84/126
#   3) cascata (E0_s0..2 x 3 classificadores, label_only) em VAL e TEST com FLYDET_CROP_PAD=<pad>
# Reentrante (marcadores DONE/summary.json, last.pt, SIGUSR1 -> rc 75 -> requeue). Sem W&B no treino;
# a matriz de avaliacao exige W&B online (mesmo contrato da cadeia D).
# Submeter (protocolo): sbatch --hold ...; gpu_watch no node; Monitor; scontrol release.
# =============================================================================
#SBATCH --job-name=p2-pad-abl
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGUSR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/%x_%j.out

REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
export GPU_AUDIT_LOG="/raid/user_marcospaulo/slurm_logs/gpu_audit_${SLURM_JOB_ID:-x}.log"
source "${REPO}/common/gpu/slurm/gpu_env.sh"   # re-executa o script dentro do jail de GPU
set -euo pipefail
export FLYDET_PARTITION=h100n2
export IMAGE=/raid/user_marcospaulo/containers/flydet-train.sif
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
export WANDB_PROJECT="${WANDB_PROJECT_RQ2}"; export WANDB_RUN_GROUP="pad-ablation"
export WANDB_TAGS="rq2,pad-ablation,h100,DS-F2_v9.0.5"
PADS="${PADS:-00 25}"
DETS=(); DET_NAMES=()
for n in E0_s0 E0_s1 E0_s2; do
    f="${RUNS_ROOT}/${n}/best.pt"; [[ -f "${f}" ]] || { echo "ABORT: ${f} ausente"; exit 2; }
    DETS+=("${f}"); DET_NAMES+=("${n}")
done
echo "PAD ABLATION inicio $(date) job=${SLURM_JOB_ID} pads=${PADS}"
for P in ${PADS}; do
    PADF="0.${P}"; [[ "${P}" == "00" ]] && PADF="0.0"
    CROPS_P="${USER_ROOT}/datasets/DS-F2_v9.0.5_crops_pad${P}"
    if ! step_done "${CROPS_P}/DONE"; then
        chain_run apptainer exec --bind "${USER_ROOT}:${USER_ROOT}" "${IMAGE}" /opt/venv/bin/python \
            "${REPO}/rq2_cascade/extract_crops.py" --src "${DATA_ROOT}" --dst "${CROPS_P}" --pad "${PADF}"
        touch "${CROPS_P}/DONE"
    fi
    CLFS=(); CLF_NAMES=()
    for S in 42 84 126; do
        RUN="clf_convnext_t_ce_pad${P}_s${S}"; OUT="${RUNS_ROOT}/${RUN}"
        if [[ -f "${OUT}/summary.json" && -f "${OUT}/best.pt" ]] && grep -q '"export_complete": *true' "${OUT}/summary.json"; then
            echo "SKIP ${RUN}"
        else
            mkdir -p "${OUT}"; RESUME=(); [[ -f "${OUT}/last.pt" ]] && RESUME+=(--resume)
            chain_run b200_apptainer --pwd "${REPO}/rq2_cascade" /opt/venv/bin/python -m architecture_lab.train \
                --data "${CROPS_P}" --out "${OUT}" --variant baseline --epochs 24 --patience 5 --batch 32 \
                --img-size 384 --lr 8e-5 --seed "${S}" --workers 8 --max-hours 1.25 --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" \
                --wandb-run-name "${RUN}-h100" "${RESUME[@]}"
        fi
        [[ -f "${OUT}/best.pt" ]] || { echo "ABORT: best.pt ausente em ${OUT}"; exit 1; }
        CLFS+=("${OUT}/best.pt"); CLF_NAMES+=("${RUN}")
    done
    for SPLIT in val test; do
        RUN="cascade_pad${P}_${SPLIT}"; OUT="${RUNS_ROOT}/${RUN}"
        if step_done "${OUT}/summary.json"; then echo "SKIP ${RUN}"; continue; fi
        mkdir -p "${OUT}"
        export FLYDET_CROP_PAD="${PADF}"
        chain_run b200_apptainer --pwd "${REPO}" /opt/venv/bin/python \
            "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
            --detector "${DETS[@]}" --detector-names "${DET_NAMES[@]}" \
            --classifier "${CLFS[@]}" --classifier-names "${CLF_NAMES[@]}" --policies label_only \
            --data "${DATA_ROOT}" --split "${SPLIT}" --out "${OUT}" \
            --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}-h100"
    done
done
touch "${RUNS_ROOT}/PAD_ABL_CHAIN_DONE"; echo "PAD_ABL_CHAIN_DONE $(date)"
