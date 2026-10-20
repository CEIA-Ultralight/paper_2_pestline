#!/usr/bin/env bash
# chain_pad75_ref_h100.sh — controle de hardware da ablacao de margem: reavalia na h100n2 a cascata do paper
# (E0_s0..2 x clf_convnext_t_ce_s{42,84,126}, pad 0.75, treinados na B200) em VAL e TEST. So avaliacao, 1 GPU.
# Saida: $RUNS_ROOT/cascade_pad75_{val,test}. Mesmo protocolo de submissao (hold, jail, gpu_watch, Monitor).
#SBATCH --job-name=p2-pad75-ref
#SBATCH --partition=h100n2
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

REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
export GPU_AUDIT_LOG="/raid/user_marcospaulo/slurm_logs/gpu_audit_${SLURM_JOB_ID:-x}.log"
source "${REPO}/common/gpu/slurm/gpu_env.sh"
set -euo pipefail
export FLYDET_PARTITION=h100n2
export IMAGE=/raid/user_marcospaulo/containers/flydet-train.sif
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
export WANDB_PROJECT="${WANDB_PROJECT_RQ2}"; export WANDB_RUN_GROUP="pad-ablation"
export WANDB_TAGS="rq2,pad-ablation,pad75-ref,h100,DS-F2_v9.0.5"
DETS=(); DET_NAMES=(); CLFS=(); CLF_NAMES=()
for n in E0_s0 E0_s1 E0_s2; do DETS+=("${RUNS_ROOT}/${n}/best.pt"); DET_NAMES+=("${n}"); done
for s in 42 84 126; do n="clf_convnext_t_ce_s${s}"; CLFS+=("${RUNS_ROOT}/${n}/best.pt"); CLF_NAMES+=("${n}"); done
for f in "${DETS[@]}" "${CLFS[@]}"; do [[ -f "$f" ]] || { echo "ABORT: $f ausente"; exit 2; }; done
export FLYDET_CROP_PAD=0.75
for SPLIT in val test; do
    RUN="cascade_pad75_${SPLIT}"; OUT="${RUNS_ROOT}/${RUN}"
    if step_done "${OUT}/summary.json"; then echo "SKIP ${RUN}"; continue; fi
    mkdir -p "${OUT}"
    chain_run b200_apptainer --pwd "${REPO}" /opt/venv/bin/python "${REPO}/rq2_cascade/eval_cascade_matrix.py" \
        --detector "${DETS[@]}" --detector-names "${DET_NAMES[@]}" \
        --classifier "${CLFS[@]}" --classifier-names "${CLF_NAMES[@]}" --policies label_only \
        --data "${DATA_ROOT}" --split "${SPLIT}" --out "${OUT}" \
        --wandb --wandb-entity "${WANDB_ENTITY}" --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${RUN}-h100"
done
echo "PAD75_REF_DONE $(date)"
