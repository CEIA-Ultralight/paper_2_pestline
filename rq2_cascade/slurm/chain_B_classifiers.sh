#!/usr/bin/env bash
# =============================================================================
# chain_B_classifiers.sh — Cadeia B (b200n1): classificadores ConvNeXt-T da cascata.
#
#   variant=baseline (CE)      seeds 42, 84, 126 -> clf_convnext_t_ce_s<seed>
#   variant=parts ("partes")   seeds 42, 84, 126 -> clf_convnext_t_parts_s<seed>
# Dados: $CROPS = /raid/user_marcospaulo/datasets/DS-F2_crops_pad75 (ImageFolder
# train/val/test, 7 classes; verificado no no dgx-B200-1). Avaliacao no VAL
# (selecao por species_macro_f1 dentro de architecture_lab.train).
# Independente da cadeia A. Reentrante (summary.json succeeded -> SKIP; last.pt -> --resume).
#
# Submissao: sbatch rq2_cascade/slurm/chain_B_classifiers.sh
# Custo estimado: 6 x <= 1.25 GPU-h (MAX_HOURS) ~= <= 7.5 GPU-h (B200 tende a ~0.5 h cada).
# Saidas: $RUNS_ROOT/clf_convnext_t_{ce,parts}_s{42,84,126}/{best.pt,last.pt,summary.json}
# W&B: pestline/paper2-rq2-cascade runs clf_convnext_t_*_s*.
# =============================================================================
#SBATCH --job-name=p2-chainB-classifiers
#SBATCH --partition=b200n1
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

set -euo pipefail
# sbatch copia o script para /var/spool/slurmd/<job>/ -> nao usar dirname(BASH_SOURCE)
REPO="${REPO:-/raid/user_marcospaulo/paper_2_pestline}"
source "${REPO}/common/slurm/b200_common.sh"
b200_guard
TRAIN="${REPO}/rq2_cascade/slurm/train_classifier_b200.sh"
export WANDB_PROJECT="${WANDB_PROJECT_RQ2}"
SEEDS="${SEEDS:-42 84 126}"
VARIANTS="${VARIANTS:-baseline parts}"

clf_done() {  # $1 = OUT dir; architecture_lab grava status completed|early_stop|budget|interrupted|failed
    [[ -f "$1/summary.json" ]] && grep -q '"exit_code": *0[,}]' "$1/summary.json" && grep -q '"export_complete": *true' "$1/summary.json"
}

echo "CHAIN B inicio $(date) job=${SLURM_JOB_ID} variants=${VARIANTS} seeds=${SEEDS}"
for v in ${VARIANTS}; do
    tag="${v}"; [[ "${v}" == baseline ]] && tag=ce
    for s in ${SEEDS}; do
        RUN="clf_convnext_t_${tag}_s${s}"
        if clf_done "${RUNS_ROOT}/${RUN}"; then echo "SKIP ${RUN}"; continue; fi
        chain_run env VARIANT="${v}" SEED="${s}" RUN_NAME="${RUN}" WANDB_RUN_NAME="${RUN}" bash "${TRAIN}"
    done
done

touch "${RUNS_ROOT}/CHAIN_B_DONE"
echo "CHAIN_B_DONE $(date)"
