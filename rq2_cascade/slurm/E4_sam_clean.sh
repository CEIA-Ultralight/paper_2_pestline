#!/usr/bin/env bash
# =============================================================================
# E4-prep — Limpa os crops com SAM (guiado por bbox) p/ remover fundo/cola.
# Gera DS-F2_crops_sam. GPU leve, via Slurm, partição h100n2.
# Submeter:   sbatch fly-det/slurm/E4_sam_clean.sh
# =============================================================================
#SBATCH --job-name=E4-sam-clean
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail
USER_ROOT="/raid/user_marcospaulo"
IMAGE="${USER_ROOT}/containers/flydet-train.sif"
REPO="${USER_ROOT}/fly-det"
SRC="${USER_ROOT}/datasets/DS-F2_crops_pad75"
DST="${USER_ROOT}/datasets/DS-F2_crops_sam"
CKPT="${USER_ROOT}/models/sam_vit_b_01ec64.pth"
PYLIBS="${USER_ROOT}/pylibs"
mkdir -p "${DST}" "${REPO}/slurm_logs"

echo "==== E4 SAM clean | src=${SRC} -> dst=${DST} | $(date) ===="

apptainer exec --nv \
  --bind "${USER_ROOT}:${USER_ROOT}" \
  --env PYTHONPATH="${PYLIBS}" \
  --env HF_HOME="${USER_ROOT}/.cache/huggingface" \
  --env TORCH_HOME="${USER_ROOT}/.cache/torch" \
  "${IMAGE}" \
  /opt/venv/bin/python "${REPO}/fly-det/scripts/sam_clean_crops.py" \
    --src "${SRC}" \
    --dst "${DST}" \
    --ckpt "${CKPT}" \
    --model-type vit_b \
    --device cuda

echo "Fim: $(date)"
