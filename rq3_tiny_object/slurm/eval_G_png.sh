#!/usr/bin/env bash
# Reavalia no TEST completo (73 imgs; 9 HEIC convertidas para PNG) os best.pt da cadeia G. 1 GPU (harness exige CUDA), idempotente.
#SBATCH --job-name=p2-evalG
#SBATCH --gres=gpu:1
#SBATCH --partition=h100n2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/raid/user_marcospaulo/slurm_logs/p2-evalG_%j.out
set -uo pipefail
export GPU_AUDIT_LOG="/raid/user_marcospaulo/slurm_logs/gpu_audit_${SLURM_JOB_ID:-x}.log"
source /raid/user_marcospaulo/paper_2_pestline/common/gpu/slurm/gpu_env.sh
export FLYDET_PARTITION=h100n2 IMAGE=/raid/user_marcospaulo/containers/flydet-train.sif
source /raid/user_marcospaulo/paper_2_pestline/common/slurm/b200_common.sh
U=/raid/user_marcospaulo; R=$U/flydet_runs_v9; REPO=$U/paper_2_pestline
for RUN in G0_e0_h100_s{0,1,2} G1_nwd_w0.5_C22_s{0,1,2} G2_nwd_w1.0_C44_s{0,1,2} G3_nwd_w0.5_C88_s{0,1,2}; do
  [[ -f $R/$RUN/DONE ]] || continue
  [[ -f $R/$RUN/test_eval_png/summary.json ]] && { echo "SKIP $RUN"; continue; }
  echo "EVAL $RUN"
  b200_apptainer /opt/venv/bin/python $REPO/rq2_cascade/eval_cascade_matrix.py \
    --detector $R/$RUN/best.pt --detector-names $RUN --data $U/datasets/DS-F2_v9.0.5_evalpng --split test \
    --out $R/$RUN/test_eval_png || echo "FALHA $RUN"
done
echo EVAL_G_DONE
