#!/usr/bin/env bash
# =============================================================================
# Item 6 — Avaliacao TTA + WBF no VAL dos detectores campeoes.
# Sem treino: consome checkpoints existentes (E0, E10, e quaisquer novos).
# Submetido ao FINAL da cadeia E10x (quando E11/E12/E13 ja existem).
# =============================================================================
#SBATCH --job-name=flydet-tta-wbf
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:SIGTERM@300
#SBATCH --output=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.out
#SBATCH --error=/raid/user_marcospaulo/fly-det/slurm_logs/%x_%j.err

set -euo pipefail
ROOT=/raid/user_marcospaulo
OUT="$ROOT/flydet_runs/tta-wbf-20260921"
DATA_YAML="$ROOT/datasets/DS-F2_v8.1.1/data.yaml"
mkdir -p "$OUT" "$ROOT/fly-det/slurm_logs"

DET_E0="$ROOT/flydet_runs/E0_yolo26m_baseline/runs/yolo26m.pt/weights/best.pt"
DET_E10="$ROOT/flydet_runs/E10_yolo26m_nwd/runs/yolo26m.pt/weights/best.pt"
DET_E11="$ROOT/flydet_runs/E10_yolo26m_nwd_w075/runs/yolo26m.pt/weights/best.pt"
DET_E12="$ROOT/flydet_runs/E10_yolo26m_nwd_w100/runs/yolo26m.pt/weights/best.pt"
DET_E10B="$ROOT/flydet_runs/E10b_yolo26m_gcd/runs/yolo26m.pt/weights/best.pt"
DET_E13="$ROOT/flydet_runs/E13_yolo26m_multiscale/runs/yolo26m.pt/weights/best.pt"
DET_E7="$ROOT/flydet_runs/E7_yolo26l_nwd/runs/yolo26m.pt/weights/best.pt"
DET_E14="$ROOT/flydet_runs/E14_nwd_rfla/runs/yolo26m.pt/weights/best.pt"
DET_E10S1="$ROOT/flydet_runs/E10_nwd_seed1/runs/yolo26m.pt/weights/best.pt"
DET_E10S2="$ROOT/flydet_runs/E10_nwd_seed2/runs/yolo26m.pt/weights/best.pt"
DET_E0S1="$ROOT/flydet_runs/E0_baseline_seed1/runs/yolo26m.pt/weights/best.pt"
DET_E0S2="$ROOT/flydet_runs/E0_baseline_seed2/runs/yolo26m.pt/weights/best.pt"

DETS=()
for spec in "e0=$DET_E0" "e10=$DET_E10" "e11=$DET_E11" "e12=$DET_E12" "e10b=$DET_E10B" \
            "e13=$DET_E13" "e7=$DET_E7" "e14=$DET_E14" \
            "e10s1=$DET_E10S1" "e10s2=$DET_E10S2" "e0s1=$DET_E0S1" "e0s2=$DET_E0S2"; do
  ck="${spec#*=}"
  [[ -f "$ck" ]] && DETS+=("$spec") || echo "SKIP $spec (sem checkpoint)"
done
echo "Detectores: ${DETS[*]}"

apptainer exec --nv \
  --bind "$ROOT:$ROOT" \
  "$ROOT/containers/flydet-train.sif" \
  /opt/venv/bin/python "$ROOT/fly-det/fly-det/scripts/eval_wbf_tta.py" \
    --data "$DATA_YAML" \
    --detectors "${DETS[@]}" \
    --out "$OUT" \
    --imgsz 1920

echo "TTA_WBF_COMPLETE $OUT"
