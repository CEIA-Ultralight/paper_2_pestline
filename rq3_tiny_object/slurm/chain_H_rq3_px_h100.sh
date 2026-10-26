#!/usr/bin/env bash
# =============================================================================
# chain_H_rq3_px_h100.sh — RQ3: variantes NWD/GCD/RFLA com a loss em PIXELS, h100n2, 3 seeds.
# Refaz os runs afetados pelo bug de unidade (PAPER_FACTS §12.1): a cadeia G (campanha c) e o
# NWD-loss / NWD+RFLA / GCD da campanha (a) rodaram com a loss em grid. Controle: G0_e0_h100_s{0,1,2}
# (cadeia G, mesmo no, container e receita; E0 nao usa patch, nao foi afetado).
#   H1 NWD w0.5 C44 both | H2 NWD w0.5 C44 loss | H3 NWD w0.5 C44 both + RFLA | H4 NWD w0.5 C22 both
#   H5 NWD w0.5 C88 both | H6 NWD w1.0 C44 both | H7 GCD w0.5 both
# Bracos e ordem via env H_ARMS (ex.: H_ARMS="H2 H1 H5 H7"); seeds via H_SEEDS (default "0 1 2").
# Blindagem: FLYDET_NWD_LOSS_UNITS=px + FLYDET_REQUIRE_PX=1 (patch aborta em grid, precheck numerico
# nos strides 8/16/32, gate antes do DONE: nwd_config/nwd_diag em px e TEST 73 imgs / 1210 especie).
# Idempotente (DONE) e reentrante (rc 75 -> requeue). Submissao: hold, jail, gpu_watch, Monitor, release.
# =============================================================================
#SBATCH --job-name=p2-chainH-rq3
#SBATCH --partition=h100n2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
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
TRAIN="${REPO}/common/slurm/train_detector_b200.sh"
export WANDB_PROJECT="${WANDB_PROJECT_RQ3}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-rq3_px_h100}"
export WANDB_TAGS="rq3,px,h100,DS-F2_v9.0.5"
export FLYDET_NWD_LOSS_UNITS=px FLYDET_REQUIRE_PX=1 FLYDET_NWD_DIAG=20
export REEVAL_RUNS=""
export EVAL_DATA_ROOT="${USER_ROOT}/datasets/DS-F2_v9.0.5_evalpng"
n_test=$(ls "${EVAL_DATA_ROOT}/images/test" | wc -l)
[[ "${n_test}" == 73 ]] || { echo "ABORT: ${EVAL_DATA_ROOT}/images/test tem ${n_test} imagens (esperado 73)"; exit 2; }
: "${H_ARMS:?Defina H_ARMS (ex.: H2 H1 H5 H7)}"
H_SEEDS="${H_SEEDS:-0 1 2}"
echo "CHAIN H inicio $(date) job=${SLURM_JOB_ID} arms=[${H_ARMS}] seeds=[${H_SEEDS}] RUNS_ROOT=${RUNS_ROOT}"

declare -A ARM_NAME=( [H1]=H1_nwd_px_w0.5_C44 [H2]=H2_nwdloss_px_w0.5_C44 [H3]=H3_nwd_rfla_px_w0.5_C44
                      [H4]=H4_nwd_px_w0.5_C22 [H5]=H5_nwd_px_w0.5_C88 [H6]=H6_nwd_px_w1.0_C44 [H7]=H7_gcd_px_w0.5 )
declare -A ARM_CFG=(  [H1]="nwd 0.5 44.0 both 0" [H2]="nwd 0.5 44.0 loss 0" [H3]="nwd 0.5 44.0 both 1"
                      [H4]="nwd 0.5 22.0 both 0" [H5]="nwd 0.5 88.0 both 0" [H6]="nwd 1.0 44.0 both 0"
                      [H7]="gcd 0.5 44.0 both 0" )

train_step() {  # ARM SEED
    local arm="$1" s="$2" mode w c scope rfla
    read -r mode w c scope rfla <<< "${ARM_CFG[$arm]}"
    local RUN="${ARM_NAME[$arm]}_s${s}"
    if step_done "${RUNS_ROOT}/${RUN}/DONE"; then echo "SKIP ${RUN}"; return 0; fi
    chain_run env RUN_NAME="${RUN}" FLYDET_SEED="${s}" FLYDET_NWD_MODE="${mode}" FLYDET_NWD_W="${w}" \
        FLYDET_NWD_C="${c}" FLYDET_NWD_SCOPE="${scope}" FLYDET_RFLA="${rfla}" WANDB_RUN_NAME="${RUN}" bash "${TRAIN}"
}

for arm in ${H_ARMS}; do
    [[ -n "${ARM_CFG[$arm]:-}" ]] || { echo "ABORT: braco desconhecido ${arm}"; exit 2; }
done
for arm in ${H_ARMS}; do
    for s in ${H_SEEDS}; do train_step "${arm}" "${s}"; done
done

touch "${RUNS_ROOT}/CHAIN_H_DONE_${H_ARMS// /_}"
echo "CHAIN_H_DONE arms=[${H_ARMS}] $(date)"
