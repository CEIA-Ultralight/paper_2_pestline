#!/usr/bin/env bash
#
# Sweep sequencial YOLO em VARIOS datasets.
#
# Para cada subdiretorio de DATASETS_DIR que contenha um data.yaml, executa
# train-many-yolos.sh com o nome do dataset como prefixo do run_name no W&B.
# O nome final da run fica: "<dataset>_<familia>_<tamanho>" (ex: "ds_v2_yolo12_n").
#
# Requisitos:
#   - train-many-yolos.sh no mesmo diretorio que este script.
#   - fly-det-train (ou o comando em FLYDET_TRAIN_CMD) no PATH.
#   - Autenticacao W&B via WANDB_API_KEY ou `wandb login`.
#
# Variaveis de ambiente (principais):
#   DATASETS_DIR       — pasta com um subdiretorio por dataset (obrigatorio ou inferido).
#                        Default: <raiz do repo>/datasets  (dois niveis acima deste script).
#   FLYDET_TRAIN_CMD   — comando para chamar o trainer (default: fly-det-train).
#                        Use "uv run fly-det-train" para ambientes gerenciados com uv.
#   WANDB_ENTITY       — team/user W&B
#   WANDB_PROJECT      — projeto W&B
#   WANDB_API_KEY      — chave de autenticacao W&B
#   EPOCHS, BATCH, PATIENCE, IMGSZ, IOU, PROJECT, GPUS, TOPK — repassados ao sweep
#   SWEEP_FAMILIES     — familias separadas por espaco (default: yolo12 yolo26)
#   SWEEP_SIZES        — sufixos de tamanho (default: n s m l x)
#   EXTRA_TRAIN_ARGS   — argumentos extras para fly-det-train
#   WANDB_SKIP_EXISTING_RUN — 1/true (default): pula run se ja existir no W&B
#   FLYDET_DATASET_ROOT — se definido, sobrepoe o root auto-detectado do dataset
#
# Uso:
#   train-many-datasets.sh [opcoes] <num_instances> <cur_instance>
#
# Opcoes (repassadas ao train-many-yolos.sh):
#   --wandb-project <nome>
#   --wandb-entity  <slug>
#   --state-dir     <dir>    pasta para arquivos .done (namespace por dataset automatico)
#
# Exemplo (uma maquina, todos os datasets e modelos, usando uv):
#   export WANDB_API_KEY="..." WANDB_ENTITY=meu-time WANDB_PROJECT=fly-yolo
#   export FLYDET_TRAIN_CMD="uv run fly-det-train"
#   bash scripts/train-many-datasets.sh --wandb-entity meu-time --wandb-project fly-yolo 1 0
#
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SCRIPT_DIR}/train-many-yolos.sh"

if [ ! -f "$SWEEP_SCRIPT" ]; then
  echo "Error: train-many-yolos.sh not found at: ${SWEEP_SCRIPT}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Parse CLI args (forwarded verbatim to train-many-yolos.sh)
# ---------------------------------------------------------------------------
usage() {
  echo "Usage: $0 [options] <num_instances> <cur_instance>" >&2
  echo "  Options:" >&2
  echo "    --wandb-project <name>   W&B project (overrides WANDB_PROJECT env)" >&2
  echo "    --wandb-entity <slug>    W&B entity/team (overrides WANDB_ENTITY env)" >&2
  echo "    --state-dir <dir>        base dir for .done markers (namespaced per dataset)" >&2
  exit 1
}

num_instances=""
cur_instance=""
state_dir_base=""
passthrough_args=()   # forwarded as-is to train-many-yolos.sh

while [ "$#" -gt 0 ]; do
  case "$1" in
    --wandb-project)
      [ "$#" -lt 2 ] && { echo "Error: --wandb-project requires a value" >&2; exit 1; }
      passthrough_args+=("$1" "$2")
      shift 2
      ;;
    --wandb-entity)
      [ "$#" -lt 2 ] && { echo "Error: --wandb-entity requires a value" >&2; exit 1; }
      passthrough_args+=("$1" "$2")
      shift 2
      ;;
    --state-dir)
      [ "$#" -lt 2 ] && { echo "Error: --state-dir requires a value" >&2; exit 1; }
      state_dir_base="$2"
      shift 2
      ;;
    -*)
      echo "Error: unknown option: $1" >&2
      usage
      ;;
    *)
      if [ -z "$num_instances" ]; then
        num_instances="$1"
      elif [ -z "$cur_instance" ]; then
        cur_instance="$1"
      else
        echo "Error: unexpected argument: $1" >&2
        usage
      fi
      shift
      ;;
  esac
done

if [ -z "$num_instances" ] || [ -z "$cur_instance" ]; then
  usage
fi

if ! [[ "$num_instances" =~ ^[0-9]+$ ]] || [ "$num_instances" -lt 1 ]; then
  echo "Error: num_instances must be a positive integer" >&2
  exit 1
fi

if ! [[ "$cur_instance" =~ ^[0-9]+$ ]]; then
  echo "Error: cur_instance must be a non-negative integer (0-based)" >&2
  exit 1
fi

if [ "$cur_instance" -ge "$num_instances" ]; then
  echo "Error: cur_instance must satisfy 0 <= cur_instance < num_instances" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Resolve DATASETS_DIR
# ---------------------------------------------------------------------------
# Default: <repo_root>/datasets  (script lives in fly-det/scripts/, repo root is 2 dirs up)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASETS_DIR="${DATASETS_DIR:-${REPO_ROOT}/datasets}"

if [ ! -d "$DATASETS_DIR" ]; then
  echo "Error: DATASETS_DIR not found: ${DATASETS_DIR}" >&2
  echo "  Set DATASETS_DIR to the folder containing one subdirectory per dataset." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Collect dataset directories (those containing data.yaml), sorted
# ---------------------------------------------------------------------------
mapfile -t DATASET_DIRS < <(
  find "$DATASETS_DIR" -mindepth 1 -maxdepth 1 -type d | sort
)

VALID_DATASETS=()
for d in "${DATASET_DIRS[@]}"; do
  if [ -f "${d}/data.yaml" ]; then
    VALID_DATASETS+=("$d")
  elif [ -f "${d}/data.yml" ]; then
    VALID_DATASETS+=("$d")
  fi
done

if [ "${#VALID_DATASETS[@]}" -eq 0 ]; then
  echo "Error: no dataset directories with data.yaml found in: ${DATASETS_DIR}" >&2
  exit 1
fi

echo "Found ${#VALID_DATASETS[@]} dataset(s) in ${DATASETS_DIR}:"
for d in "${VALID_DATASETS[@]}"; do
  echo "  - $(basename "$d")"
done
echo ""

# ---------------------------------------------------------------------------
# Iterate over datasets
# ---------------------------------------------------------------------------
for ds_dir in "${VALID_DATASETS[@]}"; do
  ds_name="$(basename "$ds_dir")"

  # Locate data.yaml (prefer .yaml over .yml)
  if [ -f "${ds_dir}/data.yaml" ]; then
    ds_yaml="${ds_dir}/data.yaml"
  else
    ds_yaml="${ds_dir}/data.yml"
  fi

  echo "########## DATASET: ${ds_name} ##########"
  echo "  data.yaml : ${ds_yaml}"
  echo ""

  # Build per-dataset state-dir to avoid collisions across datasets
  sweep_args=("${passthrough_args[@]+"${passthrough_args[@]}"}")
  if [ -n "$state_dir_base" ]; then
    ds_state_dir="${state_dir_base}/${ds_name}"
    sweep_args+=(--state-dir "$ds_state_dir")
  fi

  sweep_args+=("$num_instances" "$cur_instance")

  # Export dataset-specific variables; allow FLYDET_DATASET_ROOT override from env
  export DATA_YAML="$ds_yaml"
  export DATASET_PREFIX="$ds_name"
  if [ -z "${FLYDET_DATASET_ROOT:-}" ]; then
    export FLYDET_DATASET_ROOT="$ds_dir"
  fi

  bash "$SWEEP_SCRIPT" "${sweep_args[@]}"

  # Unset per-iteration overrides so next iteration starts clean
  unset DATA_YAML DATASET_PREFIX
  # Only unset FLYDET_DATASET_ROOT if we set it (i.e., it was not in the original env)
  if [ -z "${_FLYDET_DATASET_ROOT_ORIGINAL:-}" ]; then
    unset FLYDET_DATASET_ROOT
  fi

  echo ""
done

echo "########## All datasets finished (instance ${cur_instance}/${num_instances}) ##########"
