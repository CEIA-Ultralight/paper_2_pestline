#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  yolo_legacy_links.sh DATASET_ROOT [--force] [--dry-run]

What it does:
  Creates legacy YOLO structure expected by some tools:
    DATASET_ROOT/
      data.yaml
      train/images -> ...
      train/labels -> ...
      valid/images -> ...
      valid/labels -> ...
      test/images  -> ... (optional)
      test/labels  -> ... (optional)

It detects common modern layouts and creates symlinks accordingly.

Options:
  --force    Remove existing symlinks at target locations before recreating.
             (Will NOT delete real directories/files; it refuses if target is not a symlink.)
  --dry-run  Print actions without changing anything.
USAGE
}

log() { echo "[yolo-legacy-links] $*"; }
die() { echo "[yolo-legacy-links] ERROR: $*" >&2; exit 1; }

FORCE=0
DRY=0

if [[ $# -lt 1 ]]; then usage; exit 1; fi

ROOT="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
  shift || true
done

ROOT="$(readlink -f "$ROOT" 2>/dev/null || realpath "$ROOT")"
[[ -d "$ROOT" ]] || die "Dataset root is not a directory: $ROOT"

# Helper: run or echo (dry-run)
run() {
  if [[ "$DRY" -eq 1 ]]; then
    echo "DRY: $*"
  else
    eval "$@"
  fi
}

# Helper: create legacy split symlinks
# args: legacy_split (train|valid|test), src_images, src_labels
link_split() {
  local split="$1"
  local src_img="$2"
  local src_lbl="$3"

  [[ -d "$src_img" ]] || die "Missing images dir for '$split': $src_img"
  [[ -d "$src_lbl" ]] || die "Missing labels dir for '$split': $src_lbl"

  run "mkdir -p \"$ROOT/$split\""

  for sub in images labels; do
    local tgt="$ROOT/$split/$sub"
    local src=""
    [[ "$sub" == "images" ]] && src="$src_img" || src="$src_lbl"

    # If target exists:
    if [[ -e "$tgt" || -L "$tgt" ]]; then
      if [[ "$FORCE" -eq 1 ]]; then
        if [[ -L "$tgt" ]]; then
          run "rm -f \"$tgt\""
        else
          die "Refusing to overwrite non-symlink target: $tgt (use manual cleanup)"
        fi
      else
        log "Skip existing target (use --force to replace): $tgt"
        continue
      fi
    fi

    run "ln -s \"$(readlink -f "$src")\" \"$tgt\""
    log "Linked $tgt -> $src"
  done
}

# Detect helpers: return first matching dir that exists
pick_dir() {
  for d in "$@"; do
    [[ -d "$ROOT/$d" ]] && { echo "$ROOT/$d"; return 0; }
  done
  return 1
}

# ---- Validate presence of data.yaml (required by validator) ----
if [[ ! -f "$ROOT/data.yaml" && ! -f "$ROOT/data.yml" ]]; then
  log "WARNING: no data.yaml/data.yml found in $ROOT (validator will fail)."
  log "         Create $ROOT/data.yaml with 'names:' at minimum."
fi

# ---- Detect train/valid/test sources in common layouts ----
# Common "modern" layout you have:
#   ROOT/images/train, ROOT/labels/train
#   ROOT/images/val or valid, ROOT/labels/val or valid
#
# Common "legacy" layout already:
#   ROOT/train/images, ROOT/train/labels, ROOT/valid/images, ...
#
# Some datasets use "val" instead of "valid" at top-level:
#   ROOT/val/images, ROOT/val/labels

# Train sources
train_img="$(pick_dir "train/images" "images/train" "train" "images/training" "training/images" "images/trn" || true)"
train_lbl="$(pick_dir "train/labels" "labels/train" "labels/training" "training/labels" "labels/trn" || true)"

# Valid sources (validator wants 'valid', not 'val')
valid_img="$(pick_dir "valid/images" "val/images" "images/valid" "images/val" "valid" "val" || true)"
valid_lbl="$(pick_dir "valid/labels" "val/labels" "labels/valid" "labels/val" || true)"

# Test sources (optional)
test_img="$(pick_dir "test/images" "images/test" "test" || true)"
test_lbl="$(pick_dir "test/labels" "labels/test" || true)"

# Sanity
[[ -n "${train_img:-}" && -n "${train_lbl:-}" ]] || die "Could not detect train images/labels dirs under $ROOT"
[[ -n "${valid_img:-}" && -n "${valid_lbl:-}" ]] || die "Could not detect valid/val images/labels dirs under $ROOT"

log "Detected:"
log "  train images: $train_img"
log "  train labels: $train_lbl"
log "  valid images: $valid_img"
log "  valid labels: $valid_lbl"
if [[ -n "${test_img:-}" && -n "${test_lbl:-}" ]]; then
  log "  test  images: $test_img"
  log "  test  labels: $test_lbl"
else
  log "  test split: not detected (optional)"
fi

# ---- Create legacy links expected by rf-detr validator ----
link_split "train" "$train_img" "$train_lbl"
link_split "valid" "$valid_img" "$valid_lbl"

if [[ -n "${test_img:-}" && -n "${test_lbl:-}" ]]; then
  link_split "test" "$test_img" "$test_lbl"
fi

log "Done."

