#!/usr/bin/env python3
"""
Replay YOLO dataset file mapping from a JSON/JSONL audit, with optional destination
dataset-version folder rewrite.

Examples:
  Replay everything as-is:
    python replay_yolo_map.py rename_audit.jsonl --mode all

  Replay into a new dataset version (rewrite dst paths for both images+labels):
    python replay_yolo_map.py rename_audit.jsonl --mode all \
        --dst-version DS-F2_v5.2

  Rewrite only images' destination version:
    python replay_yolo_map.py rename_audit.jsonl --mode images \
        --dst-version DS-F2_v5.2 --dst-version-scope images

  Rewrite only labels' destination version:
    python replay_yolo_map.py rename_audit.jsonl --mode labels \
        --dst-version DS-F2_v5.2 --dst-version-scope labels

  Dry-run:
    python replay_yolo_map.py rename_audit.jsonl --dst-version DS-F2_v5.2 --dry-run -vv
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CopyResult:
    copied: int
    missing_src: int
    errors: int
    skipped: int


def _as_path(v: Any) -> Optional[Path]:
    if v is None:
        return None
    if not isinstance(v, str) or not v.strip():
        return None
    return Path(v)


def load_entries(path: Path) -> List[Dict[str, Any]]:
    """Load either .jsonl (JSON Lines) or .json (single JSON value)."""
    suffix = path.suffix.lower()

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Failed to read file: {path} ({e})") from e

    if suffix == ".jsonl":
        entries: List[Dict[str, Any]] = []
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to parse JSONL at {path}:{lineno}: {e}\nLine: {raw[:200]}"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected an object/dict at {path}:{lineno}, got {type(obj).__name__}"
                )
            entries.append(obj)
        return entries

    # Regular JSON
    try:
        data = json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} ({e})") from e

    if isinstance(data, list):
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Entry #{i} is not an object/dict (got {type(item).__name__})")
        return data

    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Expected JSON to be a list or object, got {type(data).__name__}")


def ensure_parent_dir(dst: Path, dry_run: bool) -> None:
    parent = dst.parent
    if parent.exists():
        return
    if dry_run:
        return
    parent.mkdir(parents=True, exist_ok=True)


def copy_one(src: Path, dst: Path, dry_run: bool) -> Tuple[bool, str]:
    """Copy src -> dst, overwriting if needed. Returns (success, message)."""
    if not src.exists():
        return False, f"missing source: {src}"
    try:
        ensure_parent_dir(dst, dry_run=dry_run)
        if not dry_run:
            shutil.copy2(src, dst)  # preserves metadata
        return True, "copied"
    except Exception as e:
        return False, f"error copying {src} -> {dst}: {e}"


def _rewrite_version_folder(dst: Path, old_version: str, new_version: str) -> Path:
    """
    Replace the first path segment that equals old_version with new_version.
    If old_version is not found as a segment, returns dst unchanged.
    """
    parts = list(dst.parts)
    for i, p in enumerate(parts):
        if p == old_version:
            parts[i] = new_version
            return Path(*parts)
    return dst


def infer_old_version_from_entry(entry: Dict[str, Any]) -> Optional[str]:
    """
    Infer dataset-version folder from dst_img or dst_lbl by taking the segment that
    immediately follows ".../datasets/Pestline/" when present, otherwise fall back
    to the first plausible "DS-" segment found.
    """
    for key in ("dst_img", "dst_lbl"):
        p = _as_path(entry.get(key))
        if p is None:
            continue
        parts = list(p.parts)
        # Try to find ".../datasets/Pestline/<VERSION>/..."
        for i in range(len(parts) - 2):
            if parts[i] == "datasets" and parts[i + 1] == "Pestline":
                return parts[i + 2]
        # Fallback: first segment that looks like DS-*
        for seg in parts:
            if seg.startswith("DS-"):
                return seg
    return None


def replay(
    entries: Iterable[Dict[str, Any]],
    mode: str,
    dry_run: bool,
    verbose: int,
    dst_version: Optional[str],
    dst_version_scope: str,
    old_version_hint: Optional[str],
) -> CopyResult:
    do_images = mode in ("all", "images")
    do_labels = mode in ("all", "labels")

    copied = 0
    missing_src = 0
    errors = 0
    skipped = 0

    # Determine old version folder (once) if we need rewriting.
    old_version: Optional[str] = None
    if dst_version:
        old_version = old_version_hint
        if old_version is None:
            # infer from first entry that yields a version
            for e in entries:
                old_version = infer_old_version_from_entry(e)
                if old_version:
                    break
            # If we consumed the iterator, re-materialize entries
            if not isinstance(entries, list):
                entries = list(entries)

        if not old_version:
            raise RuntimeError(
                "Could not infer the old dataset version folder from entries. "
                "Provide it explicitly with --old-version."
            )

        if verbose >= 1:
            print(f"[info] Rewriting dst version: {old_version} -> {dst_version} (scope={dst_version_scope})",
                  file=sys.stderr)

    def maybe_rewrite(dst: Path, kind: str) -> Path:
        # kind is "images" or "labels"
        if not dst_version or not old_version:
            return dst
        if dst_version_scope in ("both", kind):
            return _rewrite_version_folder(dst, old_version, dst_version)
        return dst

    # Ensure entries is iterable multiple times if needed
    if not isinstance(entries, list):
        entries = list(entries)

    for idx, e in enumerate(entries):
        # Images
        if do_images:
            src_img = _as_path(e.get("src_img"))
            dst_img = _as_path(e.get("dst_img"))

            if src_img is None or dst_img is None:
                skipped += 1
                if verbose >= 2:
                    print(f"[{idx}] skip image (missing src_img/dst_img fields)", file=sys.stderr)
            else:
                effective_dst_img = maybe_rewrite(dst_img, "images")
                ok, msg = copy_one(src_img, effective_dst_img, dry_run=dry_run)
                if ok:
                    copied += 1
                    if verbose >= 2:
                        print(f"[{idx}] IMG {src_img} -> {effective_dst_img} ({msg})")
                else:
                    if msg.startswith("missing source:"):
                        missing_src += 1
                    else:
                        errors += 1
                    if verbose >= 1:
                        print(f"[{idx}] IMG {msg}", file=sys.stderr)

        # Labels
        if do_labels:
            src_lbl = _as_path(e.get("src_lbl"))
            dst_lbl = _as_path(e.get("dst_lbl"))

            if src_lbl is None or dst_lbl is None:
                skipped += 1
                if verbose >= 2:
                    print(f"[{idx}] skip label (missing src_lbl/dst_lbl fields)", file=sys.stderr)
            else:
                effective_dst_lbl = maybe_rewrite(dst_lbl, "labels")
                ok, msg = copy_one(src_lbl, effective_dst_lbl, dry_run=dry_run)
                if ok:
                    copied += 1
                    if verbose >= 2:
                        print(f"[{idx}] LBL {src_lbl} -> {effective_dst_lbl} ({msg})")
                else:
                    if msg.startswith("missing source:"):
                        missing_src += 1
                    else:
                        errors += 1
                    if verbose >= 1:
                        print(f"[{idx}] LBL {msg}", file=sys.stderr)

    return CopyResult(copied=copied, missing_src=missing_src, errors=errors, skipped=skipped)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay a YOLO mapping JSON/JSONL by copying src -> dst.")
    p.add_argument("path", type=Path, help="Path to mapping audit (.json or .jsonl).")

    p.add_argument(
        "--mode",
        choices=["all", "images", "labels"],
        default="all",
        help="What to replay: all (default), images only, or labels only.",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write anything; just report actions.")
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Verbosity: -v (default) shows errors/info, -vv shows per-file operations.",
    )

    # Destination version rewrite options
    p.add_argument(
        "--dst-version",
        default=None,
        help="If set, rewrite the dataset version folder in destination paths to this value "
             "(e.g., DS-F2_v5.2).",
    )
    p.add_argument(
        "--dst-version-scope",
        choices=["both", "images", "labels"],
        default="both",
        help="Apply --dst-version rewrite to: both (default), images only, or labels only.",
    )
    p.add_argument(
        "--old-version",
        default=None,
        help="Optional: explicitly provide the old version folder to replace "
             "(otherwise inferred from entries).",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    if not args.path.exists():
        print(f"File not found: {args.path}", file=sys.stderr)
        return 2

    try:
        entries = load_entries(args.path)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    try:
        res = replay(
            entries=entries,
            mode=args.mode,
            dry_run=args.dry_run,
            verbose=args.verbose,
            dst_version=args.dst_version,
            dst_version_scope=args.dst_version_scope,
            old_version_hint=args.old_version,
        )
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    print("\nSummary")
    print(f"  mode: {args.mode}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  entries: {len(entries)}")
    print(f"  copied: {res.copied}")
    print(f"  skipped (missing fields): {res.skipped}")
    print(f"  missing sources: {res.missing_src}")
    print(f"  errors: {res.errors}")

    return 1 if res.errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())