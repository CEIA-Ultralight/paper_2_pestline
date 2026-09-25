from __future__ import annotations

import os
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional


def build_legacy_structure(
        dataset_root: str | Path,
        output_dir: Optional[str | Path] = None,
        force: bool = False,
        use_symlinks: bool = False,
) -> Path:
    """
    Transforms a modern YOLO dataset layout into the legacy format expected by RF-DETR.

    Legacy Format:
        ROOT/
          data.yaml
          train/
            images/
            labels/
          valid/
            images/
            labels/
          test/ (optional)
            images/
            labels/

    This function uses hard links by default to ensure compatibility with VM environments
    while remaining disk-space efficient.

    Args:
        dataset_root: The source directory of the dataset.
        output_dir: Where to create the legacy structure. If None, creates a 'legacy_dataset'
                    folder inside dataset_root.
        force: If True, deletes the output_dir if it already exists.
        use_symlinks: If True, uses symlinks instead of hard links (not recommended for VMs).

    Returns:
        Path: The absolute path to the generated legacy dataset root.
    """
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")

    # Determine output directory
    out = Path(output_dir).resolve() if output_dir else root / "rfdetr_dataset"

    if out.exists() and force:
        print(f"[dataset] Removing existing directory: {out}")
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    # 1. Handle data.yaml
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        yaml_path = root / "data.yml"

    if not yaml_path.exists():
        print(f"[dataset] WARNING: data.yaml not found in {root}. RF-DETR validator may fail.")
    else:
        shutil.copy2(yaml_path, out / "data.yaml")

    # 2. Define source discovery patterns
    # (Mapping internal split name to possible directory names)
    patterns = {
        "train": {
            "images": ["train/images", "images/train", "train", "images/training"],
            "labels": ["train/labels", "labels/train", "labels/training", "training/labels"]
        },
        "valid": {
            "images": ["valid/images", "val/images", "images/valid", "images/val", "valid", "val"],
            "labels": ["valid/labels", "val/labels", "labels/valid", "labels/val", "labels/valid"]
        },
        "test": {
            "images": ["test/images", "images/test", "test"],
            "labels": ["test/labels", "labels/test"]
        }
    }

    def link_file(src: Path, dst: Path):
        """Helper to create hard link or symlink."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return

        if use_symlinks:
            os.symlink(src, dst)
        else:
            try:
                os.link(src, dst)
            except OSError as e:
                # Fallback to copy if cross-device link (standard error 18)
                if e.errno == 18:
                    shutil.copy2(src, dst)
                else:
                    raise

    # 3. Process each split
    for split, dirs in patterns.items():
        src_img_dir = None
        src_lbl_dir = None

        # Try to find valid directories
        for p in dirs["images"]:
            if (root / p).is_dir():
                src_img_dir = root / p
                break

        for p in dirs["labels"]:
            if (root / p).is_dir():
                src_lbl_dir = root / p
                break

        if not src_img_dir or not src_lbl_dir:
            if split != "test":
                print(f"[dataset] Skip split '{split}': Could not find both images and labels.")
            continue

        print(f"[dataset] Linking split '{split}'...")

        # Link images
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        count = 0
        for img_path in src_img_dir.iterdir():
            if img_path.suffix.lower() in img_exts:
                target = out / split / "images" / img_path.name
                link_file(img_path, target)

                # Try to link corresponding label
                lbl_name = img_path.stem + ".txt"
                lbl_path = src_lbl_dir / lbl_name
                if lbl_path.exists():
                    link_file(lbl_path, out / split / "labels" / lbl_name)
                    count += 1

        print(f"[dataset]   -> Linked {count} pairs for {split}.")

    print(f"[dataset] Done. Legacy dataset ready at: {out}")
    return out


def build_dataset():
    """CLI Entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset for RF-DETR (Legacy Format)")
    parser.add_argument("root", help="Current dataset root")
    parser.add_argument("-o", "--output", help="Output directory (default: root/rfdetr_dataset)")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of hard links")

    args = parser.parse_args()

    try:
        build_legacy_structure(
            dataset_root=args.root,
            output_dir=args.output,
            force=args.force,
            use_symlinks=args.symlink
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(build_dataset())