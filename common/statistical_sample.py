#!/usr/bin/env python3
"""
sample_objects_for_review.py

Object-level stratified sampling for YOLO label datasets with finite population correction (FPC).

This version creates an annotator-friendly review package:
- one folder per sampled object
- a cropped patch image around the object, with the object bbox drawn
- a minimal label file containing only the class id
- a header line listing class index -> class name

Dataset layout assumed:
  <root>/labels/{train,val,test}/... .txt
  <root>/images/{train,val,test}/... <image with same relative stem>

Output package layout:
  <out>/
    manifest_objects.csv
    class_population.csv
    class_sample_plan.csv
    samples/
      <sample_key>/
        object_info.txt
        patch.jpg
        original_label.txt
    review_labels/<annotator>/
      <sample_key>/
        object_info.txt
        patch.jpg
        review_label.txt

Annotator instruction:
- Look at patch.jpg
- Edit review_label.txt
- Only the class id line should be changed
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from PIL import Image, ImageDraw

import json

try:
    import yaml
except ImportError:
    yaml = None

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def z_value(confidence: float) -> float:
    table = {
        0.80: 1.2816,
        0.85: 1.4395,
        0.90: 1.6449,
        0.95: 1.9600,
        0.98: 2.3263,
        0.99: 2.5758,
        0.995: 2.8070,
    }
    if confidence in table:
        return table[confidence]
    c = min(max(confidence, 0.80), 0.995)
    if c <= 0.90:
        return 1.2816 + (c - 0.80) * (1.6449 - 1.2816) / (0.90 - 0.80)
    if c <= 0.95:
        return 1.6449 + (c - 0.90) * (1.9600 - 1.6449) / (0.95 - 0.90)
    if c <= 0.99:
        return 1.9600 + (c - 0.95) * (2.5758 - 1.9600) / (0.99 - 0.95)
    return 2.5758 + (c - 0.99) * (2.8070 - 2.5758) / (0.995 - 0.99)


def fpc_sample_size(N: int, confidence: float, margin: float, p: float) -> int:
    """
    Sample size for proportion with finite population correction (FPC).
      n0 = z^2 p(1-p) / E^2
      n  = n0 / (1 + (n0-1)/N)
    """
    if N <= 0:
        return 0
    z = z_value(confidence)
    n0 = (z * z * p * (1 - p)) / (margin * margin)
    n = n0 / (1 + (n0 - 1) / N)
    return max(1, math.ceil(n))


@dataclass(frozen=True)
class Obj:
    object_id: str
    subset: str
    label_rel: str
    label_abs: Path
    line_idx: int
    cls: int
    xc: float
    yc: float
    w: float
    h: float


def iter_label_files(root: Path, subset: str) -> Iterable[Path]:
    base = root / "labels" / subset
    if not base.exists():
        return
    yield from (p for p in base.rglob("*.txt") if p.is_file())


def parse_yolo_label_file(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Returns list of boxes: (cls, xc, yc, w, h) for valid lines.
    Ignores empty lines and comment lines starting with '#'.
    """
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        out.append((cls, xc, yc, w, h))
    return out

def normalize_class_map(names_obj) -> Dict[int, str]:
    """
    Accepts YOLO-style names from data.yaml:
      - list: ["cat", "dog"]
      - dict: {0: "cat", 1: "dog"} or {"0": "cat", "1": "dog"}
    Returns {int: str}.
    """
    if names_obj is None:
        return {}

    if isinstance(names_obj, list):
        return {i: str(v) for i, v in enumerate(names_obj)}

    if isinstance(names_obj, dict):
        out: Dict[int, str] = {}
        for k, v in names_obj.items():
            out[int(k)] = str(v)
        return dict(sorted(out.items(), key=lambda x: x[0]))

    raise ValueError(f"Unsupported 'names' format in data.yaml: {type(names_obj).__name__}")


def load_class_map_from_data_yaml(root: Path) -> Dict[int, str]:
    """
    Loads class names from <root>/data.yaml if it exists.
    Returns {} if the file does not exist.
    """
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return {}

    if yaml is None:
        raise RuntimeError(
            "Found data.yaml but PyYAML is not installed. "
            "Install it with: pip install pyyaml"
        )

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {data_yaml}: expected a mapping at top level")

    names_obj = data.get("names")
    class_map = normalize_class_map(names_obj)

    # Optional consistency check with nc
    nc = data.get("nc")
    if nc is not None and class_map and int(nc) != len(class_map):
        raise ValueError(
            f"Inconsistent data.yaml: nc={nc} but resolved {len(class_map)} class names from 'names'"
        )

    return class_map


def class_name_for(class_id: int, class_map: Dict[int, str]) -> str:
    return class_map.get(class_id, f"class_{class_id}")

def build_population(root: Path, subsets: List[str]) -> Tuple[List[Obj], Dict[int, int]]:
    """
    Build object-level population and per-class counts (Nc).
    """
    population: List[Obj] = []
    class_counts: Dict[int, int] = {}

    for subset in subsets:
        base = root / "labels" / subset
        if not base.exists():
            raise FileNotFoundError(f"Missing subset labels dir: {base}")

        for label_abs in iter_label_files(root, subset):
            label_rel = str(label_abs.relative_to(base))
            boxes = parse_yolo_label_file(label_abs)
            for i, (cls, xc, yc, w, h) in enumerate(boxes):
                object_id = f"{subset}/{label_rel}::L{i}"
                population.append(
                    Obj(
                        object_id=object_id,
                        subset=subset,
                        label_rel=label_rel,
                        label_abs=label_abs,
                        line_idx=i,
                        cls=cls,
                        xc=xc,
                        yc=yc,
                        w=w*1.15, # Give 15% more room to the BBox
                        h=h*1.15,
                    )
                )
                class_counts[cls] = class_counts.get(cls, 0) + 1

    return population, class_counts


def write_csv(path: Path, header: List[str], rows: Iterable[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def stratified_sample(
    objs_by_class: Dict[int, List[Obj]],
    n_by_class: Dict[int, int],
    rng: random.Random,
    max_per_image: int,
) -> Tuple[List[Obj], List[str]]:
    """
    Greedy sampling within each class, with optional cap max_per_image.
    Here, "image" == label file (since boxes map to one label txt).
    If max_per_image <= 0, no cap.
    """
    warnings: List[str] = []
    sampled: List[Obj] = []
    per_label_count: Dict[Tuple[str, str], int] = {}

    for cls, candidates in sorted(objs_by_class.items(), key=lambda x: x[0]):
        need = n_by_class.get(cls, 0)
        if need <= 0:
            continue
        if need > len(candidates):
            warnings.append(f"class {cls}: requested {need} but only {len(candidates)} available; taking all.")
            need = len(candidates)

        cand = candidates[:]
        rng.shuffle(cand)

        picked: List[Obj] = []
        if max_per_image and max_per_image > 0:
            for o in cand:
                key = (o.subset, o.label_rel)
                if per_label_count.get(key, 0) < max_per_image:
                    picked.append(o)
                    per_label_count[key] = per_label_count.get(key, 0) + 1
                    if len(picked) >= need:
                        break

            if len(picked) < need:
                short = need - len(picked)
                warnings.append(
                    f"class {cls}: max_per_image cap prevented meeting target; "
                    f"missing {short}. Filling without cap for this class."
                )
                already = {p.object_id for p in picked}
                for o in cand:
                    if o.object_id in already:
                        continue
                    picked.append(o)
                    if len(picked) >= need:
                        break
        else:
            picked = cand[:need]

        sampled.extend(picked)

    return sampled, warnings


def sanitize_sample_key(object_id: str) -> str:
    key = object_id.replace("::", "__")
    key = key.replace("/", "__")
    key = key.replace("\\", "__")
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return key


def load_class_map(path: Optional[Path]) -> Dict[int, str]:
    """
    Supported formats:
    - plain text: one class name per line
    - csv-ish: 'idx,name'
    """
    if path is None:
        return {}

    if not path.exists():
        raise FileNotFoundError(f"Class names file not found: {path}")

    text = path.read_text(encoding="utf-8").splitlines()
    lines = [ln.strip() for ln in text if ln.strip()]
    if not lines:
        return {}

    # try idx,name
    class_map: Dict[int, str] = {}
    csv_like = True
    for ln in lines:
        if "," not in ln:
            csv_like = False
            break

    if csv_like:
        for ln in lines:
            left, right = ln.split(",", 1)
            class_map[int(left.strip())] = right.strip()
        return class_map

    # fallback: one name per line
    for i, ln in enumerate(lines):
        class_map[i] = ln
    return class_map


def class_help_line(class_map: Dict[int, str], present_classes: List[int]) -> str:
    if class_map:
        pairs = [f"{cid}={class_map.get(cid, f'class_{cid}')}" for cid in sorted(present_classes)]
    else:
        pairs = [f"{cid}=class_{cid}" for cid in sorted(present_classes)]
    return "# classes: " + ", ".join(pairs)


def find_image_for_label(root: Path, subset: str, label_rel: str) -> Path:
    """
    Find image corresponding to a label path by matching stem under images/<subset>.
    Example:
      labels/train/foo/bar.txt -> images/train/foo/bar.jpg (or png, ...)
    """
    rel_no_ext = Path(label_rel).with_suffix("")
    img_base = root / "images" / subset
    for ext in IMAGE_EXTS:
        cand = img_base / rel_no_ext.with_suffix(ext)
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"Could not find source image for label '{subset}/{label_rel}' under {img_base} with extensions {IMAGE_EXTS}"
    )


def yolo_to_xyxy(obj: Obj, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x1 = (obj.xc - obj.w / 2.0) * img_w
    y1 = (obj.yc - obj.h / 2.0) * img_h
    x2 = (obj.xc + obj.w / 2.0) * img_w
    y2 = (obj.yc + obj.h / 2.0) * img_h
    return x1, y1, x2, y2


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_patch(
    img_path: Path,
    obj: Obj,
    patch_size: int,
    bbox_width: int = 3,
) -> Image.Image:
    """
    Creates a square patch centered on the object center.
    The object bbox is drawn on the patch for unambiguous identification.
    """
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    cx = obj.xc * img_w
    cy = obj.yc * img_h

    half = patch_size / 2.0
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + patch_size
    bottom = top + patch_size

    # shift crop window back inside the image
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        shift = right - img_w
        left -= shift
        right = img_w
    if bottom > img_h:
        shift = bottom - img_h
        top -= shift
        bottom = img_h

    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    crop = img.crop((left, top, right, bottom))

    # If near borders and crop is smaller than desired, pad to exact square
    if crop.size != (patch_size, patch_size):
        padded = Image.new("RGB", (patch_size, patch_size), (0, 0, 0))
        padded.paste(crop, (0, 0))
        crop = padded

    x1, y1, x2, y2 = yolo_to_xyxy(obj, img_w, img_h)
    x1p = clamp(x1 - left, 0, patch_size - 1)
    y1p = clamp(y1 - top, 0, patch_size - 1)
    x2p = clamp(x2 - left, 0, patch_size - 1)
    y2p = clamp(y2 - top, 0, patch_size - 1)

    draw = ImageDraw.Draw(crop)
    for k in range(bbox_width):
        draw.rectangle(
            [x1p - k, y1p - k, x2p + k, y2p + k],
            outline=(255, 0, 0),
        )

    return crop


def write_sample_folders(
    root: Path,
    sampled: List[Obj],
    out_samples_dir: Path,
    class_map: Dict[int, str],
    patch_size: int,
) -> List[str]:
    """
    Writes one folder per sampled object:
      <samples>/<sample_key>/
        object_info.txt
        patch.jpg
        original_label.txt
    """
    warnings: List[str] = []
    present_classes = sorted({o.cls for o in sampled})
    help_line = class_help_line(class_map, present_classes)

    for o in sampled:
        sample_key = sanitize_sample_key(o.object_id)
        sample_dir = out_samples_dir / sample_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            img_path = find_image_for_label(root, o.subset, o.label_rel)
            patch = make_patch(img_path, o, patch_size=patch_size, bbox_width=2)
            patch.save(sample_dir / "patch.jpg", quality=95)
        except Exception as e:
            warnings.append(f"{o.object_id}: failed to create patch: {e}")

        class_name = class_map.get(o.cls, f"class_{o.cls}")

        info_lines = [
            f"object_id={o.object_id}",
            f"subset={o.subset}",
            f"label_rel={o.label_rel}",
            f"line_idx={o.line_idx}",
            f"class_id={o.cls}",
            f"class_name={class_name}",
            f"xc={o.xc:.8f}",
            f"yc={o.yc:.8f}",
            f"w={o.w:.8f}",
            f"h={o.h:.8f}",
        ]
        (sample_dir / "object_info.txt").write_text("\n".join(info_lines) + "\n", encoding="utf-8")

        label_lines = [
            help_line,
            f"# object_id={o.object_id}",
            str(o.cls),
        ]
        (sample_dir / "original_label.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return warnings


def create_annotator_review_dirs(
    out_samples_dir: Path,
    review_root: Path,
    annotators: List[str],
) -> None:
    """
    For each annotator, create:
      review_labels/<annotator>/<sample_key>/
        object_info.txt
        patch.jpg
        review_label.txt
    """
    for annotator in annotators:
        annot_dir = review_root / annotator
        if annot_dir.exists():
            shutil.rmtree(annot_dir)
        annot_dir.mkdir(parents=True, exist_ok=True)

        for sample_dir in sorted(p for p in out_samples_dir.iterdir() if p.is_dir()):
            dst = annot_dir / sample_dir.name
            dst.mkdir(parents=True, exist_ok=True)

            for fname in ("patch.jpg", "object_info.txt"):
                srcf = sample_dir / fname
                if srcf.exists():
                    shutil.copy2(srcf, dst / fname)

            src_label = sample_dir / "original_label.txt"
            if src_label.exists():
                shutil.copy2(src_label, dst / "review_label.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified object-level sampling for YOLO labels + review package.")
    ap.add_argument("--root", type=Path, required=True, help="Dataset root containing labels/{train,val,test}/ and images/{train,val,test}/")
    ap.add_argument("--subset", type=str, default="train", choices=["train", "val", "test", "all"],
                    help="Which subset(s) to include as the population")
    ap.add_argument("--out", type=Path, required=True, help="Output review package directory")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")

    # stats
    ap.add_argument("--confidence", type=float, default=0.95, help="Confidence level for per-class sizing")
    ap.add_argument("--margin", type=float, default=0.10, help="Margin of error per class (e.g. 0.10, 0.05)")
    ap.add_argument("--p", type=float, default=0.5, help="Expected proportion (0.5 worst-case)")

    # practical controls
    ap.add_argument("--min-per-class", type=int, default=30, help="Minimum samples per class (if possible)")
    ap.add_argument("--max-per-class", type=int, default=0, help="Optional cap samples per class (0 = no cap)")
    ap.add_argument("--max-per-image", type=int, default=10, help="Cap sampled objects per label file (0 = no cap)")
    ap.add_argument("--total-max", type=int, default=0,
                    help="Optional cap on total sampled objects (0 = no cap). If set, downscales per class proportionally while respecting min-per-class when possible.")

    # review package controls
    ap.add_argument("--annotators", type=str, default="", help="Comma-separated annotator ids, e.g. a1,a2,a3")
    ap.add_argument("--class-names", type=Path, default=None,
                    help="Optional class names file. Supported: one name per line, or csv lines 'idx,name'")
    ap.add_argument("--patch-size", type=int, default=512,
                    help="Square patch size in pixels centered on the object")
    ap.add_argument("--bbox-width", type=int, default=3,
                    help="Bounding box line width in the patch")

    args = ap.parse_args()

    if not (0 < args.confidence < 1):
        raise ValueError("--confidence must be in (0,1)")
    if not (0 < args.margin < 1):
        raise ValueError("--margin must be in (0,1)")
    if not (0 < args.p < 1):
        raise ValueError("--p must be in (0,1)")
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be > 0")
    if args.bbox_width <= 0:
        raise ValueError("--bbox-width must be > 0")

    subsets = ["train", "val", "test"] if args.subset == "all" else [args.subset]
    rng = random.Random(args.seed)
    if args.class_names:
        class_map = load_class_map(args.class_names)
    else:
        class_map = load_class_map_from_data_yaml(args.root)

    population, class_counts = build_population(args.root, subsets)
    if not population:
        raise RuntimeError("No labeled objects found in the selected subset(s).")

    n_by_class: Dict[int, int] = {}
    for cls, Nc in sorted(class_counts.items(), key=lambda x: x[0]):
        n = fpc_sample_size(Nc, args.confidence, args.margin, args.p)
        n = max(1, n)
        if args.min_per_class > 0:
            n = max(n, args.min_per_class)
        if args.max_per_class and args.max_per_class > 0:
            n = min(n, args.max_per_class)
        n = min(n, Nc)
        n_by_class[cls] = n

    total_planned = sum(n_by_class.values())
    if args.total_max and args.total_max > 0 and total_planned > args.total_max:
        classes = sorted(n_by_class.keys())
        mins = {
            c: min(
                n_by_class[c],
                max(1, min(args.min_per_class, class_counts[c]) if args.min_per_class > 0 else 1),
            )
            for c in classes
        }
        current = dict(n_by_class)

        sum_mins = sum(mins.values())
        if sum_mins > args.total_max:
            scale = args.total_max / total_planned
            for c in classes:
                current[c] = max(1, min(class_counts[c], int(round(current[c] * scale))))
        else:
            excess = total_planned - args.total_max
            reducible = {c: current[c] - mins[c] for c in classes}
            reducible_total = sum(reducible.values())

            if reducible_total > 0:
                for c in classes:
                    if reducible[c] <= 0:
                        continue
                    take = int(round(excess * (reducible[c] / reducible_total)))
                    take = min(take, reducible[c])
                    current[c] -= take

        def total_cur() -> int:
            return sum(current.values())

        while total_cur() > args.total_max:
            c = max(classes, key=lambda x: current[x])
            if current[c] > 1:
                current[c] -= 1
            else:
                break

        while total_cur() < args.total_max:
            candidates = [c for c in classes if current[c] < class_counts[c]]
            if not candidates:
                break
            c = max(candidates, key=lambda x: class_counts[x] - current[x])
            current[c] += 1

        n_by_class = current

    objs_by_class: Dict[int, List[Obj]] = {}
    for o in population:
        objs_by_class.setdefault(o.cls, []).append(o)

    sampled, warnings = stratified_sample(
        objs_by_class=objs_by_class,
        n_by_class=n_by_class,
        rng=rng,
        max_per_image=args.max_per_image,
    )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    write_csv(
        out / "class_population.csv",
        ["class_id", "class_name", "Nc_objects"],
        ([cls, class_name_for(cls, class_map), class_counts[cls]] for cls in sorted(class_counts.keys())),
    )

    write_csv(
        out / "class_sample_plan.csv",
        ["class_id", "class_name", "Nc_objects", "n_sampled"],
        ([cls, class_name_for(cls, class_map), class_counts[cls], n_by_class.get(cls, 0)] for cls in
         sorted(class_counts.keys())),
    )

    write_csv(
        out / "manifest_objects.csv",
        ["object_id", "subset", "label_rel", "line_idx", "class_id", "class_name", "xc", "yc", "w", "h"],
        (
            [o.object_id, o.subset, o.label_rel, o.line_idx, o.cls, class_name_for(o.cls, class_map), o.xc, o.yc, o.w,
             o.h]
            for o in sampled
        ),
    )

    samples_dir = out / "samples"
    patch_warnings = write_sample_folders(
        root=args.root,
        sampled=sampled,
        out_samples_dir=samples_dir,
        class_map=class_map,
        patch_size=args.patch_size,
    )
    warnings.extend(patch_warnings)

    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    if annotators:
        review_root = out / "review_labels"
        create_annotator_review_dirs(
            out_samples_dir=samples_dir,
            review_root=review_root,
            annotators=annotators,
        )

    if class_map:
        print(f"Loaded class names from: {args.root / 'data.yaml'}")
    else:
        print("No data.yaml found in dataset root; class names not loaded.")

    print("=== Sampling complete ===")
    print(f"Population subsets: {subsets}")
    print(f"Population objects (N): {len(population)}")
    print(f"Sampled objects (n): {len(sampled)}")
    print(f"Per-class sizing: confidence={args.confidence}, margin={args.margin}, p={args.p}")
    if args.total_max and args.total_max > 0:
        print(f"Total cap: {args.total_max}")
    print(f"Max per image(label file): {args.max_per_image}")
    print(f"Patch size: {args.patch_size}")
    print(f"Output: {out}")
    if annotators:
        print(f"Annotators: {annotators} (review folders created)")
    if warnings:
        print("Warnings:")
        for w in warnings[:50]:
            print(f"  - {w}")
        if len(warnings) > 50:
            print(f"  ... {len(warnings)-50} more")


if __name__ == "__main__":
    main()