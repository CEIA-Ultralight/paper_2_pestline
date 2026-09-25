#!/usr/bin/env python3
"""Clean and inspect YOLO label files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    line_number: int
    class_id: str
    box: tuple[float, float, float, float]
    raw: str

    @property
    def exact_key(self) -> tuple[str, tuple[float, float, float, float]]:
        return self.class_id, self.box


@dataclass(frozen=True)
class NearDuplicate:
    first: Annotation
    second: Annotation
    iou: float


@dataclass(frozen=True)
class ExactDuplicate:
    original: Annotation
    duplicate: Annotation


def calculate_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    # Convert from YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2).
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def parse_yolo_file(file_path: Path) -> list[Annotation]:
    annotations = []

    with file_path.open("r", encoding="utf-8") as labels_file:
        for line_number, line in enumerate(labels_file, start=1):
            raw = line.strip()
            if not raw:
                continue

            parts = raw.split()
            if len(parts) != 5:
                raise ValueError(
                    f"{file_path}:{line_number}: expected 5 YOLO fields, got {len(parts)}"
                )

            class_id = parts[0]
            try:
                coords = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"{file_path}:{line_number}: coordinates must be numeric"
                ) from exc
            box = (coords[0], coords[1], coords[2], coords[3])

            annotations.append(
                Annotation(
                    line_number=line_number,
                    class_id=class_id,
                    box=box,
                    raw=raw,
                )
            )

    return annotations


def remove_exact_duplicates(
    annotations: list[Annotation],
) -> tuple[list[Annotation], list[ExactDuplicate]]:
    seen: dict[tuple[str, tuple[float, float, float, float]], Annotation] = {}
    kept = []
    removed = []

    for annotation in annotations:
        original = seen.get(annotation.exact_key)
        if original is not None:
            removed.append(ExactDuplicate(original=original, duplicate=annotation))
            continue

        seen[annotation.exact_key] = annotation
        kept.append(annotation)

    return kept, removed


def find_near_duplicates(
    annotations: list[Annotation],
    iou_threshold: float,
    *,
    same_class_only: bool,
) -> list[NearDuplicate]:
    near_duplicates = []

    for first, second in combinations(annotations, 2):
        if same_class_only and first.class_id != second.class_id:
            continue

        iou = calculate_iou(first.box, second.box)
        if iou >= iou_threshold:
            near_duplicates.append(NearDuplicate(first=first, second=second, iou=iou))

    return near_duplicates


def write_annotations(file_path: Path, annotations: list[Annotation]) -> None:
    with file_path.open("w", encoding="utf-8") as labels_file:
        for annotation in annotations:
            labels_file.write(f"{annotation.raw}\n")


def iter_label_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"label path does not exist: {path}")

    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(file_path for file_path in path.glob(pattern) if file_path.is_file())


def process_file(
    file_path: Path,
    *,
    remove_exact: bool,
    write: bool,
    iou_threshold: float,
    same_class_only: bool,
) -> tuple[int, int]:
    annotations = parse_yolo_file(file_path)
    unique_annotations, exact_duplicates = remove_exact_duplicates(annotations)
    near_duplicates = find_near_duplicates(
        unique_annotations,
        iou_threshold,
        same_class_only=same_class_only,
    )

    if exact_duplicates or near_duplicates:
        print(file_path)

    for duplicate in exact_duplicates:
        print(
            "  exact duplicate "
            f"line={duplicate.duplicate.line_number} "
            f"duplicates_line={duplicate.original.line_number} "
            f"class={duplicate.duplicate.class_id} "
            f"box={format_box(duplicate.duplicate.box)}"
        )

    for duplicate in near_duplicates:
        print(
            "  near duplicate "
            f"lines={duplicate.first.line_number},{duplicate.second.line_number} "
            f"classes={duplicate.first.class_id},{duplicate.second.class_id} "
            f"iou={duplicate.iou:.4f} "
            f"boxes={format_box(duplicate.first.box)} | {format_box(duplicate.second.box)}"
        )

    if remove_exact and write and exact_duplicates:
        write_annotations(file_path, unique_annotations)

    return len(exact_duplicates), len(near_duplicates)


def format_box(box: tuple[float, float, float, float]) -> str:
    return " ".join(f"{coord:.6f}" for coord in box)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove exact YOLO annotation duplicates and report near-duplicate boxes "
            "using pairwise IoU."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="YOLO label file or directory containing .txt label files.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan .txt label files when path is a directory.",
    )
    parser.add_argument(
        "--remove-exact",
        action="store_true",
        help="Remove duplicate annotations with identical class and box coordinates.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite files after --remove-exact. Without this, the command only reports.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.95,
        help="IoU threshold used to report near-duplicate unique box pairs. Default: 0.95.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Compare boxes across classes. By default, only same-class pairs are compared.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not 0 <= args.iou_threshold <= 1:
        raise SystemExit("--iou-threshold must be between 0 and 1")

    label_files = iter_label_files(args.path, args.recursive)
    if not label_files:
        print(f"No .txt label files found under {args.path}")
        return 0

    total_exact_duplicates = 0
    total_near_duplicates = 0
    for file_path in label_files:
        if args.quiet:
            annotations = parse_yolo_file(file_path)
            unique_annotations, exact_duplicates = remove_exact_duplicates(annotations)
            near_duplicates = find_near_duplicates(
                unique_annotations,
                args.iou_threshold,
                same_class_only=not args.all_classes,
            )
            if args.remove_exact and args.write and exact_duplicates:
                write_annotations(file_path, unique_annotations)
            exact_count = len(exact_duplicates)
            near_count = len(near_duplicates)
        else:
            exact_count, near_count = process_file(
                file_path,
                remove_exact=args.remove_exact,
                write=args.write,
                iou_threshold=args.iou_threshold,
                same_class_only=not args.all_classes,
            )

        total_exact_duplicates += exact_count
        total_near_duplicates += near_count

    mode = "updated" if args.remove_exact and args.write else "checked"
    print(
        f"{mode} {len(label_files)} files; "
        f"exact_duplicates={total_exact_duplicates}; "
        f"near_duplicates={total_near_duplicates}"
    )
    if args.remove_exact and not args.write and total_exact_duplicates:
        print("Run again with --write to remove exact duplicates.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
