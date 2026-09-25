import cv2
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Sequence


# ---------------------------------------------------------
# 0. Arguments
# ---------------------------------------------------------
def get_args():
    """
    Configures and parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Try to automatically recover and fix YOLO labels that are rotated."
    )

    # Required argument: The path (file or folder)
    parser.add_argument(
        "path",
        type=str,
        help="Path to a specific .txt file OR a folder containing .txt files."
    )

    parser.add_argument(
        "--recurse",
        action="store_true",
        help="When path is a folder, process .jpg files in all subfolders too. Images and labels must be in the same folder."
    )

    return parser.parse_args()

# ---------------------------------------------------------
# 1. Core Math Functions
# ---------------------------------------------------------
def adjust_box_exif(
    exif_tag: int, original_bbox: Sequence[float]
) -> Tuple[int, float, float, float, float]:
    """Rotates a YOLO box based on EXIF tag."""
    cls, cx, cy, w, h = original_bbox
    if exif_tag == 1:
        pass  # Normal
    elif exif_tag == 3:
        cx, cy = 1.0 - cx, 1.0 - cy  # 180
    elif exif_tag == 6:
        cx, cy, w, h = 1.0 - cy, cx, h, w  # 90 CW
    elif exif_tag == 8:
        cx, cy, w, h = cy, 1.0 - cx, h, w  # 90 CCW
    return int(cls), float(cx), float(cy), float(w), float(h)


def yolo_to_pixels(box: Sequence[float], H: int, W: int) -> Tuple[int, int, int, int]:
    cls, cx, cy, w, h = box
    x0 = int(round((cx - w / 2) * W))
    y0 = int(round((cy - h / 2) * H))
    x1 = int(round((cx + w / 2) * W))
    y1 = int(round((cy + h / 2) * H))
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


# ---------------------------------------------------------
# 2. Scoring Logic
# ---------------------------------------------------------
def score_label_rotation(
    img_gray: np.ndarray, labels: List[List[float]], tag: int
) -> int:
    """
    Applies a rotation to the labels and scores how many 'dark' pixels
    are captured inside the bounding boxes.
    """
    H, W = img_gray.shape
    rotated_labels = [adjust_box_exif(tag, lbl) for lbl in labels]

    score = 0
    for box in rotated_labels:
        x0, y0, x1, y1 = yolo_to_pixels(box, H, W)
        crop = img_gray[y0:y1, x0:x1]

        if crop.size > 0:
            # Count pixels darker than 120 (insects, text, thick borders)
            dark_pixels = np.sum(crop < 120)
            score += dark_pixels

    return score


# ---------------------------------------------------------
# 3. Main Processing Loop
# ---------------------------------------------------------
def process_dataset(images: list[Path]):

    # Possible orientations: Normal, 180, 90 CW, 90 CCW
    exif_tags_to_test = [1, 3, 6, 8]
    fixed_count = 0

    for img_path in images:
        lbl_path = img_path.with_suffix(".txt")

        # Read image in grayscale for fast pixel thresholding
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue

        # Read original labels
        with open(lbl_path, "r") as f:
            raw_labels = [
                list(map(float, line.strip().split())) for line in f.readlines()
            ]

        if not raw_labels:
            continue

        # Find the rotation that captures the most dark pixels
        best_tag = 1
        best_score = -1

        for tag in exif_tags_to_test:
            score = score_label_rotation(img_gray, raw_labels, tag)
            if score > best_score:
                best_score = score
                best_tag = tag

        # If the best orientation isn't 1 (Normal), fix it!
        if best_tag != 1:
            print(f"Fixing {img_path.name} (Requires tag {best_tag})")

            # Backup original
            backup_path = lbl_path.with_suffix(".txt.bak")
            if not backup_path.exists():
                lbl_path.rename(backup_path)

            # Generate and save corrected labels
            corrected_labels = [adjust_box_exif(best_tag, lbl) for lbl in raw_labels]
            with open(lbl_path, "w") as f:
                for lbl in corrected_labels:
                    f.write(
                        f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n"
                    )

            fixed_count += 1
        else:
            print(f"{img_path.name} is already correct.")

    print(f"\nDone! Automatically recovered and fixed {fixed_count} label files.")


if __name__ == "__main__":
    args = get_args()

    # Determine the list of files to process
    imgs_to_process = []

    if os.path.isfile(args.path):
        # User provided a single file
        imgs_to_process.append(args.path)
    elif os.path.isdir(args.path):
        # User provided a directory
        if args.recurse:
            for root, _, filenames in os.walk(args.path):
                for f in filenames:
                    img_file = Path(os.path.join(root, f))
                    label_file = Path(img_file).with_suffix(".txt")
                    if f.endswith(".jpg") and os.path.exists(label_file):
                        imgs_to_process.append(img_file)
        else:
            for f in os.listdir(args.path):
                img_file = Path(os.path.join(args.path,f))
                label_file = img_file.with_suffix(".txt")
                if f.endswith(".jpg") and os.path.exists(label_file):
                    imgs_to_process.append(img_file)
    else:
        print(f"Error: The path '{args.path}' does not exist.")
        sys.exit(1)
        
    # Point these to your actual dataset folders
    process_dataset(imgs_to_process)
