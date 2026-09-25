#!/usr/bin/env python3
"""
plot_agreement_report.py

Generate report figures from agreement-analysis CSV outputs.

Usage:
    python plot_agreement_report.py --analysis-dir /path/to/analysis

Expected inputs inside --analysis-dir (best effort; script skips what is missing):
    - overall_weighted_metrics.csv
    - inter_annotator_metrics.csv
    - per_class_agreement_ci.csv
    - per_class_agreement_point.csv
    - class_population.csv
    - class_sample_plan.csv
    - confusion_original_vs_consensus.csv
    - confusion_original_vs_<annotator>.csv
    - audit_object_ratings.csv

Outputs:
    - <analysis-dir>/images/*.png

Notes:
- Uses matplotlib only.
- Handles class labels directly from CSV row/column names.
- Tries to infer annotator columns from audit_object_ratings.csv.
- Optionally creates a gallery of most ambiguous objects if patches exist under:
      <analysis-dir>/../samples/<sample_key>/patch.jpg
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def sanitize_sample_key(object_id: str) -> str:
    key = object_id.replace("::", "__")
    key = key.replace("/", "__")
    key = key.replace("\\", "__")
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return key


def readable_metric_name(col: str) -> str:
    return (
        col.replace("_", " ")
           .replace("ci lo", "CI low")
           .replace("ci hi", "CI high")
           .replace("boot mean", "bootstrap mean")
           .replace("vs", "vs.")
           .title()
    )


def infer_annotator_columns(df: pd.DataFrame) -> List[str]:
    known = {
        "object_id", "subset", "label_rel", "line_idx", "orig_cls", "class_name",
        "agree_vs_orig_mean", "relabel_rate_mean", "consensus_cls",
        "consensus_agrees_with_orig", "disagreement_entropy"
    }
    ann = []
    for c in df.columns:
        if c in known:
            continue
        # annotator columns should be integer-like class ids
        s = df[c].dropna()
        if len(s) == 0:
            continue
        try:
            pd.to_numeric(s, errors="raise")
            ann.append(c)
        except Exception:
            pass
    return ann


def heatmap_from_df(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    normalize_rows: bool = False,
    annotate: bool = True,
    fmt_counts: bool = True,
) -> None:
    values = df.to_numpy(dtype=float)
    shown = values.copy()
    if normalize_rows:
        row_sums = shown.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        shown = shown / row_sums

    n_rows, n_cols = shown.shape
    fig_w = max(8, 0.55 * n_cols + 3)
    fig_h = max(6, 0.45 * n_rows + 3)

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(shown, aspect="auto", cmap="Blues")
    plt.colorbar()
    plt.title(title)

    plt.xticks(np.arange(n_cols), df.columns.tolist(), rotation=90)
    plt.yticks(np.arange(n_rows), df.index.tolist())

    if annotate and n_rows <= 25 and n_cols <= 25:
        for i in range(n_rows):
            for j in range(n_cols):
                if normalize_rows:
                    text = f"{shown[i, j]:.2f}"
                else:
                    text = f"{int(values[i, j])}" if fmt_counts else f"{shown[i, j]:.2f}"
                plt.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.xlabel("Reviewed / predicted class")
    plt.ylabel("Original / true class")
    savefig(out_path)


# ----------------------------
# Plot families
# ----------------------------

def plot_overall_summary(analysis_dir: Path, images_dir: Path) -> None:
    overall = load_csv(analysis_dir / "overall_weighted_metrics.csv")
    inter = load_csv(analysis_dir / "inter_annotator_metrics.csv")
    if overall is None and inter is None:
        return

    rows = []

    if overall is not None and len(overall):
        r = overall.iloc[0].to_dict()

        metric_specs = [
            ("Weighted overall agreement",
             r.get("weighted_overall_agreement_point"),
             r.get("weighted_overall_agreement_ci_lo"),
             r.get("weighted_overall_agreement_ci_hi")),
            ("Weighted overall relabel rate",
             r.get("weighted_overall_relabel_point"),
             r.get("weighted_overall_relabel_ci_lo"),
             r.get("weighted_overall_relabel_ci_hi")),
            ("Weighted consensus vs original",
             r.get("weighted_consensus_vs_original_point"),
             r.get("weighted_consensus_vs_original_ci_lo"),
             r.get("weighted_consensus_vs_original_ci_hi")),
            ("Weighted disagreement entropy",
             r.get("weighted_disagreement_entropy_point"),
             r.get("weighted_disagreement_entropy_ci_lo"),
             r.get("weighted_disagreement_entropy_ci_hi")),
        ]
        for name, pt, lo, hi in metric_specs:
            if pt is not None and lo is not None and hi is not None:
                rows.append((name, float(pt), float(lo), float(hi)))

    if inter is not None and len(inter):
        r = inter.iloc[0].to_dict()
        metric_specs = [
            ("Fleiss' kappa",
             r.get("fleiss_kappa_point"),
             r.get("fleiss_kappa_ci_lo"),
             r.get("fleiss_kappa_ci_hi")),
            ("Krippendorff's alpha",
             r.get("krippendorff_alpha_point"),
             r.get("krippendorff_alpha_ci_lo"),
             r.get("krippendorff_alpha_ci_hi")),
        ]
        for name, pt, lo, hi in metric_specs:
            if pt is not None and lo is not None and hi is not None:
                rows.append((name, float(pt), float(lo), float(hi)))

    if not rows:
        return

    names = [x[0] for x in rows]
    pts = np.array([x[1] for x in rows], dtype=float)
    los = np.array([x[2] for x in rows], dtype=float)
    his = np.array([x[3] for x in rows], dtype=float)
    xerr = np.vstack([pts - los, his - pts])

    plt.figure(figsize=(10, max(4.5, 0.75 * len(rows) + 1.5)))
    y = np.arange(len(rows))
    plt.errorbar(pts, y, xerr=xerr, fmt="o", capsize=4)
    plt.yticks(y, names)
    plt.xlabel("Value")
    plt.title("Overall agreement metrics with confidence intervals")
    plt.xlim(min(0, float(np.nanmin(los)) - 0.05), max(1.0, float(np.nanmax(his)) + 0.05))
    plt.grid(axis="x", alpha=0.3)
    savefig(images_dir / "overall_metrics_ci.png")


def plot_per_class_metrics(analysis_dir: Path, images_dir: Path) -> None:
    df = load_csv(analysis_dir / "per_class_agreement_ci.csv")
    if df is None or df.empty:
        df = load_csv(analysis_dir / "per_class_agreement_point.csv")
    if df is None or df.empty:
        return

    sort_base = "relabel_rate_mean" if "relabel_rate_mean" in df.columns else (
        "mean_disagreement_entropy" if "mean_disagreement_entropy" in df.columns else "n_sampled"
    )
    df = df.sort_values(sort_base, ascending=False).reset_index(drop=True)

    label_col = "class_name" if "class_name" in df.columns else "class_id"
    labels = df[label_col].astype(str).tolist()

    specs = [
        ("agreement_vs_orig_mean", "agreement_ci_lo", "agreement_ci_hi", "Per-class agreement vs original", "per_class_agreement_ci.png"),
        ("relabel_rate_mean", "relabel_ci_lo", "relabel_ci_hi", "Per-class relabel rate", "per_class_relabel_ci.png"),
        ("consensus_vs_orig_mean", "consensus_ci_lo", "consensus_ci_hi", "Per-class consensus vs original", "per_class_consensus_ci.png"),
        ("mean_disagreement_entropy", "entropy_ci_lo", "entropy_ci_hi", "Per-class disagreement entropy", "per_class_entropy_ci.png"),
    ]

    for pt_col, lo_col, hi_col, title, fname in specs:
        if pt_col not in df.columns:
            continue

        pts = df[pt_col].astype(float).to_numpy()
        plt.figure(figsize=(11, max(5, 0.45 * len(df) + 1.5)))
        y = np.arange(len(df))

        if lo_col in df.columns and hi_col in df.columns:
            los = df[lo_col].astype(float).to_numpy()
            his = df[hi_col].astype(float).to_numpy()
            xerr = np.vstack([pts - los, his - pts])
            plt.barh(y, pts, xerr=xerr, capsize=3)
        else:
            plt.barh(y, pts)

        tick_labels = []
        for _, row in df.iterrows():
            label = str(row[label_col])
            if "n_sampled" in df.columns:
                label += f" (n={int(row['n_sampled'])})"
            tick_labels.append(label)

        plt.yticks(y, tick_labels)
        plt.xlabel("Value")
        plt.title(title)
        plt.grid(axis="x", alpha=0.3)
        savefig(images_dir / fname)


def plot_population_vs_sample(analysis_dir: Path, images_dir: Path) -> None:
    pop = load_csv(analysis_dir / "class_population.csv")
    plan = load_csv(analysis_dir / "class_sample_plan.csv")
    if pop is None or plan is None or pop.empty or plan.empty:
        return

    merged = pd.merge(
        pop,
        plan[["class_id", "n_sampled"]],
        on="class_id",
        how="left",
    )
    if "class_name" not in merged.columns:
        merged["class_name"] = merged["class_id"].map(lambda x: f"class_{x}")

    pop_col = "Nc_objects" if "Nc_objects" in merged.columns else None
    if pop_col is None:
        return

    merged = merged.sort_values(pop_col, ascending=False).reset_index(drop=True)
    labels = merged["class_name"].astype(str).tolist()
    y = np.arange(len(merged))
    h = 0.38

    plt.figure(figsize=(11, max(5, 0.45 * len(merged) + 1.5)))
    plt.barh(y + h / 2, merged[pop_col].astype(float).to_numpy(), height=h, label="Population")
    plt.barh(y - h / 2, merged["n_sampled"].fillna(0).astype(float).to_numpy(), height=h, label="Sampled")
    plt.yticks(y, labels)
    plt.xlabel("Object count")
    plt.title("Population vs sampled objects per class")
    plt.legend()
    plt.grid(axis="x", alpha=0.3)
    savefig(images_dir / "class_population_vs_sampled.png")


def plot_confusion_matrices(analysis_dir: Path, images_dir: Path) -> None:
    for path in sorted(analysis_dir.glob("confusion_*.csv")):
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            continue
        stem = path.stem
        title = stem.replace("_", " ")
        heatmap_from_df(df, f"{title} (counts)", images_dir / f"{stem}_counts.png", normalize_rows=False)
        heatmap_from_df(df, f"{title} (row-normalized)", images_dir / f"{stem}_normalized.png", normalize_rows=True)

    consensus_path = analysis_dir / "confusion_original_vs_consensus.csv"
    if consensus_path.exists():
        df = pd.read_csv(consensus_path, index_col=0)
        vals = df.to_numpy(dtype=float)
        pairs = []
        for i, rlab in enumerate(df.index.tolist()):
            for j, clab in enumerate(df.columns.tolist()):
                if i == j:
                    continue
                count = vals[i, j]
                if count > 0:
                    pairs.append((float(count), str(rlab), str(clab)))

        if pairs:
            pairs = sorted(pairs, reverse=True)[:15]
            counts = [p[0] for p in pairs][::-1]
            labels = [f"{p[1]} → {p[2]}" for p in pairs][::-1]
            plt.figure(figsize=(10, max(5, 0.4 * len(labels) + 1.5)))
            y = np.arange(len(labels))
            plt.barh(y, counts)
            plt.yticks(y, labels)
            plt.xlabel("Count")
            plt.title("Top off-diagonal confusion pairs (original vs consensus)")
            plt.grid(axis="x", alpha=0.3)
            savefig(images_dir / "top_confusion_pairs_consensus.png")


def plot_object_level(analysis_dir: Path, images_dir: Path) -> None:
    audit = load_csv(analysis_dir / "audit_object_ratings.csv")
    if audit is None or audit.empty:
        return

    if "agree_vs_orig_mean" in audit.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(audit["agree_vs_orig_mean"].astype(float).to_numpy(), bins=20)
        plt.xlabel("Object-level agreement vs original")
        plt.ylabel("Number of objects")
        plt.title("Distribution of object-level agreement")
        savefig(images_dir / "hist_object_agreement.png")

    if "relabel_rate_mean" in audit.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(audit["relabel_rate_mean"].astype(float).to_numpy(), bins=20)
        plt.xlabel("Object-level relabel rate")
        plt.ylabel("Number of objects")
        plt.title("Distribution of object-level relabel rate")
        savefig(images_dir / "hist_object_relabel_rate.png")

    if "disagreement_entropy" in audit.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(audit["disagreement_entropy"].astype(float).to_numpy(), bins=20)
        plt.xlabel("Object-level disagreement entropy")
        plt.ylabel("Number of objects")
        plt.title("Distribution of disagreement entropy")
        savefig(images_dir / "hist_disagreement_entropy.png")

    if "disagreement_entropy" in audit.columns and "consensus_agrees_with_orig" in audit.columns:
        x = audit["disagreement_entropy"].astype(float).to_numpy()
        y = audit["consensus_agrees_with_orig"].astype(float).to_numpy()
        # deterministic jitter
        offsets = np.linspace(-0.03, 0.03, len(y)) if len(y) > 1 else np.array([0.0])
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y + offsets)
        plt.xlabel("Disagreement entropy")
        plt.ylabel("Consensus agrees with original (jittered)")
        plt.title("Entropy vs consensus correctness")
        plt.yticks([0, 1], ["No", "Yes"])
        savefig(images_dir / "scatter_entropy_vs_consensus.png")


def plot_annotator_specific(analysis_dir: Path, images_dir: Path) -> None:
    audit = load_csv(analysis_dir / "audit_object_ratings.csv")
    if audit is None or audit.empty:
        return

    annotators = infer_annotator_columns(audit)
    if not annotators:
        return

    orig = audit["orig_cls"].astype(int)

    # Agreement vs original by annotator
    agree_rows = []
    relabel_rows = []
    for a in annotators:
        vals = audit[a].astype(int)
        agree = float((vals == orig).mean())
        relabel = float((vals != orig).mean())
        agree_rows.append((a, agree))
        relabel_rows.append((a, relabel))

    for title, rows, fname in [
        ("Per-annotator agreement vs original", agree_rows, "annotator_agreement_vs_original.png"),
        ("Per-annotator relabel rate", relabel_rows, "annotator_relabel_rate.png"),
    ]:
        labels = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        plt.figure(figsize=(8, max(4, 0.55 * len(rows) + 1.5)))
        y = np.arange(len(labels))
        plt.barh(y, vals)
        plt.yticks(y, labels)
        plt.xlabel("Value")
        plt.title(title)
        plt.xlim(0, 1)
        plt.grid(axis="x", alpha=0.3)
        savefig(images_dir / fname)

    # Pairwise raw agreement heatmap
    mat = np.zeros((len(annotators), len(annotators)), dtype=float)
    for i, a in enumerate(annotators):
        for j, b in enumerate(annotators):
            mat[i, j] = float((audit[a].astype(int) == audit[b].astype(int)).mean())

    df = pd.DataFrame(mat, index=annotators, columns=annotators)
    heatmap_from_df(df, "Pairwise annotator raw agreement", images_dir / "annotator_pairwise_agreement.png", normalize_rows=False, annotate=True, fmt_counts=False)


def plot_ambiguous_gallery(analysis_dir: Path, images_dir: Path, top_k: int = 12) -> None:
    audit = load_csv(analysis_dir / "audit_object_ratings.csv")
    if audit is None or audit.empty:
        return
    if "disagreement_entropy" not in audit.columns or "object_id" not in audit.columns:
        return

    samples_dir = analysis_dir.parent / "samples"
    if not samples_dir.exists():
        return

    annotators = infer_annotator_columns(audit)
    audit = audit.sort_values("disagreement_entropy", ascending=False).head(top_k).reset_index(drop=True)

    items = []
    for _, row in audit.iterrows():
        key = sanitize_sample_key(str(row["object_id"]))
        patch = samples_dir / key / "patch.jpg"
        if patch.exists():
            items.append((row, patch))

    if not items:
        return

    n = len(items)
    ncols = 3
    nrows = math.ceil(n / ncols)
    plt.figure(figsize=(5 * ncols, 4.5 * nrows))

    for idx, (row, patch_path) in enumerate(items, start=1):
        ax = plt.subplot(nrows, ncols, idx)
        img = Image.open(patch_path).convert("RGB")
        ax.imshow(img)
        ax.axis("off")

        title_lines = []
        cname = row["class_name"] if "class_name" in row else row["orig_cls"]
        title_lines.append(f"Orig: {cname}")
        if "consensus_cls" in row:
            title_lines.append(f"Consensus: {row['consensus_cls']}")
        if "disagreement_entropy" in row:
            title_lines.append(f"Entropy: {float(row['disagreement_entropy']):.3f}")
        if annotators:
            votes = ", ".join(f"{a}={int(row[a])}" for a in annotators if a in row)
            title_lines.append(votes)
        ax.set_title("\n".join(title_lines), fontsize=9)

    plt.suptitle("Most ambiguous sampled objects", y=1.01)
    savefig(images_dir / "gallery_most_ambiguous_objects.png")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate report figures from agreement-analysis CSVs.")
    ap.add_argument("--analysis-dir", type=Path, required=True, help="Folder containing analysis CSV files")
    args = ap.parse_args()

    analysis_dir = args.analysis_dir
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory not found: {analysis_dir}")

    images_dir = analysis_dir / "images"
    ensure_dir(images_dir)

    plot_overall_summary(analysis_dir, images_dir)
    plot_per_class_metrics(analysis_dir, images_dir)
    plot_population_vs_sample(analysis_dir, images_dir)
    plot_confusion_matrices(analysis_dir, images_dir)
    plot_object_level(analysis_dir, images_dir)
    plot_annotator_specific(analysis_dir, images_dir)
    plot_ambiguous_gallery(analysis_dir, images_dir, top_k=12)

    print(f"Plots written to: {images_dir}")


if __name__ == "__main__":
    main()
