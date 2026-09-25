#!/usr/bin/env python3
"""
agreement_analysis.py

Analyze annotator review packages in the per-sampled-object folder format.

Expected package layout:
  <pkg>/
    manifest_objects.csv
    class_population.csv
    review_labels/
      <annotator>/
        <sample_key>/
          review_label.txt
          object_info.txt
          patch.jpg

review_label.txt is expected to contain:
  # classes: ...
  # object_id=...
  <class_id>

Important limitation:
- If review_label.txt is prefilled with the original class, this script cannot
  distinguish "reviewed and agreed" from "never actually reviewed".
  All metrics therefore reflect recorded labels, not verified review completion.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import confusion_matrix
except Exception as e:
    raise RuntimeError("Requires scikit-learn. Install: pip install scikit-learn") from e


# ----------------------------
# Parsing
# ----------------------------

def parse_review_label_file(path: Path) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse one review_label.txt file.

    Expected format:
      # classes: ...
      # object_id=...
      <class_id>

    Returns:
      (object_id, class_id)
    """
    if not path.exists():
        return None, None

    object_id: Optional[str] = None
    class_id: Optional[int] = None

    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if "object_id=" in s:
                object_id = s.split("object_id=", 1)[1].strip()
            continue

        # first non-comment, non-empty line that parses as int is the label
        try:
            class_id = int(s)
            break
        except ValueError:
            continue

    return object_id, class_id


def parse_annotator_dir(root: Path) -> Dict[str, int]:
    """
    Reads:
      root/<sample_key>/review_label.txt

    Returns:
      object_id -> class_id
    """
    out: Dict[str, int] = {}
    for sample_dir in root.iterdir():
        if not sample_dir.is_dir():
            continue
        label_path = sample_dir / "review_label.txt"
        object_id, class_id = parse_review_label_file(label_path)
        if object_id is None or class_id is None:
            continue
        out[object_id] = class_id
    return out


# ----------------------------
# Agreement metrics
# ----------------------------

def fleiss_kappa_complete(ratings: np.ndarray) -> float:
    """
    ratings: shape (m_items, n_raters), integer category ids, complete matrix.
    """
    m, n = ratings.shape
    if m == 0 or n < 2:
        return float("nan")

    k = int(ratings.max()) + 1
    counts = np.zeros((m, k), dtype=float)

    for i in range(m):
        for r in range(n):
            counts[i, ratings[i, r]] += 1.0

    P_i = np.sum(counts * (counts - 1), axis=1) / (n * (n - 1))
    P_bar = float(np.mean(P_i))

    p_j = np.sum(counts, axis=0) / (m * n)
    P_e = float(np.sum(p_j * p_j))

    denom = 1.0 - P_e
    if denom == 0.0:
        return float("nan")
    return (P_bar - P_e) / denom


def krippendorff_alpha_nominal_complete(ratings: np.ndarray, k: int) -> float:
    """
    Nominal Krippendorff's alpha for a complete ratings matrix.
    """
    m, n = ratings.shape
    if m == 0 or n < 2:
        return float("nan")

    # Observed disagreement
    Do = 0.0
    pairs = 0
    for i in range(m):
        for a in range(n):
            for b in range(a + 1, n):
                Do += 0.0 if ratings[i, a] == ratings[i, b] else 1.0
                pairs += 1
    Do = Do / pairs if pairs else 0.0

    # Expected disagreement
    counts = np.zeros(k, dtype=float)
    for i in range(m):
        for r in range(n):
            counts[ratings[i, r]] += 1.0
    total = float(np.sum(counts))
    if total <= 1:
        return float("nan")
    p = counts / total
    De = 1.0 - float(np.sum(p * p))
    if De == 0.0:
        return float("nan")

    return 1.0 - (Do / De)


def normalized_entropy(labels: List[int]) -> float:
    """
    Normalized Shannon entropy of annotator labels for one object.
    0 = full agreement, 1 = maximally spread across observed categories.
    """
    if not labels:
        return float("nan")
    c = Counter(labels)
    probs = np.array([v / len(labels) for v in c.values()], dtype=float)
    H = -float(np.sum(probs * np.log2(probs)))
    Hmax = math.log2(len(c)) if len(c) > 1 else 1.0
    return H / Hmax if Hmax > 0 else 0.0


def majority_vote(labels: List[int]) -> int:
    """
    Deterministic majority vote.
    Ties are broken by the smallest class id.
    """
    c = Counter(labels)
    best_count = max(c.values())
    winners = sorted([cls for cls, cnt in c.items() if cnt == best_count])
    return winners[0]


# ----------------------------
# Bootstrap
# ----------------------------

def bootstrap_stratified(
    df: pd.DataFrame,
    class_col: str,
    stat_fn,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    """
    Stratified bootstrap: resample within each class with replacement,
    preserving sampled n per class.
    Returns bootstrap mean and percentile CI.
    """
    rng = random.Random(seed)
    classes = sorted(df[class_col].unique().tolist())
    groups = {c: df[df[class_col] == c].reset_index(drop=True) for c in classes}
    sizes = {c: len(groups[c]) for c in classes}

    vals: List[float] = []
    for _ in range(n_boot):
        parts = []
        for c in classes:
            g = groups[c]
            n = sizes[c]
            idxs = [rng.randrange(0, n) for _ in range(n)]
            parts.append(g.iloc[idxs])
        boot_df = pd.concat(parts, axis=0, ignore_index=True)
        vals.append(float(stat_fn(boot_df)))

    vals.sort()
    mean = float(sum(vals) / len(vals))
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return mean, float(lo), float(hi)


# ----------------------------
# Main analysis helpers
# ----------------------------

def compute_object_level_fields(df: pd.DataFrame, annotators: List[str]) -> pd.DataFrame:
    """
    Adds:
      agree_vs_orig_mean
      relabel_rate_mean
      consensus_cls
      consensus_agrees_with_orig
      disagreement_entropy
    """
    def object_agreement_rate(row) -> float:
        oks = [1.0 if int(row[a]) == int(row["orig_cls"]) else 0.0 for a in annotators]
        return float(sum(oks) / len(oks))

    def object_relabel_rate(row) -> float:
        vals = [1.0 if int(row[a]) != int(row["orig_cls"]) else 0.0 for a in annotators]
        return float(sum(vals) / len(vals))

    def object_consensus(row) -> int:
        labels = [int(row[a]) for a in annotators]
        return majority_vote(labels)

    def object_entropy(row) -> float:
        labels = [int(row[a]) for a in annotators]
        return normalized_entropy(labels)

    df = df.copy()
    df["agree_vs_orig_mean"] = df.apply(object_agreement_rate, axis=1)
    df["relabel_rate_mean"] = df.apply(object_relabel_rate, axis=1)
    df["consensus_cls"] = df.apply(object_consensus, axis=1)
    df["consensus_agrees_with_orig"] = (df["consensus_cls"].astype(int) == df["orig_cls"].astype(int)).astype(float)
    df["disagreement_entropy"] = df.apply(object_entropy, axis=1)
    return df


def weighted_overall_from_per_class(df: pd.DataFrame, pop_map: Dict[int, int], metric_col: str) -> float:
    N_total = sum(pop_map.values())
    s = 0.0
    for cls, Nc in pop_map.items():
        if Nc <= 0:
            continue
        sub = df[df["orig_cls"] == cls]
        if len(sub) == 0:
            continue
        s += (Nc / N_total) * float(sub[metric_col].mean())
    return float(s)


def build_complete_ratings_matrix(df: pd.DataFrame, annotators: List[str]) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Returns:
      ratings matrix with contiguous ids
      original_class_id -> contiguous_id map
    """
    all_classes = sorted(set(df["orig_cls"].astype(int).tolist()) |
                         set(int(df[a].astype(int).max()) for a in annotators) |
                         set(int(df[a].astype(int).min()) for a in annotators))
    # safer:
    all_classes = sorted(
        set(df["orig_cls"].astype(int).tolist()) |
        set(int(v) for a in annotators for v in df[a].astype(int).tolist())
    )

    cat_to_id = {c: i for i, c in enumerate(all_classes)}
    R = np.zeros((len(df), len(annotators)), dtype=int)

    for i, (_, row) in enumerate(df.iterrows()):
        for j, a in enumerate(annotators):
            R[i, j] = cat_to_id[int(row[a])]

    return R, cat_to_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze review package in per-object folder format.")
    ap.add_argument("--pkg", type=Path, required=True, help="Review package directory")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples")
    ap.add_argument("--seed", type=int, default=123, help="Bootstrap seed")
    ap.add_argument("--out", type=Path, default=None, help="Output dir (default: <pkg>/analysis)")
    args = ap.parse_args()

    pkg = args.pkg
    out_dir = args.out if args.out else (pkg / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = pkg / "manifest_objects.csv"
    pop_path = pkg / "class_population.csv"
    review_root = pkg / "review_labels"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing: {manifest_path}")
    if not pop_path.exists():
        raise FileNotFoundError(f"Missing: {pop_path}")
    if not review_root.exists():
        raise FileNotFoundError(f"Missing: {review_root}")

    df = pd.read_csv(manifest_path)
    pop = pd.read_csv(pop_path)

    annotators = sorted([p.name for p in review_root.iterdir() if p.is_dir()])
    if not annotators:
        raise RuntimeError("No annotator folders found under review_labels/")

    # authoritative original label from manifest
    df["orig_cls"] = df["class_id"].astype(int)
    if "class_name" not in df.columns:
        df["class_name"] = df["orig_cls"].map(lambda x: f"class_{x}")

    # annotator labels
    ann_maps: Dict[str, Dict[str, int]] = {a: parse_annotator_dir(review_root / a) for a in annotators}

    for a in annotators:
        m = ann_maps[a]
        df[a] = df["object_id"].map(lambda oid: m.get(oid, np.nan))

    # With the current package design, every object should have a parseable label for every annotator.
    missing_counts = {a: int(df[a].isna().sum()) for a in annotators}
    total_missing = sum(missing_counts.values())
    if total_missing > 0:
        raise RuntimeError(
            "Found missing or unparsable annotator labels. "
            f"Counts by annotator: {missing_counts}"
        )

    for a in annotators:
        df[a] = df[a].astype(int)

    # population weights
    pop_map = {int(r["class_id"]): int(r["Nc_objects"]) for _, r in pop.iterrows()}

    # object-level derived metrics
    df = compute_object_level_fields(df, annotators)

    # per-class point estimates
    per_class_rows = []
    for cls in sorted(df["orig_cls"].unique().tolist()):
        sub = df[df["orig_cls"] == cls].copy()
        cname = str(sub["class_name"].iloc[0]) if len(sub) else f"class_{cls}"
        per_class_rows.append({
            "class_id": int(cls),
            "class_name": cname,
            "n_sampled": int(len(sub)),
            "agreement_vs_orig_mean": float(sub["agree_vs_orig_mean"].mean()),
            "relabel_rate_mean": float(sub["relabel_rate_mean"].mean()),
            "consensus_vs_orig_mean": float(sub["consensus_agrees_with_orig"].mean()),
            "mean_disagreement_entropy": float(sub["disagreement_entropy"].mean()),
        })
    per_class_df = pd.DataFrame(per_class_rows)

    # weighted overall point estimates
    overall_agreement_point = weighted_overall_from_per_class(df, pop_map, "agree_vs_orig_mean")
    overall_relabel_point = weighted_overall_from_per_class(df, pop_map, "relabel_rate_mean")
    overall_consensus_point = weighted_overall_from_per_class(df, pop_map, "consensus_agrees_with_orig")
    overall_entropy_point = weighted_overall_from_per_class(df, pop_map, "disagreement_entropy")

    # bootstrap overall CIs
    def stat_agree(boot_df: pd.DataFrame) -> float:
        return weighted_overall_from_per_class(boot_df, pop_map, "agree_vs_orig_mean")

    def stat_relabel(boot_df: pd.DataFrame) -> float:
        return weighted_overall_from_per_class(boot_df, pop_map, "relabel_rate_mean")

    def stat_consensus(boot_df: pd.DataFrame) -> float:
        return weighted_overall_from_per_class(boot_df, pop_map, "consensus_agrees_with_orig")

    def stat_entropy(boot_df: pd.DataFrame) -> float:
        return weighted_overall_from_per_class(boot_df, pop_map, "disagreement_entropy")

    agree_mean, agree_lo, agree_hi = bootstrap_stratified(df, "orig_cls", stat_agree, args.bootstrap, args.seed)
    relabel_mean, relabel_lo, relabel_hi = bootstrap_stratified(df, "orig_cls", stat_relabel, args.bootstrap, args.seed)
    consensus_mean, consensus_lo, consensus_hi = bootstrap_stratified(df, "orig_cls", stat_consensus, args.bootstrap, args.seed)
    entropy_mean, entropy_lo, entropy_hi = bootstrap_stratified(df, "orig_cls", stat_entropy, args.bootstrap, args.seed)

    # per-class bootstrap CIs
    rng = random.Random(args.seed)
    per_class_ci_rows = []
    for cls in sorted(df["orig_cls"].unique().tolist()):
        sub = df[df["orig_cls"] == cls].copy()
        cname = str(sub["class_name"].iloc[0]) if len(sub) else f"class_{cls}"

        def boot_mean(vals: List[float]) -> Tuple[float, float, float]:
            if not vals:
                return float("nan"), float("nan"), float("nan")
            boots = []
            for _ in range(args.bootstrap):
                samp = [vals[rng.randrange(0, len(vals))] for _ in range(len(vals))]
                boots.append(sum(samp) / len(samp))
            boots.sort()
            return (
                float(sum(vals) / len(vals)),
                float(boots[int(0.025 * len(boots))]),
                float(boots[int(0.975 * len(boots)) - 1]),
            )

        agree_pt, agree_ci_lo, agree_ci_hi = boot_mean(sub["agree_vs_orig_mean"].astype(float).tolist())
        relabel_pt, relabel_ci_lo, relabel_ci_hi = boot_mean(sub["relabel_rate_mean"].astype(float).tolist())
        cons_pt, cons_ci_lo, cons_ci_hi = boot_mean(sub["consensus_agrees_with_orig"].astype(float).tolist())
        ent_pt, ent_ci_lo, ent_ci_hi = boot_mean(sub["disagreement_entropy"].astype(float).tolist())

        per_class_ci_rows.append({
            "class_id": int(cls),
            "class_name": cname,
            "n_sampled": int(len(sub)),
            "agreement_vs_orig_mean": agree_pt,
            "agreement_ci_lo": agree_ci_lo,
            "agreement_ci_hi": agree_ci_hi,
            "relabel_rate_mean": relabel_pt,
            "relabel_ci_lo": relabel_ci_lo,
            "relabel_ci_hi": relabel_ci_hi,
            "consensus_vs_orig_mean": cons_pt,
            "consensus_ci_lo": cons_ci_lo,
            "consensus_ci_hi": cons_ci_hi,
            "mean_disagreement_entropy": ent_pt,
            "entropy_ci_lo": ent_ci_lo,
            "entropy_ci_hi": ent_ci_hi,
        })

    per_class_ci_df = pd.DataFrame(per_class_ci_rows)

    # confusion matrices: original vs each annotator
    all_classes = sorted(set(df["orig_cls"].astype(int).tolist()) |
                         set(int(v) for a in annotators for v in df[a].astype(int).tolist()) |
                         set(df["consensus_cls"].astype(int).tolist()))
    class_names_by_index = [cd["class_name"] for cd in per_class_rows]

    for a in annotators:
        y_true = df["orig_cls"].astype(int).tolist()
        y_pred = df[a].astype(int).tolist()
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        cm_df = pd.DataFrame(
            cm,
            index=class_names_by_index,
            columns=class_names_by_index,
        )
        cm_df.to_csv(out_dir / f"confusion_original_vs_{a}.csv", index=True)

    # confusion matrix: original vs consensus
    cm_cons = confusion_matrix(
        df["orig_cls"].astype(int).tolist(),
        df["consensus_cls"].astype(int).tolist(),
        labels=all_classes,
    )
    cm_cons_df = pd.DataFrame(
        cm_cons,
        index=class_names_by_index,
        columns=class_names_by_index,
    )
    cm_cons_df.to_csv(out_dir / "confusion_original_vs_consensus.csv", index=True)

    # inter-annotator agreement on complete matrix
    R, cat_to_id = build_complete_ratings_matrix(df, annotators)
    kappa_point = fleiss_kappa_complete(R)
    alpha_point = krippendorff_alpha_nominal_complete(R, k=len(cat_to_id))

    def kappa_stat(boot_df: pd.DataFrame) -> float:
        Rb, _ = build_complete_ratings_matrix(boot_df, annotators)
        return float(fleiss_kappa_complete(Rb))

    def alpha_stat(boot_df: pd.DataFrame) -> float:
        Rb, mapp = build_complete_ratings_matrix(boot_df, annotators)
        return float(krippendorff_alpha_nominal_complete(Rb, k=len(mapp)))

    kappa_mean, kappa_lo, kappa_hi = bootstrap_stratified(df, "orig_cls", kappa_stat, args.bootstrap, args.seed)
    alpha_mean, alpha_lo, alpha_hi = bootstrap_stratified(df, "orig_cls", alpha_stat, args.bootstrap, args.seed)

    # outputs
    per_class_df.to_csv(out_dir / "per_class_agreement_point.csv", index=False)
    per_class_ci_df.to_csv(out_dir / "per_class_agreement_ci.csv", index=False)

    overall_df = pd.DataFrame([{
        "weighted_overall_agreement_point": overall_agreement_point,
        "weighted_overall_agreement_boot_mean": agree_mean,
        "weighted_overall_agreement_ci_lo": agree_lo,
        "weighted_overall_agreement_ci_hi": agree_hi,
        "weighted_overall_relabel_point": overall_relabel_point,
        "weighted_overall_relabel_boot_mean": relabel_mean,
        "weighted_overall_relabel_ci_lo": relabel_lo,
        "weighted_overall_relabel_ci_hi": relabel_hi,
        "weighted_consensus_vs_original_point": overall_consensus_point,
        "weighted_consensus_vs_original_boot_mean": consensus_mean,
        "weighted_consensus_vs_original_ci_lo": consensus_lo,
        "weighted_consensus_vs_original_ci_hi": consensus_hi,
        "weighted_disagreement_entropy_point": overall_entropy_point,
        "weighted_disagreement_entropy_boot_mean": entropy_mean,
        "weighted_disagreement_entropy_ci_lo": entropy_lo,
        "weighted_disagreement_entropy_ci_hi": entropy_hi,
        "num_objects_sampled": int(len(df)),
        "annotators": ",".join(annotators),
        "bootstrap": int(args.bootstrap),
    }])
    overall_df.to_csv(out_dir / "overall_weighted_metrics.csv", index=False)

    inter_df = pd.DataFrame([{
        "fleiss_kappa_point": kappa_point,
        "fleiss_kappa_boot_mean": kappa_mean,
        "fleiss_kappa_ci_lo": kappa_lo,
        "fleiss_kappa_ci_hi": kappa_hi,
        "krippendorff_alpha_point": alpha_point,
        "krippendorff_alpha_boot_mean": alpha_mean,
        "krippendorff_alpha_ci_lo": alpha_lo,
        "krippendorff_alpha_ci_hi": alpha_hi,
        "num_objects_sampled": int(len(df)),
        "num_annotators": int(len(annotators)),
        "note": "Metrics reflect recorded labels; actual review completion cannot be verified from prefilled files.",
    }])
    inter_df.to_csv(out_dir / "inter_annotator_metrics.csv", index=False)

    audit_cols = (
        ["object_id", "subset", "label_rel", "line_idx", "orig_cls", "class_name"]
        + annotators
        + ["agree_vs_orig_mean", "relabel_rate_mean", "consensus_cls", "consensus_agrees_with_orig", "disagreement_entropy"]
    )
    df[audit_cols].to_csv(out_dir / "audit_object_ratings.csv", index=False)

    print("=== Analysis complete ===")
    print(f"Package: {pkg}")
    print(f"Annotators: {annotators}")
    print(f"Objects analyzed: {len(df)}")
    print(f"Weighted overall agreement: {overall_agreement_point:.4f}")
    print(f"Weighted overall agreement CI: {agree_mean:.4f} [{agree_lo:.4f}, {agree_hi:.4f}]")
    print(f"Weighted overall relabel rate: {overall_relabel_point:.4f}")
    print(f"Consensus vs original: {overall_consensus_point:.4f}")
    print(f"Fleiss' kappa: {kappa_point:.4f}")
    print(f"Krippendorff's alpha: {alpha_point:.4f}")
    print("Note: recorded labels do not prove the item was actually reviewed if files were prefilled.")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()