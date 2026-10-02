#!/usr/bin/env python3
"""Rótulos suspeitos (confident learning, Northcutt et al. 2021) nos crops GT de VAL/TEST.

Usa as probabilidades (softmax) dos classificadores JÁ treinados (CE e partes, 3 seeds), sem treino novo: os crops
de VAL/TEST nunca foram vistos no treino, então as probabilidades são out-of-sample. Média dos classificadores,
limiar por classe t_j = média de p_j entre os exemplos rotulados j, junta confiante C[i][j] e lista dos exemplos
cujo rótulo difere da classe confiante (ordenados por 1 - autoconfiança). Saída: CSV para a re-anotação cega
(sem rótulo) e para o analista.

Uso (job Slurm, container, GPU):
  python label_issues.py --crops $CROPS --split test --classifier best1.pt best2.pt ... --out DIR
As funções puras (confident_joint, find_label_issues) rodam em CPU e são testadas em tests/test_label_issues.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def class_thresholds(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """t_j = média de p_j entre os exemplos rotulados j (autoconfiança média por classe)."""
    n_classes = probs.shape[1]
    t = np.empty(n_classes)
    for j in range(n_classes):
        mask = labels == j
        if not mask.any():
            raise ValueError(f"classe {j} sem exemplos")
        t[j] = probs[mask, j].mean()
    return t


def confident_classes(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Classe confiante por exemplo: argmax entre as classes com p_j >= t_j; -1 se nenhuma passa o limiar."""
    above = probs >= thresholds[None, :]
    masked = np.where(above, probs, -np.inf)
    out = masked.argmax(axis=1)
    out[~above.any(axis=1)] = -1
    return out


def confident_joint(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """C[i][j] = nº de exemplos com rótulo i que são confiantemente da classe j (linhas: rótulo dado)."""
    t = class_thresholds(labels, probs)
    conf = confident_classes(probs, t)
    n = probs.shape[1]
    joint = np.zeros((n, n), dtype=int)
    for given, c in zip(labels, conf):
        if c >= 0:
            joint[given, c] += 1
    return joint


def find_label_issues(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Exemplos suspeitos = classe confiante != rótulo dado. `score` = 1 - p(rótulo dado) (maior = mais suspeito)."""
    labels = np.asarray(labels)
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or len(labels) != len(probs):
        raise ValueError("labels [N] e probs [N,C] incompatíveis")
    t = class_thresholds(labels, probs)
    conf = confident_classes(probs, t)
    issue = (conf >= 0) & (conf != labels)
    given_p = probs[np.arange(len(labels)), labels]
    return {"issue": issue, "confident_class": conf, "score": 1.0 - given_p,
            "thresholds": t, "joint": confident_joint(labels, probs)}


def _write_outputs(out: Path, split: str, files, labels, probs, classes, result, model_names, species):
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-result["score"])
    with open(out / f"label_issues_{split}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "given_label", "confident_label", "given_prob", "top_label", "top_prob", "issue", "is_species_confusion"])
        for i in order:
            top = int(probs[i].argmax())
            conf = int(result["confident_class"][i])
            given = classes[labels[i]]
            confl = classes[conf] if conf >= 0 else ""
            w.writerow([files[i], given, confl, f"{probs[i, labels[i]]:.4f}", classes[top], f"{probs[i, top]:.4f}",
                        bool(result["issue"][i]), bool(given in species and confl in species and given != confl)])
    summary = {
        "split": split, "n": int(len(labels)), "classifiers": model_names, "classes": list(classes),
        "n_issues": int(result["issue"].sum()), "issue_rate": float(result["issue"].mean()),
        "thresholds": dict(zip(classes, map(float, result["thresholds"]))),
        "confident_joint_rows_given_cols_confident": result["joint"].tolist(),
        "note": "Probabilidades out-of-sample (crops de VAL/TEST nao vistos no treino); suspeito != rotulo errado: "
                "a confirmacao exige re-anotacao por especialista.",
    }
    (out / f"label_issues_{split}.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():  # pragma: no cover - GPU/Slurm
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--classifier", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("requer SLURM_JOB_ID (inferencia em GPU dentro do Slurm)")
    import torch
    from torchvision import datasets, transforms

    from eval_cascade_matrix import load_classifier

    folder = datasets.ImageFolder(str(Path(args.crops) / args.split))
    files = [Path(p).name for p, _ in folder.samples]
    labels = np.array([t for _, t in folder.samples])
    classes = tuple(folder.classes)
    acc = np.zeros((len(files), len(classes)))
    names = []
    for ck in args.classifier:
        model, transform, order, info = load_classifier(Path(ck), torch, transforms)
        if tuple(order) != classes:
            raise SystemExit(f"class_order {order} != {classes} em {ck}")
        probs = []
        with torch.no_grad():
            for s in range(0, len(files), args.batch):
                batch = torch.stack([transform(folder.loader(p)) for p, _ in folder.samples[s:s + args.batch]]).cuda()
                probs.append(model(batch)["logits"].float().softmax(1).cpu().numpy())
        acc += np.concatenate(probs)
        names.append(Path(ck).parent.name)
        print(f"ok {ck}", flush=True)
    probs = acc / len(args.classifier)
    result = find_label_issues(labels, probs)
    s = _write_outputs(Path(args.out), args.split, files, labels, probs, classes, result, names, ("MD", "MV", "MC", "MF"))
    print(json.dumps({k: s[k] for k in ("split", "n", "n_issues", "issue_rate")}))


if __name__ == "__main__":
    main()
