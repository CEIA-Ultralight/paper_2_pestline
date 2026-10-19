"""Variantes de pos-processamento da cascata (CPU) sobre as predicoes por proposta ja gravadas.

Le per_image_preds.jsonl de cascade_cross_{val,test}/E0_s*/clf_convnext_t_ce_s*/label_only (propostas do
detector + probabilidades do classificador, ordem YOLO) e avalia, com o mesmo matching do harness
(rq2_cascade/eval_cascade_matrix.rows_for_image):
  single      classe do detector
  cascade     argmax do classificador em toda proposta (o que o paper reporta)
  species4    reclassifica so propostas que o detector chamou de especie, argmax entre MD/MV/MC/MF
  fuse(w)     argmax_c [ w * conf_det * 1(c=classe_det) + (1-w) * p_clf(c) ]  (fusao tardia; o detector
              end-to-end so expoe a classe top-1 e seu score, nao a distribuicao completa)
  fuse4(w)    fuse(w) restrito ao conjunto de especies (propostas de especie; argmax entre as 4)
w e escolhido na VAL (maior acuracia media nos 9 pares) e aplicado ao TEST.
Saida: rq2_cascade/results/v9_cascade_variants.csv (por par) e v9_cascade_variants_summary.csv.
Uso: python3 -I common/analysis/rq2_cascade_variants.py [RUNS_ROOT]
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rq2_cascade"))
import eval_cascade_matrix as M  # noqa: E402

CLS = M.YOLO_CLASSES
SP = [CLS.index(n) for n in M.SPECIES]
WS = [round(0.05 * i, 2) for i in range(21)]
RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "/raid/user_marcospaulo/flydet_runs_v9")


def load(split, det, clf):
    p = RUNS / f"cascade_cross_{split}" / det / clf / "label_only" / "per_image_preds.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()]


def decide(prop, variant, w):
    d = CLS.index(prop["yolo_class"])
    probs = prop.get("probabilities")
    if probs is None or any(x is None for x in probs):
        return d  # crop degenerado: mantem o detector
    if variant == "single":
        return d
    if variant == "cascade":
        return max(range(len(CLS)), key=lambda c: probs[c])
    restrict = variant in ("species4", "fuse4")
    if restrict and d not in SP:
        return d
    pool = SP if restrict else range(len(CLS))
    if variant == "species4":
        return max(pool, key=lambda c: probs[c])
    return max(pool, key=lambda c: w * prop["yolo_conf"] * (c == d) + (1 - w) * probs[c])


def per_image(records, variant, w=0.0):
    """[(n_especie, n_corretas)] por imagem (ordem de records)."""
    out = []
    key = "correct_single" if variant == "single" else "correct_cascade"
    for rec in records:
        gt = [{"class": CLS.index(g["class"]), "box": g["box"]} for g in rec["gt"]]
        single = [{"box": q["box"], "class": CLS.index(q["yolo_class"]), "yolo_confidence": q["yolo_conf"]}
                  for q in rec["proposals"]]
        casc = [{"box": q["box"], "class": decide(q, variant, w), "yolo_confidence": q["yolo_conf"],
                 "probabilities": None} for q in rec["proposals"]]
        sp = [r for r in M.rows_for_image(rec["image"], gt, single, casc) if r["gt_class"] in M.SPECIES]
        out.append((len(sp), sum(r[key] for r in sp)))
    return out


def bootstrap_delta(data_split, pairs, variant, w, n=10000, seed=20261019):
    """IC95 percentil (reamostra imagens) do delta de acuracia vs single, media dos 9 pares."""
    import random
    per = {p: (per_image(data_split[p], variant, w), per_image(data_split[p], "single")) for p in pairs}
    n_img = len(per[pairs[0]][0])
    rng = random.Random(seed)
    ds = []
    for _ in range(n):
        idx = [rng.randrange(n_img) for _ in range(n_img)]
        tot = []
        for p in pairs:
            v, s = per[p]
            den = sum(v[i][0] for i in idx)
            tot.append((sum(v[i][1] for i in idx) - sum(s[i][1] for i in idx)) / den if den else 0.0)
        ds.append(sum(tot) / len(tot))
    ds.sort()
    return ds[int(0.025 * n)], ds[int(0.975 * n)]


def accuracy(records, variant, w=0.0):
    rows = []
    for rec in records:
        gt = [{"class": CLS.index(g["class"]), "box": g["box"]} for g in rec["gt"]]
        single = [{"box": q["box"], "class": CLS.index(q["yolo_class"]), "yolo_confidence": q["yolo_conf"]}
                  for q in rec["proposals"]]
        casc = [{"box": q["box"], "class": decide(q, variant, w), "yolo_confidence": q["yolo_conf"],
                 "probabilities": None} for q in rec["proposals"]]
        rows += M.rows_for_image(rec["image"], gt, single, casc)
    sp = [r for r in rows if r["gt_class"] in M.SPECIES]
    key = "correct_single" if variant == "single" else "correct_cascade"
    return sum(r[key] for r in sp) / len(sp)


def main():
    pairs = [(f"E0_s{d}", f"clf_convnext_t_ce_s{c}") for d in range(3) for c in (42, 84, 126)]
    data = {s: {p: load(s, *p) for p in pairs} for s in ("val", "test")}
    sel = {}
    for variant in ("fuse", "fuse4"):
        score = {w: st.mean(accuracy(data["val"][p], variant, w) for p in pairs) for w in WS}
        sel[variant] = max(WS, key=lambda w: (score[w], -abs(w - 0.5)))
    print("w escolhido na VAL:", sel)
    out = []
    for split in ("val", "test"):
        for p in pairs:
            recs = data[split][p]
            row = {"split": split, "detector": p[0], "classifier": p[1],
                   "single": accuracy(recs, "single"), "cascade": accuracy(recs, "cascade"),
                   "species4": accuracy(recs, "species4"),
                   "fuse": accuracy(recs, "fuse", sel["fuse"]), "fuse4": accuracy(recs, "fuse4", sel["fuse4"])}
            out.append(row)
    cols = ["split", "detector", "classifier", "single", "cascade", "species4", "fuse", "fuse4"]
    res = REPO / "rq2_cascade" / "results"
    with open(res / "v9_cascade_variants.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in out)
    summ = []
    for split in ("val", "test"):
        rs = [r for r in out if r["split"] == split]
        for v in ("single", "cascade", "species4", "fuse", "fuse4"):
            xs = [r[v] for r in rs]
            ds = [r[v] - r["single"] for r in rs]
            summ.append({"split": split, "variant": v, "w": sel.get(v, ""), "n_pairs": len(xs),
                         "acc_mean": round(st.mean(xs), 5), "acc_sd": round(st.stdev(xs), 5),
                         "delta_vs_single_mean": round(st.mean(ds), 5),
                         "pairs_better_than_single": sum(d > 0 for d in ds)})
    for r in summ:
        if r["variant"] in ("single",):
            r["ci95_lo"] = r["ci95_hi"] = ""
            continue
        lo, hi = bootstrap_delta(data[r["split"]], pairs, r["variant"], sel.get(r["variant"], 0.0))
        r["ci95_lo"], r["ci95_hi"] = round(lo, 5), round(hi, 5)
    with open(res / "v9_cascade_variants_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    for r in summ:
        print(r)


if __name__ == "__main__":
    main()
