"""Tabela por especie (TEST, E0 x ConvNeXt-T CE, 9 pares): recall, precisao, F1 e erro medio de contagem por imagem,
detector sozinho vs cascata. Mesmo matching do harness (conf>=0.25, IoU>=0.5, greedy classe-agnostico).
precisao_c = GT casados com classe correta c / propostas (conf>=0.25) rotuladas c; erro de contagem = media por
imagem de |n_pred_c - n_gt_c| (por imagem, nao por placa: o dataset nao tem id de placa).
Uso: python3 -I common/analysis/rq2_per_species_table.py [RUNS_ROOT]
Saida: rq2_cascade/results/v9_per_species_single_vs_cascade.csv"""
from __future__ import annotations
import csv, json, statistics as st, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "rq2_cascade"))
import eval_cascade_matrix as M  # noqa: E402
CLS = M.YOLO_CLASSES
RUNS = Path(sys.argv[1] if len(sys.argv) > 1 else "/raid/user_marcospaulo/flydet_runs_v9")
SPC = list(M.SPECIES)


def one(recs):
    out = {m: {s: dict(tp=0, gt=0, pred=0, err=0.0) for s in SPC} for m in ("single", "cascade")}
    for rec in recs:
        gt = [{"class": CLS.index(g["class"]), "box": g["box"]} for g in rec["gt"]]
        single = [{"box": q["box"], "class": CLS.index(q["yolo_class"]), "yolo_confidence": q["yolo_conf"]} for q in rec["proposals"]]
        probs = lambda q: q.get("probabilities")
        casc = []
        for q in rec["proposals"]:
            p = probs(q)
            c = CLS.index(q["yolo_class"]) if p is None or any(x is None for x in p) else max(range(len(CLS)), key=lambda k: p[k])
            casc.append({"box": q["box"], "class": c, "yolo_confidence": q["yolo_conf"], "probabilities": None})
        rows = M.rows_for_image(rec["image"], gt, single, casc)
        for m, key, preds in (("single", "correct_single", single), ("cascade", "correct_cascade", casc)):
            for s in SPC:
                n_gt = sum(r["gt_class"] == s for r in rows)
                n_pred = sum(CLS[p["class"]] == s and p["yolo_confidence"] >= 0.25 for p in preds)
                o = out[m][s]
                o["gt"] += n_gt; o["pred"] += n_pred
                o["tp"] += sum(r["gt_class"] == s and r[key] for r in rows)
                o["err"] += abs(n_pred - n_gt)
    return out, len(recs)


def main():
    acc = {m: {s: [] for s in SPC} for m in ("single", "cascade")}
    for d in range(3):
        for c in (42, 84, 126):
            p = RUNS / "cascade_cross_test" / f"E0_s{d}" / f"clf_convnext_t_ce_s{c}" / "label_only" / "per_image_preds.jsonl"
            o, n = one([json.loads(l) for l in p.read_text().splitlines()])
            for m in o:
                for s in SPC:
                    v = o[m][s]
                    r = v["tp"] / v["gt"]; pr = v["tp"] / v["pred"] if v["pred"] else 0.0
                    acc[m][s].append((r, pr, 2 * r * pr / (r + pr) if r + pr else 0.0, v["err"] / n))
    res = REPO / "rq2_cascade" / "results" / "v9_per_species_single_vs_cascade.csv"
    with open(res, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "species", "recall", "precision", "f1", "count_mae_per_image", "n_pairs"])
        for m in acc:
            for s in SPC:
                xs = acc[m][s]
                w.writerow([m, s] + [round(st.mean(x[i] for x in xs), 4) for i in range(4)] + [len(xs)])
    print(res.read_text())

main()
