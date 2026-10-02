"""RQ2 (B200): comparacao de politicas (label_only x product) com AP e P/R/F1 limiarizando o SCORE da politica.

O harness grava P/R/F1 limiarizando yolo_conf (>=0.25) para ambas as politicas -> identicos por construcao.
Aqui reconstruimos as propostas de per_image_preds.jsonl e (a) calculamos AP (ultralytics ap_per_class, mesmo
protocolo de eval_yolo_cascade.metrics) e (b) reavaliamos P/R/F1 com limiar 0.25 no score da politica.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from b200_lib import *  # noqa: F403

sys.path.insert(0, str(REPO / "rq2_cascade"))
import eval_cascade_matrix as M  # noqa: E402
import eval_yolo_cascade as EV  # noqa: E402
from ultralytics.utils.metrics import ap_per_class  # noqa: E402

OUT = REPO / "rq2_cascade" / "results"
DETS = [f"E0_s{s}" for s in range(3)] + [f"E10_nwd_w0.5_s{s}" for s in range(3)]
CLFS = [f"clf_convnext_t_{f}_s{s}" for f in ("ce", "parts") for s in (42, 84, 126)]
YC = list(EV.YOLO_CLASSES)
JOB = "33364"
TAU = 0.25


def det_fam(d):
    return "E0" if d.startswith("E0_") else "E10"


def clf_fam(c):
    return "ce" if "_ce_" in c else "parts"


def load_recs(path):
    recs = []
    for line in open(path):
        j = json.loads(line)
        gt = [{"class": YC.index(g["class"]), "box": g["box"]} for g in j["gt"]]
        single = [{"box": p["box"], "class": YC.index(p["yolo_class"]), "yolo_confidence": p["yolo_conf"], "probabilities": None} for p in j["proposals"]]
        casc = [{"box": p["box"], "class": YC.index(p["cascade_class"]), "yolo_confidence": p["yolo_conf"], "probabilities": p.get("probabilities"),
                 "cascade_score": p["cascade_score"]} for p in j["proposals"]]
        recs.append({"image": j["image"], "gt": gt, "single": single, "cascade": casc, "seconds": j["seconds"]})
    return recs


def ap_eval(gts, preds):
    tp, scores, classes, targets = [], [], [], []
    for gt, pred in zip(gts, preds, strict=True):
        targets.extend(g["class"] for g in gt)
        pairs = EV.candidates(gt, pred, class_aware=True)
        correct = [[False] * 10 for _ in pred]
        for t, th in enumerate(EV.THRESHOLDS):
            for _, p, _ in EV.greedy(pairs, th):
                correct[p][t] = True
        tp.extend(correct)
        scores.extend(p["confidence"] for p in pred)
        classes.extend(p["class"] for p in pred)
    out = ap_per_class(np.asarray(tp, bool).reshape(-1, 10), np.asarray(scores), np.asarray(classes, int), np.asarray(targets, int), plot=False)
    ap, ids = np.asarray(out[5]), np.asarray(out[6])
    res = {YC[i]: (float(ap[r, 0]), float(ap[r].mean()), targets.count(int(i))) for r, i in enumerate(ids)}
    sp = [r for r, i in enumerate(ids) if i < 4]
    res["species_mean"] = (float(ap[sp, 0].mean()), float(ap[sp].mean()), sum(targets.count(i) for i in range(4)))
    res["all7_mean"] = (float(ap[:, 0].mean()), float(ap.mean()), len(targets))
    return res


def score_thr_recs(recs, which):
    """which='single'|'label_only'|'product': threshold sobre o score da politica (substitui yolo_confidence)."""
    out = []
    for r in recs:
        cs = [dict(c, yolo_confidence=c["cascade_score"]) for c in r["cascade"]]
        sg = [dict(s, yolo_confidence=c["cascade_score"]) for s, c in zip(r["single"], r["cascade"], strict=True)]
        out.append({"image": r["image"], "gt": r["gt"], "single": sg, "cascade": cs, "seconds": r["seconds"]})
    return out


def task(args):
    split, d, c = args
    base = R / f"cascade_cross_{split.lower()}" / d / c
    lo, pr = load_recs(base / "label_only" / "per_image_preds.jsonl"), load_recs(base / "product" / "per_image_preds.jsonl")
    gts = [r["gt"] for r in lo]
    res = {"split": split, "det": d, "clf": c}
    res["ap_lo"] = ap_eval(gts, [[dict(p, confidence=p["cascade_score"]) for p in r["cascade"]] for r in lo])
    res["ap_pr"] = ap_eval(gts, [[dict(p, confidence=p["cascade_score"]) for p in r["cascade"]] for r in pr])
    if c == clf_first[d]:
        res["ap_single"] = ap_eval(gts, [[dict(p, confidence=p["yolo_confidence"]) for p in r["single"]] for r in lo])
    # P/R/F1 limiarizando o score da politica (label_only deve reproduzir o metrics.json)
    for name, recs in (("lo", lo), ("pr", pr)):
        _, _, m = M.assemble_outputs(score_thr_recs(recs, name), "label_only", True, TAU, 0.5)
        res[f"thr_{name}"] = harness_metrics(m["cascade"])  # noqa: F405
        if name == "lo":
            res["thr_single"] = harness_metrics(m["single"])  # noqa: F405
            ref = load_json(base / "label_only" / "metrics.json")["cascade"]  # noqa: F405
            res["lo_reproduces"] = (abs(m["cascade"]["species_correct"] - ref["species_correct"]) <= 1
                                    and abs(m["cascade"]["species_matched"] - ref["species_matched"]) <= 1)
            res["lo_delta_correct"] = m["cascade"]["species_correct"] - ref["species_correct"]
    return res


clf_first = dict.fromkeys(DETS, CLFS[0])

if __name__ == "__main__":
    t0 = time.time()
    tasks = [(s, d, c) for s in ("TEST", "VAL") for d in DETS for c in CLFS]
    with Pool(8) as pool:
        results = pool.map(task, tasks, chunksize=1)
    print("tempo", round(time.time() - t0), "s;", "label_only reproduz metrics.json (|d|<=1 GT) em", sum(r["lo_reproduces"] for r in results), "/", len(results),
          "; exato em", sum(r["lo_delta_correct"] == 0 for r in results), "; deltas:", sorted({r["lo_delta_correct"] for r in results}))
    assert all(r["lo_reproduces"] for r in results), [(r["split"], r["det"], r["clf"]) for r in results if not r["lo_reproduces"]]

    # ---------------------------------------------------------------- ap_per_class.csv
    rows = []
    store = {}
    NOTE = ("recalculado pelo analyst de per_image_preds.jsonl (propostas conf>=0.001, max_det 300) com EV.candidates/greedy (class-aware, IoU .50:.95) "
            "+ ultralytics 8.4.142 ap_per_class = mesmo protocolo de eval_yolo_cascade.metrics; yolo_conf/cascade_score arredondados a 6 casas no jsonl")
    for r in results:
        split, d, c = r["split"], r["det"], r["clf"]
        base = R / f"cascade_cross_{split.lower()}" / d / c
        variants = [("cascade", "label_only", r["ap_lo"]), ("cascade", "product", r["ap_pr"])]
        if "ap_single" in r:
            variants.append(("single", "-", r["ap_single"]))
        for pipe, pol, ap in variants:
            for cls, (a50, a5095, sup) in ap.items():
                rows.append({"level": "pair" if pipe == "cascade" else "detector", "split": split, "det_family": det_fam(d), "detector": d,
                             "classifier": c if pipe == "cascade" else "-", "clf_family": clf_fam(c) if pipe == "cascade" else "-", "policy": pol,
                             "pipeline": pipe, "class": cls, "support": sup, "AP50_pct": pct(a50), "AP50_95_pct": pct(a5095),  # noqa: F405
                             "seeds": f"det={d[-1]};clf={c.split('_s')[-1]}" if pipe == "cascade" else f"det={d[-1]}", "n_seeds": 1, "job": JOB,
                             "source_path": str(base / "label_only" / "per_image_preds.jsonl") if pol != "product" else str(base / "product" / "per_image_preds.jsonl"),
                             "note": NOTE})
                store[(split, pipe, pol, d, c if pipe == "cascade" else "-", cls)] = (a50, a5095)
    pdf = pd.DataFrame(rows)
    agg = []
    for split, fam in itertools.product(("TEST", "VAL"), ("E0", "E10")):
        dets = [d for d in DETS if det_fam(d) == fam]
        for cls in SP + ["species_mean", "all7_mean"]:  # noqa: F405
            v = [store[(split, "single", "-", d, "-", cls)] for d in dets]
            a, b = [x[0] for x in v], [x[1] for x in v]
            m1, s1 = mean_std(a)  # noqa: F405
            m2, s2 = mean_std(b)  # noqa: F405
            agg.append({"level": "aggregate_mean_std", "split": split, "det_family": fam, "detector": "+".join(dets), "classifier": "-", "clf_family": "-",
                        "policy": "-", "pipeline": "single", "class": cls, "AP50_pct": round(100 * m1, 4), "AP50_pct_std": round(100 * s1, 4),
                        "AP50_95_pct": round(100 * m2, 4), "AP50_95_pct_std": round(100 * s2, 4), "seeds": "det=0,1,2", "n_seeds": 3, "job": JOB,
                        "source_path": f"{R}/cascade_cross_{split.lower()}", "note": NOTE})
            for cf in ("ce", "parts"):
                clfs = [c for c in CLFS if clf_fam(c) == cf]
                for pol in ("label_only", "product"):
                    v = [store[(split, "cascade", pol, d, c, cls)] for d in dets for c in clfs]
                    a, b = [x[0] for x in v], [x[1] for x in v]
                    vs = [store[(split, "single", "-", d, "-", cls)] for d in dets for c in clfs]
                    da = [x[0] - y[0] for x, y in zip(v, vs, strict=True)]
                    db = [x[1] - y[1] for x, y in zip(v, vs, strict=True)]
                    m1, s1 = mean_std(a)  # noqa: F405
                    m2, s2 = mean_std(b)  # noqa: F405
                    m3, s3 = mean_std(da)  # noqa: F405
                    m4, s4 = mean_std(db)  # noqa: F405
                    agg.append({"level": "aggregate_mean_std", "split": split, "det_family": fam, "detector": "+".join(dets), "classifier": "+".join(clfs),
                                "clf_family": cf, "policy": pol, "pipeline": "cascade", "class": cls, "AP50_pct": round(100 * m1, 4),
                                "AP50_pct_std": round(100 * s1, 4), "AP50_95_pct": round(100 * m2, 4), "AP50_95_pct_std": round(100 * s2, 4),
                                "delta_vs_single_AP50_pp": round(100 * m3, 4), "delta_vs_single_AP50_pp_std": round(100 * s3, 4),
                                "delta_vs_single_AP50_95_pp": round(100 * m4, 4), "delta_vs_single_AP50_95_pp_std": round(100 * s4, 4),
                                "seeds": "det=0,1,2 x clf=42,84,126", "n_seeds": 9, "job": JOB, "source_path": f"{R}/cascade_cross_{split.lower()}",
                                "note": "n=9 pares; std ddof=1; delta pareado por detector"})
    pdf = pd.concat([pdf, pd.DataFrame(agg)], ignore_index=True)
    write_csv(pdf, OUT / "ap_per_class.csv")  # noqa: F405

    # ---------------------------------------------------------------- policy_comparison.csv
    prow = []
    keys = [f"{k}_pct" for k in METRIC_KEYS]  # noqa: F405
    thr = {}
    for r in results:
        for name, pol in (("lo", "label_only"), ("pr", "product")):
            v = {f"{k}_pct": pct(r[f"thr_{name}"][k]) for k in METRIC_KEYS}  # noqa: F405
            thr[(r["split"], r["det"], r["clf"], pol)] = v
            prow.append({"level": "pair", "split": r["split"], "det_family": det_fam(r["det"]), "detector": r["det"], "classifier": r["clf"],
                         "clf_family": clf_fam(r["clf"]), "policy": pol, "seeds": f"det={r['det'][-1]};clf={r['clf'].split('_s')[-1]}", "n_seeds": 1,
                         **v, "job": JOB, "source_path": str(R / f"cascade_cross_{r['split'].lower()}" / r["det"] / r["clf"] / pol / "per_image_preds.jsonl"),
                         "note": f"P/R/F1 (MD/MV/MC/MF) com limiar {TAU} no SCORE DA POLITICA (label_only=conf YOLO; product=conf*max softmax); match IoU>=.5 class-agnostico; reproduz metrics.json para label_only"})
    for split, fam, cf in itertools.product(("TEST", "VAL"), ("E0", "E10"), ("ce", "parts")):
        dets = [d for d in DETS if det_fam(d) == fam]
        clfs = [c for c in CLFS if clf_fam(c) == cf]
        for pol in ("label_only", "product"):
            vv = [thr[(split, d, c, pol)] for d in dets for c in clfs]
            m, s = agg_dicts(vv, keys)  # noqa: F405
            prow.append({"level": "aggregate_mean_std", "split": split, "det_family": fam, "detector": "+".join(dets), "classifier": "+".join(clfs),
                         "clf_family": cf, "policy": pol, "seeds": "det=0,1,2 x clf=42,84,126", "n_seeds": 9,
                         **{k: round(m[k], 4) for k in keys}, **{k + "_std": round(s[k], 4) for k in keys}, "job": JOB,
                         "source_path": f"{R}/cascade_cross_{split.lower()}", "note": f"n=9 pares; std ddof=1; limiar {TAU} no score da politica"})
        dd = [{k: thr[(split, d, c, "product")][k] - thr[(split, d, c, "label_only")][k] for k in keys} for d in dets for c in clfs]
        m, s = agg_dicts(dd, keys)  # noqa: F405
        prow.append({"level": "aggregate_delta_product_minus_label_only", "split": split, "det_family": fam, "detector": "+".join(dets),
                     "classifier": "+".join(clfs), "clf_family": cf, "policy": "product - label_only", "seeds": "det=0,1,2 x clf=42,84,126",
                     "n_seeds": 9, **{k: round(m[k], 4) for k in keys}, **{k + "_std": round(s[k], 4) for k in keys}, "job": JOB,
                     "source_path": f"{R}/cascade_cross_{split.lower()}", "note": "p.p.; pareado por par (det,clf)"})
    sing = {(r["split"], r["det"], r["clf"]): {f"{k}_pct": pct(r["thr_single"][k]) for k in METRIC_KEYS} for r in results}  # noqa: F405
    for split, fam, cf in itertools.product(("TEST", "VAL"), ("E0", "E10"), ("ce", "parts")):
        dets = [d for d in DETS if det_fam(d) == fam]
        clfs = [c for c in CLFS if clf_fam(c) == cf]
        for pol in ("label_only", "product"):
            dd = [{k: thr[(split, d, c, pol)][k] - sing[(split, d, c)][k] for k in keys} for d in dets for c in clfs]
            m, s = agg_dicts(dd, keys)  # noqa: F405
            prow.append({"level": "aggregate_delta_cascade_minus_single", "split": split, "det_family": fam, "detector": "+".join(dets),
                         "classifier": "+".join(clfs), "clf_family": cf, "policy": pol, "seeds": "det=0,1,2 x clf=42,84,126", "n_seeds": 9,
                         **{k: round(m[k], 4) for k in keys}, **{k + "_std": round(s[k], 4) for k in keys}, "job": JOB,
                         "source_path": f"{R}/cascade_cross_{split.lower()}", "note": f"p.p.; cascata (limiar {TAU} no score da politica) menos single (conf YOLO>={TAU}), pareado por detector"})
    qdf = pd.DataFrame(prow)
    write_csv(qdf, OUT / "policy_comparison.csv")  # noqa: F405
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(qdf[qdf.level.str.startswith("aggregate_delta_cascade")][["split", "det_family", "clf_family", "policy", "recall_micro_pct", "recall_micro_pct_std", "precision_micro_pct", "f1_micro_pct", "f1_micro_pct_std", "f1_macro_pct", "acc_loc_pct"]].round(2).to_string())
    print(qdf[qdf.level.str.startswith("aggregate")][["level", "split", "det_family", "clf_family", "policy", "recall_micro_pct", "precision_micro_pct", "f1_micro_pct", "f1_macro_pct", "f1_macro_pct_std", "acc_loc_pct"]].to_string())
    a = pdf[(pdf.level == "aggregate_mean_std") & (pdf["class"].isin(["species_mean", "MD", "MV", "MC", "MF"]))]
    print(a[["split", "det_family", "clf_family", "policy", "pipeline", "class", "AP50_pct", "AP50_pct_std", "AP50_95_pct", "delta_vs_single_AP50_pp", "delta_vs_single_AP50_95_pp"]].to_string())
