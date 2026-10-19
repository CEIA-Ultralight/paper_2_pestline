#!/usr/bin/env python3
"""RQ3 (ovx) — consolida SAHI/TTA/WBF dos 3 E0 (TEST, CPU, stdlib).
Uso: python rq3_ovx_infer_csv.py <RUNS_ROOT> <OUT_CSV>
Protocolos distintos: SAHI = sahi_inference (7 classes, conf 0, IoU .5, imgsz 512 tile 512 ov .25); TTA/WBF = eval_wbf_tta (Ultralytics, imgsz 1920, conf .001).
"""
import csv, json, os, statistics as st, sys

runs, out = sys.argv[1], sys.argv[2]
rows = []
for s in (0, 1, 2):
    m = json.load(open(f"{runs}/sahi_E0_s{s}/test/metrics/metrics.json"))
    for mode in ("plain", "sliced"):
        d = m[mode]
        rows.append(dict(split="TEST", method=f"SAHI_{mode}", seed=s, map50=d["map50"], map50_95=d["map50_95"],
                         precision=d["precision"], recall=d["recall"], tp=d["tp"], fp=d["fp"], fn=d["fn"],
                         protocol="sahi_inference conf0 IoU.5 imgsz512"))
    t = json.load(open(f"{runs}/ttawbf_E0_s{s}/summary.json"))
    for k, name in ((f"vanilla_E0_s{s}", "vanilla_1920"), (f"tta_E0_s{s}", "TTA_aug_1920")):
        rows.append(dict(split="TEST", method=name, seed=s, map50=t[k]["map50"], map50_95=t[k]["map5095"],
                         protocol="eval_wbf_tta conf.001 imgsz1920"))
e = json.load(open(f"{runs}/ttawbf_E0_ensemble/summary.json"))["wbf_ensemble"]
rows.append(dict(split="TEST", method="WBF_ensemble_3E0", seed="0+1+2", map50=e["map50"], protocol="eval_wbf_tta --wbf-only"))
cols = ["split", "method", "seed", "map50", "map50_95", "precision", "recall", "tp", "fp", "fn", "protocol"]
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
for name in ("SAHI_plain", "SAHI_sliced", "vanilla_1920", "TTA_aug_1920"):
    for key in ("map50", "map50_95", "recall"):
        v = [r[key] for r in rows if r["method"] == name and key in r]
        if v: print(name, key, f"{100*st.mean(v):.2f} ± {100*st.stdev(v):.2f} (n=3)")
print("WBF_ensemble map50", e["map50"])
