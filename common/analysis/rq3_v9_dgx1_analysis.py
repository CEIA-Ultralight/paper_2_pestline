#!/usr/bin/env python3
"""RQ3 no DS-F2_v9.0.5 (campanha dgx1/V100) — E0 x NWD (CPU).

Variantes: E0 (s0-2); E10 = NWD w=0,5, C=44, scope=both com a loss em unidade de grid (legado, s0-2);
E15 = idem com a loss em px (FLYDET_NWD_LOSS_UNITS=px, s0-2); E16 = NWD puro w=1,0 em px (s0, exploratorio).
Cada variante e comparada com o E0 pareado por seed (PAPER_FACTS §12/§12.1).

Le, por run, test_eval/<RUN>/single/label_only/{metrics.json,per_instance.csv} (harness eval_cascade_matrix:
conf>=0,25, IoU>=0,5, matching class-agnostico) e o log do treino (<LOGS>/<RUN>.log: validacao final do best.pt
no VAL, best epoch, epocas, horas). Campanha separada da B200 (V100, torch 2.6+cu124): nao misturar com PAPER_FACTS §9.

Uso: python rq3_v9_dgx1_analysis.py <RUNS_ROOT> <LOGS_DIR> <OUT_DIR>
Saidas (CSV em OUT_DIR): v9_dgx1_nwd_runs.csv (run x metricas), v9_dgx1_nwd_summary.csv (media/desvio por variante
e delta pareado por seed vs E0), v9_dgx1_nwd_paired.csv (McNemar exato por seed + IC95 bootstrap por imagem, vs E0),
v9_dgx1_nwd_size_bins.csv (recall de especie por tamanho, mesmos bins da RQ1/§9.1).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common/
from stats import mcnemar_paired  # noqa: E402

SPECIES = ["MD", "MV", "MC", "MF"]
# variante -> (padrao do run, seeds, rotulo curto usado nos deltas)
VARIANTS = {"E0": ("E0_s{s}", [0, 1, 2], "E0"),
            "E10_nwd": ("E10_nwd_w0.5_s{s}", [0, 1, 2], "E10"),
            "E15_nwd_px": ("E15_nwdfix_w0.5_s{s}", [0, 1, 2], "E15"),
            "E16_nwdpure_px": ("E16_nwdpure_w1.0_px_s{s}", [0], "E16")}
BINS = [("<32", 0, 32), ("32-48", 32, 48), (">=48", 48, 1e9)]
NBOOT = 10000
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def eval_dir(root, run):
    return os.path.join(root, run, "test_eval", run, "single", "label_only")


def parse_log(path):
    txt = ANSI.sub("", open(path, encoding="utf-8", errors="replace").read())
    out = {}
    m = re.search(r"Best results observed at epoch (\d+)", txt)
    out["best_epoch"] = int(m.group(1)) if m else ""
    m = re.search(r"(\d+) epochs completed in ([\d.]+) hours", txt)
    out["epochs"], out["train_h"] = (int(m.group(1)), float(m.group(2))) if m else ("", "")
    tail = txt[txt.rfind("Validating"):]
    m = re.search(r"^\s+all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", tail, re.M)
    if m:
        out.update(val_imgs=int(m.group(1)), val_gt=int(m.group(2)), val_P=float(m.group(3)), val_R=float(m.group(4)),
                   val_mAP50=float(m.group(5)), val_mAP50_95=float(m.group(6)))
    return out


def load_rows(d):
    with open(os.path.join(d, "per_instance.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["gt_class"] in SPECIES]
    for r in rows:
        r["ok"] = r["correct_single"] == "True"
        r["size"] = float(r["gt_size"])
    return rows


def run_metrics(d):
    m = json.load(open(os.path.join(d, "metrics.json")))
    s = m["single"]
    pc = s["per_class"]
    tp = sum(pc[c]["tp"] for c in SPECIES)
    pred = sum(pc[c]["predicted"] for c in SPECIES)
    sup = sum(pc[c]["support"] for c in SPECIES)
    mp, mr = tp / pred, tp / sup
    out = {"images": m["images"], "gt_objects": m["gt_objects"], "species_gt": s["species_support"],
           "macro_f1": s["species_macro_f1"], "macro_P": s["species_macro_precision"], "macro_R": s["species_macro_recall"],
           "micro_P": mp, "micro_R": mr, "micro_F1": 2 * mp * mr / (mp + mr),
           "acc_all_gt": s["species_accuracy_all_gt"], "acc_loc": s["species_accuracy_conditional_matched"],
           "det_recall": s["species_detection_recall"], "det_ms": 1000 * m["latency"]["detector"]["mean_s"]}
    for c in SPECIES:
        for k in ("precision", "recall", "f1"):
            out[f"{c}_{k[0].upper()}"] = pc[c][k]
    return out


def wr(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else x


def boot_delta(a_rows, b_rows, nboot=NBOOT):
    """Delta acuracia de especie (a-b), seeds agrupadas, IC95 percentil por bootstrap sobre imagens (seed 0)."""
    per_img = defaultdict(lambda: [0, 0, 0])
    for seed in a_rows:
        bmap = {(r["image"], r["gt_index"]): r["ok"] for r in b_rows[seed]}
        for r in a_rows[seed]:
            x = per_img[r["image"]]
            x[0] += r["ok"]
            x[1] += bmap[(r["image"], r["gt_index"])]
            x[2] += 1
    imgs = list(per_img.values())
    delta = (sum(x[0] for x in imgs) - sum(x[1] for x in imgs)) / sum(x[2] for x in imgs)
    rng = random.Random(0)
    ds = []
    for _ in range(nboot):
        smp = rng.choices(imgs, k=len(imgs))
        ds.append((sum(x[0] for x in smp) - sum(x[1] for x in smp)) / sum(x[2] for x in smp))
    ds.sort()
    return delta, ds[int(0.025 * (nboot - 1))], ds[math.ceil(0.975 * (nboot - 1))], len(imgs)


def main():
    root, logs, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    met, rows = {}, {}
    for var, (pat, seeds, _) in VARIANTS.items():
        for s in seeds:
            run = pat.format(s=s)
            d = eval_dir(root, run)
            met[(var, s)] = {"run": run, **parse_log(os.path.join(logs, f"{run}.log")), **run_metrics(d)}
            rows[(var, s)] = load_rows(d)
            assert met[(var, s)]["gt_objects"] == 3494 and met[(var, s)]["species_gt"] == 1210, run

    keys = ["run", "images", "gt_objects", "species_gt", "best_epoch", "epochs", "train_h", "val_P", "val_R", "val_mAP50",
            "val_mAP50_95", "macro_f1", "macro_P", "macro_R", "micro_P", "micro_R", "micro_F1", "acc_all_gt", "acc_loc",
            "det_recall", "det_ms"] + [f"{c}_{k}" for c in SPECIES for k in "PRF"]
    wr(os.path.join(out, "v9_dgx1_nwd_runs.csv"), ["variant", "seed", "split_test", "split_val", "campanha"] + keys,
       [[v, s, "TEST v9.0.5", "VAL v9.0.5 (best.pt; nao held-out)", "dgx1 V100 torch2.6+cu124"]
        + [fmt(met[(v, s)].get(k, "")) for k in keys] for (v, s) in sorted(met)])

    num = [k for k in keys if k not in ("run", "images", "gt_objects", "species_gt")]
    def std(xs):  # E16 tem 1 seed: sem desvio
        return f"{st.stdev(xs):.4f}" if len(xs) > 1 else ""

    summ = []
    for v, (_, seeds, _) in VARIANTS.items():
        for k in num:
            xs = [met[(v, s)][k] for s in seeds]
            summ.append([v, k, len(xs), f"{st.mean(xs):.4f}", std(xs), "|".join(fmt(float(x)) for x in xs)])
    for v, (_, seeds, short) in VARIANTS.items():
        if v == "E0":
            continue
        for k in num:
            ds = [met[(v, s)][k] - met[("E0", s)][k] for s in seeds]
            summ.append([f"delta_{short}_menos_E0", k, len(ds), f"{st.mean(ds):+.4f}", std(ds),
                         "|".join(f"{x:+.4f}" for x in ds)])
    wr(os.path.join(out, "v9_dgx1_nwd_summary.csv"), ["variant", "metric", "n_seeds", "mean", "std_ddof1", "por_seed"], summ)

    pr = []
    for v, (_, seeds, short) in VARIANTS.items():
        if v == "E0":
            continue
        for s in seeds:
            a, b = rows[(v, s)], rows[("E0", s)]
            bm = {(r["image"], r["gt_index"]): r["ok"] for r in b}
            m = mcnemar_paired([r["ok"] for r in a], [bm[(r["image"], r["gt_index"])] for r in a])
            acc_a, acc_b = sum(r["ok"] for r in a) / len(a), sum(r["ok"] for r in b) / len(b)
            pr.append([short, f"s{s}", len(a), f"{acc_a:.4f}", f"{acc_b:.4f}", f"{acc_a - acc_b:+.4f}", m["b"], m["c"],
                       f"{m['p_value']:.3g}", "", "", ""])
        dlt, lo, hi, nimg = boot_delta({s: rows[(v, s)] for s in seeds}, {s: rows[("E0", s)] for s in seeds})
        pr.append([short, "seeds_agrupadas", len(seeds) * 1210, "", "", f"{dlt:+.4f}", "", "", "", f"{lo:+.4f}",
                   f"{hi:+.4f}", nimg])
    wr(os.path.join(out, "v9_dgx1_nwd_paired.csv"),
       ["variant", "seed", "n_gt_especie", "acc_variant", "acc_E0", "delta_acc", "mcnemar_b_variantcerto_E0errado",
        "mcnemar_c_varianterrado_E0certo", "mcnemar_p_exato", "ic95_boot_img_lo", "ic95_boot_img_hi", "n_imagens"], pr)

    sz = []
    for (v, s), rs in sorted(rows.items()):
        for name, lo_, hi_ in BINS:
            sub = [r for r in rs if lo_ <= r["size"] < hi_]
            sz.append([v, s, name, len(sub), f"{sum(r['ok'] for r in sub) / len(sub):.4f}"])
    wr(os.path.join(out, "v9_dgx1_nwd_size_bins.csv"), ["variant", "seed", "bin_sqrtwh_px", "n_gt", "recall_especie"], sz)
    print("ok", len(met), "runs")


if __name__ == "__main__":
    main()
