#!/usr/bin/env python3
"""RQ1 — analise da escada de dados (CPU). Le per_instance.csv / per_image_preds.jsonl dos runs.

Bracos: L0, L1, L2, L2ORG (RUNS_RQ1/RQ1_<ARM>_s<seed>) e L3 = E0_s<seed> (RUNS_V9). TEST fixo = DS-F2_v9.0.5.
Metricas por especie (MD, MV, MC, MF): recall = deteccao casada (IoU>=.5, conf>=.25) E classe correta, sobre todos
os GT da especie (igual ao `single/<C>/recall` do harness). Tudo TEST, receita E0, rotulado single/multi-seed.

Uso: python rq1_ladder_analysis.py <RUNS_RQ1> <RUNS_V9> <OUT_DIR> [seeds=0,1,2]
Saidas (CSV em OUT_DIR): ablation_ladder.csv (arm x seed), ablation_ladder_summary.csv (media/desvio),
  ladder_size_bins.csv, ladder_crowding.csv, ladder_insonly_fp.csv, ladder_confusion.csv, ladder_paired.csv.
Binos e limiares fixados antes dos runs (docs/DECISIONS.md 2026-10-13): tamanho sqrt(wh) <32 / 32-48 / >=48 px;
placa densa = >=30 GT de especie; bootstrap 10000 sobre imagens (seed 0); McNemar exato/qui2 por seed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # common/
from stats import mcnemar_paired  # noqa: E402

SPECIES = ["MD", "MV", "MC", "MF"]
ARMS = ["L0", "L1", "L2", "L2ORG", "L3"]
BINS = [("<32", 0, 32), ("32-48", 32, 48), (">=48", 48, 1e9)]
DENSE = 30
CONF = 0.25
NBOOT = 10000


def run_dir(arm, seed, rq1, v9):
    if arm == "L3":
        return os.path.join(v9, f"E0_s{seed}", "test_eval", f"E0_s{seed}", "single", "label_only")
    return os.path.join(rq1, f"RQ1_{arm}_s{seed}", "test_eval", f"RQ1_{arm}_s{seed}", "single", "label_only")


def load_rows(d):
    with open(os.path.join(d, "per_instance.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["gt_class"] in SPECIES]
    for r in rows:
        r["ok"] = r["correct_single"] == "True"
        r["matched_b"] = r["matched"] == "True"
        r["size"] = float(r["gt_size"])
    return rows


def rate(rows):
    return sum(r["ok"] for r in rows) / len(rows) if rows else float("nan")


def wr(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def boot_delta(a_rows, b_rows, nboot=NBOOT):
    """Delta acuracia (a-b), IC95 bootstrap sobre imagens; a_rows/b_rows = {seed: rows} pareados por (imagem,gt_index)."""
    per_img = defaultdict(lambda: [0, 0, 0])  # corretos_a, corretos_b, n
    for seed in a_rows:
        bmap = {(r["image"], r["gt_index"]): r["ok"] for r in b_rows[seed]}
        for r in a_rows[seed]:
            k = r["image"]
            per_img[k][0] += r["ok"]
            per_img[k][1] += bmap[(r["image"], r["gt_index"])]
            per_img[k][2] += 1
    imgs = list(per_img.values())
    tot = sum(x[2] for x in imgs)
    delta = (sum(x[0] for x in imgs) - sum(x[1] for x in imgs)) / tot
    rng = random.Random(0)
    ds = []
    for _ in range(nboot):
        s = rng.choices(imgs, k=len(imgs))
        n = sum(x[2] for x in s)
        ds.append((sum(x[0] for x in s) - sum(x[1] for x in s)) / n)
    ds.sort()
    return delta, ds[int(0.025 * (nboot - 1))], ds[math.ceil(0.975 * (nboot - 1))]


def main():
    rq1, v9, out = sys.argv[1:4]
    seeds = [int(s) for s in (sys.argv[4] if len(sys.argv) > 4 else "0,1,2").split(",")]
    os.makedirs(out, exist_ok=True)
    data, avail = {}, {}
    for arm in ARMS:
        for s in seeds:
            d = run_dir(arm, s, rq1, v9)
            if os.path.exists(os.path.join(d, "per_instance.csv")):
                data[(arm, s)] = (d, load_rows(d))
                avail.setdefault(arm, []).append(s)
    # imagens densas / so-INS (do GT do proprio per_image_preds de qualquer run)
    any_d = next(iter(data.values()))[0]
    n_sp, has_sp = Counter(), {}
    with open(os.path.join(any_d, "per_image_preds.jsonl")) as f:
        for line in f:
            j = json.loads(line)
            n = sum(g["class"] in SPECIES for g in j["gt"])
            n_sp[j["image"]] = n
    dense = {i for i, n in n_sp.items() if n >= DENSE}
    insonly = {i for i, n in n_sp.items() if n == 0}

    # --- por arm x seed
    rows_out, by_arm = [], defaultdict(list)
    for (arm, s), (d, rows) in sorted(data.items()):
        rec = {c: rate([r for r in rows if r["gt_class"] == c]) for c in SPECIES}
        det = sum(r["matched_b"] for r in rows) / len(rows)
        acc = rate(rows)
        mac = sum(rec.values()) / len(SPECIES)
        rows_out.append([arm, s, len(rows), f"{acc:.4f}", f"{mac:.4f}", f"{det:.4f}"] + [f"{rec[c]:.4f}" for c in SPECIES])
        by_arm[arm].append((acc, mac, det, [rec[c] for c in SPECIES]))
    wr(os.path.join(out, "ablation_ladder.csv"),
       ["arm", "seed", "n_gt_especie", "acc_especie", "recall_macro", "recall_deteccao"] + [f"recall_{c}" for c in SPECIES], rows_out)
    summ = []
    for arm in ARMS:
        if arm not in by_arm:
            continue
        v = by_arm[arm]
        def ms(xs):
            return f"{st.mean(xs):.4f}", f"{(st.stdev(xs) if len(xs) > 1 else 0.0):.4f}"
        cols = [ms([x[0] for x in v]), ms([x[1] for x in v]), ms([x[2] for x in v])] + [ms([x[3][i] for x in v]) for i in range(4)]
        summ.append([arm, len(v)] + [t for c in cols for t in c])
    hdr = ["arm", "n_seeds"] + [f"{m}_{k}" for m in ["acc", "recall_macro", "recall_det"] + [f"recall_{c}" for c in SPECIES] for k in ("mean", "std")]
    wr(os.path.join(out, "ablation_ladder_summary.csv"), hdr, summ)

    # --- tamanho / densidade
    sz, cr = [], []
    for (arm, s), (d, rows) in sorted(data.items()):
        for name, lo, hi in BINS:
            sub = [r for r in rows if lo <= r["size"] < hi]
            sz.append([arm, s, name, len(sub), f"{rate(sub):.4f}"])
        for name, sel in (("densa(>=30)", lambda r: r["image"] in dense), ("resto", lambda r: r["image"] not in dense)):
            sub = [r for r in rows if sel(r)]
            cr.append([arm, s, name, len(sub), f"{rate(sub):.4f}"])
    wr(os.path.join(out, "ladder_size_bins.csv"), ["arm", "seed", "bin_sqrtwh_px", "n_gt", "recall_especie"], sz)
    wr(os.path.join(out, "ladder_crowding.csv"), ["arm", "seed", "placa", "n_gt", "recall_especie"], cr)

    # --- FP de especie em placas so-INS (conf>=CONF, classe de especie)
    fp = []
    for (arm, s), (d, _) in sorted(data.items()):
        n = 0
        with open(os.path.join(d, "per_image_preds.jsonl")) as f:
            for line in f:
                j = json.loads(line)
                if j["image"] in insonly:
                    n += sum(p["yolo_class"] in SPECIES and p["yolo_conf"] >= CONF for p in j["proposals"])
        fp.append([arm, s, len(insonly), n])
    wr(os.path.join(out, "ladder_insonly_fp.csv"), ["arm", "seed", "n_placas_so_INS", "fp_especie_conf>=0.25"], fp)

    # --- confusao entre localizados (matched), pooled seeds (H2)
    conf = []
    for arm in ARMS:
        cnt = Counter()
        for s in avail.get(arm, []):
            for r in data[(arm, s)][1]:
                if r["matched_b"]:
                    cnt[(r["gt_class"], r["pred_class_single"])] += 1
        for g in SPECIES:
            tot = sum(v for (a, _), v in cnt.items() if a == g)
            for p in SPECIES + ["outra"]:
                v = cnt[(g, p)] if p != "outra" else tot - sum(cnt[(g, q)] for q in SPECIES)
                conf.append([arm, g, p, v, f"{v / tot:.4f}" if tot else ""])
    wr(os.path.join(out, "ladder_confusion.csv"), ["arm", "gt", "pred", "n", "frac_dos_localizados"], conf)

    # --- pareado vs L3 (mesmas seeds disponiveis nos dois)
    pr = []
    for arm in ARMS[:-1]:
        ss = [s for s in avail.get(arm, []) if s in avail.get("L3", [])]
        if not ss:
            continue
        a = {s: data[(arm, s)][1] for s in ss}
        b = {s: data[("L3", s)][1] for s in ss}
        dlt, lo, hi = boot_delta(a, b)
        mc = []
        for s in ss:
            bm = {(r["image"], r["gt_index"]): r["ok"] for r in b[s]}
            m = mcnemar_paired([r["ok"] for r in a[s]], [bm[(r["image"], r["gt_index"])] for r in a[s]])
            mc.append(f"s{s}:b={m['b']},c={m['c']},p={m['p_value']:.2g}")
        per = [rate(a[s]) - rate(b[s]) for s in ss]
        pr.append([arm, "L3", len(ss), f"{dlt:+.4f}", f"{lo:+.4f}", f"{hi:+.4f}", f"{(hi - lo) / 2:.4f}",
                   "|".join(f"{x:+.4f}" for x in per), "; ".join(mc)])
    wr(os.path.join(out, "ladder_paired.csv"),
       ["arm_a", "arm_b", "n_seeds", "delta_acc_a_menos_b", "ic95_lo", "ic95_hi", "meia_largura", "delta_por_seed", "mcnemar_por_seed"], pr)
    print({k: v for k, v in avail.items()}, "densas", len(dense), "so_INS", len(insonly))


if __name__ == "__main__":
    main()
