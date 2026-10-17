#!/usr/bin/env python3
"""RQ3 (v9.0.5) — analise multi-seed dos treinos tiny-object (CPU). TEST fixo, receita E0, seeds pareadas.

Bracos (RUNS/<RUN>/test_eval/<RUN>/single/label_only/per_instance.csv):
  E0 baseline | E10 NWD | E11 GCD | E14 NWD+RFLA | E1 P2 | E20 WIoU | E21 2560 | E15/E16 NWD loss/assigner | E7 yolo26l | E22 YOLO11m
Metrica primaria: acuracia por especie (MD,MV,MC,MF) = deteccao casada (IoU>=.5, conf>=.25) E classe correta, sobre todos os GT
de especie; recall por especie e por faixa de tamanho sqrt(wh) <32 / 32-48 / >=48 px (DECISIONS 2026-10-13).
Pareado vs E0: delta de acuracia, IC95 bootstrap sobre imagens (10000, seed 0), p bootstrap bilateral, Holm entre os bracos,
TOST com margem +-1 pp (equivalente se o IC90 do delta cai em [-0.01, +0.01]).

Uso: python rq3_v9_analysis.py <RUNS_V9> <OUT_DIR> [seeds=0,1,2]
Saidas: rq3_v9_runs.csv, rq3_v9_summary.csv, rq3_v9_size_bins.csv, rq3_v9_paired.csv (+ split=TEST, n_seeds em todas).
"""
from __future__ import annotations

import csv
import math
import os
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_ladder_analysis import BINS, SPECIES, load_rows, rate, wr  # noqa: E402

ARMS = [("E0", "E0_s{s}"), ("E10_nwd", "E10_nwd_w0.5_s{s}"), ("E11_gcd", "E11_gcd_w0.5_s{s}"),
        ("E14_nwd_rfla", "E14_nwd_rfla_s{s}"), ("E1_p2", "E1_p2_s{s}"), ("E20_wiou", "E20_wiou_s{s}"),
        ("E21_2560", "E21_imgsz2560_s{s}"), ("E15_nwd_loss", "E15_nwd_loss_s{s}"),
        ("E16_nwd_assigner", "E16_nwd_assigner_s{s}"), ("E7_yolo26l", "E7_yolo26l_s{s}"),
        ("E22_yolo11m", "E22_yolo11m_s{s}")]
NBOOT, MARGIN = 10000, 0.01


def run_dir(runs, tmpl, s):
    run = tmpl.format(s=s)
    return os.path.join(runs, run, "test_eval", run, "single", "label_only")


def boot(a_rows, b_rows, nboot=NBOOT):
    """Delta de acuracia a-b pareado por (imagem, gt_index), reamostrando imagens (todas as seeds juntas)."""
    per_img = defaultdict(lambda: [0, 0, 0])
    for s in a_rows:
        bm = {(r["image"], r["gt_index"]): r["ok"] for r in b_rows[s]}
        for r in a_rows[s]:
            x = per_img[r["image"]]
            x[0] += r["ok"]
            x[1] += bm[(r["image"], r["gt_index"])]
            x[2] += 1
    imgs = list(per_img.values())
    tot = sum(x[2] for x in imgs)
    delta = sum(x[0] - x[1] for x in imgs) / tot
    rng, ds = random.Random(0), []
    for _ in range(nboot):
        smp = rng.choices(imgs, k=len(imgs))
        ds.append(sum(x[0] - x[1] for x in smp) / sum(x[2] for x in smp))
    ds.sort()
    q = lambda p: ds[min(nboot - 1, max(0, int(round(p * (nboot - 1)))))]  # noqa: E731
    p_two = min(1.0, 2 * min(sum(d <= 0 for d in ds), sum(d >= 0 for d in ds)) / nboot) or 1.0 / nboot
    return delta, q(0.025), q(0.975), q(0.05), q(0.95), p_two


def main():
    runs, out = sys.argv[1:3]
    seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")]
    os.makedirs(out, exist_ok=True)
    data = {}
    for arm, tmpl in ARMS:
        for s in seeds:
            d = run_dir(runs, tmpl, s)
            if os.path.exists(os.path.join(d, "per_instance.csv")):
                data[(arm, s)] = load_rows(d)
    avail = defaultdict(list)
    for (arm, s) in data:
        avail[arm].append(s)

    runs_rows, by_arm, sz = [], defaultdict(list), []
    for (arm, s), rows in sorted(data.items()):
        rec = {c: rate([r for r in rows if r["gt_class"] == c]) for c in SPECIES}
        acc, mac = rate(rows), sum(rec.values()) / len(SPECIES)
        runs_rows.append(["TEST", arm, s, len(rows), f"{acc:.4f}", f"{mac:.4f}"] + [f"{rec[c]:.4f}" for c in SPECIES])
        by_arm[arm].append((acc, mac, [rec[c] for c in SPECIES]))
        for name, lo, hi in BINS:
            sub = [r for r in rows if lo <= r["size"] < hi]
            sz.append(["TEST", arm, s, name, len(sub), f"{rate(sub):.4f}"])
    wr(os.path.join(out, "rq3_v9_runs.csv"), ["split", "arm", "seed", "n_gt_especie", "acc_especie", "recall_macro"]
       + [f"recall_{c}" for c in SPECIES], runs_rows)
    wr(os.path.join(out, "rq3_v9_size_bins.csv"), ["split", "arm", "seed", "bin_sqrtwh_px", "n_gt", "recall_especie"], sz)

    ms = lambda xs: (f"{st.mean(xs):.4f}", f"{(st.stdev(xs) if len(xs) > 1 else 0.0):.4f}")  # noqa: E731
    summ = []
    for arm, _ in ARMS:
        v = by_arm.get(arm)
        if v:
            cols = [ms([x[0] for x in v]), ms([x[1] for x in v])] + [ms([x[2][i] for x in v]) for i in range(4)]
            summ.append(["TEST", arm, len(v), "multi-seed" if len(v) > 1 else "single-seed"] + [t for c in cols for t in c])
    wr(os.path.join(out, "rq3_v9_summary.csv"),
       ["split", "arm", "n_seeds", "regime"] + [f"{m}_{k}" for m in ["acc", "recall_macro"] + [f"recall_{c}" for c in SPECIES]
                                               for k in ("mean", "std")], summ)

    pr = []
    for arm, _ in ARMS[1:]:
        ss = [s for s in avail.get(arm, []) if s in avail.get("E0", [])]
        if not ss:
            continue
        a, b = {s: data[(arm, s)] for s in ss}, {s: data[("E0", s)] for s in ss}
        dlt, lo, hi, lo90, hi90, p = boot(a, b)
        per = [rate(a[s]) - rate(b[s]) for s in ss]
        equiv = (lo90 > -MARGIN) and (hi90 < MARGIN)
        pr.append([arm, len(ss), dlt, lo, hi, p, "|".join(f"{x:+.4f}" for x in per), equiv])
    order = sorted(range(len(pr)), key=lambda i: pr[i][5])  # Holm
    m, run_max, holm = len(pr), 0.0, {}
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * pr[i][5]))
        holm[i] = run_max
    rows_p = [["TEST", r[0], "E0", r[1], f"{r[2]:+.4f}", f"{r[3]:+.4f}", f"{r[4]:+.4f}", f"{r[5]:.4f}", f"{holm[i]:.4f}", r[6],
               f"IC90 em +-{MARGIN * 100:.0f}pp: {'sim' if r[7] else 'nao'}"] for i, r in enumerate(pr)]
    wr(os.path.join(out, "rq3_v9_paired.csv"),
       ["split", "arm_a", "arm_b", "n_seeds", "delta_acc_a_menos_b", "ic95_lo", "ic95_hi", "p_boot", "p_holm", "delta_por_seed",
        "tost_equivalencia"], rows_p)
    print(dict(avail))


if __name__ == "__main__":
    main()
