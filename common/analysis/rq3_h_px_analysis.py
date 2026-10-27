#!/usr/bin/env python3
"""RQ3 cadeia H (h100n2): variantes NWD/GCD/RFLA com a loss em PIXELS vs controle G0 (cadeia G), seeds 0/1/2, TEST (CPU).

Le <RUNS>/<RUN>/test_eval_png/<RUN>/single/label_only/per_instance.csv (harness eval_cascade_matrix, conf>=0,25, IoU>=0,5).
Mesma estatistica da campanha (a) (rq3_v9_analysis.boot): delta de acuracia de especie pareado por (imagem, gt), IC95
bootstrap sobre imagens (10000, seed 0), p bilateral, Holm entre os bracos H; recall por faixa sqrt(wh) <32 / 32-48 / >=48.
Recusa runs sem DONE, sem nwd_loss_units=px (quando a loss usa NWD/GCD), com seed efetiva != declarada ou com TEST != 1210.
Uso: python3 rq3_h_px_analysis.py [RUNS_ROOT] [OUT_DIR]
Saidas: v9_h_px_{runs,summary,paired,size_bins}.csv
"""
import csv
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import types  # noqa: E402
try:
    import stats  # noqa: F401
except ImportError:  # sem pandas: mcnemar_paired nao e usado aqui
    sys.modules["stats"] = types.SimpleNamespace(mcnemar_paired=None)
from rq1_ladder_analysis import BINS, SPECIES, load_rows, rate, wr  # noqa: E402
from rq3_v9_analysis import boot  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/raid/user_marcospaulo/flydet_runs_v9"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                          "rq3_tiny_object", "results")
SEEDS = (0, 1, 2)
CTRL = "G0_E0_h100"
ARMS = {CTRL: "G0_e0_h100_s{s}", "H1_nwd_C44": "H1_nwd_px_w0.5_C44_s{s}", "H2_nwd_loss": "H2_nwdloss_px_w0.5_C44_s{s}",
        "H3_nwd_rfla": "H3_nwd_rfla_px_w0.5_C44_s{s}", "H4_nwd_C22": "H4_nwd_px_w0.5_C22_s{s}",
        "H5_nwd_C88": "H5_nwd_px_w0.5_C88_s{s}", "H6_nwd_pure_w1": "H6_nwd_px_w1.0_C44_s{s}", "H7_gcd": "H7_gcd_px_w0.5_s{s}"}


def check_run(work, run, s):
    """Gates de rastreabilidade; devolve (best_epoch, n_epochs)."""
    assert os.path.exists(os.path.join(work, "DONE")), f"{run}: sem DONE"
    cfg = json.load(open(os.path.join(work, "nwd_config.json")))
    if cfg.get("nwd") and cfg.get("nwd_mode") in ("nwd", "gcd") and cfg.get("nwd_scope") != "assigner":
        assert cfg.get("nwd_loss_units") == "px", f"{run}: loss fora de px {cfg}"
    assert cfg.get("seed") == s, f"{run}: seed {cfg.get('seed')} != {s}"
    args = open(os.path.join(work, "runs", "yolo26m.pt", "args.yaml")).read()
    assert re.search(rf"^seed: {s}$", args, re.M), f"{run}: args.yaml sem seed {s}"
    with open(os.path.join(work, "runs", "yolo26m.pt", "results.csv")) as f:
        fit = [float(r["metrics/mAP50-95(B)"]) for r in csv.DictReader(f)]  # fitness do ultralytics 8.4.142 (best.pt)
    fin = [x for x in fit if x == x]  # nan (divergencia) nao e best
    return fit.index(max(fin)) + 1, len(fit)


def main():
    os.makedirs(OUT, exist_ok=True)
    data, ep = {}, {}
    for arm, pat in ARMS.items():
        for s in SEEDS:
            run = pat.format(s=s)
            work = os.path.join(ROOT, run)
            d = os.path.join(work, "test_eval_png", run, "single", "label_only")
            if not os.path.exists(os.path.join(d, "per_instance.csv")):
                continue
            ep[(arm, s)] = check_run(work, run, s)
            data[(arm, s)] = load_rows(d)
            assert len(data[(arm, s)]) == 1210, f"{run}: {len(data[(arm, s)])} GT de especie"
    runs_rows, sz, acc = [], [], {}
    for (arm, s), rows in sorted(data.items()):
        rec = {c: rate([r for r in rows if r["gt_class"] == c]) for c in SPECIES}
        acc[(arm, s)] = rate(rows)
        runs_rows.append(["TEST", "h100n2", arm, s, len(rows), f"{acc[(arm, s)]:.4f}", f"{sum(rec.values()) / 4:.4f}"]
                         + [f"{rec[c]:.4f}" for c in SPECIES] + list(ep[(arm, s)]))
        for name, lo, hi in BINS:
            sub = [r for r in rows if lo <= r["size"] < hi]
            sz.append(["TEST", arm, s, name, len(sub), f"{rate(sub):.4f}"])
    wr(os.path.join(OUT, "v9_h_px_runs.csv"), ["split", "hw", "arm", "seed", "n_gt_especie", "acc_especie", "recall_macro"]
       + [f"recall_{c}" for c in SPECIES] + ["best_epoch", "n_epochs"], runs_rows)
    wr(os.path.join(OUT, "v9_h_px_size_bins.csv"), ["split", "arm", "seed", "bin_sqrtwh_px", "n_gt", "recall_especie"], sz)

    summ, pr = [], []
    for arm in ARMS:
        ss = [s for s in SEEDS if (arm, s) in data]
        if not ss:
            continue
        a = [acc[(arm, s)] for s in ss]
        bins = []
        for name, lo, hi in BINS:
            v = [rate([r for r in data[(arm, s)] if lo <= r["size"] < hi]) for s in ss]
            bins.append(f"{st.mean(v):.4f}")
        summ.append(["TEST", arm, len(ss), "multi-seed" if len(ss) > 1 else "single-seed", f"{st.mean(a):.4f}",
                     f"{st.stdev(a):.4f}" if len(a) > 1 else ""] + bins)
        if arm == CTRL:
            continue
        ps = [s for s in ss if (CTRL, s) in data]
        if ps:
            dl, lo, hi, _, _, p = boot({s: data[(arm, s)] for s in ps}, {s: data[(CTRL, s)] for s in ps})
            per = "|".join(f"{acc[(arm, s)] - acc[(CTRL, s)]:+.4f}" for s in ps)
            pr.append([arm, len(ps), dl, lo, hi, p, per])
    order = sorted(range(len(pr)), key=lambda i: pr[i][5])  # Holm entre os bracos H
    m, run_max, holm = len(pr), 0.0, {}
    for rank, i in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * pr[i][5]))
        holm[i] = run_max
    wr(os.path.join(OUT, "v9_h_px_summary.csv"), ["split", "arm", "n_seeds", "regime", "acc_mean", "acc_std"]
       + [f"recall_{b[0]}" for b in BINS], summ)
    wr(os.path.join(OUT, "v9_h_px_paired.csv"),
       ["split", "arm_a", "arm_b", "n_seeds", "delta_acc_a_menos_b", "ic95_lo", "ic95_hi", "p_boot", "p_holm", "delta_por_seed"],
       [["TEST", r[0], CTRL, r[1], f"{r[2]:+.4f}", f"{r[3]:+.4f}", f"{r[4]:+.4f}", f"{r[5]:.4f}", f"{holm[i]:.4f}", r[6]]
        for i, r in enumerate(pr)])
    for r in summ:
        print(r)
    for i, r in enumerate(pr):
        print(r[0], f"{r[2] * 100:+.2f} [{r[3] * 100:+.2f}, {r[4] * 100:+.2f}] p={r[5]:.4f} holm={holm[i]:.4f}", r[6])


if __name__ == "__main__":
    main()
