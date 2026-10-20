#!/usr/bin/env python3
"""RQ3 cadeia G (h100n2): E0 vs bracos NWD ajustados (C/peso), seeds 0/1/2, TEST (CPU).

Le test_eval_png/<RUN>/single/label_only/{metrics.json,per_instance.csv} (harness eval_cascade_matrix, conf>=0,25, IoU>=0,5).
Usa so as seeds disponiveis nos dois lados (pareado por seed). Reutiliza boot_delta/mcnemar do gerador dgx1.
Uso: python3 rq3_g_tuned_analysis.py [RUNS_ROOT] [OUT_DIR]
Saidas: rq3_tiny_object/results/v9_g_tuned_runs.csv e v9_g_tuned_summary.csv
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import types  # noqa: E402
try:
    import stats  # noqa: F401
except ImportError:  # sem pandas: so boot_delta e necessario aqui
    sys.modules['stats'] = types.SimpleNamespace(mcnemar_paired=None)
import rq3_v9_dgx1_analysis as D  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/raid/user_marcospaulo/flydet_runs_v9"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "rq3_tiny_object", "results")
ARMS = {"G0_E0_h100": "G0_e0_h100_s{s}", "G1_nwd_w0.5_C22": "G1_nwd_w0.5_C22_s{s}",
        "G2_nwd_w1.0_C44": "G2_nwd_w1.0_C44_s{s}", "G3_nwd_w0.5_C88": "G3_nwd_w0.5_C88_s{s}"}


def main():
    met, rows = {}, {}
    for arm, pat in ARMS.items():
        for s in D.SEEDS:
            run = pat.format(s=s)
            d = os.path.join(ROOT, run, "test_eval_png", run, "single", "label_only")
            if not os.path.exists(os.path.join(ROOT, run, "DONE")) or not os.path.exists(os.path.join(d, "per_instance.csv")):
                continue
            met[(arm, s)] = {"run": run, **D.run_metrics(d)}
            rows[(arm, s)] = D.load_rows(d)
            assert met[(arm, s)]["species_gt"] == 1210, run
    keys = ["run", "macro_f1", "macro_P", "macro_R", "acc_all_gt", "det_recall"] + [f"{c}_R" for c in D.SPECIES]
    D.wr(os.path.join(OUT, "v9_g_tuned_runs.csv"), ["arm", "seed", "split", "campanha"] + keys,
         [[a, s, "TEST v9.0.5", "h100n2 torch2.6+cu124"] + [D.fmt(met[(a, s)][k]) for k in keys] for (a, s) in sorted(met)])
    summ = []
    for arm in ARMS:
        seeds = [s for s in D.SEEDS if (arm, s) in met]
        if not seeds:
            continue
        acc = [met[(arm, s)]["acc_all_gt"] for s in seeds]
        r = [arm, len(seeds), f"{st.mean(acc):.4f}", f"{st.stdev(acc):.4f}" if len(acc) > 1 else "",
             "|".join(f"{x:.4f}" for x in acc), "", "", "", "", ""]
        if arm != "G0_E0_h100":
            ps = [s for s in seeds if ("G0_E0_h100", s) in met]
            if ps:
                a = {s: rows[(arm, s)] for s in ps}
                b = {s: rows[("G0_E0_h100", s)] for s in ps}
                dl, lo, hi, _ = D.boot_delta(a, b)
                per = [met[(arm, s)]["acc_all_gt"] - met[("G0_E0_h100", s)]["acc_all_gt"] for s in ps]
                r[5:] = [len(ps), f"{dl*100:+.2f}", f"{lo*100:+.2f}", f"{hi*100:+.2f}", "|".join(f"{x*100:+.2f}" for x in per)]
        summ.append(r)
    D.wr(os.path.join(OUT, "v9_g_tuned_summary.csv"),
         ["arm", "n_seeds", "acc_mean", "acc_std", "acc_por_seed", "n_pareadas", "delta_pp_vs_G0", "ic95_lo_pp", "ic95_hi_pp", "delta_pp_por_seed"], summ)
    for r in summ:
        print(r)


if __name__ == "__main__":
    main()
