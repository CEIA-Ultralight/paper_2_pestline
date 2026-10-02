"""RQ3 (B200): nwd_3x3, tiny_variants_multiseed, paired tests, ablation s0, SAHI, recall_by_size, TTA (pendente)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from b200_lib import *  # noqa: F403

OUT = REPO / "rq3_tiny_object" / "results"
FAM = {
    "E0": ("baseline CIoU (yolo26m)", [f"E0_s{s}" for s in range(3)]),
    "E10_NWD": ("NWD w=0.5 C=44 scope=both", [f"E10_nwd_w0.5_s{s}" for s in range(3)]),
    "E11_GCD": ("GCD w=0.5 C=44 scope=both", [f"E11_gcd_w0.5_s{s}" for s in range(3)]),
    "E1_P2": ("cabeca P2 (yolo26m-p2.yaml, pretrained yolo26m.pt, batch 8)", [f"E1_p2_s{s}" for s in range(3)]),
}
ABL = {
    "E10_NWD_loss_only": ("NWD w=0.5 scope=loss", "E10_nwd_w0.5_loss_s0"),
    "E10_NWD_assigner_only": ("NWD w=0.5 scope=assigner", "E10_nwd_w0.5_assigner_s0"),
    "E14_NWD_RFLA": ("NWD w=0.5 scope=both + RFLA", "E14_nwd_rfla_s0"),
}
VAL_KEYS = ["map50", "map50_95", "precision_u", "recall_u"]


def load_run(run: str) -> dict:
    base = R / run
    res = glob.glob(str(base / "runs" / "*" / "results.csv"))
    assert len(res) == 1, (run, res)
    df = pd.read_csv(res[0])
    df.columns = [c.strip() for c in df.columns]
    fit = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    i = int(fit.idxmax())
    val = {"map50": df.loc[i, "metrics/mAP50(B)"], "map50_95": df.loc[i, "metrics/mAP50-95(B)"],
           "precision_u": df.loc[i, "metrics/precision(B)"], "recall_u": df.loc[i, "metrics/recall(B)"]}
    te = base / "test_eval" / run / "single" / "label_only"
    m = load_json(te / "metrics.json")
    pi = pd.read_csv(te / "per_instance.csv")
    ok = check_pi_vs_metrics(pi, m["single"], "single", run)
    man = load_json(base / "test_eval" / "manifest.json")
    cfg = load_json(base / "nwd_config.json")
    sp = species_rows(pi)
    return {"run": run, "seed": cfg["seed"], "val": val, "best_epoch": int(df.loc[i, "epoch"]),
            "epochs_logged": len(df), "test": harness_metrics(m["single"]), "pi": sp, "ok": ok,
            "results_csv": res[0], "metrics_json": str(te / "metrics.json"), "job": man.get("slurm_job_id"),
            "cfg": cfg, "latency": m["latency"]}


RUNS = {}
for fam, (_, runs) in FAM.items():
    for r in runs:
        RUNS[r] = load_run(r)
for _, (_, r) in ABL.items():
    RUNS[r] = load_run(r)
for r, d in RUNS.items():
    exp = int(r.split("_s")[-1])
    assert d["seed"] == exp, (r, d["seed"])

# conferir alinhamento dos GT entre runs
ref = RUNS["E0_s0"]["pi"][["image", "gt_index", "gt_class", "gt_box"]]
for r, d in RUNS.items():
    assert d["pi"][["image", "gt_index", "gt_class", "gt_box"]].equals(ref), r
print("suporte TEST:", ref.gt_class.value_counts().to_dict(), "imgs", ref.image.nunique())

# -------------------------------------------------------------------------- tabela larga
COLS = (["level", "family", "variante", "run", "seeds", "n_seeds", "split", "protocol", "best_epoch_val", "epochs_logged"]
        + [f"{k}_pct" for k in VAL_KEYS] + [f"{k}_pct" for k in METRIC_KEYS] + COUNT_KEYS  # noqa: F405
        + [f"{k}_pct_std" for k in VAL_KEYS] + [f"{k}_pct_std" for k in METRIC_KEYS]  # noqa: F405
        + ["job", "source_path", "note"])
PROTO_VAL = "Ultralytics results.csv, best epoch (max fitness 0.1*mAP50+0.9*mAP50-95), imgsz 1920, conf 0.001"
PROTO_TEST = "harness eval_cascade_matrix single/label_only (conf>=0.25, IoU>=0.5, matching class-agnostico; MD/MV/MC/MF)"


def run_rows(fam, variant, d, seeds_label=None):
    out = []
    v = {f"{k}_pct": pct(d["val"][k]) for k in VAL_KEYS}
    out.append({"level": "run", "family": fam, "variante": variant, "run": d["run"], "seeds": d["seed"],
                "n_seeds": 1, "split": "VAL", "protocol": PROTO_VAL, "best_epoch_val": d["best_epoch"],
                "epochs_logged": d["epochs_logged"], **v, "job": d["job"], "source_path": d["results_csv"],
                "note": "VAL (35 imgs) tambem usado para escolher best.pt; nao e held-out"})
    t = {f"{k}_pct": pct(d["test"][k]) for k in METRIC_KEYS}  # noqa: F405
    t.update({k: d["test"][k] for k in COUNT_KEYS})  # noqa: F405
    out.append({"level": "run", "family": fam, "variante": variant, "run": d["run"], "seeds": d["seed"],
                "n_seeds": 1, "split": "TEST", "protocol": PROTO_TEST, "job": d["job"],
                "source_path": d["metrics_json"], "note": "TEST 55 imgs/918 GT especie; consolidado (nao cego)", **t})
    return out


def agg_row(level, fam, variant, split, vals_list, runs_label, seeds_label, protocol, job, src, note=""):
    keys = [f"{k}_pct" for k in VAL_KEYS] if split == "VAL" else [f"{k}_pct" for k in METRIC_KEYS]  # noqa: F405
    m, s = agg_dicts(vals_list, keys)
    row = {"level": level, "family": fam, "variante": variant, "run": runs_label, "seeds": seeds_label,
           "n_seeds": len(vals_list), "split": split, "protocol": protocol, "job": job, "source_path": src, "note": note}
    for k in keys:
        row[k] = None if m[k] is None else round(m[k], 4)
        row[k + "_std"] = None if s[k] is None else round(s[k], 4)
    return row


def vals_of(d, split):
    if split == "VAL":
        return {f"{k}_pct": pct(d["val"][k]) for k in VAL_KEYS}
    return {f"{k}_pct": pct(d["test"][k]) for k in METRIC_KEYS}  # noqa: F405


rows = []
agg_cache = {}
for fam, (variant, runs) in FAM.items():
    for r in runs:
        rows += run_rows(fam, variant, RUNS[r])
    jobs = "/".join(sorted({str(RUNS[r]["job"]) for r in runs}))
    for split, proto, src in (("VAL", PROTO_VAL, "runs/*/results.csv"), ("TEST", PROTO_TEST, "test_eval/<run>/single/label_only/metrics.json")):
        vl = [vals_of(RUNS[r], split) for r in runs]
        rows.append(agg_row("aggregate_mean_std", fam, variant, split, vl, "+".join(runs), "0,1,2", proto, jobs,
                            f"{R}/<run>/{src}", "std amostral ddof=1; n=3 seeds"))
        agg_cache[(fam, split)] = vl
# deltas vs E0 (por seed e agregado)
for fam in ("E10_NWD", "E11_GCD", "E1_P2"):
    for split, proto in (("VAL", PROTO_VAL), ("TEST", PROTO_TEST)):
        dl = []
        for s in range(3):
            a, b = vals_of(RUNS[FAM["E0"][1][s]], split), vals_of(RUNS[FAM[fam][1][s]], split)
            dd = {k: (round(b[k] - a[k], 4)) for k in a}
            dl.append(dd)
            row = {"level": "delta_vs_E0_by_seed", "family": fam, "variante": FAM[fam][0],
                   "run": f"{FAM[fam][1][s]} - {FAM['E0'][1][s]}", "seeds": s, "n_seeds": 1, "split": split,
                   "protocol": proto, "job": "33321/33363", "source_path": "ver runs", "note": "p.p.; pareado pelo indice de seed"}
            row.update(dd)
            rows.append(row)
        m, sd = agg_dicts(dl, list(dl[0].keys()))
        row = {"level": "delta_vs_E0_aggregate", "family": fam, "variante": FAM[fam][0],
               "run": f"{fam} - E0 (por seed)", "seeds": "0,1,2", "n_seeds": 3, "split": split, "protocol": proto,
               "job": "33321/33363", "source_path": "ver runs",
               "note": "media +- std (ddof=1) dos 3 deltas por seed, p.p.; |media|<std => sem efeito"}
        for k in dl[0]:
            row[k] = round(m[k], 4)
            row[k + "_std"] = round(sd[k], 4)
        rows.append(row)
for name, why in (("E7_yolo26l", "E7 yolo26l baseline s0: ultimo passo da cadeia C 33363; NAO RODOU (cadeia abortou antes)"),
                  ("ttawbf_E0_ensemble", "WBF ensemble dos 3 E0: IndexError no job 33363; NAO RODOU")):
    rows.append({"level": "not_run", "family": name, "variante": why, "run": name, "split": "TEST", "job": "33363",
                 "source_path": "/raid/user_marcospaulo/slurm_logs/p2-chainC-extra_33363.out", "note": "nao rodou"})
big = pd.DataFrame(rows).reindex(columns=COLS)
write_csv(big, OUT / "tiny_variants_multiseed.csv")
write_csv(big[big.family.isin(["E0", "E10_NWD"])].copy(), OUT / "nwd_3x3.csv")

# -------------------------------------------------------------------------- testes pareados
def pair_test(a, b, label_a, label_b):
    A, B = RUNS[a], RUNS[b]
    assert A["pi"][["image", "gt_index"]].equals(B["pi"][["image", "gt_index"]])
    t = mcnemar_exact(A["ok"], B["ok"])  # noqa: F405
    pt, lo, hi, nimg = boot_delta_acc(A["pi"].image, A["ok"], B["ok"])  # noqa: F405
    return {"A": a, "B": b, "rel": f"{label_b} vs {label_a}", "n_gt": len(A["ok"]), "n_imgs_boot": nimg,
            "acc_A_pct": pct(A["ok"].mean()), "acc_B_pct": pct(B["ok"].mean()), "b_A_ok_B_err": t["b"],
            "c_A_err_B_ok": t["c"], "p_mcnemar_exact": round(t["p_exact"], 6),
            "delta_acc_B_minus_A_pp": round(100 * pt, 4), "ci95_lo_pp": round(100 * lo, 4), "ci95_hi_pp": round(100 * hi, 4),
            "boot_n": BOOT_N, "boot_seed": BOOT_SEED}  # noqa: F405


prow = []
for fam in ("E10_NWD", "E11_GCD", "E1_P2"):
    for s in range(3):
        r = pair_test(FAM["E0"][1][s], FAM[fam][1][s], "E0", fam)
        r["tipo"] = "variante vs E0 (mesmo indice de seed)"
        prow.append(r)
for s0, s1 in ((0, 1), (0, 2), (1, 2)):
    for fam in ("E0", "E10_NWD"):
        r = pair_test(FAM[fam][1][s0], FAM[fam][1][s1], f"{fam}_s{s0}", f"{fam}_s{s1}")
        r["tipo"] = "controle de ruido: mesma config, seeds diferentes"
        prow.append(r)
pt = pd.DataFrame(prow)
pt["split"], pt["seeds"], pt["n_seeds"], pt["job"] = "TEST", "pareado por GT (mesma imagem/instancia)", 1, "33321/33363"
pt["source_path"] = f"{R}/<run>/test_eval/<run>/single/label_only/per_instance.csv"
write_csv(pt, OUT / "tiny_variants_paired_tests.csv")

# -------------------------------------------------------------------------- ablacao single-seed
std_ref = {}
for k in METRIC_KEYS:  # noqa: F405
    std_ref[k] = {fam: agg_dicts(agg_cache[(fam, "TEST")], [f"{k}_pct"])[1][f"{k}_pct"] for fam in ("E0", "E10_NWD")}
arows = []
ref_runs = [("E0", "E0_s0", "referencia E0_s0"), ("E10_NWD", "E10_nwd_w0.5_s0", "referencia E10_s0 (NWD both)")]
all_abl = [(k, v[0], v[1]) for k, v in ABL.items()]
for fam, r, lab in ref_runs:
    for row in run_rows(fam, lab, RUNS[r]):
        row["level"] = "run_referencia"
        arows.append(row)
for fam, var, r in all_abl:
    for row in run_rows(fam, var, RUNS[r]):
        arows.append(row)
    for refname, refrun in (("E0_s0", "E0_s0"), ("E10_s0", "E10_nwd_w0.5_s0")):
        for split in ("VAL", "TEST"):
            a, b = vals_of(RUNS[refrun], split), vals_of(RUNS[r], split)
            row = {"level": f"delta_vs_{refname}", "family": fam, "variante": var, "run": f"{r} - {refrun}", "seeds": 0,
                   "n_seeds": 1, "split": split, "job": "33321/33363", "source_path": "ver run_referencia",
                   "protocol": PROTO_VAL if split == "VAL" else PROTO_TEST, "note": "p.p.; single-seed"}
            row.update({k: round(b[k] - a[k], 4) for k in a})
            if split == "TEST":
                t = pair_test(refrun, r, refname, fam)
                row["note"] += (f"; McNemar exato vs {refname}: b={t['b_A_ok_B_err']} c={t['c_A_err_B_ok']} p={t['p_mcnemar_exact']}; "
                                f"dAcc={t['delta_acc_B_minus_A_pp']} IC95[{t['ci95_lo_pp']},{t['ci95_hi_pp']}] pp")
                for kk in ("f1_macro", "f1_micro"):
                    noise = max(std_ref[kk]["E0"], std_ref[kk]["E10_NWD"])
                    row[f"{kk}_pct_noise_std_3seeds"] = round(noise, 4)
                    row[f"{kk}_within_noise"] = abs(row[f"{kk}_pct"]) < noise
            arows.append(row)
acols = COLS[:-3] + ["f1_macro_pct_noise_std_3seeds", "f1_macro_within_noise", "f1_micro_pct_noise_std_3seeds",
                     "f1_micro_within_noise", "job", "source_path", "note"]
write_csv(pd.DataFrame(arows).reindex(columns=acols), OUT / "tiny_ablation_single_seed.csv")

# -------------------------------------------------------------------------- recall por tamanho
BINS = (0, 32, 48, math.inf)
BIN_LABEL = {"[0,32)": "<32", "[32,48)": "32-48", "[48,inf)": ">=48"}
brow = []
per = {}
for r, d in RUNS.items():
    sp = d["pi"].copy()
    sp["_ok"] = d["ok"]
    sp["pred_class_single"] = sp["pred_class_single"].fillna("")
    rs = stats.recall_by_size_bins(sp, BINS, pred_col="pred_class_single")
    rd = stats.recall_by_size_bins(sp, BINS, pred_col=None)
    for (_, a), (_, b) in zip(rs.iterrows(), rd.iterrows(), strict=True):
        assert a.n_gt == b.n_gt
        per[(r, a.bin)] = (a.recall, b.recall, a.n_gt, a.n_hit, b.n_hit)
        fam = next((f for f, (_, rr) in FAM.items() if r in rr), None) or next(k for k, v in ABL.items() if v[1] == r)
        brow.append({"level": "run", "family": fam, "run": r, "seeds": d["seed"], "n_seeds": 1, "split": "TEST",
                     "bin": BIN_LABEL[a.bin], "bin_def_sqrt_wh_px": a.bin, "n_gt_especie": int(a.n_gt),
                     "n_hit_especie": int(a.n_hit), "recall_especie_pct": pct(a.recall), "n_hit_deteccao": int(b.n_hit),
                     "recall_deteccao_pct": pct(b.recall), "job": d["job"], "source_path": d["metrics_json"].replace("metrics.json", "per_instance.csv"),
                     "note": "recall especie = matched & classe correta; recall deteccao = matched (class-agnostico)"})
bins_ = [BIN_LABEL[k] for k in BIN_LABEL]
raw = {BIN_LABEL[k]: k for k in BIN_LABEL}
for fam, (variant, runs) in FAM.items():
    for b in bins_:
        rs = [per[(r, raw[b])][0] for r in runs]
        rd = [per[(r, raw[b])][1] for r in runs]
        m1, s1 = mean_std(rs)
        m2, s2 = mean_std(rd)
        row = {"level": "aggregate_mean_std", "family": fam, "run": "+".join(runs), "seeds": "0,1,2", "n_seeds": 3,
               "split": "TEST", "bin": b, "bin_def_sqrt_wh_px": raw[b], "n_gt_especie": int(per[(runs[0], raw[b])][2]),
               "recall_especie_pct": round(100 * m1, 4), "recall_especie_pct_std": round(100 * s1, 4),
               "recall_deteccao_pct": round(100 * m2, 4), "recall_deteccao_pct_std": round(100 * s2, 4), "job": "33321/33363",
               "source_path": f"{R}/<run>/test_eval/<run>/single/label_only/per_instance.csv", "note": "std ddof=1"}
        brow.append(row)
        if fam != "E0":
            dl = [per[(runs[s], raw[b])][0] - per[(FAM["E0"][1][s], raw[b])][0] for s in range(3)]
            dd = [per[(runs[s], raw[b])][1] - per[(FAM["E0"][1][s], raw[b])][1] for s in range(3)]
            m1, s1 = mean_std(dl)
            m2, s2 = mean_std(dd)
            brow.append({"level": "delta_vs_E0_aggregate", "family": fam, "run": f"{fam} - E0 (por seed)", "seeds": "0,1,2",
                         "n_seeds": 3, "split": "TEST", "bin": b, "bin_def_sqrt_wh_px": raw[b],
                         "n_gt_especie": int(per[(runs[0], raw[b])][2]), "recall_especie_pct": round(100 * m1, 4),
                         "recall_especie_pct_std": round(100 * s1, 4), "recall_deteccao_pct": round(100 * m2, 4),
                         "recall_deteccao_pct_std": round(100 * s2, 4), "job": "33321/33363", "source_path": "ver run",
                         "note": "delta em p.p. (media +- std dos 3 deltas por seed)"})
bdf = pd.DataFrame(brow)
write_csv(bdf, OUT / "recall_by_size.csv")

# -------------------------------------------------------------------------- SAHI
srow = []
CLS7 = ["MD", "MV", "MC", "MF", "INS", "NOISE", "MAR"]
sah = {}
for s in range(3):
    base = R / f"sahi_E0_s{s}"
    args = load_json(base / "sahi_args.json")
    assert args["model"].endswith(f"E0_s{s}/best.pt"), args
    m = load_json(base / "test" / "metrics" / "metrics.json")
    for var in ("plain", "sliced"):
        cm = pd.read_csv(base / "test" / "metrics" / f"confusion_{var}_bg.csv", header=None).to_numpy()
        # linhas = GT (7 classes + BG/FP), colunas = pred (7 classes + BG/miss)
        assert cm.shape == (8, 8)
        row = {"level": "run", "variante": var, "run": f"sahi_E0_s{s}", "seeds": s, "n_seeds": 1, "split": "TEST",
               "protocol": "sahi_inference evaluator, 7 classes, eval_conf 0.0, IoU 0.5 (NAO e o harness conf>=0.25)",
               "imgsz_infer": args["imgsz"], "tile": args["tile"] if var == "sliced" else "n/a (frame inteiro)",
               "overlap": args["overlap"] if var == "sliced" else "n/a",
               "map50_pct": pct(m[var]["map50"]), "map50_95_pct": pct(m[var]["map50_95"]), "map75_pct": pct(m[var]["map75"]),
               "precision7_conf0_pct": pct(m[var]["precision"]), "recall7_conf0_pct": pct(m[var]["recall"]),
               "tp7": m[var]["tp"], "fp7": m[var]["fp"], "fn7": m[var]["fn"]}
        rec = []
        for i, sname in enumerate(SP):
            tot = cm[i].sum()
            row[f"{sname}_recall_conf0_pct"] = pct(cm[i, i] / tot)
            row[f"{sname}_miss_pct"] = pct(cm[i, 7] / tot)
            rec.append(cm[i, i] / tot)
        sup = sum(cm[i].sum() for i in range(4))
        row["species_recall_micro_conf0_pct"] = pct(sum(cm[i, i] for i in range(4)) / sup)
        row["species_recall_macro_conf0_pct"] = pct(float(np.mean(rec)))
        row["species_miss_micro_pct"] = pct(sum(cm[i, 7] for i in range(4)) / sup)
        row["n_gt_especie"] = int(sup)
        row["job"], row["source_path"] = "33363", str(base / "test" / "metrics")
        row["note"] = ("plain = frame inteiro inferido a imgsz 512 (reduz 1920->512, ~3.75x): baseline DEGENERADO, nao e o E0@1920; "
                       "recall conf0 nao e comparavel ao recall do harness (conf 0.25)")
        srow.append(row)
        sah[(s, var)] = row
    d = RUNS[f"E0_s{s}"]
    srow.append({"level": "run", "variante": "referencia: E0 plain imgsz 1920 (harness)", "run": f"E0_s{s}", "seeds": s,
                 "n_seeds": 1, "split": "TEST", "protocol": PROTO_TEST + " [outro protocolo, nao comparavel com linhas SAHI]",
                 "imgsz_infer": 1920, "tile": "n/a", "overlap": "n/a",
                 "species_recall_micro_conf0_pct": None, "n_gt_especie": 918,
                 "recall_micro_harness_pct": pct(d["test"]["recall_micro"]), "precision_micro_harness_pct": pct(d["test"]["precision_micro"]),
                 "f1_micro_harness_pct": pct(d["test"]["f1_micro"]), "species_det_recall_harness_pct": pct(d["test"]["det_recall"]),
                 "job": d["job"], "source_path": d["metrics_json"], "note": "referencia de contexto; mAP nao existe neste harness"})
agg_keys = ["map50_pct", "map50_95_pct", "map75_pct", "precision7_conf0_pct", "recall7_conf0_pct", "species_recall_micro_conf0_pct",
            "species_recall_macro_conf0_pct", "species_miss_micro_pct"] + [f"{s}_recall_conf0_pct" for s in SP]
for var in ("plain", "sliced"):
    rr = [sah[(s, var)] for s in range(3)]
    m, sd = agg_dicts(rr, agg_keys)
    row = {"level": "aggregate_mean_std", "variante": var, "run": "sahi_E0_s0+s1+s2", "seeds": "0,1,2", "n_seeds": 3, "split": "TEST",
           "protocol": rr[0]["protocol"], "imgsz_infer": rr[0]["imgsz_infer"], "tile": rr[0]["tile"], "overlap": rr[0]["overlap"],
           "n_gt_especie": 918, "job": "33363", "source_path": f"{R}/sahi_E0_s*/test/metrics", "note": "std ddof=1"}
    for k in agg_keys:
        row[k], row[k + "_std"] = round(m[k], 4), round(sd[k], 4)
    srow.append(row)
dl = {k: [sah[(s, "sliced")][k] - sah[(s, "plain")][k] for s in range(3)] for k in agg_keys}
row = {"level": "delta_sliced_minus_plain_by_seed_aggregate", "variante": "sliced - plain", "run": "por seed", "seeds": "0,1,2",
       "n_seeds": 3, "split": "TEST", "protocol": rr[0]["protocol"], "job": "33363", "source_path": f"{R}/sahi_E0_s*/test/metrics",
       "note": "p.p.; ATENCAO: plain@512 e baseline degenerado; delta nao mede o ganho do SAHI vs E0@1920"}
for k in agg_keys:
    m, sd = mean_std(dl[k])
    row[k], row[k + "_std"] = round(m, 4), round(sd, 4)
srow.append(row)
write_csv(pd.DataFrame(srow), OUT / "sahi_multiseed.csv")

# -------------------------------------------------------------------------- TTA pendente
trow = []
for s in range(3):
    sm = load_json(R / f"ttawbf_E0_s{s}" / "summary.json")
    for var in ("vanilla", "tta"):
        v = sm[f"{var}_E0_s{s}"]
        trow.append({"status": "PENDENTE_DE_VALIDACAO (possivel bug de indexacao no avaliador; nao usar como fato)",
                     "run": f"ttawbf_E0_s{s}", "seeds": s, "n_seeds": 1, "variante": var, "split": "TEST",
                     "protocol": "Ultralytics val no TEST (avaliador do eval_wbf_tta.py), imgsz 1920, conf 0.001; NAO e harness",
                     "map50_pct": pct(v["map50"]), "map50_95_pct": pct(v["map5095"]), "job": "33363",
                     "source_path": str(R / f"ttawbf_E0_s{s}" / "summary.json")})
write_csv(pd.DataFrame(trow), OUT / "tta_vanilla_multiseed_PENDENTE_validacao.csv")

# -------------------------------------------------------------------------- resumo p/ console
pd.set_option("display.width", 250, "display.max_columns", 50)
print(big[(big.level == "aggregate_mean_std") & (big.split == "TEST")][["family", "n_seeds", "f1_macro_pct", "f1_macro_pct_std", "f1_micro_pct", "f1_micro_pct_std", "recall_micro_pct", "precision_micro_pct", "acc_loc_pct", "acc_loc_pct_std"]].to_string())
print(big[(big.level == "aggregate_mean_std") & (big.split == "VAL")][["family", "map50_pct", "map50_pct_std", "map50_95_pct", "map50_95_pct_std"]].to_string())
print(big[(big.level == "delta_vs_E0_aggregate")][["family", "split", "f1_macro_pct", "f1_macro_pct_std", "map50_pct", "map50_pct_std", "map50_95_pct", "map50_95_pct_std"]].to_string())
print(big[(big.level == "run") & (big.split == "TEST")][["run", "f1_macro_pct", "f1_micro_pct", "recall_micro_pct", "precision_micro_pct", "acc_loc_pct", "det_recall_pct"]].to_string())
print(pt[["A", "B", "b_A_ok_B_err", "c_A_err_B_ok", "p_mcnemar_exact", "delta_acc_B_minus_A_pp", "ci95_lo_pp", "ci95_hi_pp", "tipo"]].to_string())
a = pd.DataFrame(arows)
print(a[a.level.str.startswith("delta") & (a.split == "TEST")][["level", "family", "f1_macro_pct", "f1_micro_pct", "recall_micro_pct", "precision_micro_pct", "f1_macro_within_noise", "note"]].to_string())
print(bdf[bdf.level != "run"][["level", "family", "bin", "n_gt_especie", "recall_especie_pct", "recall_especie_pct_std", "recall_deteccao_pct"]].to_string())
print(pd.DataFrame(srow)[["level", "variante", "run", "map50_pct", "map50_95_pct", "species_recall_micro_conf0_pct", "species_recall_macro_conf0_pct", "recall_micro_harness_pct"]].to_string())
