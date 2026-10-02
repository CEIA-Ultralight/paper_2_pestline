"""RQ2 (B200): cascade_{test,val}_seeds, mcnemar, confusion, oracle_crops (+decomposicao), latency, classifiers VAL."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from b200_lib import *  # noqa: F403

OUT = REPO / "rq2_cascade" / "results"
DETS = [f"E0_s{s}" for s in range(3)] + [f"E10_nwd_w0.5_s{s}" for s in range(3)]
CLFS = [f"clf_convnext_t_{f}_s{s}" for f in ("ce", "parts") for s in (42, 84, 126)]
POLS = ["label_only", "product"]
CLS7 = SP + NONSP
JOB = "33364"


def det_fam(d):
    return "E0" if d.startswith("E0_") else "E10"


def clf_fam(c):
    return "ce" if "_ce_" in c else "parts"


def clf_seed(c):
    return int(c.split("_s")[-1])


def det_seed(d):
    return int(d.split("_s")[-1])


def load_split(split):
    base = R / f"cascade_cross_{split}"
    man = load_json(base / "manifest.json")
    assert str(man.get("slurm_job_id")) == JOB, man.get("slurm_job_id")
    combos = {}
    for d in DETS:
        for c in ["single"] + CLFS:
            for p in (["label_only"] if c == "single" else POLS):
                folder = base / d / c / p
                m = load_json(folder / "metrics.json")
                pi = pd.read_csv(folder / "per_instance.csv")
                e = {"m": m, "pi": species_rows(pi), "folder": folder, "pi_all": pi}
                e["ok_single"] = check_pi_vs_metrics(pi, m["single"], "single", f"{split}/{d}/{c}/{p}")
                if c != "single":
                    e["ok_casc"] = check_pi_vs_metrics(pi, m["cascade"], "cascade", f"{split}/{d}/{c}/{p}")
                    disc = m["paired_discordant"]
                    # paired_discordant do metrics.json cobre os 7 classes (todos os GT), nao so especies
                    r = mcnemar_exact(is_true(pi.correct_single), is_true(pi.correct_cascade))  # noqa: F405
                    assert (r["b"], r["c"]) == (disc["single_right_cascade_wrong"], disc["single_wrong_cascade_right"])
                combos[(d, c, p)] = e
    return man, combos


def check_consistency(combos):
    """single identico entre combinacoes do mesmo detector; policy nao altera P/R/F1 do harness?"""
    notes = {"policy_identical": True}
    for d in DETS:
        s0 = combos[(d, "single", "label_only")]["m"]["single"]
        for c in CLFS:
            for p in POLS:
                assert combos[(d, c, p)]["m"]["single"]["species_correct"] == s0["species_correct"], (d, c, p)
    for d, c in itertools.product(DETS, CLFS):
        a, b = combos[(d, c, "label_only")], combos[(d, c, "product")]
        if not (a["ok_casc"] == b["ok_casc"]).all() or a["m"]["cascade"]["per_class"] != b["m"]["cascade"]["per_class"]:
            notes["policy_identical"] = False
        sa, sb = a["pi"].score_cascade.to_numpy(), b["pi"].score_cascade.to_numpy()
        notes.setdefault("score_differs", False)
        if not np.allclose(sa, sb):
            notes["score_differs"] = True
    return notes


COLS_BASE = ["level", "split", "det_family", "detector", "det_seed", "clf_family", "classifier", "clf_seed", "policy", "branch",
             "n_units", "seeds", "n_seeds"]
MCOLS = [f"{k}_pct" for k in METRIC_KEYS]  # noqa: F405
COLS_ALL = COLS_BASE + MCOLS + COUNT_KEYS + [f"{k}_std" for k in MCOLS] + ["job", "source_path", "note"]  # noqa: F405


def metric_vals(block):
    h = harness_metrics(block)  # noqa: F405
    out = {f"{k}_pct": pct(h[k]) for k in METRIC_KEYS}  # noqa: F405
    out.update({k: h[k] for k in COUNT_KEYS})  # noqa: F405
    return out


def build_cascade(split, combos, man):
    rows = []
    pair = {}
    base_note = ("VAL (35 imgs/617 GT): detectores e classificadores usaram o VAL p/ escolher best.pt -> nao e held-out"
                 if split == "VAL" else "TEST 55 imgs/918 GT; consolidado (nao cego)")
    for d in DETS:
        e = combos[(d, "single", "label_only")]
        v = metric_vals(e["m"]["single"])
        rows.append({"level": "pair", "split": split, "det_family": det_fam(d), "detector": d, "det_seed": det_seed(d),
                     "clf_family": "-", "classifier": "-", "clf_seed": None, "policy": "label_only", "branch": "single",
                     "n_units": 1, "seeds": f"det={det_seed(d)}", "n_seeds": 1, **v, "job": JOB,
                     "source_path": str(e["folder"] / "metrics.json"), "note": base_note})
        pair[("single", d)] = v
    for d, c, p in itertools.product(DETS, CLFS, POLS):
        e = combos[(d, c, p)]
        vs, vc = pair[("single", d)], metric_vals(e["m"]["cascade"])
        pair[("cascade", d, c, p)] = vc
        common = {"level": "pair", "split": split, "det_family": det_fam(d), "detector": d, "det_seed": det_seed(d),
                  "clf_family": clf_fam(c), "classifier": c, "clf_seed": clf_seed(c), "policy": p, "n_units": 1,
                  "seeds": f"det={det_seed(d)};clf={clf_seed(c)}", "n_seeds": 1, "job": JOB,
                  "source_path": str(e["folder"] / "metrics.json")}
        rows.append({**common, "branch": "cascade", **vc, "note": base_note})
        dl = {k: (round(vc[k] - vs[k], 4) if k.endswith("_pct") else vc[k] - vs[k]) for k in vc}
        pair[("delta", d, c, p)] = dl
        rows.append({**common, "branch": "delta_cascade_minus_single", **dl, "note": "p.p. (contagens: diferenca)"})
    # agregados
    def agg(level, sel, label, extra):
        vals = [pair[k] for k in sel]
        m, s = agg_dicts(vals, MCOLS)
        cm, _ = agg_dicts(vals, COUNT_KEYS)
        r = {"level": level, "split": split, "job": JOB, "source_path": str(R / f"cascade_cross_{split}"), **extra}
        for k in MCOLS:
            r[k], r[k + "_std"] = round(m[k], 4), round(s[k], 4)
        for k in COUNT_KEYS:
            r[k] = round(cm[k], 2)
        r["n_units"] = len(vals)
        return r
    for fam in ("E0", "E10"):
        dets = [d for d in DETS if det_fam(d) == fam]
        rows.append(agg("aggregate_pairs", [("single", d) for d in dets], "", {
            "det_family": fam, "detector": "+".join(dets), "clf_family": "-", "classifier": "-", "policy": "label_only", "branch": "single",
            "seeds": "det=0,1,2", "n_seeds": 3, "note": "n=3 detectores; std ddof=1; " + base_note}))
        for cf in ("ce", "parts"):
            clfs = [c for c in CLFS if clf_fam(c) == cf]
            for p in POLS:
                for br in ("cascade", "delta"):
                    sel = [(br, d, c, p) for d in dets for c in clfs]
                    rows.append(agg("aggregate_pairs", sel, "", {
                        "det_family": fam, "detector": "+".join(dets), "clf_family": cf, "classifier": "+".join(clfs), "policy": p,
                        "branch": br if br == "cascade" else "delta_cascade_minus_single", "seeds": "det=0,1,2 x clf=42,84,126",
                        "n_seeds": 9, "note": "n=9 pares (3 det x 3 clf, cruzados; nao independentes); std ddof=1 entre os 9 pares; " + base_note}))
                    # medias por detector (std entre as 3 seeds de detector) e por classificador
                    for axis, groups in (("det_means", [[(br, d, c, p) for c in clfs] for d in dets]),
                                         ("clf_means", [[(br, d, c, p) for d in dets] for c in clfs])):
                        gm = [{k: float(np.mean([pair[x][k] for x in g])) for k in MCOLS} for g in groups]
                        m, s = agg_dicts(gm, MCOLS)
                        r = {"level": f"aggregate_{axis}", "split": split, "job": JOB, "det_family": fam, "detector": "+".join(dets),
                             "clf_family": cf, "classifier": "+".join(clfs), "policy": p,
                             "branch": br if br == "cascade" else "delta_cascade_minus_single", "n_units": len(gm),
                             "seeds": "det=0,1,2" if axis == "det_means" else "clf=42,84,126", "n_seeds": 3,
                             "source_path": str(R / f"cascade_cross_{split}"),
                             "note": ("media sobre classificadores por detector; std entre as 3 seeds de detector"
                                      if axis == "det_means" else "media sobre detectores por classificador; std entre as 3 seeds de clf")}
                        for k in MCOLS:
                            r[k], r[k + "_std"] = round(m[k], 4), round(s[k], 4)
                        rows.append(r)
    return pd.DataFrame(rows).reindex(columns=COLS_ALL), pair


# ============================================================================================ main
out = {}
for split in ("TEST", "VAL"):
    man, combos = load_split(split.lower())
    notes = check_consistency(combos)
    print(split, "manifest versions", man.get("versions"), "job", man.get("slurm_job_id"), notes)
    df, pair = build_cascade(split, combos, man)
    write_csv(df, OUT / f"cascade_{split.lower()}_seeds.csv")
    out[split] = (man, combos, df, pair, notes)

pd.set_option("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 40)
for split in ("TEST", "VAL"):
    df = out[split][2]
    sel = df[df.level.isin(["aggregate_pairs", "aggregate_det_means"]) & (df.policy == "label_only")]
    print(split)
    print(sel[["level", "det_family", "clf_family", "branch", "n_units", "recall_micro_pct", "recall_micro_pct_std", "precision_micro_pct", "precision_micro_pct_std", "f1_micro_pct", "f1_micro_pct_std", "f1_macro_pct", "f1_macro_pct_std", "acc_loc_pct", "acc_loc_pct_std"]].to_string())

# ----------------------------------------------------------------------------------- McNemar
mrows = []
for split in ("TEST", "VAL"):
    _, combos, _, _, _ = out[split]
    for d, c, p in itertools.product(DETS, CLFS, POLS):
        e = combos[(d, c, p)]
        t = mcnemar_exact(e["ok_single"], e["ok_casc"])  # noqa: F405
        pt, lo, hi, nimg = boot_delta_acc(e["pi"].image, e["ok_single"], e["ok_casc"])  # noqa: F405
        mrows.append({"row_type": "pair", "split": split, "det_family": det_fam(d), "detector": d, "classifier": c,
                      "clf_family": clf_fam(c), "policy": p, "seeds": f"det={det_seed(d)};clf={clf_seed(c)}", "n_seeds": 1,
                      "n_gt_species": len(e["ok_single"]), "acc_single_pct": pct(e["ok_single"].mean()),
                      "acc_cascade_pct": pct(e["ok_casc"].mean()), "b_single_ok_cascade_err": t["b"],
                      "c_single_err_cascade_ok": t["c"], "p_exact": t["p_exact"], "p_chi2_cc_ref": t["p_chi2_cc"],
                      "delta_acc_pp": round(100 * pt, 4), "ci95_lo_pp": round(100 * lo, 4), "ci95_hi_pp": round(100 * hi, 4),
                      "boot_unit": "imagem", "boot_n_images": nimg, "boot_n": BOOT_N, "boot_seed": BOOT_SEED,  # noqa: F405
                      "cascade_better": t["c"] > t["b"], "job": JOB, "source_path": str(e["folder"] / "per_instance.csv")})
mdf = pd.DataFrame(mrows)
# Holm por grupo (9 pares: det_family x clf_family x policy x split) e sobre todos os 36 pares label_only (TEST)
mdf["p_holm_group9"] = np.nan
for _, g in mdf.groupby(["split", "det_family", "clf_family", "policy"]):
    adj = holm(g.p_exact.tolist())  # noqa: F405
    mdf.loc[g.index, "p_holm_group9"] = adj
mdf["p_holm_all36_same_policy"] = np.nan
for _, g in mdf.groupby(["split", "policy"]):
    mdf.loc[g.index, "p_holm_all36_same_policy"] = holm(g.p_exact.tolist())  # noqa: F405
srows = []
for (split, fam, cf, pol), g in mdf.groupby(["split", "det_family", "clf_family", "policy"]):
    srows.append({"row_type": "summary_9pairs", "split": split, "det_family": fam, "clf_family": cf, "policy": pol,
                  "seeds": "det=0,1,2 x clf=42,84,126", "n_seeds": 9, "n_pairs": len(g),
                  "n_sig_raw_p<0.05": int((g.p_exact < 0.05).sum()),
                  "n_sig_raw_cascade_better": int(((g.p_exact < 0.05) & g.cascade_better).sum()),
                  "n_sig_raw_cascade_worse": int(((g.p_exact < 0.05) & ~g.cascade_better).sum()),
                  "n_sig_holm_group9": int((g.p_holm_group9 < 0.05).sum()),
                  "n_sig_holm_all36": int((g.p_holm_all36_same_policy < 0.05).sum()),
                  "n_cascade_better_any": int(g.cascade_better.sum()),
                  "n_ci95_excl_zero_pos": int((g.ci95_lo_pp > 0).sum()),
                  "delta_acc_pp": round(g.delta_acc_pp.mean(), 4), "delta_acc_pp_std": round(g.delta_acc_pp.std(ddof=1), 4),
                  "delta_acc_pp_min": g.delta_acc_pp.min(), "delta_acc_pp_max": g.delta_acc_pp.max(),
                  "job": JOB, "source_path": str(R / f"cascade_cross_{split.lower()}")})
sdf = pd.DataFrame(srows)
write_csv(pd.concat([mdf, sdf], ignore_index=True), OUT / "mcnemar.csv")
print(sdf.to_string())
print(mdf[(mdf.split == "TEST") & (mdf.det_family == "E0") & (mdf.policy == "label_only") & (mdf.clf_family == "ce")][["detector", "classifier", "b_single_ok_cascade_err", "c_single_err_cascade_ok", "p_exact", "p_holm_group9", "delta_acc_pp", "ci95_lo_pp", "ci95_hi_pp"]].to_string())

# ----------------------------------------------------------------------------------- confusao
crow = []
OTHER = "OUTRA_CLASSE(INS/NOISE/MAR)"
COLS6 = SP + [OTHER, "MISS(nao localizado)"]


def conf_counts(pi_all, branch):
    sp = pi_all[pi_all.gt_class.isin(SP)]
    cm = stats.confusion_from_per_instance(sp, classes=CLS7, pred_col=f"pred_class_{branch}")
    cm = cm.loc[SP]
    out_ = cm[SP].copy()
    out_[OTHER] = cm[NONSP].sum(axis=1)
    out_[COLS6[-1]] = cm["BG"]
    assert (out_.sum(axis=1) == sp.gt_class.value_counts().reindex(SP)).all()
    return out_


def conf_rows(level, split, branch, det, clf, cm, n_units, src, fam, note=""):
    r_ = []
    for g in SP:
        tot = cm.loc[g].sum()
        for p_ in COLS6:
            r_.append({"level": level, "split": split, "branch": branch, "det_family": fam[0], "clf_family": fam[1], "detector": det,
                       "classifier": clf, "policy": "label_only", "n_units": n_units, "gt": g, "pred": p_, "count": int(cm.loc[g, p_]),
                       "row_total": int(tot), "row_pct": round(100 * cm.loc[g, p_] / tot, 4), "job": JOB, "source_path": src, "note": note})
    return r_


rate_store = {}
for split in ("TEST",):
    _, combos, _, _, _ = out[split]
    for fam in ("E0", "E10"):
        dets = [d for d in DETS if det_fam(d) == fam]
        sing = {d: conf_counts(combos[(d, "single", "label_only")]["pi_all"], "single") for d in dets}
        for d in dets:
            crow += conf_rows("pair", split, "single", d, "-", sing[d], 1, str(combos[(d, "single", "label_only")]["folder"] / "per_instance.csv"), (fam, "-"))
        crow += conf_rows("pooled", split, "single", "+".join(dets), "-", sum(sing.values()), 3, f"{R}/cascade_cross_test/<det>/single/label_only", (fam, "-"), "soma dos 3 detectores (3 x 918 GT)")
        for cf in ("ce", "parts"):
            clfs = [c for c in CLFS if clf_fam(c) == cf]
            casc = {}
            for d, c in itertools.product(dets, clfs):
                casc[(d, c)] = conf_counts(combos[(d, c, "label_only")]["pi_all"], "cascade")
                crow += conf_rows("pair", split, "cascade", d, c, casc[(d, c)], 1, str(combos[(d, c, "label_only")]["folder"] / "per_instance.csv"), (fam, cf))
            crow += conf_rows("pooled", split, "cascade", "+".join(dets), "+".join(clfs), sum(casc.values()), 9, f"{R}/cascade_cross_test/<det>/<clf>/label_only", (fam, cf), "soma dos 9 pares (9 x 918 GT); row_pct comparavel a single")
            # highlight: taxas de confusao especificas, media +- std
            for g, p_ in (("MC", "MV"), ("MV", "MC"), ("MF", "MC"), ("MC", "MF"), ("MD", "MV"), ("MV", "MD"), ("MF", OTHER), ("MD", COLS6[-1])):
                rs = [100 * sing[d].loc[g, p_] / sing[d].loc[g].sum() for d in dets]
                rc = [100 * casc[k].loc[g, p_] / casc[k].loc[g].sum() for k in casc]
                dl = [100 * (casc[(d, c)].loc[g, p_] / casc[(d, c)].loc[g].sum() - sing[d].loc[g, p_] / sing[d].loc[g].sum()) for d, c in casc]
                ms, ss = mean_std(rs)
                mc, sc = mean_std(rc)
                md, sd = mean_std(dl)
                crow.append({"level": "highlight_rate", "split": split, "branch": "single_vs_cascade", "det_family": fam, "clf_family": cf,
                             "detector": "+".join(dets), "classifier": "+".join(clfs), "policy": "label_only", "n_units": 9, "gt": g, "pred": p_,
                             "rate_single_pct": round(ms, 4), "rate_single_pct_std": round(ss, 4), "rate_cascade_pct": round(mc, 4),
                             "rate_cascade_pct_std": round(sc, 4), "delta_pp": round(md, 4), "delta_pp_std": round(sd, 4),
                             "row_total": int(sing[dets[0]].loc[g].sum()), "job": JOB, "source_path": f"{R}/cascade_cross_test",
                             "note": "taxa = GT classe `gt` rotulada como `pred` (% do suporte da classe); single n=3, cascade n=9, delta n=9 pares"})
                rate_store[(fam, cf, g, p_)] = (ms, ss, mc, sc, md, sd)
cdf = pd.DataFrame(crow)
cdf["seeds"] = [(f"det={str(d)[-1]};clf={str(c).split('_s')[-1]}" if c != "-" else f"det={str(d)[-1]}") if lv == "pair" else ("det=0,1,2 x clf=42,84,126" if c != "-" and "+" in str(c) else "det=0,1,2")
                 for lv, d, c in zip(cdf.level, cdf.detector, cdf.classifier, strict=True)]
cdf["n_seeds"] = cdf.n_units
write_csv(cdf, OUT / "confusion_single_vs_cascade.csv")
for k, v in rate_store.items():
    if k[0] == "E0" and k[1] == "ce":
        print(k, [round(x, 2) for x in v])
print(cdf[(cdf.level == "pooled") & (cdf.det_family == "E0") & (cdf.clf_family.isin(["-", "ce"]))].pivot_table(index=["branch", "gt"], columns="pred", values="row_pct").round(1).to_string())

# ----------------------------------------------------------------------------------- oracle
orows = []
orc = {}
for split in ("test", "val"):
    base = R / f"oracle_crops_{split}"
    for c in CLFS:
        m = load_json(base / c / "metrics.json")
        pi = pd.read_csv(base / c / "per_instance.csv")
        assert len(pi) == m["species_support"] == (918 if split == "test" else 617), (split, c, len(pi))
        assert int(pi.correct.sum()) == m["species_correct"]
        orc[(split, c)] = (m, pi)
        pc = m["per_class"]
        row = {"row_type": "oracle", "split": split.upper(), "classifier": c, "clf_family": clf_fam(c), "clf_seed": clf_seed(c),
               "seeds": f"clf={clf_seed(c)}", "n_seeds": 1, "n_gt_species": m["species_support"],
               "acc_species_pct": pct(m["species_accuracy"]), "precision_macro_pct": pct(m["species_macro_precision"]),
               "recall_macro_pct": pct(m["species_macro_recall"]), "f1_macro_pct": pct(m["species_macro_f1"]),
               "n_pred_outside_species": m["n_pred_outside_species"], "job": JOB, "source_path": str(base / c / "metrics.json"),
               "note": "classificador sobre crops das caixas GT (localizacao perfeita); precisao so sobre os crops GT de especie (sem FP de deteccao)"}
        for s in SP:
            row[f"{s}_recall_pct"], row[f"{s}_precision_pct"], row[f"{s}_f1_pct"] = pct(pc[s]["recall"]), pct(pc[s]["precision"]), pct(pc[s]["f1"])
        if split == "val":
            row["note"] += "; VAL tambem escolheu o best.pt do classificador (nao held-out)"
        orows.append(row)
okeys = ["acc_species_pct", "precision_macro_pct", "recall_macro_pct", "f1_macro_pct"] + [f"{s}_{k}_pct" for s in SP for k in ("recall", "precision", "f1")]
for split in ("TEST", "VAL"):
    for cf in ("ce", "parts"):
        rr = [r for r in orows if r["split"] == split and r["clf_family"] == cf and r["row_type"] == "oracle"]
        m, s = agg_dicts(rr, okeys)
        row = {"row_type": "oracle_aggregate", "split": split, "clf_family": cf, "classifier": "+".join(r["classifier"] for r in rr),
               "seeds": "clf=42,84,126", "n_seeds": 3, "n_gt_species": rr[0]["n_gt_species"], "job": JOB,
               "source_path": f"{R}/oracle_crops_{split.lower()}", "note": "std ddof=1 entre 3 seeds de classificador"}
        for k in okeys:
            row[k], row[k + "_std"] = round(m[k], 4), round(s[k], 4)
        orows.append(row)

# decomposicao do erro: localizacao x identificacao
drows = []
dec = {}
for split in ("TEST", "VAL"):
    _, combos, _, _, _ = out[split]
    for d in DETS:
        e = combos[(d, "single", "label_only")]
        sp = e["pi"]
        N = len(sp)
        matched = is_true(sp.matched).to_numpy()  # noqa: F405
        ok = e["ok_single"]
        dec[(split, d, "single")] = {"n": N, "loc_err": (N - matched.sum()) / N, "id_err_all": (matched.sum() - ok.sum()) / N,
                                     "acc_all": ok.mean(), "acc_loc": ok.sum() / matched.sum(), "det_recall": matched.mean()}
        row = {"row_type": "decomposition", "split": split, "det_family": det_fam(d), "detector": d, "classifier": "-", "clf_family": "-",
               "branch": "single", "seeds": f"det={det_seed(d)}", "n_seeds": 1, "n_gt_species": N,
               **{f"{k}_pct": pct(v) for k, v in dec[(split, d, "single")].items() if k != "n"},
               "n_localizados": int(matched.sum()), "n_corretos": int(ok.sum()), "job": JOB, "source_path": str(e["folder"] / "per_instance.csv"),
               "note": "erro_loc = GT nao localizado (sem proposta conf>=0.25, IoU>=0.5); erro_id = localizado com especie errada"}
        drows.append(row)
        for c in CLFS:
            ec = combos[(d, c, "label_only")]
            okc = ec["ok_casc"]
            m_, pio = orc[(split.lower(), c)]
            j = sp[["image", "gt_index", "gt_box", "gt_class"]].merge(pio[["image", "gt_index", "gt_box", "gt_class", "correct"]], on=["image", "gt_index"], suffixes=("", "_o"), how="left")
            assert j.correct.notna().all() and (j.gt_box == j.gt_box_o).all() and (j.gt_class == j.gt_class_o).all()
            oc = j.correct.astype(bool).to_numpy()
            nm = int(matched.sum())
            dec[(split, d, c)] = {"n": N, "loc_err": (N - nm) / N, "id_err_all": (nm - okc.sum()) / N, "acc_all": okc.mean(),
                                  "acc_loc": okc.sum() / nm, "det_recall": nm / N, "oracle_acc_all": oc.mean(),
                                  "oracle_acc_on_localized": oc[matched].mean(), "gap_casc_to_oracle_on_localized_pp": 100 * (oc[matched].mean() - okc.sum() / nm),
                                  "teto_pipeline_acc_all": matched.mean() * oc[matched].mean()}
            row = {"row_type": "decomposition", "split": split, "det_family": det_fam(d), "detector": d, "classifier": c,
                   "clf_family": clf_fam(c), "branch": "cascade", "seeds": f"det={det_seed(d)};clf={clf_seed(c)}", "n_seeds": 1, "n_gt_species": N,
                   **{f"{k}_pct": pct(v) for k, v in dec[(split, d, c)].items() if k not in ("n", "gap_casc_to_oracle_on_localized_pp")},
                   "gap_casc_to_oracle_on_localized_pp": round(dec[(split, d, c)]["gap_casc_to_oracle_on_localized_pp"], 4),
                   "n_localizados": nm, "n_corretos": int(okc.sum()),
                   "n_casc_errado_oraculo_certo_loc": int((~okc & oc & matched).sum()),
                   "n_casc_certo_oraculo_errado_loc": int((okc & ~oc & matched).sum()),
                   "job": JOB, "source_path": f"{ec['folder'] / 'per_instance.csv'} + {R}/oracle_crops_{split.lower()}/{c}/per_instance.csv",
                   "note": "oraculo = mesmo classificador sobre crop GT; acc_on_localized restringe o oraculo aos GT localizados (pareado por image,gt_index)"}
            drows.append(row)
dkeys = ["loc_err_pct", "id_err_all_pct", "acc_all_pct", "acc_loc_pct", "det_recall_pct", "oracle_acc_all_pct", "oracle_acc_on_localized_pct",
         "teto_pipeline_acc_all_pct", "gap_casc_to_oracle_on_localized_pp"]
for split in ("TEST", "VAL"):
    for fam in ("E0", "E10"):
        sel = [r for r in drows if r["split"] == split and r["det_family"] == fam and r["branch"] == "single"]
        m, s = agg_dicts(sel, dkeys[:5])
        row = {"row_type": "decomposition_aggregate", "split": split, "det_family": fam, "detector": "+".join(r["detector"] for r in sel), "classifier": "-",
               "clf_family": "-", "branch": "single", "seeds": "det=0,1,2", "n_seeds": 3, "n_gt_species": sel[0]["n_gt_species"], "job": JOB,
               "source_path": f"{R}/cascade_cross_{split.lower()}", "note": "n=3 detectores; std ddof=1"}
        for k in dkeys[:5]:
            row[k], row[k + "_std"] = round(m[k], 4), round(s[k], 4)
        drows.append(row)
        for cf in ("ce", "parts"):
            sel = [r for r in drows if r["split"] == split and r["det_family"] == fam and r["branch"] == "cascade" and r["clf_family"] == cf and r["row_type"] == "decomposition"]
            m, s = agg_dicts(sel, dkeys)
            row = {"row_type": "decomposition_aggregate", "split": split, "det_family": fam, "detector": "+".join(sorted({r["detector"] for r in sel})),
                   "classifier": "+".join(sorted({r["classifier"] for r in sel})), "clf_family": cf, "branch": "cascade",
                   "seeds": "det=0,1,2 x clf=42,84,126", "n_seeds": 9, "n_gt_species": sel[0]["n_gt_species"], "job": JOB,
                   "source_path": f"{R}/cascade_cross_{split.lower()}", "note": "n=9 pares; std ddof=1"}
            for k in dkeys:
                row[k], row[k + "_std"] = round(m[k], 4), round(s[k], 4)
            drows.append(row)
odf = pd.concat([pd.DataFrame(orows), pd.DataFrame(drows)], ignore_index=True)
write_csv(odf, OUT / "oracle_crops.csv")
print(odf[odf.row_type.isin(["oracle_aggregate", "decomposition_aggregate"])][["row_type", "split", "det_family", "clf_family", "branch", "acc_species_pct", "acc_species_pct_std", "f1_macro_pct", "loc_err_pct", "loc_err_pct_std", "id_err_all_pct", "id_err_all_pct_std", "acc_all_pct", "acc_all_pct_std", "acc_loc_pct", "oracle_acc_all_pct", "oracle_acc_on_localized_pct", "oracle_acc_on_localized_pct_std", "gap_casc_to_oracle_on_localized_pp", "teto_pipeline_acc_all_pct"]].to_string())

# ----------------------------------------------------------------------------------- latencia
lrows = []
for split in ("TEST", "VAL"):
    man, combos, _, _, _ = out[split]
    ver = man["versions"]
    hw = f"{ver['gpu']}; torch {ver['torch']}; ultralytics {ver['ultralytics']}; imgsz {man['predict']['imgsz']}; half={man['predict']['half']}"
    for (d, c, p), e in combos.items():
        lat = e["m"]["latency"]
        row = {"level": "pair", "split": split, "det_family": det_fam(d), "detector": d, "classifier": "-" if c == "single" else c,
               "clf_family": "-" if c == "single" else clf_fam(c), "policy": p, "n_images": lat["detector"]["n"], "hardware": hw,
               "job": JOB, "source_path": str(e["folder"] / "metrics.json")}
        for st in ("detector", "classifier_extra", "end_to_end"):
            if st in lat:
                for q in ("mean_s", "std_s", "p50_s", "p95_s"):
                    row[f"{st}_{q.replace('_s', '')}_ms"] = round(1000 * lat[st][q], 3)
        lrows.append(row)
ldf = pd.DataFrame(lrows)
agg_rows = []
for (split, fam, cf), g in ldf[ldf.classifier != "-"].groupby(["split", "det_family", "clf_family"]):
    r = {"level": "aggregate_pairs", "split": split, "det_family": fam, "classifier": "+".join(sorted(g.classifier.unique())), "clf_family": cf,
         "policy": "label_only+product", "n_images": int(g.n_images.iloc[0]), "n_pairs": len(g), "hardware": g.hardware.iloc[0], "job": JOB,
         "source_path": f"{R}/cascade_cross_{split.lower()}",
         "note": "media +- std (ddof=1) das medias por imagem de cada combinacao (det x clf x policy); std_img = media dos desvios por imagem"}
    for st in ("detector", "classifier_extra", "end_to_end"):
        m, s = mean_std(g[f"{st}_mean_ms"])
        r[f"{st}_mean_ms"], r[f"{st}_mean_ms_std_across_pairs"] = round(m, 3), round(s, 3)
        r[f"{st}_std_ms"] = round(g[f"{st}_std_ms"].mean(), 3)
        r[f"{st}_p50_ms"], r[f"{st}_p95_ms"] = round(g[f"{st}_p50_ms"].mean(), 3), round(g[f"{st}_p95_ms"].mean(), 3)
    agg_rows.append(r)
for (split, fam), g in ldf[ldf.classifier == "-"].groupby(["split", "det_family"]):
    r = {"level": "aggregate_pairs", "split": split, "det_family": fam, "classifier": "-", "clf_family": "-", "policy": "label_only",
         "n_images": int(g.n_images.iloc[0]), "n_pairs": len(g), "hardware": g.hardware.iloc[0], "job": JOB,
         "source_path": f"{R}/cascade_cross_{split.lower()}", "note": "single-stage (so detector), n=3 detectores"}
    m, s = mean_std(g["detector_mean_ms"])
    r["detector_mean_ms"], r["detector_mean_ms_std_across_pairs"] = round(m, 3), round(s, 3)
    r["detector_std_ms"] = round(g["detector_std_ms"].mean(), 3)
    r["detector_p50_ms"], r["detector_p95_ms"] = round(g["detector_p50_ms"].mean(), 3), round(g["detector_p95_ms"].mean(), 3)
    agg_rows.append(r)
ldf = pd.concat([ldf, pd.DataFrame(agg_rows)], ignore_index=True)
ldf["seeds"] = ldf["level"].map(lambda x: "por combinacao" if x == "pair" else "det=0,1,2 (x clf=42,84,126)")
ldf["note"] = ldf.get("note")
ldf["n_seeds"] = [1 if lv == "pair" else (9 if c != "-" else 3) for lv, c in zip(ldf.level, ldf.classifier, strict=True)]
write_csv(ldf, OUT / "latency.csv")
print(ldf[ldf.level == "aggregate_pairs"][["split", "det_family", "clf_family", "detector_mean_ms", "detector_std_ms", "classifier_extra_mean_ms", "classifier_extra_mean_ms_std_across_pairs", "end_to_end_mean_ms", "end_to_end_std_ms", "end_to_end_p95_ms"]].to_string())

# ----------------------------------------------------------------------------------- classificadores VAL (cadeia B)
crow2 = []
for c in CLFS:
    s_ = load_json(R / c / "summary.json")
    bm = s_["best_metrics"]
    assert s_["status"] in ("completed", "early_stop"), (c, s_["status"])
    crow2.append({"classifier": c, "clf_family": clf_fam(c), "seeds": clf_seed(c), "n_seeds": 1, "split": "VAL",
                  "protocol": "crop-level (DS-F2_crops_pad75), best epoch pela regra do summary.json", "status": s_["status"],
                  "best_epoch": s_["best_epoch"], "epochs_ran": s_["epochs_ran"], "elapsed_h": round(s_["elapsed_seconds"] / 3600, 3),
                  "selection_metric": s_["selection_metric"], "n_samples": bm["num_samples"], "acc_all7_pct": pct(bm["overall_accuracy"]),
                  "acc_species_pct": pct(bm["species_accuracy"]), "species_macro_recall_pct": pct(bm["species_macro_recall"]),
                  "species_macro_f1_pct": pct(bm["species_macro_f1"]), "n_species_support": bm["species_support"],
                  **{f"{s}_{k}_pct": pct(bm["per_class"][s][k]) for s in SP for k in ("recall", "precision", "f1")},
                  "oracle_val_acc_species_pct": pct(orc[("val", c)][0]["species_accuracy"]),
                  "oracle_minus_crops_pad75_pp": round(100 * (orc[("val", c)][0]["species_accuracy"] - bm["species_accuracy"]), 4),
                  "job": "33322", "source_path": str(R / c / "summary.json"),
                  "note": "VAL tambem escolheu o best epoch (nao held-out); oraculo VAL usa crops GT extraidos em memoria (1649 GT), summary usa crops pad75 pre-extraidos (1645)"})
cdf2 = pd.DataFrame(crow2)
write_csv(cdf2, OUT / "classifiers_b200_val_multiseed.csv")
print(cdf2[["classifier", "status", "best_epoch", "epochs_ran", "acc_species_pct", "species_macro_f1_pct"]].to_string())
