"""Gera os CSVs de rq2_cascade/results e rq3_tiny_object/results a partir do dump W&B
(_scratch/fly_species_detail.json, produzido por wandb_fly_species_detail.py) + valores
citados em docs/*.md que não são verificáveis neste nó.  CPU only, sem rede.
"""
import csv, json, math, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = {r["name"]: r for r in json.load(open(os.path.join(ROOT, "_scratch/fly_species_detail.json")))}
FS = "pestline/fly-species"
TC = RUNS["test-campaign-036ef018"]["summary"]
EV = RUNS["EVAL_E0_cascade_val_20260909T134157Z"]["summary"]
TC_RUN = f"{FS}/test-campaign-036ef018 (036ef018cfcf494d8e15eb49e95acc1e)"
EV_RUN = f"{FS}/EVAL_E0_cascade_val_20260909T134157Z (114dbfba)"
PF = "docs/PAPER_FACTS.md"
NV = "não verificável neste nó (fly-det/flydet_runs ausentes em dgx-B200-1)"
SPECIES = ["MD", "MV", "MC", "MF"]


def pct(x, nd=2):
    return "" if x is None else f"{100 * x:.{nd}f}"


def num(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def write(path, header, rows):
    path = os.path.join(ROOT, path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"{path}: {len(rows)} linhas")


# --------------------------------------------------------------------------- 1
def det_row(exp, variante, w, name, job, seed_decl, seed_ef, obs, protocolo="wandb_summary"):
    r = RUNS.get(name)
    if r is None:
        return dict(exp=exp, variante=variante, loss_w=w, seed_declarada=seed_decl, seed_efetiva="",
                    split="VAL", protocolo=protocolo, wandb_run="", state="não rodou", job=job,
                    origem="W&B (ausência de run)", observacao=obs)
    s = r["summary"]
    rt = s.get("_runtime")
    return dict(
        exp=exp, variante=variante, loss_w=w, seed_declarada=seed_decl, seed_efetiva=seed_ef,
        split="VAL", protocolo=protocolo,
        mAP50=num(s.get("metrics/mAP50(B)"), 4), mAP50_95=num(s.get("metrics/mAP50-95(B)"), 4),
        precision=num(s.get("metrics/precision(B)"), 4), recall=num(s.get("metrics/recall(B)"), 4),
        best_fitness=num(s.get("best/best_fitness"), 4), epochs_logged=s.get("_step", ""),
        runtime_h=num(rt / 3600, 2) if rt else "",
        wandb_run=f"{FS}/{name} ({r['id']})", state=r["state"], job=job,
        origem="W&B summary (última validação do best.pt ao fim do treino)", observacao=obs,
    )


def pf_row(exp, variante, w, mAP50, mAP5095, job, obs, protocolo="results_csv_best", origem=f"{PF} §1 / rq3 README"):
    return dict(exp=exp, variante=variante, loss_w=w, seed_declarada="default", seed_efetiva="0 (default Ultralytics)",
                split="VAL", protocolo=protocolo, mAP50=mAP50, mAP50_95=mAP5095, job=job,
                state="finished", origem=origem, observacao=f"{obs}; {NV}".strip("; "))


nwd_rows = [
    det_row("E0", "baseline CIoU (yolo26m)", "0", "E0_yolo26m_baseline", "n/d (2026-09-06)", "default", "0 (default)",
            "receita comum: 150 ép., batch 8, imgsz 1920, patience 30"),
    det_row("E1", "cabeça P2 (yolo26m-p2.yaml, 200 ép.)", "0", "E1_yolo26m_p2", "n/d (2026-09-06)", "default", "0 (default)",
            "alternativa single-stage; TEST F1 em sahi_tta_single_seed.csv"),
    det_row("E2", "treino em tiles 512 (imgsz 512, 60 ép.)", "0", "E2_yolo26m_sahi", "n/d (2026-09-06)", "default", "0 (default)",
            "VAL em tiles — NÃO comparável ao imgsz 1920; TEST F1 em sahi_tta_single_seed.csv"),
    det_row("E10", "NWD w=0,5, C=44", "0.5", "E10_yolo26m_nwd", "32933", "default", "0 (default)",
            "results.csv best (PAPER_FACTS) = 0,8706 é máximo por época, não o best.pt — ver linhas results_csv_*"),
    det_row("E11", "NWD w=0,75", "0.75", "E10_w075", "32947 (E10x)", "default", "0 (default)", ""),
    det_row("E12", "NWD w=1,0 (sem CIoU)", "1.0", "E10_w100", "32947 (E10x)", "default", "0 (default)",
            "colapso; parou em 33 steps logados (early stop)"),
    det_row("E10b", "GCD w=0,5", "0.5", "E10b_yolo26m_gcd", "32947 (E10x)", "default", "0 (default)", ""),
    det_row("E13", "multi_scale + rect=false", "0.5", "E13_yolo26m_multiscale", "32947 (E10x)", "default", "",
            "CUDA OOM após 127 s; abortou a cadeia E10x (E7 e TTA/WBF do 32947 não rodaram)"),
    det_row("E10 s0", "NWD w=0,5, 'seed 0'", "0.5", "E10_nwd_seed0", "33015 (E10y)", "0", "0",
            "summary idêntico ao E10_yolo26m_nwd → mesma trajetória"),
    det_row("E10 s1", "NWD w=0,5, 'seed 1'", "0.5", "E10_nwd_seed1", "33015 (E10y)", "1", "0 (bug)",
            "BUG DE SEED: summary idêntico ao seed0 (init_seeds antes do trainer, sobrescrito por args.seed=0)"),
    det_row("E10 s2", "NWD w=0,5, 'seed 2'", "0.5", "E10_nwd_seed2", "33015 (E10y)", "2", "0 (bug)",
            "BUG DE SEED: summary idêntico ao seed0"),
    det_row("E0 s0", "baseline CIoU, 'seed 0'", "0", "E0_baseline_seed0", "33015 (E10y)", "0", "0",
            "summary idêntico ao E0_yolo26m_baseline → reprodução determinística confirmada"),
    det_row("E0 s1", "baseline CIoU, 'seed 1'", "0", "E0_baseline_seed1", "33015 (E10y)", "1", "0 (bug)",
            "CRASHED no step 85 (best_fitness 0,6198 = idêntico ao s0 → mesma trajetória); métricas são da última época logada, não best.pt"),
    det_row("E0 s2", "baseline CIoU, 'seed 2'", "0", "E0_baseline_seed2", "33015 (E10y)", "2", "",
            "não rodou — cadeia 33015 abortou após crash do E0 s1"),
    det_row("E7", "yolo26l + NWD w=0,5", "0.5", "E7_yolo26l_nwd", "32947 e 33015", "default", "",
            "não rodou em nenhuma das duas cadeias (32947 abortou no E13; 33015 abortou no E0 s1)"),
    det_row("E14", "NWD w=0,5 + RFLA", "0.5", "E14_nwd_rfla", "33015 (E10y)", "default", "", "não rodou (cadeia abortada)"),
    det_row("TTA/WBF det.", "ensemble TTA+WBF dos detectores (VAL)", "", "tta_wbf_eval", "32947 e 33015", "", "",
            "não rodou; não confundir com E3_ensemble_tta (ensemble de classificadores no TEST)"),
    # results.csv (PAPER_FACTS §1) — não acessível aqui
    pf_row("E0", "baseline CIoU (yolo26m)", "0", "0.8565", "", "n/d", "best epoch = ep. 80 segundo R3 (coincide com best.pt)"),
    pf_row("E10", "NWD w=0,5, C=44", "0.5", "0.8706", "", "32933", "máximo por época (ep. 64 de 137, R3); +1,4 p.p. NÃO é claim (DECISIONS 2026-09-25)"),
    pf_row("E11", "NWD w=0,75", "0.75", "0.8642", "", "32947", ""),
    pf_row("E12", "NWD w=1,0 (sem CIoU)", "1.0", "0.6719", "", "32947", "colapso; R3: 32 épocas, best fitness ép. 2, máx. mAP50 ép. 10 (PAPER_FACTS diz 'early stop ép. 10')"),
    pf_row("E10b", "GCD w=0,5", "0.5", "0.8619", "0.6261", "32947", "melhor mAP50:95 no results.csv; +0,6 p.p. vs E0 (0,6198) — dentro do ruído"),
    # best.pt segundo R3 (leu results.csv) — registrar para não misturar
    pf_row("E0", "baseline CIoU (yolo26m)", "0", "0.8565", "0.6198", "n/d", "best.pt = ép. 80 (fitness)",
           protocolo="results_csv_bestpt", origem="docs/reviews/round-01/R3-rigor-stats.md W1"),
    pf_row("E10", "NWD w=0,5, C=44", "0.5", "0.8484", "0.6177", "32933",
           "best.pt = ép. 107 (fitness); sob regra consistente E10 fica −0,8 p.p. mAP50 vs E0; W&B summary do mesmo best.pt dá 0,8544 (passes de validação diferentes)",
           protocolo="results_csv_bestpt", origem="docs/reviews/round-01/R3-rigor-stats.md W1"),
]
NWD_HDR = ["exp", "variante", "loss_w", "seed_declarada", "seed_efetiva", "split", "protocolo", "mAP50", "mAP50_95",
           "precision", "recall", "best_fitness", "epochs_logged", "runtime_h", "wandb_run", "state", "job", "origem", "observacao"]
write("rq3_tiny_object/results/nwd_sweep_single_seed.csv", NWD_HDR, nwd_rows)


# --------------------------------------------------------------------------- helpers TEST/VAL
def tc_species(prefix):
    """prefix termina antes de 'species'. Retorna dict com micro P/R/F1, tp/fn/fp, matched, cond."""
    g = lambda k: TC.get(f"{prefix}/{k}")
    return dict(precision=g("species/micro/precision"), recall=g("species/micro/recall"), f1=g("species/micro/f1"),
                macro_f1=g("species/macro/f1"), tp=g("species/tp"), fn=g("species/fn"), fp=g("species/fp"),
                matched=g("species/matched"), support=g("species/support"),
                cond=g("species/classification_accuracy_conditional_matched"),
                mAP50=g("mAP50"), mAP50_95=g("mAP50_95"), species_mAP50=g("species_mAP50"))


def tc_per_class(prefix, cls):
    g = lambda k: TC.get(f"{prefix}/species/per_class/{cls}/{k}")
    return dict(precision=g("precision"), recall=g("recall"), f1=g("f1"), tp=g("tp"), fn=g("fn"), fp=g("fp"), support=g("support"))


def timing(prefix):
    return TC.get(f"{prefix}/timing/mean")


# --------------------------------------------------------------------------- 2
sahi_rows = []


def sahi_row(exp, variante, prefix, obs, f1_pf, origem_pf, tipo="detector", policy=""):
    m = tc_species(prefix)
    t = timing(prefix.split("/metrics")[0])
    sahi_rows.append(dict(
        exp=exp, variante=variante, tipo=tipo, politica=policy, split="TEST", protocolo="harness eval_test_campaign (score≥0,25, IoU≥0,5, 4 espécies micro)",
        precision=pct(m["precision"]), recall=pct(m["recall"]), f1=pct(m["f1"]), tp=m["tp"], fn=m["fn"], fp=m["fp"],
        n_gt_especies=m["support"], mAP50_harness_7cls=pct(m["mAP50"]), mAP50_95_harness_7cls=pct(m["mAP50_95"]),
        ms_img=num(t * 1000) if t else "", f1_paper_facts=f1_pf, origem_paper_facts=origem_pf,
        seed="default (0)" if tipo == "detector" else "ver classifiers_single_seed.csv",
        wandb_run=TC_RUN, origem="W&B summary test-campaign", observacao=obs))


sahi_row("E0", "YOLO26m baseline, foto inteira 1920", "detector/E0/metrics", "referência", "73,2", f"{PF} §4")
sahi_row("E1", "YOLO26m + cabeça P2, foto inteira", "detector/E1/metrics", "piorou (−5,7 p.p. F1)", "67,5", f"{PF} §4")
sahi_row("E2", "treino em tiles 512, aval. foto inteira 1920", "detector/E2/metrics", "igual ao baseline", "73,2", f"{PF} §4")
sahi_row("E2_sliced", "treino em tiles 512, inferência fatiada (SAHI: 512/stride 384/NMS 0,5/cap 300)", "detector/E2_sliced/metrics",
         "piorou: FPs (466 vs 285) — efeito de borda", "66,2", f"{PF} §4")
sahi_row("E3_ensemble", "ensemble 3 classificadores (swin_t 224 + convnext_t 384 + convnext_s) sobre propostas E0", "cascade/E3_ensemble/metrics/label_only",
         "ensemble de CLASSIFICADORES (não TTA/WBF do detector); escolhido após ver o TEST → exploratório", "", "", tipo="cascata", policy="label_only")
sahi_row("E3_ensemble", "idem, score produto", "cascade/E3_ensemble/metrics/product",
         "= 'Ensemble 3 classif.' do PAPER_FACTS §3 (R 75,5 / P 77,4 / F1 76,4 / 662 ms); exploratório", "76,4", f"{PF} §3", tipo="cascata", policy="product")
sahi_row("E3_ensemble_tta", "ensemble 3 classificadores + TTA", "cascade/E3_ensemble_tta/metrics/label_only",
         "PAPER_FACTS §4 'Ensemble + TTA 75,4–75,6' = label_only 75,4 / product 75,6; 2427 ms = 3,7× o ensemble sem TTA", "75,4", f"{PF} §4", tipo="cascata", policy="label_only")
sahi_row("E3_ensemble_tta", "idem, score produto", "cascade/E3_ensemble_tta/metrics/product", "exploratório", "75,6", f"{PF} §4", tipo="cascata", policy="product")
# E2_eval_sahi (run antigo, protocolo próprio)
s = RUNS["E2_eval_sahi"]["summary"]
for mode in ("plain", "sliced"):
    sahi_rows.append(dict(exp="E2_eval_sahi", variante=f"modelo E2 (tiles 512), modo {mode}", tipo="detector", split="n/d (não declarado no run)",
                          protocolo="avaliador SAHI legado (fly_det/sahi_inference.py) — NÃO comparável ao harness", precision=pct(s[f"{mode}/precision"]),
                          recall=pct(s[f"{mode}/recall"]), f1=pct(s[f"{mode}/f1"]), tp=s[f"{mode}/tp"], fn=s[f"{mode}/fn"], fp=s[f"{mode}/fp"],
                          mAP50_harness_7cls=pct(s[f"{mode}/map50"]), mAP50_95_harness_7cls=pct(s[f"{mode}/map50_95"]), seed="default (0)",
                          wandb_run=f"{FS}/E2_eval_sahi (v40xeurj)", origem="W&B summary", observacao="2026-09-06; classes/split não registrados no run; só para histórico"))
sahi_rows.append(dict(exp="TTA/WBF detector", variante="ensemble TTA+WBF dos detectores E0/E10/… (eval_wbf_tta.py)", tipo="detector", split="VAL",
                      protocolo="avaliador WBF próprio", wandb_run="", origem="W&B (ausência de run)", observacao="NUNCA RODOU (cadeias 32947 e 33015 abortaram antes da etapa 9)"))
SAHI_HDR = ["exp", "variante", "tipo", "politica", "split", "protocolo", "precision", "recall", "f1", "tp", "fn", "fp", "n_gt_especies",
            "mAP50_harness_7cls", "mAP50_95_harness_7cls", "ms_img", "seed", "f1_paper_facts", "origem_paper_facts", "wandb_run", "origem", "observacao"]
write("rq3_tiny_object/results/sahi_tta_single_seed.csv", SAHI_HDR, sahi_rows)


# --------------------------------------------------------------------------- 3
ct_rows = []


def add(pipeline, clf, policy, metrica, especie, valor, unidade, seed_clf, exploratorio, obs="", origem="W&B summary test-campaign", run=TC_RUN, pf=""):
    ct_rows.append(dict(pipeline=pipeline, classificador=clf, politica=policy, metrica=metrica, especie=especie, valor=valor, unidade=unidade,
                        split="TEST", n_gt_especies=918, seed_det="default (0)", seed_clf=seed_clf, exploratorio=exploratorio,
                        valor_paper_facts=pf, origem=origem, wandb_run=run, observacao=obs))


def pipeline_block(pipeline, clf, policy, prefix, seed_clf, exploratorio, pf_all=None, pf_sp=None, obs=""):
    m = tc_species(prefix)
    pf_all = pf_all or {}
    add(pipeline, clf, policy, "localizadas_e_especie_correta", "ALL", m["tp"], "contagem", seed_clf, exploratorio, obs, pf=pf_all.get("tp", ""))
    add(pipeline, clf, policy, "recall", "ALL", pct(m["recall"]), "%", seed_clf, exploratorio, obs, pf=pf_all.get("recall", ""))
    add(pipeline, clf, policy, "precision", "ALL", pct(m["precision"]), "%", seed_clf, exploratorio, obs, pf=pf_all.get("precision", ""))
    add(pipeline, clf, policy, "f1_micro", "ALL", pct(m["f1"]), "%", seed_clf, exploratorio, obs, pf=pf_all.get("f1", ""))
    add(pipeline, clf, policy, "f1_macro", "ALL", pct(m["macro_f1"]), "%", seed_clf, exploratorio, obs)
    add(pipeline, clf, policy, "acerto_especie_entre_localizadas", "ALL", pct(m["cond"]), "%", seed_clf, exploratorio,
        f"{m['tp']}/{m['matched']} localizadas (IoU≥0,5, score≥0,25). {obs}".strip(), pf=pf_all.get("cond", ""))
    add(pipeline, clf, policy, "localizadas_class_agnostic", "ALL", m["matched"], "contagem", seed_clf, exploratorio, obs)
    add(pipeline, clf, policy, "fp_especie", "ALL", m["fp"], "contagem", seed_clf, exploratorio, obs)
    add(pipeline, clf, policy, "mAP50_harness_7cls", "ALL", pct(m["mAP50"]), "%", seed_clf, exploratorio,
        f"harness eval_test_campaign, 7 classes; NÃO comparável ao results.csv. {obs}".strip(), pf=pf_all.get("mAP50", ""))
    add(pipeline, clf, policy, "mAP50_95_harness_7cls", "ALL", pct(m["mAP50_95"]), "%", seed_clf, exploratorio, obs)
    t = timing(prefix.split("/metrics")[0])
    add(pipeline, clf, policy, "ms_img", "ALL", num(t * 1000) if t else "", "ms", seed_clf, exploratorio,
        f"H100 FP32, imagem decodificada, média 55 imgs; cascata reclassifica todas as propostas. {obs}".strip(), pf=pf_all.get("ms", ""))
    for cls in SPECIES:
        c = tc_per_class(prefix, cls)
        p = (pf_sp or {}).get(cls, ("", ""))
        add(pipeline, clf, policy, "precision", cls, pct(c["precision"]), "%", seed_clf, exploratorio, obs, pf=p[0])
        add(pipeline, clf, policy, "recall", cls, pct(c["recall"]), "%", seed_clf, exploratorio, obs, pf=p[1])
        add(pipeline, clf, policy, "f1", cls, pct(c["f1"]), "%", seed_clf, exploratorio, obs)
        add(pipeline, clf, policy, "tp", cls, c["tp"], "contagem", seed_clf, exploratorio, f"suporte {c['support']}. {obs}".strip())


PF_E0 = dict(tp="695", recall="75,7", precision="70,9", f1="73,2", cond="77,4", mAP50="76,8", ms="25")
PF_PARTS = dict(tp="735", recall="80,1", precision="71,2", f1="75,3", cond="81,8", mAP50="72,1", ms="463")
PF_SP_E0 = {"MD": ("69,5", "54,3"), "MV": ("56,1", "83,6"), "MC": ("82,3", "85,3"), "MF": ("79,7", "78,3")}
PF_SP_PARTS = {"MD": ("66,8", "66,2"), "MV": ("60,8", "86,9"), "MC": ("87,6", "81,2"), "MF": ("73,5", "84,4")}
NIGHT = "cascade/architecture-night-v1-20260907/crops"
FOLLOW = "cascade/architecture-followup-v1-20260908"

pipeline_block("YOLO_E0", "—", "—", "detector/E0/metrics", "—", "false", PF_E0, PF_SP_E0,
               obs="single-stage; resultado principal PAPER_FACTS §3")
pipeline_block("YOLO_E0+ConvNeXt-T_partes", "A1_parts (ConvNeXt-T 384, atenção a partes)", "label_only",
               f"{NIGHT}/parts/metrics/label_only", "42", "false", PF_PARTS, PF_SP_PARTS,
               obs="resultado principal PAPER_FACTS §3; classificador escolhido na VAL antes do TEST")
pipeline_block("YOLO_E0+ConvNeXt-T_partes", "A1_parts", "product", f"{NIGHT}/parts/metrics/product", "42", "false",
               obs="política secundária pré-definida (score×max softmax)")
pipeline_block("YOLO_E0+ConvNeXt-T_CE", "A1_baseline (ConvNeXt-T 384, CE)", "label_only", f"{NIGHT}/baseline/metrics/label_only", "42", "false",
               obs="controle: cascata com CE simples — separa efeito 'cascatear' de 'atenção a partes' (R3 W6)")
pipeline_block("YOLO_E0+ConvNeXt-T_CE", "A1_baseline", "product", f"{NIGHT}/baseline/metrics/product", "42", "false")
pipeline_block("YOLO_E0+ConvNeXt-T_partes", "A2_parts_seed84", "label_only", f"{FOLLOW}/seed84/parts/metrics/label_only", "84", "false",
               obs="réplica seed 84 (PAPER_FACTS §3: 'F1 70,1–72,6 %')")
pipeline_block("YOLO_E0+ConvNeXt-T_partes", "A2_parts_seed84", "product", f"{FOLLOW}/seed84/parts/metrics/product", "84", "false",
               obs="réplica seed 84")
pipeline_block("YOLO_E0+ConvNeXt-T_CE", "A2_baseline_seed84", "label_only", f"{FOLLOW}/seed84/baseline/metrics/label_only", "84", "false",
               obs="réplica seed 84")
pipeline_block("YOLO_E0+ConvNeXt-T_CE", "A2_baseline_seed84", "product", f"{FOLLOW}/seed84/baseline/metrics/product", "84", "false",
               obs="réplica seed 84")
pipeline_block("YOLO_E0+ensemble3", "E3_ensemble (swin_t+convnext_t+convnext_s)", "product", "cascade/E3_ensemble/metrics/product",
               "n/d (E3 legado, seed não registrada)", "true",
               dict(recall="75,5", precision="77,4", f1="76,4", ms="662"), obs="ESCOLHIDO APÓS VER O TEST — exploratório, não usar como principal")
pipeline_block("YOLO_E0+ensemble3", "E3_ensemble", "label_only", "cascade/E3_ensemble/metrics/label_only", "n/d", "true", obs="exploratório")

# faixa seed 84 (PAPER_FACTS §3) — intervalo explícito
f1s = {k: TC.get(f"{FOLLOW}/seed84/{v}/metrics/{p}/species/micro/f1") for k, (v, p) in
       {"parts/label_only": ("parts", "label_only"), "parts/product": ("parts", "product"),
        "baseline/label_only": ("baseline", "label_only"), "baseline/product": ("baseline", "product")}.items()}
add("YOLO_E0+ConvNeXt-T_{partes,CE}", "A2_parts_seed84 + A2_baseline_seed84", "label_only e product", "f1_micro_intervalo", "ALL",
    f"{pct(min(f1s.values()), 1)}–{pct(max(f1s.values()), 1)}", "%", "84", "false",
    obs="intervalo = min–max sobre 2 classificadores × 2 políticas (" + "; ".join(f"{k}={pct(v, 1)}" for k, v in f1s.items()) + "); todos ABAIXO do YOLO E0 (73,2)",
    pf="70,1–72,6")
CT_HDR = ["pipeline", "classificador", "politica", "metrica", "especie", "valor", "unidade", "split", "n_gt_especies", "seed_det", "seed_clf",
          "exploratorio", "valor_paper_facts", "origem", "wandb_run", "observacao"]
write("rq2_cascade/results/cascade_test_single_seed.csv", CT_HDR, ct_rows)


# --------------------------------------------------------------------------- 4
cv_rows = []


def addv(pipeline, policy, metrica, valor, unidade, obs="", pf="", origem="W&B summary EVAL_E0_cascade_val", run=EV_RUN):
    cv_rows.append(dict(pipeline=pipeline, politica=policy, metrica=metrica, valor=valor, unidade=unidade, split="VAL", n_gt_especies=617,
                        n_gt_total=1649, n_imgs=35, job="32088", seed_det="default (0)", seed_clf="42" if pipeline != "YOLO_E0" else "—",
                        valor_paper_facts=pf, origem=origem, wandb_run=run, observacao=obs))


PFV = {"yolo": dict(mAP50="81,18", mAP50_95="60,12", correct="496", acc="80,39"),
       "parts_label_only": dict(mAP50="76,37", mAP50_95="56,91", correct="522", acc="84,60"),
       "parts_product": dict(mAP50="77,28", mAP50_95="57,60", correct="516", acc="83,63"),
       "arcface_label_only": dict(mAP50="75,68", mAP50_95="56,56"),
       "baseline_label_only": dict(mAP50="75,14", mAP50_95="55,68"), "baseline_product": dict(mAP50="76,44", mAP50_95="56,68"),
       "arcface_product": dict(mAP50="76,17", mAP50_95="56,96")}
NAMES = {"yolo": ("YOLO_E0", "—"), "baseline_label_only": ("YOLO_E0+ConvNeXt-T_CE", "label_only"), "baseline_product": ("YOLO_E0+ConvNeXt-T_CE", "product"),
         "parts_label_only": ("YOLO_E0+ConvNeXt-T_partes", "label_only"), "parts_product": ("YOLO_E0+ConvNeXt-T_partes", "product"),
         "arcface_label_only": ("YOLO_E0+ConvNeXt-T_ArcFace", "label_only"), "arcface_product": ("YOLO_E0+ConvNeXt-T_ArcFace", "product")}
for key, (pl, pol) in NAMES.items():
    g = lambda k: EV.get(f"{key}/{k}")
    pf = PFV.get(key, {})
    addv(pl, pol, "mAP50_harness_7cls", pct(g("mAP50")), "%", "harness eval_yolo_cascade, 7 classes, conf≥0,001, max_det 300", pf.get("mAP50", ""))
    addv(pl, pol, "mAP50_95_harness_7cls", pct(g("mAP50_95")), "%", "", pf.get("mAP50_95", ""))
    addv(pl, pol, "mAP50_harness_4especies", pct(g("species_mAP50")), "%", "média das 4 espécies")
    addv(pl, pol, "especies_corretas", g("species_correct"), "contagem", "class-agnostic IoU≥0,5, score≥0,25", pf.get("correct", ""))
    addv(pl, pol, "especies_pareadas", g("species_matched"), "contagem", "propostas pareadas a GT de espécie")
    addv(pl, pol, "acerto_sobre_617_gt", pct(g("species_accuracy_all_gt_including_misses")), "%", "inclui misses", pf.get("acc", ""))
    addv(pl, pol, "acerto_entre_pareadas", pct(g("species_classification_accuracy_conditional_matched")), "%")
# latência (VAL, ms/img)
for key, pl in (("detector", "YOLO_E0"), ("baseline", "YOLO_E0+ConvNeXt-T_CE"), ("parts", "YOLO_E0+ConvNeXt-T_partes"), ("arcface", "YOLO_E0+ConvNeXt-T_ArcFace")):
    for stat, pf in (("mean", {"detector": "22,88", "parts": "416,53", "baseline": "431,39", "arcface": "415,18"}[key]), ("p50", ""), ("p95", "")):
        v = EV.get(f"evaluation.latency.{key}.decoded_image_e2e_seconds.{stat}")
        addv(pl, "—", f"ms_img_{stat}", num(v * 1000) if v else "", "ms", "H100 FP32, imagem decodificada, e2e detector(+classificador)", pf)
# cobertura (só no .md)
addv("propostas_YOLO_E0", "—", "cobertura_class_agnostic_gt", "94.91", "%", "conf≥0,001, IoU≥0,5; 4.197 propostas; NÃO está no summary W&B", "94,91",
     origem="docs/archive/CASCADE_RESULTS_20260909.md (report.md do job 32088, não acessível aqui)")
addv("propostas_YOLO_E0", "—", "cobertura_class_agnostic_especies", "97.89", "%", "idem", "97,89",
     origem="docs/archive/CASCADE_RESULTS_20260909.md")
addv("propostas_YOLO_E0", "—", "n_propostas", "4197", "contagem", "", "4.197", origem="docs/archive/CASCADE_RESULTS_20260909.md")
CV_HDR = ["pipeline", "politica", "metrica", "valor", "unidade", "split", "n_gt_especies", "n_gt_total", "n_imgs", "job", "seed_det", "seed_clf",
          "valor_paper_facts", "origem", "wandb_run", "observacao"]
write("rq2_cascade/results/cascade_val_single_seed.csv", CV_HDR, cv_rows)


# --------------------------------------------------------------------------- 5
cl_rows = []


def clf(label, train_run, tc_key, seed, f1_pf, obs="", gt_only=False, roi=False):
    r = RUNS.get(train_run)
    s = r["summary"] if r else {}
    c = r["config"] if r else {}
    a = c.get("args", {}) if isinstance(c.get("args"), dict) else {}
    arch = c.get("architecture", {}) if isinstance(c.get("architecture"), dict) else {}
    rt = s.get("_runtime")
    row = dict(classificador=label, wandb_run_treino=f"{FS}/{train_run} ({r['id']})" if r else "", state=r["state"] if r else "",
               data=r["created"][:10] if r else "", seed=seed, img_size=a.get("img_size", c.get("img_size", "")),
               backbone=arch.get("backbone", c.get("model", "")), variante=arch.get("variant", ""),
               epochs_config=a.get("epochs", c.get("epochs", "")), epochs_done=s.get("epoch", ""), best_epoch=s.get("best/epoch", ""),
               max_hours=a.get("max_hours", ""), runtime_h=num(rt / 3600, 2) if rt else "",
               val_species_macro_f1=pct(s.get("best/metrics.species_macro_f1")), val_species_acc=pct(s.get("best/metrics.species_accuracy")),
               val_acc_all7_legacy=pct(s.get("val_acc")), test_acc_all7_legacy=pct(s.get("test_acc")),
               f1_paper_facts=f1_pf, wandb_run_eval=TC_RUN, observacao=obs)
    if tc_key:
        if roi:
            g = tc_species(f"gt_roi/{tc_key}/metrics")
            row.update(test_gtcrop_species_f1=pct(g["f1"]), test_gtcrop_species_acc=pct(g["cond"]))
        else:
            g = tc_species(f"gt_crop/{tc_key}/metrics")
            row.update(test_gtcrop_species_f1=pct(g["f1"]), test_gtcrop_species_acc=pct(g["cond"]),
                       test_gtcrop_acc_all7=pct(TC.get(f"gt_crop/{tc_key}/metrics/crop_accuracy_all7")))
            if not gt_only:
                lo = tc_species(f"cascade/{tc_key}/metrics/label_only")
                pr = tc_species(f"cascade/{tc_key}/metrics/product")
                t = timing(f"cascade/{tc_key}")
                row.update(test_cascade_label_only_P=pct(lo["precision"]), test_cascade_label_only_R=pct(lo["recall"]), test_cascade_label_only_F1=pct(lo["f1"]),
                           test_cascade_product_P=pct(pr["precision"]), test_cascade_product_R=pct(pr["recall"]), test_cascade_product_F1=pct(pr["f1"]),
                           test_cascade_mAP50_label_only=pct(lo["mAP50"]), test_cascade_ms_img=num(t * 1000) if t else "")
    cl_rows.append(row)


LEG = "E3–E5 legado: seed não registrada no run; treino train_classifier.py"
clf("Swin-T 224 (E3)", "E3_swin", "E3_swin", "n/d", "74,4", LEG)
clf("ConvNeXt-T 384 CE (E3b)", "E3b_convnext", "E3b_convnext", "n/d", "76,0", LEG + "; 'melhor classificador individual' PAPER_FACTS §4")
clf("ConvNeXt-T 384 focal+cw (E3c)", "E3c_focal", "E3c_focal", "n/d", "", LEG + "; 'focal etc.'")
clf("ConvNeXt-S 384 focal+cw (E3d)", "E3d_convnext_s", "E3d_convnext_s", "n/d", "72,4", LEG)
clf("ConvNeXt-T 384 SAM-limpo (E4)", "E4_convnext_sam", "E4_convnext_sam", "n/d", "71,8",
    LEG + "; só GT-crops (diagnóstico); PAPER_FACTS '71,8 vs ≈79' = gt_crop species F1 vs E3b/E3c gt_crop", gt_only=True)
clf("ConvNeXt-T 512 focal+cw (E5)", "E5_convnext_512", "E5_convnext_512", "n/d", "70,3", LEG)
clf("Ensemble E3 (swin_t+convnext_t+convnext_s)", "E3_ensemble_tta", "E3_ensemble", "n/d", "76,4 (product)", "run E3_ensemble_tta é só avaliação (2 s); exploratório")
clf("Ensemble E3 + TTA", "E3_ensemble_tta", "E3_ensemble_tta", "n/d", "75,4–75,6", "exploratório; 3,7× custo do ensemble")
N1 = "architecture-night-v1-20260907/crops"
F1_ = "architecture-followup-v1-20260908"
A = "campanha architecture-night (job 32005), snapshot imutável, 24 ép./patience 5, budget 1,25 h"
clf("ConvNeXt-T 384 CE (A1 baseline)", "A1_baseline_architecture-night-v1-20260907", f"{N1}/baseline", 42, "", A + "; controle da cascata")
clf("ConvNeXt-T 384 SupCon (A1)", "A1_supcon_architecture-night-v1-20260907", f"{N1}/supcon", 42, "72,0–73,3", A + "; best epoch 1 (6 ép. logadas)")
clf("ConvNeXt-T 384 ArcFace (A1)", "A1_arcface_architecture-night-v1-20260907", f"{N1}/arcface", 42, "", A)
clf("ConvNeXt-T 384 atenção a partes (A1) — PRINCIPAL", "A1_parts_architecture-night-v1-20260907", f"{N1}/parts", 42, "75,3",
    A + "; recebeu extensão de 1,5 h (resume=True); selecionado na VAL por species_macro_f1")
clf("ConvNeXt-T 384 hires stem/2 (A1)", "A1_hires_architecture-night-v1-20260907", f"{N1}/hires", 42, "", A)
clf("RoI head sobre features YOLO E0 congelado (A1 roi)", "A1_roi_architecture-night-v1-20260907", "original", 42, "—",
    "GT-RoI, população 2.864; protocolo distinto; runtime W&B 50 s (treino em cache); 'pior que classificadores dedicados'", roi=True)
B = "followup (job 32009), budget 1 h"
clf("parts+SupCon (A2 combo)", "A2_parts_supcon_seed42_architecture-followup-v1-20260908", f"{F1_}/combos/parts_supcon", 42, "", B)
clf("hires+parts (A2 combo)", "A2_hires_parts_seed42_architecture-followup-v1-20260908", f"{F1_}/combos/hires_parts", 42, "", B)
clf("hires+SupCon (A2 combo)", "A2_hires_supcon_seed42_architecture-followup-v1-20260908", f"{F1_}/combos/hires_supcon", 42, "", B)
clf("ConvNeXt-T 384 CE seed 84 (A2)", "A2_baseline_seed84_architecture-followup-v1-20260908", f"{F1_}/seed84/baseline", 84, "70,1–72,6 (faixa)", "réplica fresca seed 84, budget 1,25 h")
clf("ConvNeXt-T 384 partes seed 84 (A2)", "A2_parts_seed84_architecture-followup-v1-20260908", f"{F1_}/seed84/parts", 84, "70,1–72,6 (faixa)", "réplica fresca seed 84, budget 1,25 h")
CL_HDR = ["classificador", "wandb_run_treino", "state", "data", "seed", "img_size", "backbone", "variante", "epochs_config", "epochs_done", "best_epoch", "max_hours",
          "runtime_h", "val_species_macro_f1", "val_species_acc", "val_acc_all7_legacy", "test_acc_all7_legacy", "test_gtcrop_species_f1", "test_gtcrop_species_acc",
          "test_gtcrop_acc_all7", "test_cascade_label_only_P", "test_cascade_label_only_R", "test_cascade_label_only_F1", "test_cascade_product_P",
          "test_cascade_product_R", "test_cascade_product_F1", "test_cascade_mAP50_label_only", "test_cascade_ms_img", "f1_paper_facts", "wandb_run_eval", "observacao"]
write("rq2_cascade/results/classifiers_single_seed.csv", CL_HDR, cl_rows)

# --------------------------------------------------------------------------- GPU-h
tot = {"finished": 0.0, "crashed": 0.0, "failed": 0.0}
det = 0.0
for n, r in RUNS.items():
    rt = r["summary"].get("_runtime") or 0
    tot[r["state"]] = tot.get(r["state"], 0) + rt
    if r["job_type"] == "train" and n.startswith(("E0", "E1", "E2_yolo", "E10", "E13")):
        det += rt
print("GPU-s por estado:", {k: round(v) for k, v in tot.items()}, "→ total h:", round(sum(tot.values()) / 3600, 1),
      "| detectores YOLO h:", round(det / 3600, 1))
