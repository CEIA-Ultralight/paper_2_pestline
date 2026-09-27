# RQ2 — Arquitetura: cascata detector → classificador fine-grained

**Pergunta.** Desacoplar detecção de identificação — YOLO26m como detector (propostas) seguido de um
classificador dedicado sobre os crops — supera o single-stage em **recall, precisão e F1 por espécie**?
A que custo (mAP, latência, estabilidade entre seeds)? A confusão MC↔MV / MF↔BF tida como "irredutível" no
PACBB se reduz?

**Hipótese.** Uma rede que precisa localizar um inseto de ~40 px e discriminar espécie ao mesmo tempo compromete
a representação; um classificador dedicado sobre o crop recupera a identificação — mas re-rotular sem re-ranquear
reduz AP.

## O que já temos (single-seed, seed 42 salvo indicação)

> **Exportado em 2026-09-26** (analyst): `results/cascade_test_single_seed.csv`, `cascade_val_single_seed.csv`,
> `classifiers_single_seed.csv`, `runs.md`. Achados: o resultado principal é a política **label_only**; o controle CE
> seed 42 (F1 74,95) ≈ partes (75,35) → o ganho vem de **cascatear**, não da cabeça de partes; partes seed 84
> label_only dá P 63,7 / R 77,9 / **F1 70,1 (abaixo do E0)**. Seções 4.2/4.4 e 5.2 escritas (manuscript 77459a1).

| Item | Onde | Valor |
|---|---|---|
| Extração de crops (GT e propostas do YOLO) | `extract_crops.py`, `extract_proposal_crops.py`, `slurm/extract_proposal_crops.sh` | propostas cobrem 94,9 % dos GT / 97,9 % das espécies (VAL) |
| Treino de classificadores (Swin-T, ConvNeXt-T/S, focal, SupCon, ArcFace, "atenção a partes") | `train_classifier.py`, `architecture_lab/{models,train,campaign,followup,diagnose,roi}.py`, `slurm/E3*.sh E4*.sh E5*.sh` | melhor individual: **ConvNeXt-T 384 partes** (F1 76,0 % TEST) |
| Avaliação da cascata (harness próprio, 7 classes) | `eval_yolo_cascade.py`, `slurm/eval_yolo_cascade.sh`, `val_cascade_study.py`, `slurm/val_study.sh` | VAL: YOLO 80,4 % → cascata 84,6 % espécie correta; mAP50 81,2 → 77,3 |
| Políticas de fusão (só classe / score produto / limiar) | `policy_study.py`, `slurm/policy_study.sh` | score produto ligeiramente melhor que só classe |
| Cabeças alternativas / RoI sobre features YOLO | `test_campaign_heads.py`, `test_campaign_roi.py` | RoI head pior que classificador dedicado |
| Limpeza de fundo com SAM | `sam_clean_crops.py`, `slurm/E4_sam_clean.sh` | **piorou** (71,8 % vs ≈79 %) |
| DINOv2 como backbone | `slurm/dinov2.sh` | ver `docs/ARCHITECTURE_NIGHT.md` |
| Cabeça P2 no YOLO (alternativa single-stage) | `slurm/E1_yolo26m_p2.sh` | **piorou** (F1 67,5 %) |
| Orquestração noturna / monitor | `architecture_lab/campaign.py`, `monitor_*.py`, `slurm/architecture_*.sh`, `slurm/followup*.sh` | histórico em `docs/ARCHITECTURE_NIGHT.md` |
| **Resultado principal (TEST, 918 espécies)** | `docs/PAPER_FACTS.md §3`, `docs/CASCADE_RESULTS_20260909.md` | recall 75,7 → **80,1 %**; precisão 70,9 → 71,2 %; F1 73,2 → **75,3 %**; mAP50 76,8 → 72,1 % (−4,7); 25 → 463 ms/img (≈18×) |
| Por espécie (P/R %) | idem | MD 69,5/54,3 → 66,8/**66,2**; MV 56,1/83,6 → 60,8/86,9; MC 82,3/85,3 → 87,6/**81,2** (recall cai); MF 79,7/78,3 → 73,5/**84,4** |
| Seed 84 (baseline e partes) | `docs/RELATORIO_TESTE_20260909*.md` (no fly-det) | F1 70,1–72,6 % — **abaixo** da seed 42; sinal do Δ F1 pode inverter |
| Ensemble 3 classif. + TTA (exploratório, escolhido após ver o TEST) | `docs/PAPER_FACTS.md §3–4` | F1 76,4 %, 662 ms/img — **não usar como resultado principal** |
| Testes CPU | `../tests/test_yolo_cascade.py`, `test_policy_study.py`, `test_test_campaign_heads.py`, `test_extract_proposal_crops.py` | |

## O que falta

- [ ] **Multi-seed do classificador** (≥3 seeds: 42, 84 + 1) com o mesmo detector E0, avaliado no TEST →
      média ± std de recall/precisão/F1 por espécie. É o pré-requisito da contribuição 1.
- [ ] **Multi-seed do detector** na cascata (usar os E0 seeds 0–2 do RQ3) — cruzamento detector × classificador.
- [ ] **McNemar pareado** (por GT de espécie: acertou/errou no single-stage vs cascata) → `results/mcnemar.csv`.
- [ ] **Matriz de confusão** single-stage vs cascata no TEST (4 espécies + INS/NOISE), com foco em MC↔MV e MF↔MC
      — é a evidência direta contra o claim de irredutibilidade. Script existente: `../common/log_confusion_wandb.py`.
- [ ] Reconciliar protocolos: mAP50 do harness (76,8 %) **não é comparável** ao `results.csv` do Ultralytics
      (85,65 % VAL) — a tabela do paper usa um só protocolo, com nota.
- [ ] Declarar seleção de modelo: classificador escolhido na VAL antes de olhar o TEST (limpo); TEST já
      consultado → chamar de "consolidado", não "cego". Verificar se existe holdout nunca consultado.
- [ ] Custo: registrar GPU-horas e latência por estágio → `results/latency.csv`.
- [ ] Figuras: `fig_cascade_tradeoff.pdf` (recall/F1 ↑ vs mAP ↓ por seed), `fig_confusion.pdf`, exemplos
      qualitativos MC↔MV corrigidos pela cascata.

## Saídas esperadas em `results/`

- `cascade_test_seeds.csv` — seed_det, seed_clf, classificador, recall/precisão/F1 por espécie, mAP50, ms/img.
- `confusion_single_vs_cascade.csv`, `mcnemar.csv`, `latency.csv`.
