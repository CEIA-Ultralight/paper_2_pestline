# LEDGER — histórico do paper 2 (VISAPP 2027)

Append-only. Escrito pelo `orchestrator` (ciclos de RQ) e pelo `area-chair` (rodadas de review).
Fixes têm ID `F{rodada}-{k}` e status `open` → `fixed@<hash manuscript>` | `wontfix (motivo)` | `superseded by F..`.

## Rodadas de review

| Rodada | Data | Manuscript | Venue-alvo | R1 | R2 | R3 | R4 | R5 | Média | Decisão | Fix nº 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 (legado CVPR) | 2026-09-22 | cvpr2027 draft | CVPR 2027 | 2 | 2 | 2 | 2 | 2 | 2,0 | Reject | multi-seed + seed bug; reviews em `reviews/round-01/` (personas antigas) |

## Fixes abertos

| ID | Fix | Origem | Status |
|---|---|---|---|
| F01-01 | Corrigir bug de seed em `train_nwd.py` e rerodar 3×3 E0 vs E10 | round-01 META (CVPR) | fixed (seed via `--yolo-config`; 3×3 B200 cadeia A 33321; manuscript a235828) |
| F01-02 | Multi-seed do classificador da cascata + McNemar no TEST | round-01 META | fixed (cadeias B 33322 / D 33364; manuscript a235828) |
| F01-03 | Tabela de mapeamento de classes STID ↔ DS-F2 (colisão de sigla MF) | round-01 R4/R5 | open |
| F01-04 | Não misturar mAP do `results.csv` com mAP do harness na mesma tabela | round-01 R3 | open |
| F01-05 | Remover "same setting/partner" e placeholders de anonimato | round-01 R5 | open (manuscrito reescrito para VISAPP — reverificar) |

## Ciclos de RQ

| Data | RQ | O que fechou | Evidência (results/) | Registrado por |
|---|---|---|---|---|
| 2026-09-25 | — | Repo criado; tese e RQs definidas; abstract + Sec. 1–2 escritos (manuscript @ 2466fe0) | — | orchestrator (sessão inicial) |
| 2026-09-26 | RQ2, RQ3 | Inventário W&B (129 runs, 8 projetos); cadeia 33015 declarada **inválida** (seed bug confirmado: seeds idênticos; E0 s1 crashed; s2/E7/E14/TTA não rodaram); "+1,4 pp" NWD contradito sob protocolo único (E10 0,8544 < E0 0,8605 W&B). Analyst exportou single-seed → CSVs; PAPER_FACTS §1b/§6/§8. Writer escreveu Sec. 4 e 5 (RQ1 em aberto), tudo marcado single-seed/preliminary (manuscript @ 77459a1, 12 pp., tectonic). Infra: workspace em dgx-B200-1, dados/checkpoints/.sif no /raid de dgx-H100-02 (h100n2). | `rq3_tiny_object/results/{nwd_sweep,sahi_tta}_single_seed.csv`, `rq2_cascade/results/{cascade_test,cascade_val,classifiers}_single_seed.csv`, `*/results/runs.md` | orchestrator |
| 2026-09-30 | RQ2, RQ3 | Campanha B200 multi-seed (cadeias A 33321, B 33322, C 33363 [falhou só no WBF ensemble; resubmetida 33620 com E7 yolo26l], D 33364). RQ3: NWD/GCD/P2/RFLA sem ganho (E0 72,4±2,9 macro-F1 TEST). RQ2: cascata R +3,0 / P −2,2 / F1 +0,1; oráculo 79,6 % = teto; MD→MV não muda (CE) → **tese contradita**, `\TODO{AUTHOR DECISION}` no manuscrito. Writer: Sec. 4/5 com B200 (manuscript a235828, 16 pp). Scholar: revisão RQ2 + ~28 refs no main.bib. Agentes com modelo por tier. | `rq2_cascade/results/{cascade_*_seeds,mcnemar,confusion_single_vs_cascade,oracle_crops,latency,policy_comparison,ap_per_class}.csv`, `rq3_tiny_object/results/{nwd_3x3,tiny_variants_*,tiny_ablation_single_seed,recall_by_size,sahi_multiseed}.csv`, PAPER_FACTS §9 | orchestrator |
| 2026-10-06 | RQ2 | Campanha v9 (DS-F2_v9.0.5): cascata não melhora o detector em nenhuma das 5 backbones × 3 detectores (ΔF1 TEST −1,1 a −5,1; precisão cai, recall ≈ igual); abstenção por confiança nunca supera o detector; oráculo ≈ 83 % (teto), 69 % dos erros com confiança ≥ 0,8, acurácia de família MD/MV/MC 96 % vs espécie 81 %. Análises CPU geradas; BG multi-seed e curva de aprendizado em job 34026 (pendentes). Seção RQ2 escrita com `\TODO` nos pendentes. | `rq2_cascade/results/v9_*.csv`, PAPER_FACTS §10 | orchestrator |
| 2026-10-06 | RQ2 | v9: cascata não supera o detector em 5 backbones × 3 detectores (ΔF1 −1,1 a −5,1, TEST); oráculo ≈ 83%; 4,9% das instâncias erradas por todos os 15 clf, 69% dos erros com confiança ≥0,8, 79% dos erros dentro de MD/MV/MC; rótulos suspeitos 3,7%. BG multi-seed e curva de aprendizado em andamento (job 34026) | `rq2_cascade/results/v9_*.csv`; PAPER_FACTS §10 | claude (a pedido do autor) |
| 2026-10-08 | RQ2 | Fechamento v9: BG ×3 seeds (TEST ΔF1 −2,75, BCa [−4,78; −0,44], oráculo 77,6±2,8, sem ganho); curva de aprendizado 25/50/100 % (TEST 83,0/81,7/82,6, platô → compatível com ambiguidade, single-seed); subseção RQ2 completa no manuscrito | `rq2_cascade/results/v9_learning_curve.csv`, `v9_oracle_bg.csv`, `v9_bootstrap_delta_f1.csv` | orchestrator |
