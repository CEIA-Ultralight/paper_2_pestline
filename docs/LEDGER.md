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
| F01-01 | Corrigir bug de seed em `train_nwd.py` e rerodar 3×3 E0 vs E10 | round-01 META (CVPR) | open |
| F01-02 | Multi-seed do classificador da cascata + McNemar no TEST | round-01 META | open |
| F01-03 | Tabela de mapeamento de classes STID ↔ DS-F2 (colisão de sigla MF) | round-01 R4/R5 | open |
| F01-04 | Não misturar mAP do `results.csv` com mAP do harness na mesma tabela | round-01 R3 | open |
| F01-05 | Remover "same setting/partner" e placeholders de anonimato | round-01 R5 | open (manuscrito reescrito para VISAPP — reverificar) |

## Ciclos de RQ

| Data | RQ | O que fechou | Evidência (results/) | Registrado por |
|---|---|---|---|---|
| 2026-09-25 | — | Repo criado; tese e RQs definidas; abstract + Sec. 1–2 escritos (manuscript @ 2466fe0) | — | orchestrator (sessão inicial) |
| 2026-09-26 | RQ2, RQ3 | Inventário W&B (129 runs, 8 projetos); cadeia 33015 declarada **inválida** (seed bug confirmado: seeds idênticos; E0 s1 crashed; s2/E7/E14/TTA não rodaram); "+1,4 pp" NWD contradito sob protocolo único (E10 0,8544 < E0 0,8605 W&B). Analyst exportou single-seed → CSVs; PAPER_FACTS §1b/§6/§8. Writer escreveu Sec. 4 e 5 (RQ1 em aberto), tudo marcado single-seed/preliminary (manuscript @ 77459a1, 12 pp., tectonic). Infra: workspace em dgx-B200-1, dados/checkpoints/.sif no /raid de dgx-H100-02 (h100n2). | `rq3_tiny_object/results/{nwd_sweep,sahi_tta}_single_seed.csv`, `rq2_cascade/results/{cascade_test,cascade_val,classifiers}_single_seed.csv`, `*/results/runs.md` | orchestrator |
