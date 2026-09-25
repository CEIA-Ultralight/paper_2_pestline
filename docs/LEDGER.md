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
