# runs — rq1_data

Registro job → run W&B (escrito pelo `slurm-runner`; coluna "usado em" pelo `analyst`).

| Data | Job | Script | Commit (repo / baseline) | Seed | Variante | Run W&B (pestline/<projeto>/<nome>) | Status | Usado em |
|---|---|---|---|---|---|---|---|---|
| 2026-10-13 | 34630 | rq1_data/slurm/chain_rq1_ladder.sh | (a preencher) | 0,1,2 | RQ1 L0/L1/L2/L2ORG (12 runs) | pestline/paper2-rq1-data/RQ1_<ARM>_s<seed> | concluído 2026-10-14 (12/12 DONE, RQ1_CHAIN_DONE) | commit desta etapa |
| 2026-10-19 | 34950 | rq1_data/slurm/chain_rq1_equalsize_h100.sh | 068d1b4 | 2 / 0,1,2 | RQ1 controle de mesmo tamanho EQ869_s2, EQ103_s0–2 (h100n2, jail GPU) | pestline/paper2-rq1-data/EQ<N>_s<seed>-h100 | concluído 2026-10-19 (4/4 DONE) | PAPER_FACTS §11b |
