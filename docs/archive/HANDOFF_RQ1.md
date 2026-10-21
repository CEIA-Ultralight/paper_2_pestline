# HANDOFF RQ1 (escada de dados) — retomar daqui

Autorização do autor (2026-10-13): terminar a RQ1 de ponta a ponta (treino, análise, PAPER_FACTS, manuscrito RQ1), em loop, só parar ao final. W&B só `paper2-rq1-data`. Nunca `git push`; commits locais em PT sem atribuição a IA.

## Design
Ver `docs/DECISIONS.md` (2026-10-13) e `rq1_data/results/rq1_versions_summary.md`. Braços L0/L1/L2/L2ORG em `/raid/user_marcospaulo/datasets/rq1_ladder/<ARM>` (gerados por `rq1_data/make_ladder.py`); L3 = `E0_s{0,1,2}` em `/raid/user_marcospaulo/flydet_runs_v9`.
Hipóteses: H1 recall sobe onde as coletas miram (bins de tamanho √wh, placas densas, FP de espécie em placas só-INS); H2 confusão MC↔MV/MF↔MC não muda.

## Jobs
- Cadeia: `sbatch rq1_data/slurm/chain_rq1_ladder.sh` (12 treinos seed-major, idempotente, requeue). RUNS_ROOT `/raid/user_marcospaulo/flydet_runs_rq1`; marcador final `RQ1_CHAIN_DONE`.
- Regra: no máx. 1 job meu RUNNING. Ver `squeue -u $USER`; se o job sumiu sem `RQ1_CHAIN_DONE`, reenviar o mesmo sbatch (retoma).
- Registrar job→run em `rq1_data/results/runs.md`.

## Próximos passos (após a cadeia)
1. `common/analysis/rq1_ladder_analysis.py` → `rq1_data/results/ablation_ladder.csv`, `rq1_confusion.csv`, `dataset_stats.csv` (por `test_eval/per_instance.csv`; bootstrap IC por imagem, McNemar).
2. Linhas em `docs/PAPER_FACTS.md`; atualizar `rq1_data/README.md`, `docs/LEDGER.md`.
3. Figura de confusão L0 vs L3 + figura das coletas.
4. `manuscript/sec/3_dataset.tex` e `5_experiments.tex` §RQ1 (tab:rq1_mapping, tab:rq1_ablation, fig:rq1_confusion); tectonic; commit no submodule e `git add manuscript`. Não alterar tese sem o autor: sinalizar.
## Pendências abertas
607 vs 869 imagens copy-paste (PAPER_FACTS vs nomes `-instance`); "graduated-density" vs "graduada"; remoção do DS-ORG na v9.0.5 sem justificativa documentada; acrônimo MF (flesh fly no PACBB vs fruit fly aqui).

## ESTADO FINAL (2026-10-14) — RQ1 CONCLUÍDA
- Job 34630 terminou (12/12 DONE, RQ1_CHAIN_DONE). CSVs em `rq1_data/results/` (gerador `common/analysis/rq1_ladder_analysis.py`); fatos em `docs/PAPER_FACTS.md` §11; LEDGER e runs.md atualizados.
- Manuscrito: §RQ1 de `manuscript/sec/5_experiments.tex` reescrita (escada L0–L3, tabela, figura `figs/rq1_confusion.pdf` via `common/analysis/rq1_confusion_fig.py`). Compila.
- Pendências para o AUTOR: ver lista no relatório final (STID "subconjunto próprio" em §3, definição de "graduada", DS-ORG removido na v9.0.5, sigla MF, framing "alavancas", IC das taxas de confusão, controle de subamostra de L2 não rodado, tabela de lotes em §3 e contagem de págs).
