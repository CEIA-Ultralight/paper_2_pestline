# common — compartilhado por todas as RQs

| Arquivo | Origem (fly-det) | Função |
|---|---|---|
| `eval_test_campaign.py` | `scripts/` | Métricas por espécie no TEST (recall / precisão / F1, 4 espécies, matching IoU≥0,5) — **o harness oficial do paper** |
| `slurm/eval_test_campaign.sh` | `slurm/` | sbatch do harness |
| `slurm/E0_yolo26m_baseline.sh` | `slurm/` | Baseline YOLO26m (150 ép., batch 8, imgsz 1920, patience 30) — a referência de todas as RQs |
| `log_confusion_wandb.py`, `slurm/log_confusion.sh` | `scripts/`, `slurm/` | Matriz de confusão para o W&B |
| `wandb_run_exists.py` | `scripts/` | Idempotência de jobs (não repetir run já existente) |
| `statistical_sample.py`, `agreement_analysis.py`, `plot_agreement_report.py` | `scripts/` | Amostragem e concordância entre anotadores/modelos |
| `containers/flydet-train.def` | `containers/` | Definição do Apptainer `flydet-train.sif` (Ultralytics 8.4.142, torch 2.6.0+cu124) |

## Falta

- [ ] `stats.py` — média ± std multi-seed, McNemar pareado, IC bootstrap (usado por RQ1/RQ2/RQ3).
- [ ] `wandb_export.py` — run W&B → CSV em `rqN/results/`.
- [ ] `slurm/_header.sh` — cabeçalho comum (`--partition=h100n2`, `--signal=B:SIGUSR1@300`, `source wandb.env`,
      `apptainer exec`, resume) para os sbatch das RQs pararem de repetir boilerplate.
- [ ] `datasets.yaml` — caminhos canônicos em `/raid` (DS-F2_v8.1.1, subset STID, TEST fixo).
