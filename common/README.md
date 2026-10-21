# common — código compartilhado pelas RQs

| Caminho | Função |
|---|---|
| `eval_test_campaign.py`, `slurm/eval_test_campaign.sh` | Harness oficial de avaliação por espécie no TEST (recall, precisão, F1, matching IoU ≥ 0,5) |
| `stats.py` | Média ± desvio entre seeds, McNemar pareado, bootstrap, confusão e recall por faixa de tamanho |
| `statistical_sample.py`, `agreement_analysis.py`, `plot_agreement_report.py` | Amostragem e concordância entre modelos/anotadores |
| `log_confusion_wandb.py`, `slurm/log_confusion.sh`, `wandb_run_exists.py` | Matriz de confusão no W&B; idempotência de jobs |
| `monitor_v9.sh` | Monitor de cadeias da campanha v9 |
| `slurm/b200_common.sh` | Ambiente comum dos jobs (caminhos, caches em `/raid`, W&B, `chain_run`, `step_done`) |
| `slurm/train_detector_b200.sh`, `slurm/E0_yolo26m_baseline.sh` | Receita E0 do detector (150 ép., batch 8, imgsz 1920, patience 30) |
| `slurm/build_container_b200.sh`, `slurm/smoke_b200.sh`, `containers/*.def` | Definição e build do container Apptainer (Ultralytics 8.4.142) |
| `gpu/gpu_watch.py`, `gpu/gpu_audit.py`, `gpu/slurm/gpu_env.sh` | Isolamento de GPU por UUID e auditoria dos processos do job |
| `analysis/` | Geradores de CSV e figuras, um por tabela do artigo (ver READMEs das RQs) |
