---
paths:
  - "common/**"
  - "rq*/**/*.py"
  - "rq*/slurm/**"
  - "tests/**"
---
# ML (treino/avaliação de detectores e classificadores)

Antes de qualquer job caro (um treino E0 custa 0,6–1,5 GPU-h; cadeias custam dezenas):
- Smoke test em CPU (`tests/`) e, se precisar de GPU, um job curto via `sbatch` — nunca no login node.
- Conferir splits do DS-F2_v8.1.1 (train 1.259 / val 35 / test 55), índices de classe 0–6 e `imgsz` 1920.
- Conferir `FLYDET_SEED`, `MODEL_CFG`, caminho do container B200 e `RUNS_ROOT` no script antes de submeter.
- Nunca usar TEST para treino, seleção de `best.pt`, limiares, calibração ou ensemble. Seleção/ajuste só na VAL, declarada.
- Comparações entre configs: multi-seed (42/84/126) e teste pareado; single-seed não é evidência (McNemar entre seeds da mesma config já dá p<0,01).
- Estatística reutilizável em `common/stats.py` (mcnemar_paired, bootstrap_ci por imagem, mean_std); não reimplementar.
