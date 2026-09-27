# RQ1 — Dados: evolução dirigida do dataset (STID → DS-F2_v8.1.1)

**Pergunta.** Estender o dataset STID (paper anterior, PACBB 2026) com *coletas físicas dirigidas* —
(1) placas de **crowding** da mesma espécie, (2) placa **graduada** (escala), (3) **papel em branco** (negativo puro),
(4) **copy-paste de instâncias reais** anotadas — melhora a detecção por espécie?

**Hipótese pré-registrada (autor, 26/09; já escrita como H1/H2 em `manuscript/sec/5_experiments.tex` @ `09eba69`).**
- **H1 (detecção melhora, onde esperado):** as coletas dirigidas atacam falhas do PACBB (oclusão em crowding,
  variação de escala, FP em fundo limpo, espécie rara); mAP@50 e recall geral sobem, com ganho *localizado* nas
  métricas correspondentes (recall por bin de tamanho, recall em placas de crowding, FP em placas em branco).
- **H2 (identificação não se move):** a confusão MC↔MV / MF↔MC entre objetos localizados **não** muda de forma
  mensurável (IC bootstrap da diferença cobre zero) entre STID-only e DS-F2 completo. Só a cascata (RQ2) a reduz.
- **Desenho:** STID-only vs DS-F2_v8.1.1, YOLO26m receita E0, seeds 0/1/2, mesmo TEST (55 img / 918 inst.),
  média±std + McNemar pareado + IC bootstrap (`common/stats.py`, `rq2_cascade/eval_cascade_matrix.py`).

## O que já temos

| Item | Onde | Observação |
|---|---|---|
| Linhagem confirmada: DS-F2_v8.1.1 = STID (PACBB) + coletas novas | usuário, 25/09 | STID é **subconjunto** do DS-F2 → ablação é viável |
| Dataset atual: 1.259 train / 35 val (617 GT espécies) / 55 test (918 GT espécies), 7 classes, 1920×1080 | `docs/PAPER_FACTS.md §0` | mean √wh = 44,4 px |
| Scripts de construção/curadoria do YOLO dataset | `yolo_restructure.py`, `filter_classes.py`, `change_class.py`, `clean_labels.py`, `recover_labels.py`, `rename_files.py`, `replay_renaming.py`, `update_overwrite.py`, `yolo_legacy_dataset.sh` | copiados de `fly-det/scripts/` sem alteração |
| Fatiamento (para experimentos em tiles) | `slice_dataset.py` | usado em "treino em tiles 512" (F1 igual ao baseline) |
| Copy-paste de instâncias | `baseline/fly-det/fly_det/augment.py` (E9) | run E9: **nome e mAP50 desconhecidos** (job do usuário) |
| Treino em múltiplos datasets | `train-many-datasets.sh`, `slurm/yolo_extended.sh` | base para a ablação STID-only vs full |
| Números do PACBB para a seção de evolução | `docs/prior_work/pacbb2026_extract.txt` | Clean 0,476 → Onto 0,381 → SOP **0,608** (YOLO26-S); classes FF/HF/BF/MF/OI/Noise |

## O que falta

- [ ] **Rastreio das coletas novas dentro do DS-F2_v8.1.1** — lista de imagens (por nome/pasta/metadado) que
      pertencem a cada tipo de coleta (crowding / graduada / branco / copy-paste). Sem isso a ablação não existe.
- [ ] **Mapeamento de classes STID ↔ DS-F2** (⚠️ colisão de siglas: no PACBB `MF` = flesh fly; aqui `MF` = fruit fly,
      `MC` = flesh fly). Tabela obrigatória na Seção 3 do paper.
- [ ] **Regra do split** (por placa/período?) — confirmar procedimento e descrever no paper.
- [ ] **Ablação (Opção B — decidida 26/09)**: treinar YOLO26m em STID-only vs DS-F2 completo, mesmo TEST, seeds 0/1/2
      (6 treinos, ~1–1,8 h cada no B200 ≈ 8 GPU-h). Entra na fila após as cadeias C/D. O braço DS-F2 completo
      **reutiliza os 3 E0 da cadeia A** (mesma receita/seeds) → só 3 treinos novos (STID-only).
- [ ] **Fixar antes dos treinos:** limites dos bins de tamanho √wh (candidato 32/48 px, coerente com RQ3); nº de
      reamostragens do bootstrap; McNemar por par de seeds vs pooled.
- [ ] **Verificar se o TEST tem placas em branco e de crowding em número suficiente**; senão, held-out dedicado
      (dizer na legenda de `tab:rq1_ablation`).
- [ ] Identificar a **espécie menos frequente** (linha copy-paste de `tab:rq1_mapping`).
- [ ] Seção 3: descrever como os rótulos STID foram mapeados/reanotados nas 7 classes do DS-F2.
- [ ] Preencher `tab:rq1_ablation` (todas as células), `tab:rq1_mapping`, `fig:rq1_confusion` (duas matrizes 7×7
      entre objetos localizados, seeds pooled) e reescrever o parágrafo de falsificação após os resultados.
- [ ] Recuperar o run **E9 (copy-paste)**: nome no W&B e mAP50, ou reexecutar.
- [ ] Estatísticas do dataset para a tabela do paper: nº de placas por tipo de coleta, instâncias por espécie
      por split, histograma de tamanho de objeto → `results/dataset_stats.csv`.
- [ ] Decisão: **liberar o DS-F2 publicamente?** (parceiro industrial) — define se o dataset é contribuição.
- [ ] Figura: exemplos de cada tipo de coleta dirigida → `figures/out/fig_collections.pdf`.

## Saídas esperadas em `results/`

- `dataset_stats.csv` — contagens por split × classe × tipo de coleta.
- `ablation_stid_vs_full.csv` — run, seed, subset, mAP50 (harness 7 classes), P/R/F1 micro, recall por bin de tamanho,
  recall em crowding, FP por placa em branco, correct|localized, taxas MC↔MV e MF↔MC (TEST); média±std e Δ com IC 95 %.
- `stid_subset.txt` — lista de imagens do subconjunto STID-only (saída do rastreio).
