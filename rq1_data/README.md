# RQ1 — Dados: evolução dirigida do dataset (STID → DS-F2_v8.1.1)

**Pergunta.** Estender o dataset STID (paper anterior, PACBB 2026) com *coletas físicas dirigidas* —
(1) placas de **crowding** da mesma espécie, (2) placa **graduada** (escala), (3) **papel em branco** (negativo puro),
(4) **copy-paste de instâncias reais** anotadas — melhora a detecção por espécie?

**Hipótese.** As coletas dirigidas atacam falhas identificadas no PACBB (oclusão em alta densidade, variação de
escala, FP em fundo limpo); o ganho deve aparecer principalmente em recall de MD/MV e em precisão (menos FP em NOISE).

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
- [ ] **Ablação (Opção B)**: treinar YOLO26m em STID-only vs DS-F2 completo, mesmo TEST, ≥3 seeds cada
      (~6 treinos, ~6–10 h cada em H100). Se não couber na fila até ~10/out, RQ1 fica **descritiva** (Opção A).
- [ ] Recuperar o run **E9 (copy-paste)**: nome no W&B e mAP50, ou reexecutar.
- [ ] Estatísticas do dataset para a tabela do paper: nº de placas por tipo de coleta, instâncias por espécie
      por split, histograma de tamanho de objeto → `results/dataset_stats.csv`.
- [ ] Decisão: **liberar o DS-F2 publicamente?** (parceiro industrial) — define se o dataset é contribuição.
- [ ] Figura: exemplos de cada tipo de coleta dirigida → `figures/out/fig_collections.pdf`.

## Saídas esperadas em `results/`

- `dataset_stats.csv` — contagens por split × classe × tipo de coleta.
- `ablation_stid_vs_full.csv` — se Opção B: run, seed, subset, mAP50, recall/precisão/F1 por espécie (TEST).
