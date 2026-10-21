# RQ1 — Dados: escada de conjuntos de treino

**Pergunta.** Estender o dataset com coletas físicas dirigidas (placas de crowding, placa graduada, papel em branco, copy-paste de instâncias reais) melhora a leitura por espécie?

**Desenho.** Cinco conjuntos de treino cumulativos de DS-F2_v9.0.5 (L0, L1, L2, L2ORG, L3), YOLO26m com a receita E0, seeds 0/1/2, mesmo VAL e TEST (73 imagens) em todos os braços; só o treino muda.

| Braço | Conteúdo |
|---|---|
| L0 | lote v5.1 (originais + aumentos de imagem) |
| L1 | L0 + v6.0 + v7 |
| L2 | L1 + v8 + v9 (sem copy-paste) |
| L2ORG | L2 + DS-ORG (papel em branco, v8.1.1) |
| L3 | DS-F2_v9.0.5 completo = L2 + copy-paste (reaproveita os runs `E0_s{0,1,2}` da RQ2) |

O controle de mesmo tamanho treina subamostras aleatórias do L3 com 103, 244 e 869 imagens (3 seeds cada) e compara com L0, L1 e L2.

## Resultado

- Acurácia de espécie (= recall micro nas quatro espécies) sobe de **56,1 %** (L0) para **82,2 %** (L3); os maiores ganhos são em objetos < 32 px e em placas só com insetos.
- Em 244 e 869 imagens, a subamostra aleatória do L3 iguala L1 e L2: de L1 a L2 o ganho é majoritariamente quantidade. Em 103 imagens a subamostra aleatória fica **19 pp** acima de L0, então o degrau L0 não é só quantidade.
- Quantidade, qualidade do rótulo e domínio variam juntos; o resultado é reportado como associação, não como causalidade. Números e IC em `docs/PAPER_FACTS.md` §11, §11b, §11c.

## Scripts

| Etapa | Arquivo |
|---|---|
| Montar os braços (hardlinks a partir do v9.0.5) | `make_ladder.py`, `make_equalsize.py` |
| Treino em cadeia (12 treinos; 4 do controle) | `slurm/chain_rq1_ladder.sh`, `slurm/chain_rq1_equalsize_h100.sh` |
| Análise da escada (acurácia, IC por imagem, McNemar, recall por tamanho) | `common/analysis/rq1_ladder_analysis.py` |
| Controle de mesmo tamanho, composição por lote, curva de aprendizado | `common/analysis/rq1_equalsize.py`, `ladder_composition.py`, `rq1_learning_curve.py` |
| Inventário das versões do dataset (sobreposição, vazamento, origem) | `common/analysis/rq1_version_*.py`, `rq1_origin_profile.py`, `rq1_v4_in_v9.py`, `leakage_check.py` |
| Figura de confusão L0 × L3 | `common/analysis/rq1_confusion_fig.py` |
| Construção do dataset a partir do STID (legado, copiado do baseline) | `yolo_restructure.py`, `filter_classes.py`, `change_class.py`, `clean_labels.py`, `recover_labels.py`, `rename_files.py`, `replay_renaming.py`, `update_overwrite.py`, `slice_dataset.py`, `yolo_legacy_dataset.sh`, `train-many-datasets.sh`, `slurm/yolo_extended.sh` |

## `results/`

| CSV | Conteúdo |
|---|---|
| `ablation_ladder.csv`, `ablation_ladder_summary.csv` | acurácia e recall por espécie, por seed e agregados |
| `ladder_paired.csv` | Δ pareado entre braços, IC 95 % por imagem, McNemar |
| `ladder_size_bins.csv`, `ladder_crowding.csv`, `ladder_insonly_fp.csv`, `ladder_confusion.csv` | recortes por tamanho, densidade, placas só-INS e confusão |
| `equalsize_summary.csv`, `equalsize_prior_ovx.csv` | controle de mesmo tamanho (a segunda tabela traz runs anteriores de outra GPU) |
| `ladder_composition.csv`, `learning_curve.csv` | composição por lote e curva de aprendizado |
| `rq1_versions_*.csv`, `rq1_profile_*.csv`, `rq1_v4_in_v9.csv`, `rq1_versions_summary.md` | inventário e sobreposição entre versões do dataset |
| `runs.md` | job Slurm → run W&B (`paper2-rq1-data`) |

## Limitações declaradas

Sem ID de placa, o split por placa não pode ser verificado e vazamento entre placas não é excluído; a origem dos rótulos e as definições de NOISE/MAR dependem do autor. A versão anterior (v4, 1280×720, 6 classes, outro teste) aparece só como linha histórica.
