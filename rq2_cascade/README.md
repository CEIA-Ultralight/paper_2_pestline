# RQ2 — Identificação: cascata detector → classificador

**Pergunta.** Um detector genérico (YOLO26m, E0) seguido de um classificador fine-grained sobre os crops (margem 0,75) supera o detector único em recall, precisão e F1 por espécie? E até onde qualquer classificador chega com crops perfeitos?

## Resultado (campanha oficial v9, DS-F2_v9.0.5, TEST de 73 imagens)

- Com esta receita (detector congelado, margem de 75 %, fusão só por rótulo), a cascata **não melhora** o detector em nenhuma das 15 configurações (macro-F1 de −1,1 a −5,1 pp) e custa 30–67× o tempo de inferência. Também não ajudam o relabel só entre espécies, a fusão tardia, outras margens de crop, o fundo como classe (BG) nem a abstenção híbrida.
- A acurácia com crops de verdade (oráculo) satura em **82,7–83,8 %** para cinco classificadores em quatro backbones (três deles foundation models); 4,9 % das instâncias erram nos 15 classificadores, 69 % dos erros são confiantes e 79 % caem na família MD/MV/MC.
- A VAL inverte o sinal em alguns pares, o que confirma que ela não é held-out.
- Números e ICs em `docs/PAPER_FACTS.md` §10 e §11d–§11f.

## Scripts

| Etapa | Arquivo |
|---|---|
| Extração de crops (verdade e propostas do detector) | `extract_crops.py`, `extract_proposal_crops.py`, `sam_clean_crops.py` |
| Treino dos classificadores (ConvNeXt-T/S, Swin, foundation models, perdas CE/focal/SupCon/ArcFace) | `train_classifier.py`, `architecture_lab/`, `slurm/train_classifier_b200.sh`, `slurm/chain_B_classifiers.sh` |
| Avaliação da cascata (matriz detector × classificador, 7 classes) | `eval_cascade_matrix.py`, `eval_yolo_cascade.py`, `slurm/chain_core_v9.sh`, `slurm/chain_D_cascade_cross.sh` |
| Oráculo em crops verdadeiros | `eval_oracle_crops.py` |
| Extras v9 (BG, margens 0/25/75 %, curva de aprendizado) | `slurm/chain_extras_v9.sh`, `slurm/chain_pad_ablation_h100.sh`, `slurm/chain_pad75_ref_h100.sh` |
| Políticas de fusão e estudo de VAL | `policy_study.py`, `val_cascade_study.py`, `label_issues.py` |
| Análise (bootstrap pareado BCa, permutação, McNemar, confusão, latência, localização, fusão tardia) | `common/analysis/rq2_v9_analysis.py`, `rq2_v9_extra.py`, `rq2_v9_summary.py`, `rq2_v9_lc_bg.py`, `rq2_cascade_variants.py`, `rq2_per_species_table.py`, `make_crop_fraction.py` |
| Cabeças alternativas sobre features do YOLO (exploratório) | `test_campaign_heads.py`, `test_campaign_roi.py` |

## `results/`

Os arquivos `v9_*` são os **vigentes**; os demais (sem prefixo) são das campanhas preliminares em DS-F2_v8.1.1 (H100/B200) e ficam como histórico rastreável, não entram no artigo.

| CSV | Conteúdo |
|---|---|
| `v9_cascade_summary.csv`, `v9_cascade_aggregate.csv`, `v9_bootstrap_delta_f1.csv`, `v9_mcnemar.csv` | cascata × detector: ΔF1, ICs, testes pareados |
| `v9_cascade_variants*.csv` (`pad00`, `pad25`, `pad75`), `v9_late_fusion_approx.csv`, `v9_hybrid_abstention.csv` | variantes de cascata e de margem |
| `v9_oracle_*.csv` | oráculo: resumo, por tamanho, confiança, família, concordância, risco–cobertura, fundo |
| `v9_per_species_single_vs_cascade.csv`, `v9_confusion_single_vs_cascade.csv` | por espécie e confusão |
| `v9_latency.csv`, `v9_localization.csv`, `v9_learning_curve.csv`, `v9_leakage_check.csv` | latência, localização, curva de aprendizado, verificação de vazamento |
| `runs.md` | job Slurm → run W&B (`paper2-rq2-cascade`) |

## Limitações declaradas

Receita única de cascata (detector congelado, label-only); não foram feitos TIDE, re-pontuação ciente de IoU, oráculo por nitidez ou aglomeração, fusão tardia com scores reais do detector nem re-anotação por especialista. Os 15 pares compartilham detectores e classificadores, logo não são independentes.
