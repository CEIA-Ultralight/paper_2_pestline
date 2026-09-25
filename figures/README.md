# figures

- `src/` — um script por figura, lendo **apenas** `../rq*/results/*.csv` (nunca números digitados).
- `out/` — PDFs finais; copiar para `manuscript/fig/` e referenciar no `.tex`.

## Figuras planejadas

| Figura | Fonte | Script |
|---|---|---|
| Exemplos de coletas dirigidas (crowding / graduada / branco / copy-paste) | imagens do DS-F2 | `make_collections.py` |
| Pipeline single-stage vs cascata (diagrama) | — | desenho (draw.io/TikZ) |
| Trade-off da cascata: recall/F1 ↑ vs mAP ↓ por seed | `rq2_cascade/results/cascade_test_seeds.csv` | `make_cascade_tradeoff.py` |
| Matriz de confusão single-stage vs cascata (MC↔MV, MF↔MC) | `rq2_cascade/results/confusion_single_vs_cascade.csv` | `make_confusion.py` |
| mAP50 por seed, E0 vs E10 (NWD) | `rq3_tiny_object/results/nwd_3x3.csv` | `make_nwd_seeds.py` |
| Recall por faixa de tamanho de objeto | `rq3_tiny_object/results/recall_by_size.csv` | `make_recall_by_size.py` |
| Exemplos qualitativos: erros do single-stage corrigidos pela cascata | predições no TEST | `make_qualitative.py` |

Nenhum script existe ainda — todos pendentes.
