# Resultado: YOLO E0 versus cascatas, validação original

Job Slurm **32088**, h100n2, uma H100, Apptainer, COMPLETED/0, 2m07s incluindo
15 testes, smoke de duas imagens e retomada. Nenhum retreino ou mudança nos dados.
Run: https://wandb.ai/pestline/fly-species/runs/114dbfba

35 imagens 1920×1080, 1.649 GT, sete classes; 617 GT das quatro espécies.
YOLO fornece as mesmas 4.197 propostas às três cascatas, conf>=0,001,
max_det=300, iouNMS=0,5, imgsz1920, FP32. Backend real: end2end=False, fp16=False.
Matching AP testado contra BaseValidator instalado. Ultralytics8.4.142,
torch2.6.0+cu124, torchvision0.21.0+cu124. Um crop degenerado preservado por cabeça.

## mAP (todas as sete classes)

| Pipeline | mAP50 | mAP50:95 |
|---|---:|---:|
| YOLO sozinho | **81,18%** | **60,12%** |
| + baseline, somente classe | 75,14% | 55,68% |
| + partes, somente classe | 76,37% | 56,91% |
| + ArcFace, somente classe | 75,68% | 56,56% |
| + baseline, score produto | 76,44% | 56,68% |
| + partes, score produto | 77,28% | 57,60% |
| + ArcFace, score produto | 76,17% | 56,96% |

Somente classe mantém score YOLO específico da classe original. Produto multiplica
esse score pelo máximo softmax: é uma heurística, não probabilidade calibrada.
Nenhuma das duas políticas venceu YOLO. Partes/produto fica 3,90 p.p. abaixo em
mAP50 e 2,52 p.p. abaixo em mAP50:95. Nenhuma política selecionada via test.

## Há ganho de classificação a limiar fixo, mas não de detecção global

Confusão class-agnostic IoU>=0,5, score>=0,25:

| Medida | YOLO | + partes somente classe | + partes produto |
|---|---:|---:|---:|
| Espécies pareadas | 593 | 593 | 579 |
| Espécies corretas | 496 | 522 | 516 |
| Acertos / todos os 617 GT espécies | 80,39% | 84,60% | 83,63% |
| Acertos / espécies pareadas | 83,64% | 88,03% | 89,12% |
| Predições sem GT pareado, sete classes | 309 | 309 | 280 |
| Predições sem GT pareado classificadas como espécies | 40 | 53 | 40 |

Reclassificar com partes corrige 26 espécies líquidas na população pareada, mas
também aumenta previsões não pareadas rotuladas como moscas. O AP mede a curva
precisão-recall e a ordenação por score, não só acertos nos objetos pareados.
Não há contradição em aumentar acurácia a um limiar e reduzir AP.

AP50 MD/MV no YOLO: 67,80%/85,94%; partes-produto: 64,19%/74,31%.
A cobertura class-agnostic de propostas conf0,001 a IoU0,5 é 94,91% dos GT e
97,89% das espécies, limitada pelo matching e propostas deste protocolo, não
uma prova de teto máximo de detecção.

## Custo observado sobre imagem já decodificada

| Pipeline | Média ms/imagem | p50 ms/imagem | p95 ms/imagem |
|---|---:|---:|---:|
| YOLO | 22,88 | 22,70 | 24,09 |
| YOLO + baseline | 431,39 | 355,06 | 923,21 |
| YOLO + partes | 416,53 | 362,37 | 870,73 |
| YOLO + ArcFace | 415,18 | 354,71 | 875,48 |

Cascata processa todas as propostas de score baixo (média120 por imagem), não só
detecções de produção conf0,25. CPU crops/transforms e classificação estão no
tempo; shared disk/decode, HTTP/rede estão fora. Uma observação por imagem,
FP32, todos os modelos residentes. Não extrapolar estes ~18× da média de partes
para um pipeline otimizado ou para o custo real de produção sem medir.

## Conclusão e próximo plano (não executado)

**Manter YOLO E0 como referência; não substituir globalmente suas classes pela
cascata atual.** Classificação adicional tem sinal útil, mas a substituição cega
e as regras de score testadas não produzem ganho no objetivo mAP.

Próximas hipóteses justificadas: reclassificação seletiva de detecções incertas,
fusão de scores com preservação de decisões confiáveis do YOLO e treinamento
com propostas/jitter de caixas derivadas apenas do train. Pré-definir controles
e validar seleção em protocolo apropriado; não assumir que calibração será a cura.
Isso ainda não foi implementado nem submetido nesta avaliação.

Validação já usada para seleção de modelos: resultado exploratório, não holdout
cego. Diferença de JPEG nos crops e ausência de threshold/calibração de produção
são limitações registradas. Meta0,95 não atingida; impossibilidade de melhoria
futura não demonstrada. Resultados antigos test0,816 não foram usados nesta tabela.

Artefatos completos ficam no RAID em flydet_runs/E0_cascade_val_20260909/full,
incluindo JSON por imagem, hashes, confusões, summary.json e report.md, também no W&B.