# PAPER_FACTS — fonte da verdade dos números do paper CVPR 2027

> **Regra:** todo número que aparece em `paper/cvpr2027/sec/*.tex` tem que existir aqui, com origem (job/arquivo/protocolo).
> Revisores (R1–R5) comparam o paper contra este arquivo; divergência = weakness grave.
> Atualizar **este arquivo primeiro**, depois o `.tex`. Marcar claramente `single-seed` vs `multi-seed` e `VAL` vs `TEST`.
> Datas no formato do cluster (2026).

## 0. Estado dos dados (2026-09-22)

| Item | Valor | Origem |
|---|---|---|
| Dataset atual | DS-F2_v8.1.1 | `fly_det/config/yolo.yaml` |
| Resolução nativa | 1920×1080 (nunca reduzida) | regra do projeto |
| Classes (7) | MD, MV, MC, MF, INS, NOISE, MAR — índices YOLO 0–6 | data.yaml |
| Espécies-alvo (4) | MD (housefly), MV (blowfly), MC (flesh fly), MF (fruit fly) — índices 0–3 | idem |
| Train | 1.259 imgs / 59.346 boxes | inventário |
| Val | 35 imgs / 1.649 GT / **617 espécies** | CASCADE_RESULTS_20260909.md |
| Test | 55 imgs / 2.864 GT / **918 espécies** | RELATORIO_TESTE_20260909.md |
| Tamanho dos objetos (train) | mean √(w·h) = 44,4 px; mediana 39,2 px | análise 21/09 (C do NWD = 44) |
| Split | por placa/período (**confirmar e descrever no paper**) | ⚠️ pendente confirmar |
| **Novas coletas (RQ1)** | (1) crowding mesma espécie; (2) placa graduada; (3) papel em branco (negativo); (4) copy-paste de instâncias reais | ⚠️ **não sabemos se já estão em DS-F2_v8.1.1 nem se foram treinadas/ablacionadas** — perguntar ao usuário |

## 1. Detector — VAL mAP50 (Ultralytics `results.csv`, best epoch, imgsz 1920, protocolo idêntico)

Receita comum: yolo26m.pt, 150 ép., batch 8, imgsz 1920, patience 30, H100 80 GB, Apptainer `flydet-train.sif`, Ultralytics 8.4.142.

| Exp | Variante | mAP50 | mAP50:95 | Seeds | Job / commit | Status |
|---|---|---:|---:|---|---|---|
| E0 | baseline CIoU | **0,8565** | — | 1 (default) | results.csv E0 | ✅ |
| E10 | NWD w=0,5, C=44 | **0,8706** (+1,4) | — | 1 (default) | 32933 / 29c37e8 | ✅ |
| E11 | NWD w=0,75 | 0,8642 | — | 1 | 32947 (E10x) | ✅ |
| E12 | NWD w=1,0 (sem CIoU) | 0,6719 (colapso, early stop ép. 10) | — | 1 | 32947 | ✅ negativo |
| E10b | GCD w=0,5 | 0,8619 | **0,6261** (melhor mAP50:95 de todos) | 1 | 32947 | ✅ |
| E13 | multi_scale (rect=false) | — | — | — | 32947 | ❌ CUDA OOM, descartado |
| E10 s0/s1/s2 | NWD w=0,5, seeds 0–2 | ⏳ | ⏳ | 3 | 33015 (E10y) | em andamento |
| E0 s0/s1/s2 | baseline, seeds 0–2 | ⏳ | ⏳ | 3 | 33015 | em andamento |
| E7 | yolo26l + NWD w=0,5 | ⏳ | ⏳ | 1 | 33015 | em andamento |
| E14 | NWD + RFLA | ⏳ | ⏳ | 1 | 33015 | em andamento |
| TTA/WBF | ensemble dos acima | ⏳ | ⏳ | — | 33015 | em andamento |

⚠️ **O E0 mAP50 do harness de cascata (0,8118, Tab. §2) NÃO é comparável** ao `results.csv` (0,8565): protocolos diferentes (conf/NMS/matching). Nunca misturar na mesma tabela sem nota.

⚠️ O ganho E10 (+1,4 p.p.) é **single-seed**; variância entre seeds observada no projeto ≈ vários p.p. Só afirmar ganho após E10y (3×3 seeds, média ± std, teste pareado).

## 2. Cascata detector→classificador — VAL (job 32088, harness próprio, 7 classes)

Protocolo: YOLO E0 conf≥0,001, max_det 300, NMS IoU 0,5, imgsz 1920, FP32, end2end=False; 4.197 propostas; classificador reclassifica crops.

| Pipeline | mAP50 | mAP50:95 |
|---|---:|---:|
| YOLO E0 sozinho | 81,18 % | 60,12 % |
| + partes, só classe | 76,37 % | 56,91 % |
| + partes, score produto | 77,28 % | 57,60 % |
| + ArcFace, só classe | 75,68 % | 56,56 % |

Confusão class-agnostic IoU≥0,5, score≥0,25 (617 GT espécies): YOLO 496 corretas (80,39 %) → +partes 522 (84,60 %). A cascata **melhora acurácia a limiar fixo mas reduz AP** (re-rotula sem re-ranquear). Cobertura class-agnostic das propostas: 94,91 % dos GT, 97,89 % das espécies.

Custo (H100, imagem já decodificada): YOLO 22,88 ms/img; + partes 416,53 ms (≈18×).

## 3. Cascata — TEST (RELATORIO_TESTE_20260909.md; 918 espécies; ⚠️ test já consultado antes → "consolidado", não "cego")

Seleção do classificador "atenção a partes" feita na VAL **antes** de olhar o test (limpa). Ensemble escolhido **depois** de olhar o test (exploratório — não usar como resultado principal).

| Métrica (4 espécies) | YOLO E0 | YOLO + ConvNeXt-T partes | Δ |
|---|---:|---:|---:|
| Localizadas e espécie correta | 695/918 | 735/918 | +40 |
| Recall | 75,7 % | **80,1 %** | +4,4 p.p. |
| Precisão | 70,9 % | 71,2 % | +0,2 p.p. |
| F1 | 73,2 % | **75,3 %** | +2,1 p.p. |
| Acerto de espécie entre localizadas | 77,4 % | 81,8 % | +4,5 p.p. |
| mAP50 (harness, 7 classes) | 76,8 % | 72,1 % | −4,7 p.p. |
| ms/img (H100) | 25 | 463 | ≈18× |

Por espécie (precisão / recall %): MD 69,5/54,3 → 66,8/**66,2**; MV 56,1/83,6 → 60,8/86,9; MC 82,3/85,3 → 87,6/81,2; MF 79,7/78,3 → 73,5/**84,4**.

Ensemble 3 classif. (exploratório): recall 75,5 %, precisão 77,4 %, F1 76,4 %, 662 ms/img.

Seeds: baseline e partes na seed 84 deram F1 70,1–72,6 % (**abaixo** da seed 42) → variância de vários p.p. Classificadores são single-seed salvo indicação.

## 4. Resultados negativos (TEST F1 4 espécies, single-seed, RELATORIO_TESTE_20260909.md)

| Ideia | F1 | Conclusão |
|---|---:|---|
| YOLO26m baseline | 73,2 % | referência |
| + cabeça P2 | 67,5 % | piorou |
| Treino em tiles 512 px, aval. foto inteira | 73,2 % | igual |
| Inferência fatiada (SAHI) | 66,2 % | piorou: FPs nas bordas |
| Swin-T 224 | 74,4 % | + pequena |
| ConvNeXt-T 384 | 76,0 % | melhor classificador individual |
| ConvNeXt-S | 72,4 % | maior não ajudou |
| ConvNeXt entrada 512 | 70,3 % | ampliar crop não cria detalhe |
| SAM limpeza de fundo (GT boxes) | 71,8 % vs ≈79 % | piorou |
| SupCon | 72,0–73,3 % | recall 80,4 %, precisão 65 % |
| Ensemble + TTA | 75,4–75,6 % | sem ganho, 3,7× custo |
| RoI head sobre features YOLO (GT boxes) | — | pior que classificadores dedicados |
| NWD w=1,0 (E12, VAL mAP50) | 0,6719 | colapso; CIoU precisa ficar |
| multi_scale 1920 (E13) | OOM | inviável em 80 GB sem grad. accumulation |

## 5. Paper anterior (citar em 3ª pessoa; NÃO reutilizar texto/figuras)

"A Data-Centric Pipeline for Microscale Insect Detection in Industrial Sticky Traps", PACBB 2026 (aceito set/2026), Springer LNCS. Dataset STID: Clean (~500 imgs, mAP 0,476) → Raw (>2.700 sintéticas Blender, falhou por domain gap) → Onto (classe Noise, mAP 0,381, P 0,574) → SOP (~900, foco fixo, grade impressa, **mAP 0,608** YOLO26-S). Classes FF/HF/BF/MF/OI/Noise. Single-seed admitido como limitação. Afirma confusão MF↔BF "irredutível" (nosso RQ3 confronta isso).

⚠️ STID ≠ DS-F2_v8.1.1 — confirmar relação/mapeamento de classes antes de comparar números.

## 6. Hardware / reprodutibilidade

- DGX H100 80 GB (partição h100n2), 1 GPU por job; Apptainer; Ultralytics 8.4.142; torch 2.6.0+cu124; torchvision 0.21.0.
- Patches: `fly-det/scripts/{nwd_patch.py,gcd_patch.py,rfla_patch.py,train_nwd.py}`; eval WBF/TTA `eval_wbf_tta.py`; testes `tests/test_nwd_patch.py` (7).
- GPU-horas totais da campanha: ⏳ somar ao final (compute reporting CVPR).

## 7. Perguntas abertas (bloqueiam seções do paper)

1. As novas coletas (crowding / graduada / branco / copy-paste) já estão em DS-F2_v8.1.1? Foram treinadas com/sem cada uma (ablação RQ1)? Em qual dataset (STID ou DS-F2)?
2. E9 copy-paste (job do usuário): nome do run e mAP50?
3. Split por placa/período — confirmar procedimento exato.
4. Dataset será liberado publicamente (parceiro industrial)? Define se é "contribution".
5. Existe holdout **nunca** consultado para a avaliação final?
