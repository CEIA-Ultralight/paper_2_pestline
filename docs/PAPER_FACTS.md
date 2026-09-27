# PAPER_FACTS — fonte da verdade dos números do paper VISAPP 2027

> **Regra:** todo número que aparece em `manuscript/sec/*.tex` tem que existir aqui, com origem (job/arquivo/protocolo) **e** em um CSV de `rq*/results/` (ver §8).
> Revisores (R1–R5) comparam o paper contra este arquivo; divergência = weakness grave.
> Atualizar **este arquivo primeiro**, depois o `.tex`. Marcar claramente `single-seed` vs `multi-seed` e `VAL` vs `TEST`.
> Datas no formato do cluster (2026). Alvo: VISAPP 2027 (deadline 22/out/2026); o cabeçalho anterior ("CVPR 2027", `paper/cvpr2027/`) era legado.
>
> **Auditoria 2026-09-26 (analyst):** todos os números de §2–§4 foram re-derivados do W&B (`pestline/fly-species`, runs `EVAL_E0_cascade_val_20260909T134157Z` e `test-campaign-036ef018`) e batem com os `.md` a ≤0,05 p.p. (exceções de arredondamento listadas em §8). Os números de §1 marcados `results.csv` **não são verificáveis neste nó** (`/raid/user_marcospaulo/fly-det` e `flydet_runs` ausentes em dgx-B200-1); o que existe no W&B é o *summary* (validação final do `best.pt`), que é outro protocolo — ver tabela 1b.

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
| E13 | multi_scale (rect=false) | — | — | — | 32947 | ❌ CUDA OOM (run `E13_yolo26m_multiscale`, failed após 127 s); **abortou a cadeia 32947** → E7 e TTA/WBF dessa cadeia não rodaram |
| E10 s0/s1/s2 | NWD w=0,5, "seeds 0–2" | **inválido** | **inválido** | 3 declaradas / **1 efetiva** | 33015 (E10y) | ❌ runs `E10_nwd_seed0/1/2` finished, mas summary **byte-idêntico** entre si e ao `E10_yolo26m_nwd` (mAP50 0,854434 / mAP50:95 0,618038 / P 0,783389 / R 0,831122 / best_fitness 0,61771) → bug de seed (`init_seeds` antes do trainer, sobrescrito por `args.seed=0`). Não é multi-seed. |
| E0 s0/s1/s2 | baseline, "seeds 0–2" | inválido | inválido | 3 declaradas / 1 efetiva | 33015 | ❌ `E0_baseline_seed0` finished = idêntico ao `E0_yolo26m_baseline` (0,860486 / 0,618887); `E0_baseline_seed1` **crashed** no step 85 (best_fitness 0,61982 = idêntico ao s0 → mesma trajetória); `E0_baseline_seed2` **não rodou** (cadeia abortou no crash) |
| E7 | yolo26l + NWD w=0,5 | — | — | — | 32947 e 33015 | ❌ **não rodou** em nenhuma cadeia (sem run no W&B) |
| E14 | NWD + RFLA | — | — | — | 33015 | ❌ **não rodou** (sem run no W&B) |
| TTA/WBF (detector) | ensemble TTA+WBF dos detectores no VAL (`tta_wbf_eval.sh`) | — | — | — | 32947 e 33015 | ❌ **não rodou**. Não confundir com o ensemble de *classificadores* + TTA de §4 (`E3_ensemble_tta`) |

**Status da cadeia 33015 (E10y) em 2026-09-26:** abortada após o crash do `E0_baseline_seed1`; das 9 etapas rodaram 4 completas + 1 crashed, todas com seed efetiva 0. **Nenhuma medida de variância do detector existe.** Fix F01-01 (LEDGER) continua aberto. Fonte: W&B `pestline/fly-species` (35 runs) → `rq3_tiny_object/results/nwd_sweep_single_seed.csv`.

### 1b. Mesmos experimentos, dois protocolos de leitura (nunca misturar na mesma tabela)

| Exp | run W&B | `results.csv` best (PAPER_FACTS §1, máximo por época; **não verificável neste nó**) | `results.csv` `best.pt` (R3 review, idem) | **W&B summary** (validação final do `best.pt`, verificado) mAP50 / mAP50:95 / P / R |
|---|---|---:|---:|---|
| E0 | `E0_yolo26m_baseline` (y0joklcn) | 0,8565 | 0,8565 / 0,6198 (ép. 80) | **0,8605** / 0,6189 / 0,7912 / 0,8173 |
| E10 | `E10_yolo26m_nwd` (ro6yymqk) | 0,8706 (ép. 64 de 137) | 0,8484 / 0,6177 (ép. 107) | **0,8544** / 0,6180 / 0,7834 / 0,8311 |
| E11 | `E10_w075` (4r8wbsby) | 0,8642 | — | 0,8659 / 0,6030 / 0,8075 / 0,8201 |
| E12 | `E10_w100` (ziqlwumu) | 0,6719 | — | 0,6536 / 0,3381 / 0,5614 / 0,7019 (33 steps) |
| E10b | `E10b_yolo26m_gcd` (tvpy6wmz) | 0,8619 / 0,6261 | — | 0,8572 / **0,6296** / 0,8003 / 0,8287 |
| E1 (P2) | `E1_yolo26m_p2` (shp7xlkc) | — | — | 0,8284 / 0,5905 (200 ép.) |
| E2 (tiles 512) | `E2_yolo26m_sahi` (ojzrkngx) | — | — | 0,8366 / 0,5990 (VAL em tiles, imgsz 512 — não comparável) |

⚠️ Sob o W&B summary (protocolo único e verificável) **E10 fica −0,6 p.p. abaixo de E0 em mAP50** (0,8544 vs 0,8605) e empatado em mAP50:95; sob `best.pt` do `results.csv` (R3) fica −0,8 p.p. O "+1,4 p.p." só existe comparando o máximo-por-época do E10 com o `best.pt` do E0. Coerente com DECISIONS 2026-09-25: **não é claim**.

⚠️ **O E0 mAP50 do harness de cascata (0,8118, Tab. §2) NÃO é comparável** ao `results.csv` (0,8565) nem ao W&B summary (0,8605): protocolos diferentes (conf/NMS/matching). Nunca misturar na mesma tabela sem nota.

⚠️ O ganho E10 (+1,4 p.p.) é **single-seed** e dependente do protocolo (1b); variância entre seeds do detector **ainda não foi medida** (33015 inválido). Só afirmar qualquer efeito após rerun 3×3 com seed corrigido, média ± std, teste pareado.

## 2. Cascata detector→classificador — VAL (job 32088, harness próprio, 7 classes)

Protocolo: YOLO E0 conf≥0,001, max_det 300, NMS IoU 0,5, imgsz 1920, FP32, end2end=False; 4.197 propostas; classificador reclassifica crops.
Origem verificada: W&B `pestline/fly-species/EVAL_E0_cascade_val_20260909T134157Z` (114dbfba) → `rq2_cascade/results/cascade_val_single_seed.csv` (todos os valores abaixo batem a 2 decimais). Cobertura 94,91/97,89 % e "4.197 propostas" só existem em `CASCADE_RESULTS_20260909.md` (report.md do job), não no summary W&B. Classificadores seed 42 (A1), detector seed default. **single-seed.**

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

Origem verificada (2026-09-26): W&B `pestline/fly-species/test-campaign-036ef018` (id `036ef018cfcf494d8e15eb49e95acc1e`, 2026-09-09, 1.068 s) → `rq2_cascade/results/cascade_test_single_seed.csv`. Chaves: `detector/E0/metrics/species/*` e `cascade/architecture-night-v1-20260907/crops/parts/metrics/label_only/species/*`. Política da linha principal = **label_only** (classe substituída, score YOLO mantido). Detector seed default; classificador `A1_parts` seed 42, 22 ép. (best ép. 17), 1,55 h. **single-seed** (det.) × **single-seed** (clf.). Valores exatos: recall 80,07 %, precisão 71,15 %, F1 75,35 % (arredonda para 75,4; o "75,3" abaixo veio do .md — Δ vs E0 = +2,1 p.p. em ambos os casos); acerto entre localizadas 735/898 = 81,85 %; E0 695/898 = 77,39 %.

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

Ensemble 3 classif. (exploratório): recall 75,5 %, precisão 77,4 %, F1 76,4 %, 662 ms/img. — Origem W&B: `cascade/E3_ensemble/metrics/product` (swin_t 224 + convnext_t 384 + convnext_s; **política product**, não label_only; label_only dá R 78,6 / P 73,5 / F1 76,0). Marcado `exploratorio=true` no CSV.

Seeds: baseline e partes na seed 84 deram F1 70,1–72,6 % (**abaixo** da seed 42) → variância de vários p.p. Classificadores são single-seed salvo indicação.
Desagregado (W&B `cascade/architecture-followup-v1-20260908/seed84/*`, no CSV): partes s84 label_only **P 63,7 / R 77,9 / F1 70,1** (715/918); partes s84 product 66,5 / 76,8 / 71,3; CE s84 label_only 68,9 / 75,6 / 72,1 (694/918); CE s84 product 71,2 / 74,1 / 72,6. **Todos abaixo do YOLO E0 (F1 73,2)**: na seed 84 a cascata perde 7,2 p.p. de precisão e 3,1 p.p. de F1 e ganha só +2,2 p.p. de recall. Só o sinal do ganho de recall é estável entre as duas seeds. Controle CE seed 42 (`A1_baseline`, label_only): P 70,9 / R 79,5 / F1 75,0 — i.e. "atenção a partes" (75,35) ≈ CE simples (74,95): o ganho é de **cascatear**, não da cabeça de partes.

## 4. Resultados negativos (TEST F1 4 espécies, single-seed, RELATORIO_TESTE_20260909.md)

Origem verificada: mesmo run W&B `test-campaign-036ef018` → `rq3_tiny_object/results/sahi_tta_single_seed.csv` (detectores) e `rq2_cascade/results/classifiers_single_seed.csv` (classificadores). ⚠️ Os F1 dos classificadores legados desta tabela (Swin-T 74,4; ConvNeXt-T 76,0; ConvNeXt-S 72,4; 512 70,3; SupCon 73,3; ensemble 76,4) correspondem à política **product** no W&B; a linha principal de §3 (partes 75,3) usa **label_only**. Sob product a cabeça de partes dá 75,5 e o ConvNeXt-T CE (E3b) 76,0 / A1_baseline 75,7 → não há efeito mensurável da cabeça de partes. "SAM 71,8 vs ≈79" são F1 de espécie em **GT-crops** (`gt_crop/*`), não da cascata.

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
- **Duração real por treino** (W&B `_runtime`, 1×H100, `rq3_tiny_object/results/nwd_sweep_single_seed.csv`): YOLO26m 150 ép. imgsz 1920 = **0,7–1,3 h** (E0 0,72 h / 111 ép. com early stop; E10 1,31 h / 138 ép.; E10b 1,05 h; E11 0,87 h; E12 0,26 h colapso; E1-P2 2,0 h / 200 ép.; E2 tiles 0,91 h). Classificadores legados E3–E5: 0,9–5,7 h (E3b 5,74 h); campanha A1/A2: 0,5–1,55 h cada (budget 1–1,5 h).
- **GPU-h da campanha `pestline/fly-species`** (soma de `_runtime` dos 35 runs finished/crashed/failed; exclui testes/smoke/fila do Slurm, logo é limite inferior do wall-time dos jobs): **44,5 h** = detectores YOLO 11,3 h (13 runs) + classificadores legados 22,4 h (6) + campanha A1/A2 10,4 h (12) + avaliações 0,3 h (4). Cadeia 32947: 2,2 h; cadeia 33015: 4,2 h (desperdiçadas — seed inválido).

## 7. Perguntas abertas (bloqueiam seções do paper)

1. As novas coletas (crowding / graduada / branco / copy-paste) já estão em DS-F2_v8.1.1? Foram treinadas com/sem cada uma (ablação RQ1)? Em qual dataset (STID ou DS-F2)?
2. E9 copy-paste (job do usuário): nome do run e mAP50?
3. Split por placa/período — confirmar procedimento exato.
4. Dataset será liberado publicamente (parceiro industrial)? Define se é "contribution".
5. Existe holdout **nunca** consultado para a avaliação final?

## 8. Arquivos CSV (`rq*/results/`) — gerados em 2026-09-26 por `_scratch/build_results_csvs.py` a partir do dump W&B

| CSV | Linhas | Dá suporte a | Fonte primária | Status |
|---|---:|---|---|---|
| `rq3_tiny_object/results/nwd_sweep_single_seed.csv` | 24 | §1 (tab. detector VAL), §1b, §6 (durações) | W&B summary de 17 runs/ausências (`protocolo=wandb_summary`) + 7 linhas `results_csv_best`/`results_csv_bestpt` copiadas de §1 e R3 (não verificáveis) | single-seed; seeds 33015 marcadas `seed_efetiva=0 (bug)` |
| `rq3_tiny_object/results/sahi_tta_single_seed.csv` | 11 | §4 (P2, tiles 512, SAHI, ensemble+TTA) | W&B `test-campaign-036ef018` (`detector/E0,E1,E2,E2_sliced`, `cascade/E3_ensemble*`) + `E2_eval_sahi` legado | TEST, single-seed; TTA/WBF do detector = "não rodou" |
| `rq2_cascade/results/cascade_test_single_seed.csv` | 298 (long) | §3 completo: tab. principal, por espécie MD/MV/MC/MF (P/R/F1/tp), controle CE, product, seed 84 (4 combinações + intervalo), ensemble (`exploratorio=true`) | W&B `test-campaign-036ef018` | TEST, seed_det default × seed_clf 42/84 |
| `rq2_cascade/results/cascade_val_single_seed.csv` | 64 (long) | §2 completo: mAP50/mAP50:95 7 pipelines, 496→522/617, latência mean/p50/p95, cobertura | W&B `EVAL_E0_cascade_val…` (114dbfba); cobertura/4.197 de `CASCADE_RESULTS_20260909.md` | VAL, single-seed |
| `rq2_cascade/results/classifiers_single_seed.csv` | 19 | §4 (classificadores), §3 (seeds), §6 (runtime) | runs de treino E3*/E4*/E5*/A1_*/A2_* (config seed/épocas/runtime, VAL species_macro_f1) × `test-campaign` (GT-crop e cascata label_only/product, ms/img) | TEST+VAL, single-seed; E3–E5 sem seed registrada |

Divergências de arredondamento CSV × texto acima (não alteradas): F1 partes 75,35 → "75,3"; ensemble F1 76,45 → "76,4"; ms/img 24,92 → "25", 462,74 → "463"; "4.197" vs `4197`. Nenhuma outra divergência > 0,05 p.p.
Ausentes do W&B e **não rastreáveis neste nó**: `results.csv` best-epoch/`best.pt` (§1/1b), cobertura 94,91/97,89 %, contagem de 4.197 propostas, "E12 early stop ép. 10", tamanho de objetos 44,4/39,2 px, contagens do §0 (1.259 imgs / 59.346 boxes), tudo do §5 (PACBB).
