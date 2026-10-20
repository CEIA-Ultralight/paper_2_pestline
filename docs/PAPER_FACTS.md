# PAPER_FACTS — fonte da verdade dos números do paper VISAPP 2027

> **Regra:** todo número que aparece em `manuscript/sec/*.tex` tem que existir aqui, com origem (job/arquivo/protocolo) **e** em um CSV de `rq*/results/` (ver §8).
> Revisores (R1–R5) comparam o paper contra este arquivo; divergência = weakness grave.
> Atualizar **este arquivo primeiro**, depois o `.tex`. Marcar claramente `single-seed` vs `multi-seed` e `VAL` vs `TEST`.
> **⚠️ 2026-09-30 (analyst): a campanha B200 multi-seed (§9) SUPERA os números single-seed H100 de §1–§4, que ficam apenas como histórico (vários claims foram contradidos — ver §9.3).**
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
| E12 | NWD w=1,0 (sem CIoU) | 0,6719 (colapso, early stop ép. 10) | — | 1 | 32947 | ⚠️ **artefato do bug de unidade do NWD na loss (§12.1)**; refeito como E16 (px, v9), que **também colapsou** (§12.2) |
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
| NWD w=1,0 (E12, VAL mAP50) | 0,6719 | colapso — ⚠️ artefato do bug de unidade (§12.1), não usar como evidência; refeito como E16, que também colapsou (§12.2) |
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

## 9. Campanha B200 (2026-09-26/27) — multi-seed; **supera** os números single-seed H100 de §1–§4 (mantidos como histórico)

**Setup.** dgx-B200-1 (partição b200n1), torch 2.8.0+cu128, ultralytics 8.4.142, `flydet-train-b200.sif`, FP32 no eval. Dataset DS-F2_v8.1.1; índices 0–3 = MD, MV, MC, MF (conferido no `data.yaml`). Suporte conferido nos `per_instance.csv`: **VAL 35 imgs / 617 GT de espécie** (MD 142, MV 110, MC 93, MF 272); **TEST 55 imgs / 918** (MD 210, MV 214, MC 218, MF 276; as espécies estão em 51 das 55 imgs) — sem divergência. Seeds: detectores 0/1/2 (`args.yaml` e `nwd_config.json` confirmam seed efetiva = declarada; métricas distintas entre seeds, ao contrário da cadeia 33015); classificadores 42/84/126. Commit: cadeia A (33321) submetida em 9297f13+infra (`runs.md`); cadeias C/D (33363/33364) em 47ea2c7. Média ± desvio amostral (ddof=1); **diferença < std ⇒ sem efeito**. `best.pt` e `best epoch` dos detectores e dos classificadores foram escolhidos no VAL ⇒ **VAL não é held-out**; TEST é "consolidado" (já consultado em campanhas anteriores), não cego. Nenhuma escolha de modelo foi feita olhando o TEST.

**Protocolos (nunca misturar numa tabela):** **P1** Ultralytics `results.csv` (VAL, best epoch = máx. fitness 0,1·mAP50+0,9·mAP50:95); **P2** harness `eval_cascade_matrix` (conf ≥ 0,25, IoU ≥ 0,5, matching class-agnóstico; P/R/F1 de MD/MV/MC/MF; "micro" = soma de TP/predições das 4 espécies; "macro" = média das 4; acc\|loc = corretos/localizados); **P3** avaliador `fly_det.sahi_inference` (7 classes, conf 0,0); **P4** `ap_per_class` recalculado pelo analista de `per_image_preds.jsonl` (mesmo código de `eval_yolo_cascade.metrics`); **P5** Ultralytics val no TEST de `eval_wbf_tta.py` (pendente de validação).

### 9.1 RQ3 — detectores (3 seeds por variante; E0 vs E10 vs GCD vs P2)

> **⚠️ 2026-10-18 (§12.1):** em todos os runs E10/NWD desta seção o NWD **da loss** foi calculado em unidades de grid (saturado ≈0,99); só o NWD do **assigner** estava correto. Leia "E10" como *NWD no assigner + CIoU com metade do peso na loss*. A ablação "NWD só loss" testou, na prática, *CIoU com peso 0,5*. GCD, P2, SAHI e RFLA (como candidate expansion) não são afetados.

Fonte: `rq3_tiny_object/results/tiny_variants_multiseed.csv` (e subconjunto `nwd_3x3.csv` = E0 + E10); runs `E0_s{0,1,2}`, `E10_nwd_w0.5_s{0,1,2}` (job 33321), `E11_gcd_w0.5_s{0,1,2}`, `E1_p2_s{0,1,2}` (job 33363) em `/raid/user_marcospaulo/flydet_runs/<run>/` (`runs/*/results.csv`, `test_eval/<run>/single/label_only/{metrics.json,per_instance.csv}`). **multi-seed (n=3)**.

| Variante | VAL mAP50 (P1) | VAL mAP50:95 (P1) | TEST macro-F1 (P2) | TEST micro R / P / F1 | TEST acc\|loc |
|---|---:|---:|---:|---|---:|
| E0 baseline | 86,1 ± 0,3 | 62,9 ± 0,1 | 72,4 ± 2,9 | 75,4±2,3 / 70,1±3,7 / 72,6±2,9 | 77,0 ± 2,6 |
| E10 NWD w=0,5 | 86,1 ± 0,4 | 62,0 ± 0,4 | 73,1 ± 1,4 | 75,7±1,8 / 71,5±1,0 / 73,6±1,4 | 76,9 ± 1,5 |
| E11 GCD w=0,5 | 85,6 ± 0,4 | 62,0 ± 0,6 | 70,2 ± 2,2 | 71,9±1,5 / 69,4±3,0 / 70,6±2,0 | 73,5 ± 1,6 |
| E1 P2 | 86,3 ± 1,4 | 62,1 ± 0,3 | 71,2 ± 0,9 | 75,2±1,4 / 68,3±0,3 / 71,6±0,8 | 76,8 ± 1,4 |

Δ vs E0 (média ± std dos 3 deltas por seed, p.p.): **E10** macro-F1 +0,8 ± 1,5, micro-F1 +0,9 ± 1,7, VAL mAP50 **−0,03 ± 0,12**, mAP50:95 −0,9 ± 0,5 → **sem efeito** (McNemar exato por seed, acerto de espécie: p = 0,63 / 0,63 / 0,057). **GCD** macro-F1 **−2,2 ± 0,6** (3/3 seeds negativas; McNemar p = 0,0008 / 5·10⁻⁶ / 0,29), VAL mAP50 −0,5 ± 0,1 → **piora** (único delta de TEST com |média| > std). **P2** macro-F1 −1,2 ± 1,9, VAL mAP50 +0,2 ± 1,3 → **sem efeito/inconclusivo**. No VAL mAP50:95 as três variantes ficam ≈0,9 p.p. abaixo do E0 (E10 −0,9 ± 0,5; GCD −0,9 ± 0,8; P2 −0,9 ± 0,3; std do E0 = 0,1) → nenhuma variante melhora o mAP50:95 do VAL. Controle de ruído (`tiny_variants_paired_tests.csv`): mesma config, seeds diferentes, McNemar p = 0,006 (E0 s0×s1), 0,0006 (E0 s1×s2), 0,006 (E10 s0×s1) → **a variância entre seeds do detector é detectável pelo próprio teste**; comparações single-seed (§1) não são evidência. A seed 1 do E0 (macro-F1 75,5) é a melhor de todas as 12 runs.

Por espécie (TEST, recall/precisão %, média das 3 seeds; std do recall 3–8 p.p., ver CSV): E0 MD 59,2/70,7 · MV 82,2/55,5 · MC 79,5/80,1 · MF 79,2/79,5; E10 MD 60,5/73,9 · MV 79,8/58,8 · MC 81,3/77,6 · MF 79,8/78,1.

**Ablações single-seed (s0)** — `tiny_ablation_single_seed.csv` (VAL: P1; TEST: P2). Ruído de referência = max(std macro-F1 E0, E10) = 2,9 p.p. **single-seed ⇒ exploratório.**

| Run (s0) | VAL mAP50 / mAP50:95 | TEST macro-F1 | R / P micro | Δ macro-F1 vs E0_s0 / vs E10_s0 | McNemar exato vs E0_s0 |
|---|---|---:|---|---|---|
| E0_s0 (ref.) | 86,5 / 62,8 | 70,0 | 74,6 / 66,4 | — | — |
| E10_s0 (NWD both, ref.) | 86,5 / 62,2 | 71,8 | 73,9 / 70,4 | +1,9 / — | p = 0,63 |
| NWD só loss | 85,4 / 62,6 | 73,0 | 74,1 / 72,3 | +3,0 / +1,2 | b=64, c=59, p = 0,72 (Δacc −0,5; IC95 −4,3…+4,2) |
| NWD só assigner | 86,2 / 62,9 | 69,7 | 71,6 / 68,4 | −0,3 / −2,2 | b=75, c=47, p = 0,014 (Δacc −3,1) |
| NWD + RFLA | 84,5 / **57,8** | 67,4 | 70,3 / 65,0 | −2,6 / −4,4 | b=102, c=62, p = 0,0022 (Δacc −4,4; IC95 −9,1…+0,9) |

Leitura: **RFLA piora** (VAL mAP50:95 −5,0 p.p. ≫ std de seeds 0,1–0,4; recall TEST −4,4). NWD só-loss: F1 +3,0 vem de precisão (+5,9) com recall estável (−0,5) e ≈1σ de ruído, e E0_s0 é a pior seed do E0 → **inconclusivo** (precisa de 3 seeds). Assigner-only ≈ E0/E10 em F1, recall −3,1 → sem benefício.

**Recall de espécie por tamanho** (TEST; `recall_by_size.csv`; bins √(w·h) em px: <32 [n=451], 32–48 [n=295], ≥48 [n=172]; fronteira 48 pertence ao último bin; média ± std de 3 seeds): E0 72,1±3,6 / 73,6±2,1 / 87,2±2,5; E10 73,2±4,6 / 72,5±2,6 / 88,0±0,9 (Δ vs E0 +1,0±6,4 / −1,0±4,1 / +0,8±2,9 → **sem efeito em nenhum bin**); GCD 65,8±2,7 / 72,5±1,8 / 86,8±4,7 (Δ <32: −6,4±5,5); P2 72,1±1,8 / 72,0±2,0 / 89,1±1,9. **Recall de detecção** (class-agnóstico) E0: 96,2 / 99,4 / 99,6 % → o gargalo dos objetos pequenos é **identificar a espécie** (72 % em <32 px), não localizar (96 %).

**SAHI** (`sahi_multiseed.csv`; runs `sahi_E0_s{0,1,2}`, job 33363, TEST, P3, tile 512, overlap 0,25, imgsz 512, **n=3**): sliced mAP50 59,2 ± 0,9, mAP50:95 40,0 ± 0,8 (7 classes); recall de espécie (conf 0,0) micro 77,7 ± 2,6 %. O "plain" do mesmo avaliador inferiu o frame 1920×1080 a imgsz 512 (redução ≈3,75×): mAP50 2,9 ± 0,4, recall de espécie 1,3 % → **baseline degenerado, não é o E0@1920**; o avaliador P3 (conf 0,0) **não é comparável** ao harness P2 (E0 R = 75,4 % a conf 0,25). **Veredito: inconclusivo** — não há comparação SAHI × E0@1920 no mesmo protocolo; a única evidência (§4: F1 66,2 vs 73,2, H100, single-seed, harness) segue como histórico. Pendência: rodar plain@1920 e SAHI com exportação de predições para o harness P2.

**TTA/WBF:** `ttawbf_E0_s{0,1,2}` (vanilla/TTA, P5) → `tta_vanilla_multiseed_PENDENTE_validacao.csv` — **pendente de validação (possível bug de indexação); NÃO usar como fato.** `ttawbf_E0_ensemble` (WBF) e `E7_yolo26l_s0`: **não rodaram** (job 33363 abortou no ensemble: `IndexError`).

### 9.2 RQ2 — cascata (job 33364; 6 detectores × 6 classificadores; **9 pares = 3 det × 3 clf cruzados, não independentes**)

Fonte: `rq2_cascade/results/cascade_test_seeds.csv`, `cascade_val_seeds.csv`, `mcnemar.csv`, `policy_comparison.csv`, `ap_per_class.csv`; runs `cascade_cross_{test,val}/<det>/<clf>/<policy>/` (`metrics.json`, `per_instance.csv`, `per_image_preds.jsonl`). Classificadores: ConvNeXt-T 384 CE (`clf_convnext_t_ce_s{42,84,126}`) e partes (`…parts_s{42,84,126}`), job 33322. Política principal **label_only**.

**TEST, E0 × CE, label_only (P2; single = 3 detectores; cascata = 9 pares; Δ = cascata − single do mesmo detector):**

| | micro R | micro P | micro F1 | macro-F1 | acc\|loc |
|---|---:|---:|---:|---:|---:|
| single (n=3) | 75,4 ± 2,3 | 70,1 ± 3,7 | 72,6 ± 2,9 | 72,4 ± 2,9 | 77,0 ± 2,6 |
| cascata CE (n=9) | 78,5 ± 0,3 | 67,8 ± 2,5 | 72,7 ± 1,4 | 72,5 ± 1,5 | 80,1 ± 0,3 |
| **Δ (n=9)** | **+3,0 ± 2,2** | −2,2 ± 2,6 | +0,1 ± 2,2 | +0,1 ± 2,1 | +3,1 ± 2,2 |

E0 × partes: Δ R +3,3 ± 2,4, P −0,4 ± 2,8, F1 +1,2 ± 2,1 (macro-F1 73,8 ± 1,5). E10 × CE: Δ R +3,4 ± 1,2, P −3,7 ± 1,9, F1 −0,5 ± 1,3; E10 × partes: R +3,4 ± 1,8, P −2,3 ± 2,1, F1 +0,3 ± 1,2. **Só o recall (= acerto de espécie) sobe; F1 não muda (|Δ| < std em todas as 4 combinações); precisão cai ou fica igual.** Partes ≈ CE (macro-F1 73,8 vs 72,5, Δ < std; oráculo 79,7 vs 79,6) — confirma §3.

**McNemar exato (acerto de espécie, por GT, 918) + IC95 bootstrap por imagem (10 000 reamostragens de imagens, seed 20260930):** E0 × CE (9 pares): **6/9 significativos a 0,05 (todos a favor da cascata)**, 6/9 com Holm (família de 9), **3/9 com Holm sobre os 36 pares**; IC95 do bootstrap exclui 0 em 4/9. Os 3 pares não significativos são exatamente os 3 do detector E0_s1 (Δacc −0,1 … +0,4; p ≥ 0,79), cujo single já tem acc\|loc 80,0 %; E0_s0: Δacc +3,8…+4,1 (b=57–61, c=93–96, p ≤ 0,0065); E0_s2: +4,6…+5,4 (p ≤ 0,0012). E0 × partes 5/9 (Holm 4); E10 × CE 5/9 (Holm 4); E10 × partes 5/9 (Holm 5). O McNemar trata as 918 GT como independentes e subestima a incerteza (GT agrupados em 51 imagens); o IC por imagem é o mais conservador.

**VAL inverte o sinal (P2, 617 GT; `cascade_val_seeds.csv`, VAL não é held-out):** E0 × CE Δ R **−0,7 ± 0,3** (0/9 significativos, 9/9 negativos), F1 −2,7 ± 1,1; E0 × partes −2,3 ± 0,5 (F1 −3,3 ± 1,4); E10 × CE +0,0 ± 0,8 (F1 −3,2 ± 1,0); E10 × partes −1,7 ± 0,5. **Contradiz o ganho VAL de §2 (+4,2 p.p., H100, single-seed)**. Mecanismo medido: acc\|loc da cascata ≈ acerto do classificador sobre crop GT, **independente do detector** (std entre 9 pares 0,3 p.p. vs 2,6 do single) → a cascata substitui a identificação do detector pela do classificador e só ajuda quando o classificador é melhor que o detector (TEST: oráculo 79,6 vs single 77,0; VAL: oráculo 84,6 vs single 86,8).

**Por espécie, TEST, E0 × CE label_only (R / P %; single → cascata):** MD 59,2/70,7 → 59,6/69,1; MV 82,2/55,5 → 87,7/55,8; MC 79,5/80,1 → 81,5/82,4; MF 79,2/79,5 → 83,2/**70,4** (precisão −9,2 ± 2,2: o classificador não tem classe de fundo e re-rotula como espécie as propostas FP de NOISE/INS/MAR).

**Confusão (TEST, E0 × CE label_only; `confusion_single_vs_cascade.csv`; % do suporte da classe; single n=3 → cascata n=9, Δ ± std):** MD→MV 31,3 → 30,4 (−0,9 ± 5,0); MC→MV 15,4 → 15,6 (+0,1 ± 7,3); MV→MC 7,3 → 6,3 (−1,0 ± 2,2); **MF→MC 0,0 e MC→MF 0,0 em ambos**; MF→{INS,NOISE,MAR} 15,5 → 11,6 (−3,8 ± 4,6). Erros dominantes (MD→MV, MC→MV) **não mudam com a cascata**; a confusão MF↔"MC" (análoga à MF↔BF do PACBB, §5) **não ocorre** neste dataset — o erro de MF é para classes não-espécie.

**Oráculo (classificador sobre crops das caixas GT; `oracle_crops.csv`, jobs 33364/33322):** acc de espécie TEST CE **79,6 ± 0,5** (macro-F1 81,8 ± 0,4; recall MD 61,1 · MV 87,7 · MC 82,3 · MF 85,1), partes 79,7 ± 1,8 (82,6 ± 1,3); VAL CE 84,6 ± 0,4, partes 83,4 ± 0,4 (VAL não é held-out). Precisão do oráculo só sobre crops GT (sem FP de detecção) — não comparável à precisão do pipeline.

**Decomposição do erro de espécie (TEST, E0; % dos 918 GT):** erro de **localização** (GT sem proposta conf ≥ 0,25/IoU ≥ 0,5) **2,1 ± 0,4** (E10: 1,5 ± 0,4); erro de **identificação** (localizado e espécie errada) **22,5 ± 2,7** no single → **19,4 ± 0,3** na cascata CE. Cascata acc\|loc 80,1 vs oráculo restrito aos localizados 79,9 (gap −0,3 ± 0,5 p.p.): **a cascata atinge o teto do classificador; o erro restante (≈19 %) é de identificação, não de localização**. Teto do pipeline (det.recall × oráculo\|loc) = 78,2 vs acc real 78,5. VAL: loc 3,7 %, id single 12,8 → cascata 13,5 %.

**Política (label_only × product).** No harness P2 as duas políticas dão **P/R/F1 idênticos por construção** (`policy_comparison.csv`: o limiar 0,25 é aplicado ao `yolo_conf`, não ao score da política) → não permite ranquear. Reavaliação do analista (limiar 0,25 no **score da política**, reproduz o `metrics.json` do label_only em 72/72 combinações, ≤ 1 GT de diferença por arredondamento do jsonl) e AP (P4): E0 × CE, TEST — label_only R 78,5 / P 67,8 / F1 72,7; **product R 77,6 / P 70,0 / F1 73,6** (Δ F1 product−label_only **+0,85 ± 0,39, 9/9 pares positivos**); AP50 de espécie (média das 4) single 73,1 ± 2,4 → label_only 66,8 ± 1,2 (−6,3) → **product 70,0 ± 0,7 (−3,1; product > label_only em 9/9 pares)**; AP50:95 54,2 / 49,8 / 52,0. Mesmo sentido no VAL (AP50 espécie 84,8 / 77,5 / 79,4). **Melhor política = product**, mas **nenhuma cascata supera o single em AP** (a cascata re-rotula sem re-ranquear — confirma §2); product vs single a limiar 0,25: R +2,2 ± 1,9, P −0,1, F1 +1,0 ± 2,0 (sem efeito). Política não foi usada para escolher classificador/detector.

**Latência (B200, FP32, imgsz 1920; ms por imagem; ± entre as 18 combinações det×clf×política; `latency.csv`; TEST 55 imgs):** detector E0 11,9 ± 6,0 (E10 7,9 ± 0,4); extra do classificador CE 413 ± 182 (partes 340 ± 53) para E0; end-to-end E0×CE 425 ± 186 (std por imagem 267; p95 899), E0×partes 352 ± 58, E10×CE 347 ± 17, E10×partes 335 ± 4 → cascata ≈ 30–44× o custo do detector. A latência **varia entre jobs** (detector do E0_s0: 9,1 ms na cadeia A vs 20,1 ms na D; modelos residentes e nó compartilhado) — **indicativa**, não benchmark; depende do nº de propostas por imagem (≤ 300 crops). Não comparável com os números H100 (§2/§3).

**Classificadores (VAL, crops pad75, cadeia B; `classifiers_b200_val_multiseed.csv`; n=3):** acc de espécie CE 85,4 ± 0,4 (macro-F1 83,7 ± 0,9), partes 83,5 ± 1,2 (83,6 ± 1,5); todos `early_stop` (best epoch CE 11/4/4, partes 7/1/11 — partes s84 com best epoch 1 de 6 ⇒ possivelmente sub-treinado). O oráculo VAL (crops GT em memória, 1.649 GT) difere dos crops pad75 pré-extraídos (1.645) em −1,3…+0,7 p.p.

**AP por classe (`ap_per_class.csv`):** não existe nos `metrics.json` da cadeia D; foi **recalculado** pelo analista de `per_image_preds.jsonl` (propostas conf ≥ 0,001; arredondadas a 6 casas) com `EV.candidates/greedy` + `ap_per_class` do ultralytics 8.4.142 (mesmo protocolo de `eval_yolo_cascade.metrics`; 7 classes + média de espécies).

### 9.3 Claims anteriores × B200 (para o orquestrador; o analista não edita o `.tex`)

| Claim (onde) | B200 | Status |
|---|---|---|
| NWD E10 +1,4 p.p. mAP50 (§1, README RQ3) | Δ VAL mAP50 −0,03 ± 0,12; mAP50:95 −0,9 ± 0,5; TEST macro-F1 +0,8 ± 1,5 (3 seeds) | **contradiz** |
| GCD tem o melhor mAP50:95 (§1) | VAL 62,0 ± 0,6 vs E0 62,9 ± 0,1; TEST macro-F1 −2,2 ± 0,6 | **contradiz** |
| P2 piora (§4: F1 67,5 vs 73,2) | 71,2 ± 0,9 vs 72,4 ± 2,9 (Δ −1,2 ± 1,9) | **inconclusivo** (efeito de 5,7 p.p. era ruído/config) |
| SAHI piora (§4: 66,2 vs 73,2) | sem baseline plain@1920 no mesmo protocolo | **inconclusivo** |
| Cascata melhora recall e F1 (§3: +4,4 / +2,1) | TEST: R +3,0 ± 2,2 (6/9 sig.), F1 +0,1 ± 2,2; **VAL: R −0,7 ± 0,3** | recall: **confirma só no TEST**; F1: **contradiz**; VAL: **contradiz** |
| Cascata reduz AP (§2/§3) | AP50 espécie −6,3 (label_only) / −3,1 (product) | **confirma** |
| Partes ≈ CE (§3) | macro-F1 73,8 vs 72,5; oráculo 79,7 vs 79,6 | **confirma** |
| Confusão MF↔outra espécie "irredutível" (PACBB, §5) | MF→MC = MC→MF = 0 %; MF→não-espécie 15,5 % | **não se aplica** a MF↔MC |
| NWD ajuda objetos pequenos | recall <32 px Δ +1,0 ± 6,4 | **sem efeito** |
| "NWD na loss e no assigner" (Método) / colapso w=1 por falta de termo de forma | NWD da loss em grid, saturado (§12.1) | **incorreto** — E10 = NWD só no assigner (efetivo); colapso E12 = artefato; mas o NWD puro em px (E16) também colapsa (§12.2) → mecanismo correto é "gradiente ≈ 0 longe do alvo", não "falta termo de forma" |

### 9.4 CSVs da campanha B200 (todos com `split`, `seeds`, `n_seeds`, `source_path`, `job`; script gerador local `_scratch/b200_{lib,rq3,rq2,rq2_policy}.py`, não versionado)

`rq3_tiny_object/results/`: `nwd_3x3.csv` (24), `tiny_variants_multiseed.csv` (58), `tiny_variants_paired_tests.csv` (15), `tiny_ablation_single_seed.csv` (22), `recall_by_size.csv` (66), `sahi_multiseed.csv` (12), `tta_vanilla_multiseed_PENDENTE_validacao.csv` (6, **não é fato**). `rq2_cascade/results/`: `cascade_test_seeds.csv` e `cascade_val_seeds.csv` (200 cada), `mcnemar.csv` (160), `confusion_single_vs_cascade.csv` (1.184), `oracle_crops.csv` (112, inclui decomposição), `latency.csv` (168), `classifiers_b200_val_multiseed.csv` (6), `policy_comparison.csv` (184), `ap_per_class.csv` (1.524).

Ressalvas: (i) `paired_discordant` do `metrics.json` conta os 7 classes (todos os GT), não só espécies — o analista recalcula b/c só em espécies (conferido contra o `metrics.json` em 7 classes); (ii) cascata e single usam o mesmo matching por GT (pareado por construção); (iii) a latência de detector do `E0_s0` (20,1 ms na cadeia D) é atípica frente a E0_s1/s2 e E10 (≈8 ms); (iv) TEST já foi consultado em campanhas anteriores (consolidado, não cego).

## 10. Campanha v9 (DS-F2_v9.0.5; 2026-10-05/06) — RQ2 cascata; **supera §9 para a RQ2** (dataset novo)

Fonte: `RUNS_ROOT=/raid/user_marcospaulo/flydet_runs_v9`; geradores rastreados `common/analysis/rq2_v9_summary.py` e `rq2_v9_analysis.py`. TEST = 73 imgs (1.210 GT de espécie); VAL = 52 imgs (856 GT de espécie), **VAL escolheu `best.pt` → não é held-out**. Métrica: F1 macro das 4 espécies (MD MV MC MF). Políticas `label_only` e `product` são **idênticas** no harness (artefato conhecido; reportar uma). Pares det×clf cruzados (E0 3 seeds × 3 seeds clf = 9) **não são independentes**.

| Fato | Valor | Split / seeds | CSV |
|---|---|---|---|
| F1 macro detector isolado | E0 (YOLO26m) 79,5 (seeds 79,6/80,7/78,3); YOLO26l 80,5; RT-DETR-l 72,9/77,4/73,6 (média 74,6) | TEST; E0 3 seeds, yolo26l 1 seed, RT-DETR 3 seeds | `v9_cascade_summary.csv` |
| ΔF1 cascata − detector, E0, por classificador | CE −2,7 ± 1,5; partes −1,4 ± 0,9; DINOv2 −1,7 ± 2,0; DINOv3 −2,0 ± 1,4; BioCLIP 2 −1,1 ± 1,1 | TEST, 9 combos (não indep.) | `v9_cascade_aggregate.csv` |
| ΔF1, YOLO26l | CE −4,0; partes −2,6; DINOv2 −2,7; DINOv3 −3,1; BioCLIP 2 −2,1 | TEST, 3 seeds clf, 1 seed det | idem |
| ΔF1, RT-DETR × CE | −5,1 / −4,2 / −2,3 (seeds 0/1/2) | TEST | idem |
| ΔF1 em VAL | −3,0 a −8,0 (E0×CE −5,5) | VAL (não held-out) | idem |
| Efeito da cascata | recall de espécie ≈ igual (E0×CE 80,8→79,6); **precisão cai** (79,2→75,1) | TEST | idem |
| Bootstrap pareado por imagem (10.000 reamostras, IC percentil 95 %; teste de permutação pareado) | E0×CE Δ −2,7 [−4,9; −0,4], p=0,013; E0×partes −1,4 [−3,9; 1,1], p=0,24; E0×DINOv2 −1,7 [−3,6; 0,0], p=0,037; E0×DINOv3 −2,0 [−4,1; 0,0], p=0,040; E0×BioCLIP 2 −1,1 [−2,8; 0,7], p=0,20 | TEST, 73 imgs | `v9_bootstrap_delta_f1.csv` |
| McNemar single × cascata (espécie, por combo) | maioria das combos n.s. no TEST (mediana p 0,10–0,53 por família); significativo em VAL para CE/partes/BioCLIP 2 (mediana p 0,007–0,03) | TEST/VAL | `v9_mcnemar.csv` |
| Abstenção (usar classificador só se max-softmax ≥ τ, senão rótulo do detector) | F1 sobe monotonicamente com τ e **nunca supera o detector** (E0_s0×CE: 79,6 no detector; 76,2 τ=0 → 79,3 τ=0,9 → 79,6 τ≥0,95) | TEST | `v9_hybrid_abstention.csv` |
| Oráculo (crops GT, 15 classificadores) acurácia de espécie | CE 82,7 ± 1,1; partes 83,3 ± 1,9; DINOv2 83,8 ± 2,8; DINOv3 83,5 ± 1,1; BioCLIP 2 83,6 ± 1,4 | TEST, 3 seeds cada | `v9_oracle_summary.csv` |
| Concordância entre os 15 classificadores | todos corretos 60,0 %; todos errados 4,9 % (MD 25/224, MC 17/268, MF 9/416, MV 8/302); voto majoritário 86,3 % vs média individual 83,4 % | TEST, 1.210 GT | `v9_oracle_agreement.csv` |
| Confiança dos erros | 69,4 % dos erros têm confiança ≥ 0,8 (5,3 % < 0,5); ECE 0,043 | TEST | `v9_oracle_confidence.csv` |
| Família MD/MV/MC | acurácia de família (3 espécies colapsadas) 96,0 % vs espécie 80,6 %; 79,2 % dos erros de espécie dentro da família | TEST, 794 GT | `v9_oracle_family.csv` |
| Acurácia do oráculo por tamanho (√wh, px) | <24: 85,7 (n=454); 24–32: 69,4 (159); 32–48: 81,2 (376); 48–64: 92,0 (207); ≥64: 99,5 (14). **Confundido com espécie** (MF concentra <24, MC ≥48) | TEST | `v9_oracle_by_size.csv` |
| Risco × cobertura (média dos classificadores) | ver CSV | TEST/VAL | `v9_oracle_risk_coverage.csv` |

Ressalvas: (i) IC percentil, **não BCa** (BCa e sign-flip por instância não implementados); (ii) o classificador com classe BG (8 classes) tem apenas 1 seed (126) até o re-treino; números BG não são fatos; (iii) fine-tune completo (lr 2e-5) nos modelos fundacionais, **não LP-FT**; (iv) 869 imagens de copy-paste no treino — vazamento VAL/TEST não verificado; (v) TIDE, score ciente de IoU e fusão com distribuição de classe do detector **não foram feitos** (o detector só grava top-1 + conf).

## 10. RQ2 no DS-F2_v9.0.5 (campanha oficial v9; gerado em 2026-10-06; **supera** §9 para o RQ2)

**Setup.** DS-F2_v9.0.5; TEST = 73 imagens, 1210 instâncias de espécie (MD 224, MV 302, MC 268, MF 416); VAL = 52 imagens, 856 instâncias (VAL escolheu `best.pt` → não é held-out). Harness `eval_cascade_matrix` (conf ≥ 0,25, IoU ≥ 0,5, matching class-agnóstico). Política `label_only` (idêntica a `product` no harness atual; artefato conhecido). Métrica: F1 macro das 4 espécies (MD MV MC MF). Runs em `/raid/user_marcospaulo/flydet_runs_v9` (`cascade_{cross,found,rtdetr}_{val,test}`, `oracle_*`, `label_issues`). Foundation models (DINOv2/DINOv3/BioCLIP 2): **fine-tune completo** (lr 2e-5), não LP-FT. Os 9 pares E0 (3 det × 3 clf) **não são independentes**.
**CSVs (geradores em `common/analysis/`: `rq2_v9_summary.py`, `rq2_v9_analysis.py`; testes em `tests/test_rq2_v9_analysis.py`):** `rq2_cascade/results/v9_cascade_summary.csv`, `v9_cascade_aggregate.csv`, `v9_oracle_summary.csv`, `v9_bootstrap_delta_f1.csv`, `v9_mcnemar.csv`, `v9_oracle_{by_size,confidence,agreement,family,risk_coverage}.csv`, `v9_hybrid_abstention.csv`.

### 10.1 Cascata × detector (TEST, multi-seed detector; F1 macro %, ΔF1 = cascata − detector sozinho, IC 95% percentil de bootstrap pareado sobre imagens, p de permutação pareada; 10 000 reamostragens; **IC BCa** desde 2026-10-06, `ci_low_bca/ci_high_bca` no CSV; percentil também no CSV)
| Detector | F1 detector | Clf | n pares | ΔF1 [IC] | p_perm |
|---|---|---|---|---|---|
| E0 (3 seeds) | 79,5 | ConvNeXt-T CE | 9 | −2,7 [−4,8; −0,3] | 0,013 |
| E0 | 79,5 | ConvNeXt-T partes | 9 | −1,4 [−3,7; +1,3] | 0,24 |
| E0 | 79,5 | DINOv2 | 9 | −1,7 [−3,5; +0,1] | 0,037 |
| E0 | 79,5 | DINOv3 | 9 | −2,0 [−4,0; +0,3] | 0,040 |
| E0 | 79,5 | BioCLIP 2 | 9 | −1,1 [−2,7; +0,8] | 0,20 |
| yolo26l (1 seed) | 80,5 | CE / partes / DINOv2 / DINOv3 / BioCLIP 2 | 3 cada | −4,0 / −2,6 / −2,7 / −3,1 / −2,1 | 0,008 / 0,12 / 0,047 / 0,042 / 0,12 |
| RT-DETR-l s0/s1/s2 | 72,9 / 77,4 / 73,6 (média 74,6) | CE | 3 cada | −5,1 / −4,2 / −2,3 | 0,0006 / 0,003 / 0,18 |
Nenhuma configuração (5 backbones × 3 detectores) melhora o detector sozinho. IC por linha usa 73 imagens: **TEST pequeno**. VAL (ΔF1 −3,1 a −8,0; `v9_bootstrap_delta_f1.csv`) é pior, mas VAL não é held-out. McNemar (por par, `v9_mcnemar.csv`): p varia entre pares de mesma configuração (ex.: CE+E0_s1 p∈[0,005; 0,51]); **não resumir como um único p**.

### 10.2 Oráculo (classificador em crops GT; acurácia de espécie em TEST, mean±std entre seeds)
CE 82,7 ±1,0 (n=3 seeds únicos); partes 83,3±1,5; DINOv2 83,8±2,3; DINOv3 83,5±0,9; BioCLIP 2 83,6±1,1 (todas ≈ 83%; `v9_oracle_summary.csv`). VAL: 87,1–88,3. Teto do oráculo ≈ teto do detector (F1 ≈ 80–81).

### 10.3 Diagnóstico do teto (15 classificadores, TEST; `v9_oracle_*.csv`)
- Concordância: 60,0% das instâncias acertadas por todos os 15; **4,9% erradas por todos os 15**; 35,1% mistas; voto majoritário 86,3% (média individual 83,4%). All-wrong por classe: MD 25/224, MV 8/302, MC 17/268, MF 9/416.
- Confiança dos erros: 69,4% dos erros têm max-softmax ≥ 0,8 (confiantes); só 5,3% < 0,5; ECE 0,043.
- Erros dentro da família MD/MV/MC: 79,2% dos erros de instâncias da família; acurácia no nível de família 96,0% (vs 80,6% por espécie).
- Tamanho (sqrt(wh) em px da imagem original): acurácia 85,7% (<24; n=454), 69,4% (24–32; 159), 81,2% (32–48; 376), 92,0% (48–64; 207), 99,5% (≥64; 14). **Não monotônica** (composição de classes confunde: MD tem acc 54% em <24 e 65–68% em 24–64) → tamanho sozinho **não** explica o teto de MD.
- Risco×cobertura (oráculo, média dos clf): acc 83,4% (100%) → 88,4% (80%) → 90,5% (50%).
- Rótulos suspeitos (confident learning, out-of-sample, **suspeito ≠ errado; sem especialista**): 3,7% (127/3470) em TEST e 4,1% (96/2355) em VAL; confusões dominantes NOISE↔MF, MD↔MV, MC↔MV (`label_issues_{val,test}.csv` em `flydet_runs_v9/label_issues`; ainda sem cópia em `rq2_cascade/results/`).

### 10.4 Cascata híbrida com abstenção (E0_s0, TEST; usa o clf só se max-softmax ≥ τ, senão rótulo do detector)
Detector 79,6; melhor τ=0,9 → ~79,8 (partes), 79,8 (BioCLIP 2), 79,3 (CE): ganho ≤ +0,2 pp, dentro do ruído. `v9_hybrid_abstention.csv`.


### 10.5 Verificação de vazamento treino → VAL/TEST (CPU; `common/analysis/leakage_check.py`; `rq2_cascade/results/v9_leakage_check.csv`)
Método: correlação normalizada de miniaturas cinza (crops 32×32, com e sem espelhamento; imagens 48×48) contra todo o treino (1738 imagens, 869 de copy-paste). **Imagem inteira:** nenhuma imagem de VAL/TEST tem similaridade ≥ 0,98 com o treino (máx. 0,949 VAL; 0,962 TEST); ≥ 0,95: 0/52 VAL e 5/73 TEST. **Crops:** ≥ 0,98 em 82/2347 (3,5%) VAL e 161/3460 (4,7%) TEST (53 destes têm como vizinho mais próximo um crop de imagem de copy-paste); ≥ 0,95 em 11,3% e 11,9%. **Leitura:** sem duplicata de imagem; a similaridade alta de crops é compatível com fundo de cola dominando crops pequenos e **não prova vazamento** (sem controle com crops de treino contra treino). Conclusão permitida: "no near-duplicate images between train and VAL/TEST (cosine ≥ 0.98); a small share of crops are highly similar to training crops, which we could not attribute to copy-paste".

### 10.6 Confusão, latência, localização e fusão tardia (CPU; `common/analysis/rq2_v9_extra.py`; CSVs `v9_confusion_single_vs_cascade.csv`, `v9_latency.csv`, `v9_localization.csv`, `v9_late_fusion_approx.csv`; TEST, multi-seed, 9 pares E0×CE)
- Confusão E0 → cascata CE (% das instâncias GT): MD→MV 26,3→21,0 (Δ −5,3±3,5); MD→não-espécie 7,0→15,0 (Δ +8,0±5,1); MV→MD 1,4→6,3 (Δ +4,9±1,2); MV→não-espécie 4,7→1,5; MC→MV 11,7→13,1 (Δ +1,4±6,2); MC→não-espécie 3,0→0,7; MF→não-espécie 8,7→7,5; MF→missed 5,3 (igual).
- Localização (species_detection_recall, TEST): E0 97,7±0,4 (seeds 0–2); RT-DETR 98,5.
- Latência (TEST, B200, FP32, 1920; ms/img, média entre execuções; varia entre jobs): detector 8,6–11,6; e2e E0+CE 407, parts 353, DINOv3 487, BioCLIP2 462, DINOv2 573 → ~30–67× o detector.
- Fusão tardia **APROXIMADA** (distribuição do detector inferida da conf top-1; p_clf·p_det^w), F1 macro TEST: E0 79,54; cascata (w=0) 76,87; w=0,25 77,95; 0,5 78,72; 1 79,53; 2 79,56. VAL: E0 84,67; w=2 84,35. Não é a fusão com scores reais do detector.

### 10.6b BG (classificador de 8 classes com fundo; 3 seeds de clf × detector E0 s0–2; `v9_bootstrap_delta_f1.csv`, `oracle_bg_*/clf_*/metrics.json`)
- TEST: ΔF1 −2,75 pp, IC BCa [−4,78; −0,44], p_perm 0,010 (CE: −2,67). VAL: −5,43 (CE −5,52). Oráculo BG (acurácia de espécie, crops GT): TEST 77,55±2,78; VAL 80,84±1,93.

### 10.6c Curva de aprendizado (ConvNeXt-T seed 42, single-seed; `v9_learning_curve.csv`, gerador `common/analysis/rq2_v9_lc_bg.py`; job 34130)
- Acurácia no oráculo, 25/50/100% dos crops: TEST 83,0/81,7/82,6; VAL 85,7/85,5/88,9. Platô no TEST: compatível com ambiguidade, não prova. BG oráculo em `v9_oracle_bg.csv`.

### 10.7 Pendente (não citar no paper antes de rodar)
TIDE; score ciente de IoU; fusão tardia com scores reais do detector; RF-DETR+DINOv2 (sem `rfdetr` no container); oráculo por nitidez/aglomeração

## 11. RQ1 — escada de dados (TEST DS-F2_v9.0.5 fixo, receita E0, YOLO26m, **multi-seed 0/1/2**; gerado em 2026-10-14)
Fonte: `rq1_data/results/{ablation_ladder,ablation_ladder_summary,ladder_*}.csv`; gerador `common/analysis/rq1_ladder_analysis.py`; job 34630; W&B `pestline/paper2-rq1-data`. L3 = E0_s0/1/2 da RQ2. TEST (73 imgs, 1.210 GT de espécie); best.pt escolhido na VAL, TEST não participa. Acurácia por espécie = casado (IoU≥0,5, conf≥0,25) E classe certa / GT.
- Acurácia média±dp (3 seeds): L0 0,561±0,006; L1 0,738±0,004; L2 0,797±0,015; L2ORG 0,811±0,012; L3 0,822±0,016. Recall de detecção (sem classe): 0,923 / 0,960 / 0,976 / 0,979 / 0,977.
- Δ vs L3 (bootstrap por imagem, 10k, seeds agrupadas): L0 −0,261 [−0,335;−0,189]; L1 −0,084 [−0,123;−0,049]; L2 −0,025 [−0,048;−0,001]; L2ORG −0,012 [−0,029;+0,003] (meia-largura 0,016). McNemar por seed: L0/L1 p<1e-8 nas 3 seeds; L2 p=0,019/1e-7/0,57; L2ORG p=0,055/0,15/0,93. L2ORG≈L3 **não é equivalência provada** (IC inclui 0).
- Recall por tamanho (<32 / 32–48 / ≥48 px, média de 3 seeds, ver `ladder_size_bins.csv`): ganho L0→L3 concentrado em objetos <32 px (~0,44→~0,81) e continua nas outras faixas.
- FP de espécie em 6 placas só-INS (conf≥0,25, por seed): L0 74/87/66; L1 45/57/43; L2 0/0/0; L2ORG 0/3/2; L3 0/0/0. Salto entre L1 e L2 (placas v8/v9 com INS).
- Placas densas (≥30 espécies; 7 placas, 230 GT): L0 ~0,59; L1 ~0,75; L2 ~0,79; L2ORG ~0,83; L3 ~0,84 — melhora, mas amostra pequena (7 placas), seeds ruidosas.
- Recall MD (pior espécie): L0 0,44; L1 0,66; L2 0,69; L2ORG 0,64; L3 0,65 (dp até 0,13). Papel branco (L2→L2ORG) não melhora MD; ganho L2→L2ORG está em MF/MV.
- H2 (confusão entre localizados, `ladder_confusion.csv`, seeds agrupadas): MC→MV 18,5% (L0) → 14,5% (L2) → 11,7% (L3); MV→MC 14,8% → 6,3% → 5,4%; MF→não-espécie 46% → 17% → 9%. **H2 (confusão inalterada) NÃO se sustenta integralmente**: MV↔MC cai, mas MC→MV persiste em ~12% em L3.
- Ressalvas: L0 é pequeno (lote v5.1); quantidade e qualidade estão confundidas entre braços (controle de subamostra de L2 não rodado); L2ORG tem ORG≈42% do treino; cópia-e-cola só em L3 (L2→L3 mistura copy-paste + DS-ORG removido).

### 11b. RQ1 — controle de mesmo tamanho (subamostras aleatórias do L3; TEST; receita E0; 3 seeds em todos os tamanhos; 2026-10-19)
Fonte: `rq1_data/results/equalsize_summary.csv`; gerador `common/analysis/rq1_equalsize.py` (+ `equalsize_prior_ovx.csv` para as execuções anteriores); subamostras `rq1_data/make_equalsize.py` (random.Random(1000*N+seed); seed de amostragem = seed de treino). Cadeia h100: job 34950, `rq1_data/slurm/chain_rq1_equalsize_h100.sh`, runs em `/raid/user_marcospaulo/flydet_runs_rq1_h100`; as anteriores (EQ244 s0–2, EQ869 s0–1) rodaram no nó ovx (L40S). **Hardware misto** (H100 vs L40S): ressalva obrigatória; single-seed por célula.
- EQ103 (3 seeds, H100): 0,7539±0,0067 (s0 0,7500; s1 0,7500; s2 0,7616) vs **L0 0,5609±0,0055** (103 imagens do lote v5.1): **+19 pp**.
- EQ244 (3 seeds, ovx): 0,7515±0,0766 (s0 0,6876; s1 0,8364; s2 0,7306) vs L1 0,738±0,004.
- EQ869 (3 seeds: s0/s1 ovx, s2 H100): 0,8001±0,0068 (s0 0,8017; s1 0,7926; s2 0,8060) vs L2 0,797±0,015.
- Leitura: em 244 e 869 imagens, L1 e L2 não se distinguem de subamostras aleatórias do L3 (a variância entre amostras no EQ244 é maior que o efeito) → quantidade explica a subida ali. **No menor degrau o resultado é o oposto**: 103 imagens aleatórias do L3 (0,754) superam em ~19 pp as 103 do L0 (0,561), com dp baixo — o L0 é pior por qualidade/versão/lote (rótulos, domínio), não só por quantidade. "Mostly quantity" NÃO vale para L0→L1. Não prova equivalência nos demais degraus. L3 completo: 0,8223±0,0158.
- Ressalva: o pool do L3 contém rótulos das versões finais (inclui copy-paste, DS-ORG removido) — o ganho a 103 imagens pode vir de rótulos mais limpos, não de "alvo" da coleta.

## 12. RQ3 no DS-F2_v9.0.5 — campanha dgx1/V100 (E0 × NWD, **multi-seed 0/1/2**; gerado em 2026-10-16)

> **⚠️ 2026-10-18 (§12.1):** o E10 desta tabela tem o NWD da loss saturado (bug de unidade). Os números abaixo valem para *NWD no assigner + CIoU ×0,5 na loss*, não para "NWD na loss e no assigner". A leitura "NWD não transfere" fica **suspensa** até o E15 (NWD corrigido, 3 seeds). **2026-10-19:** E15 concluído (§12.2) → leitura **confirmada** com o NWD corrigido (Δ macro-F1 +0,1 ± 2,8).
**Campanha separada** da B200 (§9/§10/§11): 1× V100 32 GB, Docker `pestline-rq3:cu124` (torch 2.6.0+cu124, ultralytics 8.4.142, fly_det b36580a); E0 aqui é o baseline **desta** campanha — não misturar com o E0 da §10 na mesma tabela. Receita E0 idêntica (yolo26m, 150 ép., batch 8, imgsz 1920, patience 30, `best.pt` na VAL); E10 = 0,5·CIoU + 0,5·NWD, C=44 px (medido no v8.1.1, não recalculado), scope=both. TEST 73 imgs / 3.494 GT / 1.210 de espécie (MD 224, MV 302, MC 268, MF 416), harness P2 (`eval_cascade_matrix`, conf ≥ 0,25, IoU ≥ 0,5, class-agnóstico). VAL 52 imgs / 2.359 GT (não held-out). 83 HEIC convertidas para PNG antes do treino (sem isso o TEST contava 2.980 GT). Fonte: `rq3_tiny_object/results/v9_dgx1_nwd_{runs,summary,paired,size_bins}.csv`; gerador `common/analysis/rq3_v9_dgx1_analysis.py`; runs em `rq3_tiny_object/results/runs.md`. Média ± desvio amostral (ddof=1); Δ = E10 − E0 pareado por seed.

| Métrica | E0 (n=3) | E10 NWD (n=3) | Δ E10−E0 (média ± std; por seed) |
|---|---:|---:|---|
| TEST macro-F1 espécie | 78,8 ± 2,6 | 79,3 ± 0,8 | +0,5 ± 3,1 (+0,6 / −2,6 / +3,5) |
| TEST micro R / P / F1 | 81,7±0,9 / 77,8±4,1 / 79,7±2,3 | 81,7±0,6 / 78,6±0,8 / 80,1±0,6 | R +0,0 ± 0,9; P +0,8 ± 4,2; F1 +0,5 ± 2,5 |
| TEST acc\|loc | 83,5 ± 0,8 | 83,2 ± 0,3 | −0,2 ± 0,7 |
| TEST recall de detecção (espécie, class-agnóstico) | 97,9 ± 0,3 | 98,2 ± 0,4 | +0,3 ± 0,6 |
| VAL mAP50 (best.pt, 7 classes) | 88,5 ± 0,4 | 87,3 ± 0,6 | **−1,1 ± 0,4** (3/3 negativos) |
| VAL mAP50:95 | 63,8 ± 0,1 | 62,0 ± 0,4 | **−1,7 ± 0,4** (3/3 negativos) |
| Recall TEST por espécie MD / MV / MC / MF | 63,7 / 86,8 / 87,7 / 83,8 | 62,5 / 91,4 / 87,8 / 81,1 | −1,2±3,9 / **+4,6±0,7** / +0,1±6,1 / −2,7±3,3 |
| Épocas até early stop (best) | 75 (45 ± 7) | 86 (56 ± 12) | — |
| Tempo de treino (V100) | 2,20 ± 0,22 h | 2,54 ± 0,31 h | +0,34 h (~+15 %) |

- **McNemar exato** (acerto de espécie, 1.210 GT, por seed): s0 b=64/c=70, p=0,67 (Δacc −0,50); s1 b=54/c=60, p=0,64 (−0,50); s2 b=80/c=67, p=0,32 (+1,07). **0/3 significativos.**
- **Bootstrap por imagem** (seeds agrupadas, 10.000 reamostras, 67 imagens com espécie, IC percentil): Δacc **+0,03 p.p. [−1,57; +1,68]** — IC inclui 0; meia-largura 1,6 p.p.
- **Recall de espécie por tamanho** (√wh px; <32 n=613, 32–48 n=376, ≥48 n=221): Δ <32 **−1,7 ± 1,1** (3/3 negativos: −1,5/−0,8/−2,9); 32–48 +2,9 ± 3,5; ≥48 +0,0 ± 2,0 → **NWD não ajuda os objetos pequenos** (se algo, piora levemente).
- Latência do detector (V100, FP32, 1920): E0 42,4 ± 1,0 ms/img; E10 43,0 ± 0,1 (NWD não muda a inferência).
- **Leitura:** reproduz a §9.1 (B200, v8.1.1: Δ macro-F1 +0,8 ± 1,5, sem efeito) no dataset do paper: **NWD não melhora a identificação de espécie** (Δ < std em todas as métricas de TEST, 0/3 McNemar) e **piora a localização fina no VAL** (mAP50:95 −1,7 ± 0,4, mesmo sinal da B200 −0,9 ± 0,5). Único efeito por espécie com |média| > std: recall MV +4,6 ± 0,7 (sem correção para múltiplas comparações; 4 espécies × várias métricas → exploratório). A variância entre seeds do E0 (macro-F1 std 2,6) é maior que o efeito.
- **Não rodado no v9:** GCD, P2, RFLA, ablações loss/assigner, SAHI — para esses, só existe a §9 (B200, v8.1.1).

## 13. RQ3 no DS-F2_v9.0.5 — campanha ovx (L40S; E0 + 8 variantes de tiny-object; **TEST, multi-seed 0/1/2**, exceto E22; gerado em 2026-10-19)
**Campanha separada** (ovx-l40s-01, Slurm `ovx01`, Apptainer ultralytics 8.4.142, torch 2.6.0+cu124): E0 desta campanha é o único controle válido para as variantes abaixo (não misturar com E0 da §10 nem da §12). Receita idêntica ao E0 (yolo26m, 150 ép., batch 8, imgsz 1920, patience 30; E21 batch 4). TEST = 73 imgs, n_gt=3494. Fonte: `rq3_tiny_object/results/v9_ovx_{runs,summary,size_bins,paired,sahi_tta}.csv` (geradores `common/analysis/rq3_v9_analysis.py`, `common/analysis/rq3_ovx_infer_csv.py`); jobs em `runs.md`. Nota: o best.pt foi escolhido em VAL (VAL não é held-out); todos os números abaixo são TEST.

| Braço (n seeds) | Acurácia espécie (média ± std) | Δ vs E0 pareado [IC95 bootstrap por imagem] | p Holm | Recall <32 px | 32–48 | ≥48 |
|---|---:|---|---:|---:|---:|---:|
| E0 controle (3) | 0,817 ± 0,012 | — | — | 79,4 | 81,0 | 89,4 |
| E11 GCD (3) | 0,815 ± 0,004 | −0,002 [−0,018; +0,016] | 1,00 | 78,3 | 80,5 | 92,3 |
| E14 NWD+RFLA (3) | 0,729 ± 0,075 | **−0,088 [−0,127; −0,049]** | 0,002 | 63,3 | 78,5 | 90,0 |
| E1 P2 (3) | 0,790 ± 0,027 | −0,028 [−0,050; −0,004] | 0,139 | 74,0 | 80,1 | 91,0 |
| E20 Wise-IoU v3 (3) | 0,814 ± 0,010 | −0,003 [−0,019; +0,015] | 1,00 | 78,9 | 79,6 | 91,4 |
| E21 imgsz 2560 (3) | 0,780 ± 0,033 | −0,038 [−0,061; −0,011] | 0,057 | 72,8 | 79,3 | 90,0 |
| E15 NWD scope=loss (3) | 0,793 ± 0,034 | −0,025 [−0,045; −0,005] | 0,084 | 74,0 | 80,8 | 91,3 |
| E16 NWD scope=assigner (3) | 0,820 ± 0,019 | +0,003 [−0,014; +0,024] | 1,00 | 79,4 | 80,6 | 91,7 |
| E22 YOLO11m (**single-seed s0**, exploratório) | 0,810 | −0,012 [−0,044; +0,021] | 1,00 | 76,3 | 83,0 | 90,5 |

- **Resultado geral: nenhuma variante melhora o E0** (nenhum Δ positivo significativo). Recall por tamanho: Δ <32 px ≤ 0 em todas (E16 = 0,0); ganhos em ≥48 px (+0,6 a +2,9) vêm com std alto (1–5 pp) e não são inferência.
- **Equivalentes ao E0 (|Δ|<1 pp, TOST ±1 pp)**: o IC95 de E11, E20 e E16 (meia-largura ≈1,6–1,9 pp) é mais largo que a margem — **não equivalência demonstrada; apenas sem diferença detectável** com 73 imgs/3 seeds.
- **E14 (NWD+RFLA) é negativo e instável**: s0 early stop na ép. 40 (best ≈ ép. 10); **s1 divergiu** (cls_loss ~2150 desde ép. 27, nan na ép. 30; receita mantida, sem retry); recall <32 px 63,3 ± 14,1. Reportar como resultado negativo/instável, com Δ por seed.
- **E21 (2560) e E1 (P2) pioram** o recall <32 px (−6,6 e −5,4 pp) apesar de mais resolução/escala; com Holm só E14 sobrevive a 5 % (E21 p=0,057).
- **Escopo do NWD**: assigner (E16) ≈ E0 (+0,3 pp), loss (E15) −2,5 pp [−4,5; −0,5], loss+assigner+RFLA (E14) −8,8 pp → o dano vem da perda NWD e do RFLA, não do assigner.
- **Inferência nos 3 E0 (TEST; `v9_ovx_sahi_tta.csv`; protocolos distintos, não comparar entre si)**: SAHI tile 512/ov 0,25 (sahi_inference, 7 classes, conf 0, IoU 0,5): recall 86,7 ± 1,8 % vs 35,5 ± 0,9 % do frame inteiro a 512 px (plain) e mAP50 69,5 ± 0,8 vs 9,5 ± 1,0 — **o plain a 512 não é baseline justo** (downscale); o baseline justo é o E0 a 1920 (mAP50 vanilla 84,2 ± 0,5; mAP50:95 61,7 ± 0,1). TTA (flip/escala) 1920: mAP50 84,6 ± 0,2 (+0,4), mAP50:95 62,2 ± 0,1 (+0,5) — ganho pequeno, 3 seeds, single-split. WBF ensemble dos 3 E0: mAP50 83,6 (< vanilla médio 84,2: **sem ganho**; calculado em conf 0,001).
- **Não rodado nesta campanha**: E10 NWD (ver §12, V100), E7 yolo26l, E22 s1/s2.

## 11c. Composição do treino por lote (DS-F2_v9.0.5) — rodada 6, M5
Fonte: `rq1_data/results/ladder_composition.csv` (gerador `common/analysis/ladder_composition.py`).
| lote | originais | aug. de imagem | reais | copy-paste |
|---|---|---|---|---|
| v5.1 | 58 | 45 | 103 | 103 |
| v6.0 | 66 | 5 | 71 | 71 |
| v7 | 56 | 14 | 70 | 70 |
| v8 | 80 | 172 | 252 | 252 |
| v9 | 139 | 234 | 373 | 373 |
L0=103; L1=103+71+70=244; L2=244+252+373=869; L3=1738 = L2 + 869 copy-paste (um por imagem real). Os "97, 83, 70, 100, 174 imagens por lote" do texto antigo não tinham rastro e foram substituídos. (O "607" antigo em §9/§14 foi corrigido para 869, a contagem por nome de arquivo e por construção L3 = 2×869.)

## 11d. Variantes da cascata (pós-processamento, CPU) — rodada 6, M2
Fonte: `rq2_cascade/results/v9_cascade_variants{,_summary}.csv`; gerador `common/analysis/rq2_cascade_variants.py`; entradas `per_image_preds.jsonl` de `cascade_cross_{val,test}` (E0 s0–2 × CE s42/84/126 = 9 pares, copiadas da B200; avaliação original na B200). **Sem GPU nova; multi-seed (9 pares não independentes); w escolhido na VAL (que também escolheu os best.pt) e aplicado ao TEST.**
Acurácia de espécie (all-GT), TEST, média dos 9 pares: single 0,8223; cascata (relabel total) 0,8135 (Δ −0,9 pp, IC95 imagem [−2,9; +1,2]); **species4** (reclassifica só propostas de espécie entre as 4) 0,8079 (Δ −1,4 pp [−2,9; +0,1]); **fusão tardia** w=0,55: 0,8353 (**Δ +1,3 pp, IC95 [+0,3; +2,3], 9/9 pares > single**); fusão restrita às 4 espécies (w=0,70): 0,8226 (Δ +0,0 pp [−0,03; +0,1]).
Fusão: argmax_c [w·conf_det·1(c=classe_det) + (1−w)·p_clf(c)]; o head end-to-end só expõe classe top-1 e score (não a distribuição completa). VAL: fuse +0,2 pp (5/9 pares) — o ganho é maior no TEST que na VAL, não é "selecionado por overfit de VAL" mas o w foi escolhido nela.

## 11e. Ablação da margem do crop e controle de hardware (H100) — rodada 7, M2
Fonte: `rq2_cascade/results/v9_cascade_variants_pad{00,25,75}{,_summary}.csv`; gerador `common/analysis/rq2_cascade_variants.py <RUNS> <PAD>`; jobs Slurm 34953 (pad 0 e 0,25: crops GT do v9.0.5, ConvNeXt-T CE 3 seeds por margem, receita do paper) e 34965 (referência pad 0,75 reavaliada na H100 com os classificadores treinados na B200). Detectores E0_s0..2 (B200). **Tudo avaliado na H100; multi-seed (9 pares E0×clf, não independentes); w da fusão escolhido na VAL; IC95 por imagem (10 000 reamostragens).**
TEST, acurácia de espécie, média dos 9 pares; Δ vs detector sozinho (single H100 = 0,8160; na B200 era 0,8223 — mesmos checkpoints, inferência difere entre GPUs):
| margem (por lado) | cascata Δ [IC95] | species4 Δ | fusão tardia Δ [IC95] |
|---|---|---|---|
| 0 | −2,5 pp [−5,3; +0,6] | −1,1 pp | +0,03 pp [−0,02; +0,09] (w=0,75) |
| 0,25 | −2,8 pp [−5,0; −0,3] | −2,3 pp [−3,8; −0,7] | +0,05 pp [−0,01; +0,13] (w=0,75) |
| 0,75 (paper) | −0,9 pp [−3,1; +1,5] | −1,5 pp [−3,2; +0,3] | +0,3 pp [−0,1; +0,7] (w=0,65) |
Leitura: mais contexto no crop reduz a perda da cascata, mas nenhuma margem supera o detector sozinho; a fusão tardia com o score do detector **não** é robusta: +1,3 pp [+0,3; +2,3] na B200 (§11d) e +0,3 pp [−0,1; +0,7] na H100 com a mesma configuração. Não reportar a fusão como ganho; reportar como "dentro do ruído de hardware". VAL: cascata −4,8 pp (0,25) e −3,4 pp (0,75) vs single.
Caveats: trainer grava "pad75" no metadado de pré-processamento mesmo nas margens 0 e 0,25 (rótulo errado; margem real está no nome do run e em FLYDET_CROP_PAD); classificadores pad 0,75 treinados na B200, pad 0/0,25 na H100; margem de inferência igual à de treino em cada caso.

## 11f. Tabela por espécie, detector vs cascata (TEST, E0 × ConvNeXt-T CE, 9 pares, multi-seed; tabela `tab:persp`)
Fonte: `rq2_cascade/results/v9_per_species_single_vs_cascade.csv`; gerador `common/analysis/rq2_per_species_table.py` (matching do harness: conf≥0,25, IoU≥0,5; precisão = GT casados com a classe correta / propostas rotuladas com a classe; erro de contagem = média por imagem de |n_pred − n_GT|, por imagem porque não há id de placa). Detector: MD R/P/F1 64,9/78,8/71,2; MV 87,9/69,0/77,3; MC 84,6/86,1/85,3; MF 86,0/82,9/84,4. Cascata: MD 61,4/70,4/65,5; MV 85,6/68,8/76,2; MC 84,3/86,3/85,3; MF 87,1/74,9/80,5. Macro-F1: 79,5 vs 76,9 (confere com §10.1). Erro de contagem (detector → cascata): MD 1,17→1,49; MV 1,67→1,96; MC 0,76→0,78; MF 0,83→1,41. Leitura: a cascata perde precisão em MF (−8,0 pp) e recall em MD (−3,5 pp); nenhuma espécie melhora de forma material. TEST, sem IC por espécie.

## 14. RQ3 cadeia G — NWD ajustado + controle E0 na H100 (TEST 73 imgs, 1210 inst. de espécie; 2026-10-20)
Fonte: `rq3_tiny_object/results/v9_g_tuned_{runs,summary}.csv`; gerador `common/analysis/rq3_g_tuned_analysis.py`; job 35018 (h100n2, jail+gpu_watch). Avaliação no TEST completo (cópia PNG `DS-F2_v9.0.5_evalpng`: 9 HEIC convertidos; sem isso o harness avaliava só 64 imagens). Todos os números TEST; H100 (não misturar com B200/L40S). Δ = diferença pareada por seed vs G0, IC95 bootstrap por imagem.
| braço | seeds | acc média±dp | por seed | Δ vs G0 [IC95] |
|---|---|---|---|---|
| G0 E0 (controle H100) | 3 | 82,2±1,9 | 84,4/80,7/81,6 | — |
| G1 NWD w0,5 C22 | 3 | 81,7±1,5 | 83,2/80,3/81,6 | −0,50 pp [−2,67; +1,78] |
| G2 NWD w1,0 C44 | 3 | 53,9±1,3 | 52,7/53,6/55,4 | −28,29 pp [−34,93; −21,44] |
| G3 NWD w0,5 C88 | **2 (s2 cancelada por prazo)** | 81,8±3,7 | 79,2/84,5 | −0,70 pp [−3,02; +1,78] (por seed −5,21/+3,80) |
Leitura: nenhum braço NWD ajustado supera o controle; w=1,0 degrada fortemente (colapso consistente nas 3 seeds). Controle H100 (82,2) coerente com L40S (81,7).
Cadeia 35018 cancelada após G3 s1 (G3 s2 não rodou). Seeds de G3 divergem em 9 pp: resultado inconclusivo, não vantagem.
### 12.1 Diagnóstico: unidade do NWD na loss (2026-10-18; levantado pelo R1 W1/Q1 da rodada 01 e por Isaac em 2026-10-16)

**Fato.** No Ultralytics 8.4.142 (`ultralytics/utils/loss.py`, `v8DetectionLoss.get_assigned_targets_and_loss`) o assigner recebe as caixas em **pixels** (`pred_bboxes.detach() * stride_tensor`, `gt_bboxes`, l. 428), mas `BboxLoss.forward` recebe as caixas em **unidades de grid** (`pred_bboxes` de `bbox_decode` + `target_bboxes / stride_tensor`, l. 449). O `nwd_patch.py` usava C = 44 nos dois pontos → **na loss, C = 44 células** = 352 / 704 / 1.408 px em P3/P4/P5. A CIoU é invariante à escala (não afetada); o GCD usa escala local (invariante, salvo `clamp(min=1)`); P2, SAHI e RFLA não usam o NWD.

**Evidência.**
- CPU, caminho real da loss (DetectionModel yolo26 → E2E loss → `BboxLoss.forward`), batch sintético 1920² com caixas 40×40 px: `target` máx. na loss = 214 (não 1.920); lado médio da caixa = 4,4 (não 40); NWD para erro W₂ ≈ 4,9 px = **0,988 em grid vs 0,895 em px**. Gradiente do termo NWD em grid = 0,14× / 0,07× / 0,035× do correto (P3/P4/P5).
- dgx1, treino real `E15_nwdfix_w0.5_s0` (log `~/pestline/logs/E15_nwdfix_w0.5_s0.log`, 2026-10-18 09:04): `target_max_in=240.0` (= 1.920/8, grid) e `target_max_px=1920.0` após a correção; `nwd_mean` 0,79–0,90 nas 4 primeiras chamadas (variando entre batches), `nwd_min` ≈ 0. As 20 linhas `[nwd_diag]` vêm do processo de treino → o patch age no treino (vale também para os E10 antigos: o NWD do assigner estava ativo).

**Consequências para os números existentes.**
- **E10** (§9.1 B200/v8.1.1 e §12 dgx1/v9): efetivamente *NWD (correto) no assigner + 0,5·CIoU + termo NWD quase constante na loss* — **não** "NWD na loss e no assigner".
- **E12** (w = 1,0): o termo IoU da loss fica ≈ constante; resta só a L1 de ltrb (YOLO26 tem `reg_max=1`, sem DFL). O colapso é **artefato de implementação**, não propriedade do NWD. **⚠️ Revisto em 2026-10-19 (§12.2):** o E16 (NWD puro em px, sem o bug) também colapsa; o E12 continua inválido como evidência, mas "o NWD puro converge sem o bug" é falso.
- **Ablação "NWD só loss"** (§9.1, B200): testou *CIoU com peso 0,5*, não NWD. **"NWD só assigner"** é válida.
- **GCD, P2, SAHI, RFLA**: inalterados.

**Correção.** `FLYDET_NWD_LOSS_UNITS=px` multiplica as caixas pelo stride antes do NWD da loss (`(pred_bboxes * stride)[fg_mask]`); CIoU e assigner inalterados. Default `grid` = legado (reproduz E10/E11/E12). Testes CPU `tests/test_nwd_patch.py::TestNWDLossUnits` (loss em px = NWD em pixels em P3/P4/P5; grid satura > 0,98). Verificado também com predições em fp16 (AMP): loss e gradientes finitos para erros de 5 a 600 px.

**Reexecução (dgx1/V100, mesma campanha da §12; ✅ concluída em 2026-10-18 — resultados na §12.2).**

| Run | Config | Seeds | Papel |
|---|---|---|---|
| `E15_nwdfix_w0.5_s{0,1,2}` | 0,5·CIoU + 0,5·NWD, C = 44 px, scope=both, **loss em px** | 0/1/2 | substitui o E10 como "NWD na loss e no assigner"; comparar com E0 da §12, pareado por seed |
| `E16_nwdpure_w1.0_px_s0` | NWD puro (w = 1,0) na loss e no assigner, loss em px | 0 | refaz o E12 sem o bug; **single-seed ⇒ exploratório** |

Mesma receita do E0 (§12); C = 44 px mantido (medido no v8.1.1, não recalculado) para mudar uma variável só. Risco conhecido do E16: com erro grande o NWD puro dá gradiente ≈ 0 (teste CPU: |grad| ~100× menor que com CIoU para erros de ~600 px); se colapsar, é resultado legítimo ("NWD sozinho não basta"), distinto do E12.

**Frase segura até o E15:** "nenhuma variante testada melhora a identificação de espécie além da variância entre seeds"; o E10 só pode ser descrito como "NWD no assigner (em pixels); na loss, a mistura reduz efetivamente o peso da CIoU pela metade".

### 12.2 Resultado do rerun E15/E16 (NWD com a loss em px; gerado em 2026-10-19)

Fila dgx1 18/out 09:04 → 17:36, os 4 runs `DONE`, `loss_units=px` confirmado no log e no `nwd_config.json`, seeds distintas, 0 erros. Mesma campanha, TEST, harness e E0 da §12 (Δ pareado por seed contra o mesmo E0). Fonte: `rq3_tiny_object/results/v9_dgx1_nwd_{runs,summary,paired,size_bins}.csv` (variantes `E15_nwd_px`, `E16_nwdpure_px`; as linhas E0/E10 foram regeneradas e são idênticas às de 2026-10-16); gerador `common/analysis/rq3_v9_dgx1_analysis.py`. VAL = validação final do `best.pt` no log (não held-out).

| Métrica | E0 (n=3) | E15 NWD px w=0,5 (n=3) | Δ E15−E0 (média ± std; por seed) | E16 NWD puro px (s0, exploratório) | Δ E16−E0 s0 |
|---|---:|---:|---|---:|---:|
| TEST macro-F1 espécie | 78,8 ± 2,6 | 78,8 ± 0,8 | +0,1 ± 2,8 (−1,4 / −1,7 / +3,3) | 66,7 | −12,7 |
| TEST micro F1 | 79,7 ± 2,3 | 79,7 ± 0,9 | −0,0 ± 2,8 (−1,5 / −1,8 / +3,2) | 68,0 | −12,2 |
| TEST micro R / P | 81,7 / 77,8 | 81,3 ± 2,2 / 78,1 ± 0,4 | R −0,4 ± 3,1; P +0,3 ± 4,2 | 69,3 / 66,6 | R −13,3; P −11,1 |
| TEST acc\|loc | 83,5 ± 0,8 | 83,2 ± 2,4 | −0,3 ± 3,2 (−3,9 / +0,8 / +2,2) | 71,7 | −12,7 |
| TEST recall de detecção (espécie) | 97,9 ± 0,3 | 97,8 ± 0,3 | −0,1 ± 0,0 | 96,8 | −1,2 |
| VAL mAP50 | 88,5 ± 0,4 | 87,5 ± 1,2 | −0,9 ± 1,1 (+0,3 / −1,4 / −1,7) | 78,2 | −10,2 |
| VAL mAP50:95 | 63,8 ± 0,1 | 62,3 ± 1,0 | **−1,5 ± 1,0** (3/3 negativos: −0,5 / −2,5 / −1,4) | 48,9 | −14,8 |
| Recall TEST MD / MV / MC / MF | 63,7 / 86,8 / 87,7 / 83,8 | 61,5 / 89,9 / 87,1 / 82,1 | −2,2±2,7 / **+3,1±0,5** (3/3 pos.) / −0,6±8,3 / −1,7±5,4 | 45,5 / 77,2 / 83,6 / 67,3 | −19,6 / −8,9 / −6,7 / −17,3 |
| F1 TEST MD | 70,5 ± 2,5 | 68,8 ± 2,2 | −1,6 ± 1,2 (3/3 negativos) | 47,2 | −24,0 |
| Recall de espécie <32 / 32–48 / ≥48 px | — | — | −1,8±4,8 / +2,0±3,8 / −0,3±3,0 | — | −20,1 / −8,0 / −3,6 |
| Épocas (best) | 75 (45 ± 7) | 83 (53 ± 23) | — | 34 (4) | — |
| Tempo de treino (V100) | 2,20 ± 0,22 h | 2,47 ± 0,67 h | +0,27 h | 1,01 h | — |

- **McNemar exato** (E15 vs E0, 1.210 GT por seed): s0 b=38/c=85, p=3,4e-5 (Δacc −3,88, **favorece E0**); s1 b=61/c=52, p=0,45 (+0,74); s2 b=72/c=47, p=0,028 (+2,07, **favorece E15**). Os dois significativos têm **sinais opostos** → efeito depende da seed, não do método.
- **Bootstrap por imagem** (seeds agrupadas, 10.000, 67 imagens): Δacc E15−E0 **−0,36 p.p. [−2,14; +1,38]**, IC inclui 0. E16 s0: **−13,31 p.p. [−19,49; −7,27]**, McNemar b=74/c=235, p=8,9e-20.
- **Leitura E15:** com o NWD em pixels na loss **e** no assigner, a identificação de espécie no TEST **não muda** (Δ macro-F1 +0,1 ± 2,8, IC do Δacc inclui 0) e a localização fina no VAL piora (mAP50:95 −1,5 ± 1,0, 3/3 negativos), igual ao E10 (§12: +0,5 ± 3,1 e −1,7 ± 0,4). O resultado nulo da §12 **não dependia do bug**. Objetos <32 px também não ganham (−1,8 ± 4,8). Efeitos por espécie com mesmo sinal em 3/3 seeds (exploratório, sem correção para múltiplas comparações): recall MV +3,1 ± 0,5 (também no E10: +4,6 ± 0,7 → provável efeito do assigner) e F1 MD −1,6 ± 1,2.
- **Leitura E16:** o NWD puro em pixels **também colapsa** (best epoch 4, early stop na 34, `train/cls_loss` final 4,7 vs 0,5–0,9 nos E15). Logo, a frase da §12.1 "o colapso do E12 é artefato de implementação" **é forte demais**: o bug de unidade tornava o NWD da loss constante, mas sem o bug o NWD puro continua sem convergir (single-seed; E12 é outra campanha/dataset, magnitudes não comparáveis). Leitura segura: **"o NWD puro não converge nesta receita; um termo de sobreposição (CIoU) continua necessário"**, compatível com o gradiente ≈ 0 do NWD longe do alvo.
- **Frase segura (substitui a anterior):** "Com o NWD calculado em pixels na loss e no assigner (3 seeds), nenhuma variante melhora a identificação de espécie além da variância entre seeds, e o NWD puro não converge."
