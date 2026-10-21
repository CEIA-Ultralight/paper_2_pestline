# RQ2 — métodos usados × cobertura da literatura (2026-10-02)

Números: B200 multi-seed, TEST (918 GT de espécie), fonte `docs/PAPER_FACTS.md` §9. Literatura: `docs/related_work_map.md` §RQ2
(2026-09-30) + busca complementar de 2026-10-02 (refs novas marcadas ★, lidas só por resumo/snippet, ainda sem entrada no `main.bib`).

## 1. Métodos já usados na RQ2

| # | Método | Resultado (TEST, B200 salvo indicação) | Veredito |
|---|---|---|---|
| 1 | Single-stage species-aware YOLO26m (E0, 3 seeds) | R 75,4±2,3 · P 70,1±3,7 · F1 72,6±2,9 · acc\|loc 77,0 | baseline |
| 2 | Cascata YOLO→ConvNeXt-T CE, política label_only (9 pares) | R +3,0±2,2 · P −2,2±2,6 · F1 +0,1±2,2 · acc\|loc 80,1; VAL inverte (R −0,7) | só recall sobe; F1 igual |
| 3 | Classificador "partes" (atenção a partes) | F1 +1,2±2,1 vs single; oráculo 79,7 vs 79,6 (CE) | ≈ CE |
| 4 | Política score-produto vs label_only | F1 +0,85±0,39 (9/9); AP50 espécie 66,8→70,0 (single 73,1) | melhor; ainda não supera AP do single |
| 5 | Oráculo de crops GT | 79,6±0,5 %; erro loc 2,1 % vs identificação ~19–22 % | cascata no teto do classificador |
| 6 | Cascata sobre E10 (NWD) | F1 −0,5±1,3 (CE) / +0,3±1,2 (partes) | sem efeito |
| 7 | Confusão single vs cascata | MD→MV 31→30 %, MC→MV 15→16 % (parts: MD→MV −12,7, MV→MD +5,9); MF↔MC = 0 | confusão dominante não muda |
| 8 | Latência | E0 ≈ 12 ms; cascata ≈ 425 ms/img (30–44×) | indicativa |
| 9 | McNemar + bootstrap por imagem | 6/9 sig. (3/9 Holm-36); bootstrap exclui 0 em 4/9 | efeito frágil |
| 10 | Swin-T, ConvNeXt-S, focal, SupCon, ArcFace (H100, single-seed, VAL/TEST) | todos convergem ao mesmo teto | não vale repetir |
| 11 | Cabeça RoI sobre features YOLO | pior que classificador dedicado | descartado |
| 12 | Limpeza de fundo com SAM | 71,8 vs ≈79 % | piorou |
| 13 | DINOv2 como backbone (H100, sem multi-seed) | ver `ARCHITECTURE_NIGHT.md` | incompleto |
| 14 | Ensemble de 3 classificadores + TTA | F1 76,4 (escolhido vendo o TEST) | exploratório, não usar |
| 15 | P2 head, GCD, NWD, RFLA, SAHI (RQ3, detector) | sem ganho / piora / inconclusivo | ver RQ3 |
| 16 | WBF ensemble + yolo26l (E7) | job 33620 pendente | aguardando |

## 2. Literatura × o que falta testar

Legenda: **Testar** = falta e vale GPU/CPU; **Citar** = só discutir; **Não** = não vale.

| Tema | Referências | O que dizem | Nosso estado | Ação |
|---|---|---|---|---|
| Cascata em insetos/fauna | Korsch 2023, Vyskočil 2024, Jain/AMI 2024 ★(arXiv:2406.12452), Bjerge 2023/26, Beery 2019 | Pipeline vs classificador de imagem inteira; Korsch: crops 93,1 vs pipeline 88,1 (gap ≈5 pp) | Lacuna confirmada: ninguém compara com single species-aware + oráculo + multi-seed. Nosso gap ≈0,4 pp | Citar (posicionamento) |
| Detecção fine-grained em 2 estágios | PETDet ★(arXiv:2312.10515), Cascade R-CNN ★(1712.00726) | Conflito de tarefa no backbone compartilhado; propostas de qualidade ajudam o classificador | Nosso detector já localiza (erro loc 2,1 %) | Citar |
| Conflito classificação×localização no detector | TSCODE ★(arXiv:2303.01047), YOLOX, TOOD | Cabeças desacopladas/alinhadas mitigam o conflito | YOLO26 já é desacoplado; não testamos alinhamento | Citar; testar só se sobrar tempo |
| Classificador sem classe de fundo | Hard FP suppression ★(arXiv:1810.04002), Cascade R-CNN | 2º estágio precisa de negativos difíceis / classe de fundo | **MF precisão −9,2 pp: classificador re-rotula FP de NOISE/INS/MAR como espécie** | **Testar** (classe background nos crops de propostas, treino com negativos do detector) |
| Fusão de scores / calibração | Zhou 2021, Guo 2017, Küppers 2020 | produto P(obj)·P(cls); temperature scaling | produto testado; calibração e re-rank não | **Testar** (CPU, P0 #3) |
| Diagnóstico de erros | TIDE (Bolya 2020), Hoiem 2012 | dAP Cls/Loc/Both/Dupe/Bkg/Miss | só decomposição própria loc×id | **Testar** (CPU, P0 #1) |
| Oráculo inverso e estratificado | Korsch 2023 (oráculo de crops) | teto do classificador | oráculo de crops feito; falta caixas do detector + classe GT, e por tamanho/nitidez/densidade | **Testar** (CPU, P0 #2, #4) |
| Contexto/padding do crop | AMI-Traps ★(Jain 2024: padding quadrado e crops maiores ajudam) | contexto e padding melhoram o classificador | pad 0,75 fixo, sem ablação | **Testar** (crop duplo justo+contexto, P2) |
| Resolução do crop | Cai 2017, Chen 2024, FixRes, thrips/whitefly ★(Frontiers 2025: mínimo 80 µm) | SR só ajuda com LR sintética; há resolução mínima | sem curva de sensibilidade | **Testar** análise (item 11, confirmar regra de resolução); SR/destilação HR→LR: **Não** |
| Abstenção/hierarquia | Bertinetto 2020, Geifman 2017, Mahfoud 2026, Open-Insect | Hierarquia reduz severidade; risco–cobertura | não feito | **Testar** (CPU, P0 #5, abstenção para família) |
| Ruído/ambiguidade de rótulo | Austen 2018, Northcutt 2021, Song 2022, Ling 2023, acordo entre especialistas em fotos ★(PMC6262731: Fleiss κ 0,64; 39 % das imagens com especialistas divergindo) | Especialistas divergem em espécies parecidas; ≥3,3 % de rótulos errados em test sets | Teto ~80 % pode ser rótulo; **não medido** | **Testar** (confident learning 5-fold + re-anotação cega de ~150 crops) |
| Foundation models | BioCLIP 2, DINOv3, DINOv2, LP-FT, VPT, LoRA | DINOv2 > BioCLIP 2 após fine-tune; BioCLIP fraco fora do domínio | DINOv2 só H100 single-seed | **Testar** DINOv3-B e BioCLIP 2 com LP-FT, 3 seeds (P1 #10); zero/few-shot: **Não** |
| Desbalanceamento / long-tail | logit adjustment, label-aware smoothing, Remix (buscas 2024–25 ★) | Ajustam margens de classes raras | MD e MV são as confundidas; focal já testado, sem ganho | Citar; **Não** gastar GPU (focal já convergiu ao teto) |
| Destilar classificador no detector | Guo 2021 ★(arXiv:2106.05209), feature imitation (Wang 2019) | Classificador de imagem como professor melhora reconhecimento do detector | não testado; é a alternativa "single-stage com teto do classificador" | Citar; **Testar só se** tempo (P2) |
| Detectores DETR em objetos pequenos/densos | DEIM ★(CVPR 2025), RT-DETRv4 ★, Dome-DETR ★(2505.05741); estudo em sticky cards ★ | Em insetos pequenos/ocluídos, DEIM/RT-DETR competem ou superam YOLO | só rtdetr-l disponível em `models/`, não rodado | **Testar** se houver GPU (baseline single-stage mais forte, junto de yolo26l) |
| Single-stage maior / tiling | SAHI, yolo26l | resolução e tiling | SAHI inconclusivo; yolo26l pendente (33620) | aguardar 33620 |
| TTA/ensemble de crops | TTA em classificadores ★ | ganhos pequenos (~0,1–1 pp) | ensemble exploratório | Citar; **Não** como resultado principal |
| Fusão tardia YOLO+classificador | — | — | não testado | **Testar** (CPU, P0 #7) |

## 3. Observações
- Todas as refs ★ foram vistas por resultado de busca (título/snippet). Verificar no arXiv/venue antes de entrar no `main.bib`; preprints com cautela.
- Não encontrei estudo de acordo inter-observador MD/MV/MC por foto nem estudo controlado single-stage species-aware × cascata em insetos pequenos; é ausência na busca, não prova.

## 4. Busca ampliada (2026-10-02, 2ª passada) — o que mudou

Cobertura: ~30 consultas em arXiv, bioRxiv, Crossref/Frontiers/MDPI/Wiley/PMC via busca web. Não acessei Scopus/Web of Science/IEEE Xplore/Semantic Scholar (429) nem li texto completo: **não há como garantir que nenhum método falte**; a evidência é que as famílias de métodos se repetem entre as buscas.
Todas as refs ★★ abaixo vieram de título/snippet; verificar antes de citar.

| Família | Refs ★★ | Relevância | Ação |
|---|---|---|---|
| Reclassificação de propostas por classificador dedicado | DCR — Decoupled Classification Refinement (arXiv:1810.04002), Region-based Ensemble (1902.03377), WS-BAN (1808.02152) | É o desenho da nossa cascata; DCR reporta ganho por suprimir FP difíceis | Citar; reforça o teste de classe de fundo |
| Score ciente de IoU / rescoring | VarifocalNet (2008.13367), Cascade-DETR (2307.11035), Calibrating Probabilistic Detectors with Annotator Disagreement (2605.24722) | O produto score_det×p_cls ignora qualidade da caixa; calibração com discordância de anotadores liga RQ2 ao ruído de rótulo | Testar IoU-aware rescoring na CPU (junto do re-rank) |
| Ruído de rótulo em detecção | Noisy Annotation Refinement (2110.10456), Noise-resistant OD (2003.01285), Tiny OD amidst label noise (2401.08056) | Treino robusto existe, mas só vale após medir o ruído | Citar; treinar só se ruído ≥ 5 % |
| Hierarquia taxonômica em insetos | TaxaNet (CEUR 3786), hierárquica multitarefa + anomalia (bioRxiv 2023), InsectDCT (bioRxiv 2026), arXiv:2607.28005 | Predição até o nível confiável; abstenção por família | Testar abstenção (CPU) |
| Foundation/VLM em insetos | Insect-Foundation (2502.09906), BIOSCAN-5M/BioAutoML-NAS (2510.05888), BioCLIP 2; VLMs genéricos < modelos específicos; Grounding DINO em MassID45 só 27,1 mask AP | Open-vocab e VLM zero-shot não servem para MD/MV/MC | Citar; só LP-FT de DINOv3/BioCLIP 2 |
| Pré-treino específico do domínio | SSL plâncton (2503.11341), moth domain adaptation (2508.20089) | Pré-treino de domínio supera ImageNet com poucos dados | Citar (backbones foundation cobrem) |
| Coarse-to-fine / zoom em alta resolução | Coarse-to-Fine HR detection (2303.01219), Dynamic Zoom-in (1711.05187) | Equivalente a tiling/SAHI; reforça "resolução é treino" | Citar |
| Detectores DETR em sticky cards | RT-DETR vs YOLO em cucumber beetles (Sci Rep 2026), DEIM, Dome-DETR | RT-DETR indicado para insetos pequenos e ocluídos | Testar rtdetr-l (já em `models/`) |
| Espécie em sticky traps | Thrips/whitefly YOLO11 (Frontiers 2025: mínimo 80 µm), Ong & Høye 2025 (cor da armadilha), Nieuwenhuizen YST | Resolução mínima e domínio importam | Citar |
| Moscas (Diptera) | Chrysomya por asas (Macleod 2018), espiráculos de larvas (Sci Rep 2022), ViT sarcófagas (Insects 2025) | Imagens de laboratório/asas isoladas, outra modalidade | Citar como contraste |
| CNN × ViT em insetos | Comparison transformers vs CNN (2307.11112), crowdsourced insect images (2404.03474) | Sem vencedor claro; depende de dados | Citar |
| Protocolo estatístico | Paired bootstrap BCa + sign-flip (arXiv:2511.19794) | 3 seeds podem esconder ou inverter efeitos; protocolo pareado | Adotar no bootstrap por imagem (CPU) |
| NMS ciente de classe, soft-NMS | nada decisivo | Nosso matching é class-agnóstico; duplicatas entram no TIDE (Dupe) | Coberto pelo TIDE |
| Patch size/resolução em ViT | Scaling laws in patchification (2502.03738) | Patch 16 sobre objetos de ~40 px upsampled a 384 pode perder detalhe | Baixa prioridade |

**Conclusão da varredura:** os métodos da lista anterior continuam sendo o universo relevante; a 2ª passada acrescentou três itens testáveis (classe de fundo/rejeição, rescoring ciente de IoU, baseline RT-DETR) e um protocolo estatístico (bootstrap pareado BCa + sign-flip). Nenhum método novo de "single-stage melhor para espécie" apareceu além de DETRs e destilação.

## 5. 3ª passada (2026-10-06) — refs verificadas no arXiv e já em `main.bib`
Verificadas por título/autores/ano na página do arXiv (não li o texto completo): CLOD 2211.13993 (`chachula2023clod`), DCR 1810.04002 (`cheng2018dcr`), bootstrap pareado BCa + sign-flip 2511.19794 (`du2025pairedbootstrap`), FathomNet CLEF 2607.18700 (`brock2026fathomnet`), Object Detection Benchmarks are Incomplete 2609.21822 (`penquitt2026benchmarks`), RF-DETR 2511.09554 (`robinson2025rfdetr`), Cascade R-CNN (`cai2018cascade`), hierárquico em insetos 2607.28005 (já era `mahfoud2026hierarchical`).
Ainda sem entrada no bib: PMC13283985, PMC6262731 (κ 0,64), TOOD, TSCODE, VarifocalNet. Fine-tune dos foundation models é **completo**, não LP-FT (corrige o plano anterior).
Estado dos testes pendentes: BCa+sign-flip **feito**; vazamento de imagem **feito** (ver PAPER_FACTS §10.5); calibração/re-rank, TIDE, score ciente de IoU, oráculo por nitidez/densidade **não feitos**; fusão tardia exige guardar a distribuição de classes do detector (GPU); BG multi-seed e curva de aprendizado no job 34026; RF-DETR+DINOv2 e latência não rodados.
