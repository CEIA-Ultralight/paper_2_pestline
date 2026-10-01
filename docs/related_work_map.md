# Mapa da literatura — fly-det (base para related work VISAPP 2027)

Compilado em 2026-09-25 (arXiv + Bing + leitura integral de Laekeman et al. 2025).
Uso: seções de Related Work / Introduction / Discussion do paper aplicado.

---

## Bloco A — Detecção de insetos em armadilhas adesivas (concorrentes diretos)

### Concorrente mais próximo (lido na íntegra)
- **Laekeman et al. 2025** — *Species-level detection of thrips and whiteflies on yellow
  sticky traps using YOLO-based deep learning detection models*. Front. Plant Sci. 16,
  doi:10.3389/fpls.2025.1668795 (publicado 18/nov/2025).
  - Claim: "primeiro sistema species-level para thrips/whitefly em sticky traps RGB".
  - **Single-stage apenas** (YOLO11 n/s/m/l/x + YOLO-NAS S/M/L); sem cascata.
  - Dataset de laboratório: insetos criados em gaiolas (ILVO, Bélgica), DSLR Sony α7R III
    42.4MP, 5 µm/pixel, patches 640×640 (3.2×3.2 mm), 1.000 labels/espécie, 4 espécies
    (F. occidentalis, E. americanus, B. tabaci, T. vaporariorum).
  - Resultados: interno mAP@50 ≥90%; **externo mAP@50 79–89%, F1 74–87%** (queda 10–20%).
  - Confusão inter-espécie B. tabaci ↔ T. vaporariorum cresce no dataset externo
    (**análogo ao nosso MC↔MV — citar a favor da motivação do classificador dedicado**).
  - Estudo de resolução mínima: 80 µm/pixel viabiliza smartphones/câmeras baratas.
  - Grad-CAM para interpretar features por espécie.
  - **Sem multi-seed, sem teste estatístico, sem tiny-object losses, sem campo real.**
  - Dataset será liberado no Zenodo em set/2026: DOI 10.5281/zenodo.15574404
    (**monitorar — possível benchmark comparável até a camera-ready**).

### Linhagem sticky traps (via survey Teixeira 2023 + buscas)
- Bauch & Rath 2005 — protótipo visão p/ whitefly (Acta Hortic).
- Xia et al. 2015 — contagem low-cost em estufa (Ecol. Inform.).
- Espinoza et al. 2016 — image processing + ANN p/ B. tabaci e F. occidentalis (CEA).
- Sun et al. 2017 — contagem via espectro de Fourier 2D (Biosyst. Eng.).
- Nieuwenhuizen et al. 2018 — detecção em yellow sticky traps (dataset público YST).
- Gutierrez et al. 2019 — benchmarking SSD/Faster R-CNN p/ whitefly (J. Sensors);
  precisão adultos modesta (27–74%).
- **Liu et al. 2019 — PestNet** (IEEE Access): E2E multi-classe em armadilhas, larga escala.
- Böckmann et al. 2021 — bag-of-visual-words p/ B. tabaci vs T. vaporariorum (Sci Rep);
  fraco (recall 72%/54%); observou decaimento de asas com tempo de residência (IRT).
- **Li et al. 2021 — TPest-RCNN** (CEA): "field detection of tiny pests from sticky trap
  images", Faster R-CNN adaptado, estufa.
- Wang et al. 2021 — YOLOv4 melhorado p/ whitefly+thrips em traps (Trans. ASABE).
- Le et al. 2021 — AlertTrap (arXiv:2112.13341): SSD-MobileNet/YOLOv4-tiny em edge p/ fruit fly.
- Albanese et al. 2021 — pest detection edge/embedded (arXiv:2108.00421, IEEE JETCAS).
- Niyigena et al. 2023 — única detecção species-level de thrips pré-Laekeman
  (S. dorsalis vs "outros thrips", Insects); mAP@50 91–97% mas com marcadores coloridos
  pré-anotados na imagem (generalização questionável — criticável em related work).
- Zhang et al. 2023 — sistema de ID de pragas em estufa (Front. Plant Sci.).
- Wang et al. 2024 — small target tomato pests em traps (Agronomy).
- Wu et al. 2024 — thrips em estufas de manga (IEEE TIM).
- **Ong & Høye 2025** (Pest Manag. Sci.): cor da armadilha afeta fortemente generalização
  de modelos; recomenda treinar com traps transparentes.
- Reviews: Lima et al. 2020 (Agriculture); Preti et al. 2021 (J. Pest Sci.);
  **Teixeira et al. 2023** (Agriculture 13:713 — systematic review, mAP@50 típico 70–95%).

### Moscas-das-frutas (Tephritidae — nosso domínio exato)
- Ding & Taylor 2016 — moth detection em trap images (arXiv:1602.07383, CEA) — o clássico.
- Rodrigues et al. 2018 — armadilha inteligente p/ fruit fly (ACM SAC, dl.acm.org/10.1145/3167132.3167155,
  **grupo brasileiro**).
- e-trap McPhail eletrônico (J. Pest Sci. 2022, doi:10.1007/s10340-022-01528-x) —
  vigilância remota em tempo real de fruit flies.
- "Smart insect monitoring based on YOLOv5: B. zonata + C. capitata" (Ain Shams Eng. J. 2023).
- Dataset p/ ID automática de fruit flies (MDPI AgriEngineering 7(12):422, dez/2025).
- Multi-angle imaging p/ quarantine true fruit fly ID (R. Soc. Open Sci. 2026).
- **FlyYOLO-SORT** (CEA 2026, S2772375526003977): detecção leve + tracking de tefritídeos.

### Lacuna do Bloco A
Nenhum trabalho combina: (i) species-level em armadilha de **campo real**; (ii) **cascata**
detector→classificador; (iii) avaliação de **tiny-object losses**; (iv) **multi-seed/estatística**.
Species-level existentes são single-stage e/ou laboratório.

---

## Bloco B — Tiny object detection (RQ2)

- **Kisantal et al. 2019** (arXiv:1902.07296) — augmentation/oversampling p/ small objects
  (origem do copy-paste em detecção).
- **NWD** — Wang et al. 2021, arXiv:2110.13389 (*A Normalized Gaussian Wasserstein Distance
  for Tiny Object Detection*).
- **NWD-RKA + AI-TOD-v2** — Xu et al., ISPRS J. P&RS 2022 (arXiv:2206.13996) — benchmark aéreo.
- **RFLA** — Xu et al., ECCV 2022 (arXiv:2208.08738) — Gaussian receptive field label assignment.
- **SAHI** — Akyon et al., ICIP 2022 (arXiv:2202.06934); avaliação YOLO+slicing: Keles et al.
  2022 (arXiv:2203.04799); variantes: ROI-Gated SAHI (arXiv:2608.23923), tiling adaptativo
  por altitude p/ SAR marítimo (arXiv:2511.19728), SAHI p/ defeitos IC (arXiv:2311.11439).
- Surveys: Muzammul & Li 2021 (arXiv:2107.07927); **Cheng et al., TPAMI 2023** (SODA benchmark,
  "Towards large-scale small object detection"); meta-review remote sensing (arXiv:2309.06751).
- Recentes: DroneScan-YOLO (arXiv:2604.13278), SME-YOLO PCB (arXiv:2601.11402),
  CoSWA-YOLOv12 malária c/ NWD (arXiv:2609.29527, MICCAI-W 2026),
  **Point-Prompted Paradigm** (Zhu et al., arXiv:2604.02773 — grupo do NWD/RFLA),
  SCAResNet (arXiv:2404.04179), MTU-Net infrared tiny ships (arXiv:2209.13756).
- **YOLO26**: análise NMS-free E2E (arXiv:2601.12882); benchmark YOLO26/27 fine-grained
  small-object em pomares (arXiv:2608.23636).

### Posicionamento
Ganhos de NWD/RFLA reportados **apenas** em aerial/remote sensing, geralmente com detectores
anchor-based/two-stage. Não há estudo em armadilhas de insetos nem com YOLO26 NMS-free →
nosso resultado (negativo, condicional ao rerun 3×3) é a primeira evidência de
não-transferência nesse regime. Publicável como achado.

---

## Bloco C — Cascata detector→classificador e fine-grained

- **MegaDetector / Beery et al. 2019** (arXiv:1907.06772, KDD-W) — o análogo canônico:
  em camera traps de fauna, o padrão é detector genérico + classificador de espécie no crop.
  Argumento: "trazemos o paradigma consolidado da ecologia de vertebrados para insetos".
- **Jain et al. 2024** (arXiv:2406.13031, NeurIPS-W Climate Change AI) — pipeline de
  monitoramento automático de mariposas: detector genérico → classificador fine-grained +
  filtro binário. **Metodologicamente o mais próximo do nosso RQ3**, mas em câmeras de luz
  e sem comparação controlada vs single-stage.
- Trotter et al. 2025 (arXiv:2507.21665, ICCVW Marine Vision) — detect+classify de organismos
  bentônicos antárticos em imagens hi-res.
- Fine-grained insetos: Pucci et al. 2023 (arXiv:2307.11112, transformers vs CNN) e 2024
  (arXiv:2404.03474, crowdsourced); moth fine-grained c/ foundation priors (arXiv:2508.20089);
  InsectMamba (arXiv:2404.03611); Ung et al. 2021 (arXiv:2107.12189, multi-CNN pest);
  **Prompt-CAM** (CVPR 2025, arXiv:2501.09333 — interpretabilidade fine-grained ViT);
  **Open-Insect** (NeurIPS 2025 D&B, arXiv:2503.01691 — OOD/espécies novas; conecta com
  roadmap de rejeição); fine-grained ZSL c/ DNA (NeurIPS 2021, arXiv:2109.14133).

### Lacuna
Ninguém quantificou o **trade-off da cascata** (recall/F1 ↑ vs mAP ↓; instabilidade entre
seeds) em tiny objects de armadilha. Nosso paradoxo é novo no domínio.

---

## Bloco D — Datasets e foundation models de insetos

- **IP102** (Wu et al., CVPR 2019) — 102 classes, benchmark clássico de classificação de pragas.
- **Pest24** (Wang et al., CEA 2020) — detecção multi-praga em armadilhas automáticas (China).
- AgriPest (Wang et al. 2021).
- **BIOSCAN-1M** (NeurIPS 2023, arXiv:2307.10455); BIOSCAN-5M (2024).
- **Insect-1M / Insect Foundation** (CVPR 2024, arXiv:2311.15206; versão VLM arXiv:2502.09906).
- **BioCLIP** (CVPR 2024) — foundation model p/ árvore da vida.
- MassID45 (arXiv:2507.06972) — segmentação/ID de amostras bulk de artrópodes.
- BuzzSet v1.0 (arXiv:2508.19762) — polinizadores em campo.
- **DAPWH** (arXiv:2602.20028, IEEE Data 2026) — vespas parasitoides, grupo brasileiro (USP/UFSCar).
- Wild bee dataset + XAI (arXiv:2206.07497).
- Dataset ILVO/Laekeman: Zenodo 10.5281/zenodo.15574404 (liberação set/2026).

### Posicionamento do DS-F2
Foundation datasets = espécimes individuais/curados; datasets de armadilha (Pest24, YST) não
têm nossas espécies (MD/MV/MC/MF) nem 1920×1080 de campo com protocolo de coleta dirigida.
Dataset é contribuição válida no VISAPP.

---

## Bloco E — Data-centric / augmentation (RQ1)

- Kisantal et al. 2019 — copy-paste p/ small objects (base direta do E9).
- **Ghiasi et al., Simple Copy-Paste, CVPR 2021** — referência canônica.
- Copy-paste p/ crowded scenes (Deng et al., AAAI 2023, arXiv:2211.12110 — **relevante:
  nossas placas de crowding**; ICD/CDD effects).
- Occlusion Copy&Paste (BMVC 2022, arXiv:2210.03686).
- Few-shot copy-paste p/ novas categorias (arXiv:2206.05730).
- Variantes por difusão: Control Copy-Paste (arXiv:2507.21816), Domain-RAG (arXiv:2506.05872),
  Synthetic Object Compositions (arXiv:2510.09110), Stable Diffusion aerial (arXiv:2311.12345).
- Coleta dirigida/rare classes: RareSpot+ (arXiv:2604.20000 — active learning fauna rara aérea);
  Traffic Context-Aware augmentation (ICRA 2022, arXiv:2205.00376).
- Small drone detection c/ multi-scale + augmentation (IJCNN 2025, arXiv:2504.19347 —
  grupo brasileiro, Laroca/Menotti).

### Lacuna
Augmentation por **coleta física dirigida** (placas crowding/graduadas/vazias) praticamente
inexiste — todos usam síntese digital. Se a ablação RQ1 mostrar efeito, ângulo original.

---

## Matriz de posicionamento (para a Introdução)

| Dimensão | Estado da arte | Nós |
|---|---|---|
| Species-level em sticky trap | Laekeman 2025: single-stage, laboratório | Cascata det→clf, campo real, 4 spp de moscas + INS/NOISE |
| Cascata p/ espécies | MegaDetector (vertebrados); Jain 2024 (mariposas, sem comparação controlada) | Primeira comparação controlada single-stage vs cascata em armadilhas, trade-off quantificado |
| Tiny-object losses | Ganhos só em aerial/remote sensing | Primeiro teste NWD/RFLA/SAHI em armadilhas + YOLO26 E2E (resultado negativo honesto) |
| Data-centric físico | Só copy-paste digital | Coletas dirigidas (crowding/graduadas/vazias) + copy-paste real |
| Rigor | Nenhum sticky-trap paper usa multi-seed/estatística | 3×3 seeds + McNemar (pré-requisito a executar) |

## Narrativa de posicionamento
"A literatura de armadilhas adesivas parou no single-stage em condições de laboratório; a
literatura de tiny objects valida métodos apenas em imagens aéreas; a de cascata de espécies
só em vertebrados/mariposas. Unimos os três em um sistema de campo real e mostramos o que
transfere (cascata, com trade-off explícito) e o que não transfere (tiny-object losses)."

## Notas táticas
1. Laekeman 2025 saiu depois do nosso ciclo experimental — citar e diferenciar explicitamente
   (cascata + campo real + rigor estatístico).
2. A confusão inter-espécie crescente no teste externo deles corrobora nossa motivação para
   o classificador dedicado — usar a favor.
3. Niyigena 2023 tem falha metodológica (marcadores coloridos) — criticável com cuidado.
4. Monitorar Zenodo ILVO (set/2026) como possível avaliação cruzada até a camera-ready.

---

## RQ2 — revisão 2026-09-30

Fontes: arXiv API (metadados/abstracts), Crossref, bioRxiv API, página do TIDE. Semantic Scholar
respondeu 429, então a cobertura de CEA / Smart Agric. Technol. / Plant Methods foi só via Crossref
(busca por título/tema): "não encontrei" não implica ausência. Leitura = abstracts/metadados, exceto
Laekeman 2025 (íntegra, ver Bloco A). Chaves em `manuscript/main.bib`.

### C.1 Cascatas detector→classificador (o que reportam)
- **Korsch et al. 2023** (`korsch2023moth`, arXiv:2307.15427, moth scanner AMMOD): detector (mAP 99,01 %)
  + classificador de 200 espécies; 93,13 % em recortes isolados; o pipeline sobe a acurácia de
  espécie nas imagens de 79,62 % para 88,05 % (baseline não especificado no abstract). Reporta
  implicitamente o "oráculo" (recortes) vs pipeline (gap ≈5 pp); no nosso caso o gap é ≈0,4 pp.
- **Vyskočil & Picek 2024** (`vyskocil2024camera`, arXiv:2410.12769): MegaDetector + 2 classificadores
  reduz o erro relativo de um classificador de imagem inteira em ~42 % (CCT20), ~48 % (CEF), ~75 % (WCT);
  remover fundo reduz à metade o erro em locais novos. Baseline = classificador de imagem inteira,
  **não** detector species-aware: comparação mais branda que a nossa.
- **Jain et al. 2024** (`jain2024`) e **Bjerge et al. 2023/2026** (`bjerge2023hierarchical`,
  `bjerge2026insectdct`, preprint): cascatas com classificação hierárquica; nenhum compara
  com single-stage species-aware nem usa AP pós-cascata.
- Achado de lacuna: não encontrei estudo controlado "single-stage species-aware vs cascata" em
  insetos pequenos com oráculo + multi-seed. Reforça a novidade, mas é ausência na busca.

### C.2 Baixa resolução em fine-grained
- Cai et al. 2017 (`cai2017lowres`): SR + classificação fine-grained conjunta supera CNN comum
  em Cars/CUB em baixa resolução. Chen et al. 2024 (`chen2024vimd`): SR + destilação HR→LR,
  SOTA em 7 benchmarks. Ambos usam baixa resolução **sintética** (downsample de HR disponível).
- FixRes (`touvron2019fixres`): discrepância de tamanho aparente treino/teste; treino em res menor
  + fine-tune na de teste melhora acurácia.
- Nekka et al. 2026 (`nekka2026counting`, **preprint SSRN**): em sticky traps (YST, moscas-brancas),
  resolução máxima não é a melhor para contagem (mais clutter); F1 máximo na menor resolução.
  Usar só como cautela, não como evidência forte.

### C.3 Fusão de scores, calibração, AP pós-cascata
- Zhou et al. 2021 (`zhou2021probabilistic`): score final = P(objeto)·P(classe|objeto); justifica
  nossa política "score-produto". Guo et al. 2017 (`guo2017calibration`): temperature scaling.
  Küppers et al. 2020 (`kuppers2020calibration`): calibração de detectores usando também posição/escala da caixa.
- Não encontrei trabalho que reporte AP pós-re-rotulagem por classificador de crops em insetos.

### C.4 Hierarquia / abstenção
- Bertinetto et al. 2020 (`bertinetto2020mistakes`): losses hierárquicas reduzem a *severidade* do erro.
  Geifman & El-Yaniv 2017 (`geifman2017selective`): classificação seletiva (risco–cobertura).
  Mahfoud et al. 2026 (`mahfoud2026hierarchical`, preprint): prediz até o nível mais profundo com
  confiança ≥0,6; acurácia por nível 80–99 % (5 níveis, 34 classes).
  Open-Insect (`opensect2025`): métodos pós-hoc simples seguem sendo baseline forte em OSR.

### C.5 Ruído de rótulo / ambiguidade de especialistas
- Austen et al. 2018 (`austen2018species`, PeerJ): especialistas identificando tritões a partir de
  fotos concordam pouco; acurácia por espécie 69–96 %, independente de experiência.
- Northcutt et al. 2021 (`northcutt2021confident`, `northcutt2021pervasive`): confident learning;
  ≥3,3 % de erro de rótulo em média nos test sets de 10 benchmarks. Song et al. 2022 (`song2022noisy`): survey.
- Ling et al. 2023 (`ling2023wing`, Med. Vet. Entomol.): venação alar de Sarcophagidae/Calliphoridae/
  Muscidae com 91,04 % (espécie) / 98,56 % (família) — mas em asas montadas de machos; em armadilhas
  adesivas asas ficam ocultas/degradadas (Böckmann 2021, Bloco A). Não encontrei estudo de
  acordo inter-observador para MD/MV/MC a partir de foto.

### C.6 Foundation models
- BioCLIP 2 (`gu2025bioclip2`), DINOv3 (`simeoni2025dinov3`), DINOv2 (`oquab2023dinov2`), LP-FT
  (`kumar2022lpft`), VPT (`jia2022vpt`), LoRA (`hu2021lora`).
- Saputra et al. 2026 (`saputra2026orchid`, preprint): fine-tune idêntico; DINOv2 > BioCLIP 2 em
  7,1 pp de top-1 macro (gêneros de orquídeas). Montanares et al. 2026 (`montanares2026planktonzilla`,
  preprint): BioCLIP/BioCLIP 2 fracos em plâncton (zero/few-shot) — domínio fora da distribuição.

### C.7 Protocolo de avaliação
- TIDE (`bolya2020tide`, ECCV 2020): dAP por Cls/Loc/Both/Dupe/Bkg/Miss + FP/FN; Hoiem et al. 2012
  (`hoiem2012diagnosing`): precursor. Aplicar aos dois sistemas (single vs cascata).

### Frases de posicionamento sugeridas (para o writer)
- Intro/Related: "Detector-then-classifier pipelines are standard for camera-trap fauna and moths
  \cite{beery2019,korsch2023moth,jain2024}, but are typically compared against whole-image classifiers
  \cite{vyskocil2024camera}, not against a species-aware single-stage detector."
- Discussion (oráculo): "As in \cite{korsch2023moth}, we report classifier accuracy on ground-truth crops as
  an upper bound; our cascade is within 0.5 pp of it, locating the bottleneck in identification."
- Discussion (rótulos): "Expert agreement from images alone is limited even for trained observers
  \cite{austen2018species}; label ambiguity may bound MD/MV/MC accuracy."
