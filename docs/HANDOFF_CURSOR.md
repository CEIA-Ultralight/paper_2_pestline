# HANDOFF — paper_2_pestline (VISAPP 2027) → continuação no Cursor

Gerado em 2026-09-30 pelo orchestrator (VS Code Copilot). Leia este arquivo inteiro antes de agir.
Responder ao autor em **português**. Deadline VISAPP 2027: **22/out/2026**.

---

## 0. Leitura obrigatória (nesta ordem)

1. `.github/copilot-instructions.md` — regras globais (cluster, W&B, cadeia de custódia dos números, anonimato, git).
2. `README.md` — tese, RQs, layout.
3. `docs/DECISIONS.md` — decisões já tomadas (**não reabrir**); `docs/LEDGER.md` — histórico.
4. `docs/PAPER_FACTS.md` **§9 (Campanha B200)** — fonte única dos números atuais.
5. `rq2_cascade/README.md`, `rq3_tiny_object/README.md`, `rq1_data/README.md` — backlog por RQ.
6. `docs/related_work_map.md` §"RQ2 — revisão 2026-09-30" — literatura (resumida na Seção 5 abaixo).
7. `.github/agents/*.agent.md` — papéis (scholar, experimenter, slurm-runner, analyst, illustrator, writer,
   fact-checker, R1–R5, area-chair). No Cursor eles não são carregados automaticamente: use o corpo de cada
   arquivo como prompt/regra quando for desempenhar aquele papel (ou copie para `.cursor/rules/`).

---

## 1. Regras do cluster (violação = bloqueio + penalidade financeira) — fonte: `orientacoes_cluster_HPC.pdf`

- **GPU só via Slurm** (`sbatch`). Nunca `python`/`torchrun`/`apptainer --nv` com GPU no login node.
- **Partição `b200n1`** (dgx-B200-1, 8× B200 178 GB). `/raid` é **disco local por nó**: dados, checkpoints e
  container estão no /raid do dgx-B200-1. Checar `sinfo -p b200n1` antes de submeter.
- **Nada na home.** Datasets, checkpoints, caches (uv, pip, HF, torch, wandb) em `/raid/user_marcospaulo/`.
- Jobs com `save_period`, `resume`, `#SBATCH --signal=B:SIGUSR1@300`, `--requeue` (manutenções cancelam sem aviso).
  Todos os scripts em `common/slurm/`, `rq*/slurm/chain_*.sh` já seguem isso e são **idempotentes** (marcadores `DONE`).
- **Nunca reduzir a resolução do dataset** (1920×1080).
- QOS da conta `onejob`: **máx. 2 jobs RUNNING** por usuário (inclui jobs de outros projetos do autor).
- W&B: entity `pestline`; credenciais **só** via `set -a; source /raid/user_marcospaulo/secrets/wandb.env; set +a`
  (já feito em `common/slurm/b200_common.sh`). Nunca passar a chave por `--env` nem copiar para o repo.
- Arquivos temporários **só dentro do workspace** (`_scratch/`, ignorado pelo git) — nunca /tmp nem outro lugar do /raid.
- **Git: nunca `git push`** (nem no submodule `manuscript/`, que é o Overleaf). Commits locais, um por etapa,
  mensagem em português. Hoje tudo está em `main` (não existe `dev`; o autor decide).
- `manuscript/` e `baseline/` são submodules: commitar dentro de `manuscript/` e depois `git add manuscript` na raiz.

## 2. Infra e ambiente

| Item | Caminho / valor |
|---|---|
| Repo | `/raid/user_marcospaulo/paper_2_pestline` (raiz `b8b62df`; manuscript `3882e20`) |
| Container B200 | `/raid/user_marcospaulo/containers/flydet-train-b200.sif` — torch 2.8.0+cu128, ultralytics 8.4.142, `/opt/venv/bin/python` (o antigo `flydet-train.sif` cu124 **não roda** em Blackwell) |
| Dataset | `/raid/user_marcospaulo/datasets/DS-F2_v8.1.1` (train 1.259 / val 35 imgs, 617 inst. espécie / test 55 imgs, 918 inst.; 7 classes MD MV MC MF INS NOISE MAR, índices 0–6) |
| Crops classificador | `/raid/user_marcospaulo/datasets/DS-F2_crops_pad75` (pad 0,75, img 384) |
| Modelos base | `/raid/user_marcospaulo/models/` (yolo26m.pt, yolo26l.pt, rtdetr-l.pt, sam_vit_b, dinov2) |
| Saídas | `RUNS_ROOT=/raid/user_marcospaulo/flydet_runs/` |
| Logs Slurm | `/raid/user_marcospaulo/slurm_logs/%x_%j.out` |
| venv CPU (testes) | `_scratch/venv_tests/bin/python` (torch CPU, ultralytics 8.4.142, pandas, pytest, ruff); `UV_CACHE_DIR=/raid/user_marcospaulo/.cache/uv` |
| LaTeX | sem pdflatex; `cd manuscript && TECTONIC_CACHE_DIR=../_scratch/tectonic_cache tectonic -X compile main.tex --outdir _build` |
| Testes | `_scratch/venv_tests/bin/python -m pytest tests -q` (125 passed, 25 skipped em 1bff714) |
| Custo de referência B200 | E0 yolo26m ≈ 37–60 min/treino; P2 ≈ 1–1,5 h; ConvNeXt-T ≈ 15–40 min |
| W&B | projetos `paper2-rq2-cascade`, `paper2-rq3-tiny-object`, `paper2-smoke` (legado: `fly-species`) |

Scripts-chave: `common/slurm/b200_common.sh` (`b200_guard`, `b200_apptainer`, `chain_run`, `step_done`),
`common/slurm/train_detector_b200.sh` (receita E0: 150 ép., batch 8, imgsz 1920, patience 30, `FLYDET_SEED`,
`FLYDET_NWD_*`, `MODEL_CFG` p/ P2), `rq2_cascade/slurm/train_classifier_b200.sh`,
`rq2_cascade/eval_cascade_matrix.py` (per_instance.csv / metrics.json), `rq2_cascade/eval_oracle_crops.py`,
`common/stats.py` (mean_std, mcnemar_paired, bootstrap_ci, confusion_from_per_instance, recall_by_size_bins).

## 3. Estado dos jobs (2026-09-30)

| Job | O quê | Estado |
|---|---|---|
| 33321 cadeia A | E0 s0–2, E10 NWD s0–2 + TEST eval | COMPLETED |
| 33322 cadeia B | ConvNeXt-T CE/parts × seeds 42/84/126 | COMPLETED |
| 33363 cadeia C | GCD s0–2, NWD loss/assigner s0, NWD+RFLA s0, SAHI 3×E0, P2 ×3, TTA 3×E0 | FAILED só no `ttawbf_E0_ensemble` (bug corrigido em `1bff714`) |
| **33620** cadeia C (resubmissão) | só `ttawbf_E0_ensemble` + `E7_yolo26l_s0` | **PENDING** (fila; outros jobs do autor 33618/33619 na frente) |
| 33364 cadeia D | 6 det × 6 clf × {label_only, product} em VAL+TEST + oráculo de crops GT | COMPLETED |

Quando 33620 terminar (`$RUNS_ROOT/CHAIN_C_DONE`): analyst valida `rq3_tiny_object/results/tta_vanilla_multiseed_PENDENTE_validacao.csv`,
exporta WBF ensemble e E7 yolo26l → PAPER_FACTS; writer remove os `\TODO{pending: chain C rerun}` da Sec. 4/5.
Se cair: `cd /raid/user_marcospaulo/paper_2_pestline && sbatch rq3_tiny_object/slurm/chain_C_extra.sh` (idempotente).

## 4. Resultados atuais (B200, multi-seed, TEST) — detalhes e fontes em PAPER_FACTS §9

**RQ3 (tiny-object, negativo confirmado)** — `rq3_tiny_object/results/`
- macro-F1 de espécie: E0 72,4±2,9; NWD 73,1±1,4 (Δ por seed +0,8±1,5, sem efeito); GCD 70,2±2,2 (−2,2±0,6, piora 3/3);
  P2 71,2±0,9 (inconclusivo). Ablações s0: NWD só-loss +3,0 (inconclusivo), só-assigner −0,3, NWD+RFLA −2,6.
- Recall de detecção em <32 px ≈ 96 % → gargalo é **identificação**, não detecção. SAHI inconclusivo (baseline plain a imgsz 512, degenerado).
- McNemar entre seeds da mesma config: p = 0,0006–0,006 → comparações single-seed não são evidência.

**RQ2 (cascata)** — `rq2_cascade/results/`
- E0×CE label_only (3 det × 3 clf = 9 pares): R 75,4→78,5 (+3,0±2,2), P 70,1→67,8 (−2,2±2,6), F1 72,6→72,7 (+0,1),
  acc|localizado 77,0→80,1 (+3,1). McNemar 6/9 pares p<0,05 (3/9 com Holm sobre 36); bootstrap por imagem exclui 0 em só 4/9.
- VAL inverte o sinal (VAL foi usada para selecionar `best.pt` → não é held-out).
- Score-produto > label_only (F1 +0,85±0,39, 9/9). Nenhuma cascata supera o single em AP50 de espécie (re-rotula sem re-ranquear).
- **Oráculo** (classificador em crops GT): 79,6±0,5 % — cascata já no teto. Erro de localização 2,1 % vs identificação ~20 %.
- Confusão: MD→MV ~30 % e MC→MV ~15 % **não mudam** com CE; com "parts" MD→MV −12,7±2,6 pp mas MV→MD +5,9±3,2 pp
  (ainda não está em PAPER_FACTS — analyst precisa registrar). MF↔MC = 0 %.
- Latência (indicativa): E0 ≈ 12 ms/img; cascata ≈ 425 ms/img (B200).

**RQ1** — não roda nesta campanha (outra pessoa). Escrita como hipótese pré-registrada (H1 detecção melhora / H2 confusão não muda),
STID-only vs DS-F2, só placeholders. Proveniência dos dados está nos prefixos dos arquivos (`DS-ORG_` 628 train; `DS-F2_v5.1_`, `v6.0_`, `v7_`, `v8_`)
e nos `audit*.jsonl` do dataset.

### ⚠️ Conflito de tese (decisão do autor pendente)
A tese atual ("a confusão é artefato da arquitetura acoplada; desacoplar resolve") é **contradita**: a cascata não muda MD→MV
e o oráculo limita a identificação a ~80 %. O writer inseriu `\TODO{AUTHOR DECISION: ...}` sem alterar as afirmações em:
`manuscript/main.tex:36` (abstract), `sec/1_intro.tex:10, :22, :23`, `sec/2_related.tex:14`, `sec/5_experiments.tex:87`
(RQ1 "Link to RQ2"), `sec/6_conclusion.tex:4`. Proposta de reenquadramento (a validar com P0 abaixo):
*"nem dados nem desacoplamento movem a identificação; o teto (~80 %) é informacional/de rótulo"* — sustentado por oráculo,
TIDE e acordo com especialista.

Manuscrito: 16 pp. com TODOs (limite VISAPP full paper = 12 pp. **incluindo referências**). Candidatos a apêndice/corte:
`tab:rq2_ablation`, `tab:rq3_ablation`, `tab:rq2_policy`, placeholders da RQ1.

---

## 5. Literatura (RQ2) — resumo; íntegra em `docs/related_work_map.md` §RQ2, chaves no `manuscript/main.bib`

Fontes: arXiv, Crossref, bioRxiv (Semantic Scholar deu 429 → cobertura parcial de CEA/Smart Agric. Tech./Plant Methods).
Leitura por abstract/metadados (exceto Laekeman 2025, íntegra).

| Tema | Refs (chave bib) | Achado | Implicação para nós |
|---|---|---|---|
| Cascatas em insetos/fauna | `korsch2023moth` (arXiv:2307.15427), `vyskocil2024camera`, `jain2024`, `bjerge2023hierarchical`, `bjerge2026insectdct` (preprint), `beery2019` | Korsch: crops isolados 93,1 % vs pipeline 88,1 % (gap ≈5 pp); Vyskočil: cascata reduz erro 42–75 % **vs classificador de imagem inteira** | Ninguém compara com single-stage *species-aware* + oráculo + multi-seed → lacuna real. Nosso gap oráculo–pipeline ≈0,4 pp |
| Baixa resolução fine-grained | `cai2017lowres`, `chen2024vimd`, `touvron2019fixres`, `nekka2026counting` (preprint) | SR/destilação ajudam só com LR **sintética** (HR disponível); FixRes: discrepância treino/teste; Nekka: resolução máxima não é melhor em sticky traps (contagem) | SR provavelmente alucina textura (sem HR real) → P2 no máximo |
| Fusão/calibração | `zhou2021probabilistic`, `guo2017calibration`, `kuppers2020calibration` | score = P(obj)·P(cls\|obj); temperature scaling; calibração usando posição/escala | Justifica score-produto; testar re-rank/calibração para recuperar AP |
| Hierarquia/abstenção | `bertinetto2020mistakes`, `geifman2017selective`, `mahfoud2026hierarchical` (preprint), `opensect2025` | Hierarquia reduz *severidade*, não contagem; risco–cobertura; limiar 0,6 dá 80–99 % por nível | Abstenção p/ família (Diptera indet.) é argumento prático forte |
| Ruído de rótulo / especialistas | `austen2018species` (PeerJ), `northcutt2021confident`, `northcutt2021pervasive`, `song2022noisy`, `ling2023wing` | Especialistas por foto: 69–96 % por espécie; ≥3,3 % de rótulos errados em test sets; venação alar resolve família (98,6 %) mas em asa montada | Teto ~80 % pode ser ambiguidade de rótulo → medir (κ especialista) |
| Foundation models | `gu2025bioclip2`, `simeoni2025dinov3`, `oquab2023dinov2`, `kumar2022lpft`, `jia2022vpt`, `hu2021lora`, `saputra2026orchid` (preprint), `montanares2026planktonzilla` (preprint) | DINOv2 > BioCLIP 2 em +7,1 pp após fine-tune; BioCLIP fraco zero/few-shot fora do domínio | Testar **com fine-tune** (LP-FT), esperar 0–2 pp |
| Protocolo de avaliação | `bolya2020tide` (ECCV 2020), `hoiem2012diagnosing` | dAP por Cls/Loc/Both/Dupe/Bkg/Miss | Aplicar a single e cascata (P0) |

Refs ainda com `note={VERIFY}`: venue de `korsch2023moth`, NeurIPS de `touvron2019fixres`/`geifman2017selective`, `pucci2023`,
`beery2019`, `deng2023crowded`. Preprints citar com cautela. Não encontrado: estudo de acordo inter-observador MD/MV/MC por foto.
Correções feitas: `jain2024` (era mistura de dois papers; agora arXiv:2406.13031), `opensect2025` (título "Open-Insect…").

Frases de posicionamento sugeridas: ver final da seção RQ2 em `docs/related_work_map.md`.

**O que reviewers VISAPP vão cobrar na RQ2 se faltar:** TIDE para os dois sistemas; oráculo nos dois sentidos; re-rank/calibração
explicando AP↓ com recall/F1↑ (PR por espécie, operating point fixo); single-stage maior (yolo26l) ou tiling; seleção na VAL
declarada e ensemble só exploratório; evidência sobre ruído de rótulo (MD→MV 30 %); custo 12→425 ms justificado; ≥1 foundation backbone.

**O que NÃO vale gastar GPU:** mais variantes de classificador (Swin/ConvNeXt-S/focal/SupCon/ArcFace já convergem ao mesmo teto);
BioCLIP zero/few-shot; SR/destilação HR→LR; treino robusto a ruído antes de medir o ruído; mais cabeças RoI sobre features YOLO
ou limpeza SAM (já pioraram).

---

## 6. Próximos passos (em ordem; RQ2 aguarda aprovação do autor)

### 6.1 Pendências imediatas (sem aprovação extra)
1. Aguardar job 33620 → analyst (TTA/WBF validados, E7 yolo26l) → PAPER_FACTS §9 → writer remove `\TODO{pending: chain C rerun}`.
2. analyst: registrar em PAPER_FACTS a confusão CE vs parts (MD→MV −12,7, MV→MD +5,9) e corrigir a frase da §9.2 que compara
   oráculo (todos os GT) com acc|loc do single (denominadores diferentes; par homogêneo VAL: oráculo 84,6 vs single 83,6).
3. Mover os scripts geradores do analyst de `_scratch/` (ignorado) para `common/` para rastreabilidade.

### 6.2 Plano de exaustão da RQ2 (proposto; total ≈ 12–16 GPU-h ≈ 1–2 dias com 2 slots)

**P0 — CPU sobre saídas existentes** (`$RUNS_ROOT/cascade_cross_{val,test}/**/per_instance.csv`, `per_image_preds.jsonl`, `oracle_crops_*`):

| # | Experimento | Saída esperada | Custo |
|---|---|---|---|
| 1 | TIDE (Cls/Loc/Both/Dupe/Bkg/Miss) single vs cascata, por seed | `rq2_cascade/results/tide.csv` | CPU < 1 h |
| 2 | Oráculo inverso: caixas do detector + classe GT (teto de localização) | `oracle_loc.csv` | CPU |
| 3 | Re-rank/calibração: temperature scaling (VAL→TEST), fusão aprendida (logística com score_det, p_cls, tamanho, entropia; CV na VAL), emitir top-2; AP por espécie e PR com operating point fixo | `rerank.csv` | CPU ~1 h |
| 4 | Oráculo estratificado por tamanho (bins <32/32–48/≥48 px), nitidez (variância do Laplaciano) e densidade (nº de vizinhos) | `oracle_strata.csv` | CPU |
| 5 | Risco–cobertura + abstenção para família (MD/MV/MC → "Diptera indet.") | `risk_coverage.csv`, figura | CPU |
| 6 | Bootstrap clusterizado por imagem como teste primário (McNemar trata GT como independentes) | atualizar `mcnemar.csv` | CPU |
| 7 | Fusão tardia: classe do YOLO como voto extra no classificador | `late_fusion.csv` | CPU |

**P0 — ruído de rótulo (GPU + humano):**
8. Confident learning: 5-fold out-of-fold ConvNeXt-T CE sobre crops GT de treino (5 treinos ≈ 3 GPU-h) → lista de rótulos suspeitos.
9. Re-anotação **cega** de ~150 crops MD/MV/MC (estratificados: concordantes, confusões MD→MV/MC→MV, suspeitos do item 8) por
   entomologista → κ especialista vs rótulo. **Depende do autor arranjar o especialista.** Preparar pacote de imagens sem rótulo.
   Se acordo ≈ 80 % → teto é intrínseco → reenquadra a tese.

**P1 — GPU:**

| # | Experimento | Custo |
|---|---|---|
| 10 | Foundation backbone com fine-tune: DINOv3-B e BioCLIP 2 (LP-FT), seeds 42/84/126, mesma avaliação de cascata + oráculo | 4–6 GPU-h |
| 11 | Curva de sensibilidade à resolução do crop (degradação controlada só dos crops para análise; **não altera o dataset** — confirmar com o autor por causa da regra de resolução) | 3–6 GPU-h |
| 12 | Cascata sobre `E7_yolo26l_s0` (single-stage mais forte) na matriz do `eval_cascade_matrix.py` | < 0,5 h |
| 13 | Benchmark de latência dedicado (GPU fixa, warmup, batch de crops; reportar média±std) | < 0,5 h |

**P2 — opcional:** crop duplo (justo + contexto, 2–4 GPU-h); perda hierárquica (1–2 GPU-h); treino robusto a ruído (só se item 8/9 ≥ 5 %).

Fluxo por item: experimenter (código + teste CPU em `tests/`) → slurm-runner (`sbatch` em b200n1, `runs.md`) → analyst (CSV + PAPER_FACTS)
→ illustrator (`figures/src/*.py` → `manuscript/fig/`, `manuscript/tab/`) → writer (Sec. 5) → atualizar `rq2_cascade/README.md` e `docs/LEDGER.md`.

### 6.3 Depois da RQ2
- Decisão do autor sobre a tese → writer reescreve abstract/intro/conclusão e remove `\TODO{AUTHOR DECISION}`.
- Cortar para 12 pp. (apêndice/suplementar).
- Figuras: `fig_cascade_tradeoff.pdf`, `fig_confusion.pdf`, `fig_oracle_decomp.pdf`, `fig_nwd_seeds.pdf`, `fig_recall_by_size.pdf`, exemplos MD→MV.
- **Mesa redonda:** fact-checker → R1–R5 em paralelo → area-chair (`docs/reviews/round-NN/`, LEDGER) → apresentar notas e fixes com ID;
  nenhum fix aplicado sem aprovação.

## 7. Decisões pendentes do autor
1. Aprovar P0/P1 da RQ2 (e o item 11, regra de resolução).
2. Especialista para re-anotação cega (item 9).
3. Reenquadramento da tese.
4. Corte para 12 pp.
5. Branch `dev` (hoje tudo em `main`); mover `orientacoes_cluster_HPC.pdf`/`PACBB26_paper_6448.pdf` (não rastreados) para `docs/`?
6. Release do DS-F2 (define se dataset é contribuição); rastreio coleta→imagem para RQ1 (outra pessoa).
7. Rotacionar os tokens W&B/Overleaf colados no chat em 2026-09-26.
8. Enviar `manuscript` ao Overleaf quando quiser (`cd manuscript && git push origin main` — só o autor).

## 8. Mapa de commits
| Hash | Conteúdo |
|---|---|
| raiz `47ea2c7` | cadeias C/D estendidas (P2, TTA/WBF, yolo26l, E10 na matriz, VAL+TEST, oráculo) |
| raiz `1bff714` | fix IndexError `eval_wbf_tta.py` + teste |
| raiz `e1d4bbe` | analyst: CSVs B200 RQ2/RQ3 + PAPER_FACTS §9 |
| raiz `69df244` / manuscript `a235828` | Sec. 4/5 com B200; TODOs AUTHOR DECISION |
| raiz `be19d2a` | runs.md: cadeia C resubmetida (33620) |
| raiz `b8b62df` / manuscript `3882e20` | agentes com modelo por tier; plano RQ2; refs scholar no main.bib; LEDGER/DECISIONS |
