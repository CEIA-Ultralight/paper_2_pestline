# Plano de implementação — o que faltou da RQ2 (DS-F2_v9.0.5)

Gerado em 2026-10-02. Complementa `docs/PLANO_JOBS_RQ2_DATASET_NOVO.md`. Nada aqui foi implementado ou submetido.
Estado: o job `p2v9-core` (33851, reservas 33852/33853) cobre E0 ×3, yolo26l, ConvNeXt-T CE/partes ×3 e a matriz da cascata + oráculo (VAL e TEST).

## O que falta e o que já existe no repositório

| # | Item | Existe? | O que implementar |
|---|---|---|---|
| X1 | Classificador com classe de fundo (BG) | **Quase tudo**: `extract_proposal_crops.py` (crops de propostas, 8ª classe BG), `architecture_lab/train.py --with-background`, fallback `background` em `eval_cascade_matrix.py` | Script de job novo (o `proposal_classifier.sh` é do dataset e cluster antigos: h100n2, DS-F2_v8.1.1, flock); parametrizar para v9/b200n1 e avaliar a cascata com o classificador de 8 classes |
| X2 | Backbone foundation DINOv2-B/14 | **Sim** (`variant dinov2`, checkpoint em `models/dinov2/`) | Só o script de job (3 seeds, LP-FT se a variante já suportar; senão fine-tune completo com lr menor) |
| X3 | BioCLIP 2 | Não. Container sem `open_clip`/`timm`/`transformers`; pesos abertos no HF (acessível pelo login node) | Instalar `open_clip_torch` em `/raid/user_marcospaulo/pylibs` (via `PYTHONPATH`, sem rebuild do SIF), baixar pesos no login node para `models/`, nova variante em `architecture_lab/models.py` + teste CPU de carga/forward |
| X3b | DINOv3-B | Não. O repositório do HF exige token e aceite de licença (HTTP 401) | **Depende do autor** (token HF com acesso). Sem token: ficamos com DINOv2 e declaramos |
| X4 | RT-DETR-l como baseline single-stage | Parcial: `rtdetr-l.pt` em `models/`; `train_detector_b200.sh` aceita `MODEL=`; o trainer `fly_det` usa `YOLO(...)` do ultralytics (que despacha RT-DETR) | **Smoke de 1 época via Slurm** para confirmar: treino a imgsz 1920, predição e avaliação em `eval_cascade_matrix.py` (limite de 300 propostas). Depois 3 seeds |
| X5 | Rótulos suspeitos (confident learning) | Não. `cleanlab` ausente | **Reformulado:** rodar sobre o **TEST/VAL** usando as probabilidades dos 6 classificadores já treinados (nenhum treino novo). Implementar `rq2_cascade/label_issues.py` (limiares por classe, ~50 linhas numpy). A versão out-of-fold no treino fica opcional (as 607 imagens com copy-paste vazam entre folds) |
| X6 | Análises CPU (TIDE, oráculo inverso, re-rank/calibração, score ciente de IoU, rejeição por entropia, oráculo por tamanho/nitidez/densidade, risco×cobertura, bootstrap BCa + sign-flip, fusão tardia) | Não; há `common/stats.py` (McNemar, bootstrap, mean_std) e `common/analysis/b200_rq2*.py` (caminhos do campanha antiga) | Novos scripts em `common/analysis/` + testes em `tests/`, lendo `per_instance.csv` / `per_image_preds.jsonl` de `flydet_runs_v9` |
| X7 | Benchmark de latência dedicado | Latência existe dentro do harness, mas varia entre jobs | Passo curto no job de extras (GPU fixa, warmup, média±desvio) |
| X8 | Analyst da campanha v9 | `common/analysis/b200_*.py` apontam para `flydet_runs` e `rq*/results/` antigos | Parametrizar `RUNS_ROOT`/`OUT` (v9) e gerar CSVs em `rq2_cascade/results/` substituindo os antigos (o histórico fica no git) |
| X9 | Documentação | `CLAUDE.md` e README citam `flydet_runs`/v8.1.1 | Atualizar para v9 (RUNS_ROOT, dataset, projetos W&B) e registrar a decisão em `docs/DECISIONS.md` |

## Ordem de implementação (CPU antes de GPU, cada passo com teste)

**Fase 0 — agora, sem GPU (enquanto o core espera na fila):**
1. X6 e X5 em CPU, desenvolvidos e testados sobre os `per_instance.csv` já gerados do E0 seeds 0/1 em `flydet_runs_v9/*/test_eval/`. Quando a matriz sair, só reexecutar.
2. X8: parametrizar os geradores de CSV e atualizar X9.
3. X3: instalar `open_clip_torch` em `/raid/.../pylibs`, baixar os pesos do BioCLIP 2 no login node, escrever a variante e o teste de forward.
4. X1/X2/X4: escrever o script de job de extras (v9, b200n1, reentrante).

**Fase 1 — um job `p2v9-extras` (1 GPU), submetido com dependência no core:**
`--dependency=afterok:33851?afterok:33852?afterok:33853` (o `?` significa "qualquer um dos três concluir bem") e `--kill-on-invalid-dep=yes`; mantém **um único job meu rodando por vez**, como pediu.

| Etapa | Conteúdo | Estimativa |
|---|---|---|
| E1 | Smoke de RT-DETR (1 época); se falhar, a etapa é pulada e registrada, sem derrubar o job | ≈ 0,3 h |
| E2 | Propostas do E0_s0 sobre treino e val → crops com BG (`extract_proposal_crops.py`) | ≈ 0,5 h |
| E3 | ConvNeXt-T com BG × 3 seeds | 3–4 h |
| E4 | DINOv2-B × 3 seeds | 3–4 h |
| E5 | BioCLIP 2 × 3 seeds (se X3 passar no teste) | 3–4 h |
| E6 | RT-DETR-l × 3 seeds (se E1 passou), com avaliação TEST | 3–5 h |
| E7 | Matriz da cascata estendida: {E0 ×3, yolo26l, RT-DETR} × {BG, DINOv2, BioCLIP 2} × {label_only, product}, VAL e TEST, + oráculo dos classificadores novos | 2–3 h |
| E8 | `label_issues.py` com as probabilidades dos classificadores no TEST/VAL | minutos |
| E9 | Benchmark de latência dedicado | < 0,5 h |

Total ≈ 16–22 GPU-h (≈ 16–22 h de relógio em 1 GPU). Marcadores `DONE` por etapa, `RUNS_ROOT` v9, e a nova repetição em rc 255 da `chain_run`.

**Fase 2 — CPU, com as saídas dos jobs:**
- Rodar os scripts de X6 sobre as saídas v9; analyst registra CSVs e `PAPER_FACTS`.
- Com o X5, montar o pacote cego de ~150 crops (concordantes, confusões MD→MV e MC→MV, suspeitos) para o especialista.
- Illustrator e writer só depois (cadeia de custódia).

## Decisões do autor
1. **Token HF com acesso ao DINOv3** (se quiser DINOv3). Sem ele, o plano usa DINOv2 e declara no paper.
2. **Especialista** para a re-anotação cega (os crops são gerados na Fase 2).
3. **Ordem de prioridade** se o tempo apertar: BG (X1) e análises CPU (X6) primeiro; depois RT-DETR (X4); por último foundation (X2/X3).
4. **RT-DETR**: aceitar a etapa condicionada ao smoke de 1 época.
5. **Remoção dos itens antigos** em `/raid` ainda depende de você rodar o `rm` (ou liberar por regra de permissão).

## Riscos
- O copy-paste do treino (607 imagens) pode inflar o treino e os folds; por isso o confident learning vai para TEST/VAL.
- VAL com 52 imagens: a seleção de `best.pt` é ruidosa; reportar sempre com multi-seed e declarar.
- RT-DETR a 1920 px pode estourar memória ou o limite de propostas; o smoke decide antes de gastar horas.
- `open_clip` em `pylibs` depende da versão do torch do container (2.8.0+cu128); testar o import dentro do SIF antes de contar com ele.
