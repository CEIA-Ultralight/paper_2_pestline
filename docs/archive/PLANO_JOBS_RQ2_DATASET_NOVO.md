# Plano de jobs — RQ2 no dataset novo (proposta, aguarda aprovação do autor)

Gerado em 2026-10-02. Base: `docs/RQ2_metodos_e_literatura.md`. Nada aqui foi submetido.
Custos de GPU são **estimativas** a partir dos tempos de referência do handoff (E0 ≈ 37–60 min; classificador ≈ 15–40 min no B200).

## Premissas e restrições
- Dataset novo (em download pelo autor): substitui o DS-F2_v8.1.1. **Todos** os números atuais viram históricos e os experimentos de RQ2 são refeitos nele.
- QOS `onejob`: máx. 2 jobs RUNNING por usuário (hoje o job 33804, de outro projeto, ocupa 1 vaga). Por isso o desenho usa **2 jobs**, que rodam em paralelo se houver 2 vagas e em sequência se houver 1.
- Cadeia C do dataset antigo terminou (`CHAIN_C_DONE`, incluindo `E7_yolo26l_s0` e `ttawbf_E0_*`); a validação do analista e o registro no PAPER_FACTS ainda estão pendentes.
- `/raid` está a 95 % (1,4 TB livres). O dataset novo, crops e checkpoints precisam caber; liberar espaço (ex.: `DS-F2_crops_sam`, `_sliced512`, `_pad50`) só com autorização do autor.
- Os marcadores `DONE` são por `RUNS_ROOT`: **usar um `RUNS_ROOT` novo** (ex.: `/raid/user_marcospaulo/flydet_runs_v9/`) para a cadeia não pular etapas do dataset antigo.

## Passo 0 — preparação (sem GPU, 0 jobs)
1. Validar o dataset novo: `data.yaml`, índices de classe, splits, resolução 1920×1080, sobreposição treino/teste (hash de imagens), contagem de GT por espécie.
2. Mapeamento de classes entre versões (atenção à colisão de siglas MF/MC).
3. Gerar os crops de classificador (pad 0,75, 384 px) a partir do GT: `rq2_cascade/extract_crops.py`, CPU.
4. Apontar `DATA_ROOT`, `CROPS`, `RUNS_ROOT` e `WANDB_PROJECT` (sufixo `-v9`) por variáveis de ambiente; nenhuma mudança de código.
5. Fase CPU (P0 do plano): desenvolver e testar os scripts de TIDE, oráculo inverso, re-rank/calibração (incl. score ciente de IoU e rejeição por entropia), oráculo estratificado, risco×cobertura, bootstrap BCa + sign-flip e fusão tardia **agora, sobre as saídas do dataset antigo**, em `common/analysis/` com testes em `tests/`. No dataset novo basta reexecutar. Também validar a cadeia C antiga e registrar no PAPER_FACTS.

## Jobs GPU (2)

### Job A — detectores (1 GPU, ≈ 9–11 GPU-h, ≈ 9–11 h)
Cadeia reentrante (`chain_run`, marcadores `DONE`, `--requeue`, `--signal=B:SIGUSR1@300`), baseada em `chain_A_detectors.sh`.

| Etapa | Conteúdo | Estimativa |
|---|---|---|
| A1 | E0 yolo26m × seeds 0/1/2 + avaliação TEST/VAL por instância | 2–3 h |
| A2 | yolo26l × seed 0 (single-stage maior) | ≈ 1 h |
| A3 | RT-DETR-l × seeds 0/1/2 (baseline DETR, ultralytics; `rtdetr-l.pt` já em `models/`) | 3–5 h |
| A4 | Propostas do E0 sobre o treino (e val) → crops de propostas rotulados espécie/fundo | ≈ 0,5 h |
| A5 | Marcador `CHAIN_A_DONE` | — |

Fora desta lista, de propósito: NWD, GCD, P2, RFLA, SAHI e TTA (RQ3; já têm resultado negativo e só entram se o autor quiser repetir no dataset novo).
Observação sobre A4: as propostas do treino vêm de um detector treinado nesses mesmos dados e tendem a ser mais limpas que no TEST; declarar essa limitação no paper (alternativa: out-of-fold, custa mais GPU).

### Job B — classificadores + avaliação (1 GPU, ≈ 14–17 GPU-h, ≈ 14–17 h)
Depende só dos crops GT até a etapa B5, então começa junto do Job A; a etapa B6 espera o `CHAIN_A_DONE` do Job A (laço com `sleep` curto, sem ocupar GPU em treino; tende a já estar pronto porque o Job A é mais curto).

| Etapa | Conteúdo | Estimativa |
|---|---|---|
| B1 | ConvNeXt-T CE × seeds 42/84/126 (classificador de referência) | 1,5–2 h |
| B2 | ConvNeXt-T "partes" × 3 seeds (ablação) | 1,5–2 h |
| B3 | DINOv3-B (LP-FT) × 3 seeds | 2–3 h |
| B4 | BioCLIP 2 (LP-FT) × 3 seeds | 2–3 h |
| B5 | Confident learning: ConvNeXt-T CE 5-fold out-of-fold nos crops GT de treino → lista de rótulos suspeitos | ≈ 3 h |
| B6a | Espera por `CHAIN_A_DONE`; treino do classificador **com classe de fundo** (crops de propostas do A4) × 3 seeds | 1–2 h |
| B6b | Matriz da cascata: detectores {E0×3, yolo26l, RT-DETR×3} × classificadores {CE, partes, DINOv3, BioCLIP 2, +fundo} × políticas {label_only, product}, em VAL e TEST (`eval_cascade_matrix.py`) | 2–3 h |
| B6c | Oráculo de crops GT para todos os classificadores (`eval_oracle_crops.py`) + benchmark de latência dedicado (GPU fixa, warmup) | < 1 h |
| B7 | Marcador `CHAIN_B_DONE` | — |

Se a matriz ficar grande demais, limitar B6b a {E0 3 seeds, yolo26l, RT-DETR s0} × {CE, partes, melhor foundation, +fundo}.

### Ordem de submissão
```
sbatch rq3_tiny_object/slurm/chain_A_detectors_v9.sh      # Job A
sbatch rq2_cascade/slurm/chain_B_classifiers_eval_v9.sh   # Job B (espera CHAIN_A_DONE só na etapa B6)
```
Total: **≈ 23–28 GPU-h em 2 jobs**. Com 2 vagas, ≈ 14–17 h de relógio; com 1 vaga, o Job B só começa quando o A terminar (≈ 25 h).
Por que não 1 job: um único job serializa tudo (≈ 25 h) e perde o paralelismo; por que não mais de 2: o QOS só roda 2 e cada job extra só aumentaria a fila.
Dependência opcional para garantir ordem sem laço de espera: `sbatch --dependency=afterok:<A>` no Job B, ao custo de B1–B5 só começarem depois do A.

## Depois dos jobs (CPU, 0 jobs)
- Rodar os scripts da Fase CPU nas saídas novas; analyst registra em `PAPER_FACTS` e nos CSVs de `rq2_cascade/results/`.
- Pacote de re-anotação cega (~150 crops MD/MV/MC estratificados entre concordantes, confusões e suspeitos do B5) para o especialista; depende do autor.
- Figuras e texto: illustrator → writer → fact-checker.

## O que precisa de decisão do autor antes do `sbatch`
1. Caminho e versão do dataset novo (e mapeamento de classes).
2. Aprovar o escopo A/B acima e o orçamento de ≈ 23–28 GPU-h.
3. Liberar espaço no `/raid` ou autorizar a remoção dos crops/datasets antigos.
4. Especialista para a re-anotação cega.
5. Se a RQ3 (NWD/GCD/P2/SAHI/TTA) será repetida no dataset novo; não está neste plano.
