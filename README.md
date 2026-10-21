# paper_2_pestline — VISAPP 2027

Código, evidências e manuscrito de

> **Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps**
> VISAPP 2027 (Praga, 26–28/fev/2027), Area 2 — Recognition and Detection · regular paper, double-blind, ≤ 12 páginas com referências.

Detecção e identificação de moscas (YOLO26, 7 classes: MD MV MC MF INS NOISE MAR) em imagens 1920×1080 de armadilhas adesivas industriais.
Três alavancas, um protocolo (mesmo dataset, mesmo conjunto de teste, métricas por espécie, três seeds):

| RQ | Pergunta | Resultado (detalhes e CSVs em `docs/PAPER_FACTS.md`) | Pasta |
|---|---|---|---|
| **RQ1 — dados** | Coletas físicas dirigidas melhoram a leitura por espécie? | Em cinco conjuntos de treino cumulativos (L0–L3), a acurácia de espécie sobe de 56,1 % para 82,2 %. Em L1 e L2, subamostras aleatórias do conjunto final igualam a acurácia; em L0 não. Quantidade e direcionamento não se separam. | [`rq1_data/`](rq1_data/) |
| **RQ2 — identificação** | Uma cascata detector → classificador supera o detector único? | Não, em nenhuma das 15 configurações (5 classificadores × 3 detectores). A acurácia com crops de verdade (oráculo) satura perto de 83 %, com erros confiantes concentrados em MD/MV/MC. | [`rq2_cascade/`](rq2_cascade/) |
| **RQ3 — escala** | Métodos tiny-object da literatura aérea (NWD, RFLA, GCD, WIoU, P2, 2560 px, SAHI/TTA) transferem? | Nenhuma variante melhora o controle da mesma campanha. NWD+RFLA é instável e NWD puro não converge. | [`rq3_tiny_object/`](rq3_tiny_object/) |

Todos os resultados vêm de uma instalação e de um conjunto de teste de 73 imagens (1210 instâncias de espécie).

## Layout

```
manuscript/        submódulo → Overleaf (texto do artigo, template SCITEPRESS)
baseline/          submódulo → fly-det (pacote fly_det: treino YOLO26m, SAHI)
common/            avaliação por espécie, estatística, containers, scripts Slurm comuns
  analysis/          geradores de CSV e figuras (leem os dumps do W&B e as saídas dos runs)
  gpu/               auditoria e isolamento de GPU para jobs Slurm
rq1_data/          construção da escada de dados e do controle de mesmo tamanho
rq2_cascade/       crops, classificadores, avaliação da cascata, oráculo, políticas
rq3_tiny_object/   patches NWD/GCD/RFLA/WIoU para o Ultralytics, treino, SAHI/TTA
tests/             testes em CPU
docs/              fonte da verdade dos números, decisões, histórico, literatura, revisões simuladas
```

Cada pasta de RQ tem um `README.md` com os scripts, os resultados finais e a lista de CSVs, e um `results/runs.md` que liga job Slurm → run W&B → CSV.

## Documentos de referência

| Arquivo | Para quê |
|---|---|
| [`docs/PAPER_FACTS.md`](docs/PAPER_FACTS.md) | Todo número do artigo, com origem. Regra: nenhum número vai para o `.tex` sem CSV em `*/results/` e linha aqui. Tem um mapa de seções no topo. |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Decisões de escopo e protocolo, com data e motivo. |
| [`docs/LEDGER.md`](docs/LEDGER.md) | Histórico cronológico (rodadas de revisão, campanhas, correções). |
| [`docs/STATISTICAL_ANALYSIS.md`](docs/STATISTICAL_ANALYSIS.md) | Procedimentos estatísticos (bootstrap pareado, permutação, McNemar). |
| [`docs/related_work_map.md`](docs/related_work_map.md) | Mapa da literatura. |
| [`docs/VISAPP2027_REQUIREMENTS.md`](docs/VISAPP2027_REQUIREMENTS.md) | Requisitos da conferência (formato, anonimato). |
| [`docs/reviews/`](docs/reviews/) | Rodadas de revisão simulada (formulário VISAPP). |
| [`docs/archive/`](docs/archive/) | Planos e handoffs antigos, só por rastreabilidade. |

## Protocolo experimental (fixo)

- Dataset **DS-F2_v9.0.5**, 1920×1080 (nunca reduzido), 7 classes; teste fixo de 73 imagens.
- Receita de detector (E0): YOLO26m, 150 épocas, batch 8, `imgsz` 1920, patience 30, seeds 0/1/2.
- A métrica primária é a acurácia (= recall micro) de espécie nas quatro espécies MD, MV, MC e MF; mAP é secundária.
- `best.pt` e as épocas foram escolhidos na validação, que portanto não é held-out.
- Cada campanha de RQ3 tem o seu próprio controle E0, e hardware diferente (B200, H100, L40S, V100) nunca é misturado numa mesma tabela.
- Em RQ3 a loss NWD das campanhas L40S e H100 usa unidades de grid (†, ver `docs/PAPER_FACTS.md` §12.1); a campanha V100 usa pixels.

## Reprodução

```bash
git clone --recurse-submodules <url-deste-repositório>
cd paper_2_pestline
export UV_CACHE_DIR=/raid/$USER/.cache/uv        # nada de cache na home (regra do cluster)
uv sync                                          # instala fly_det (editável) e dependências
```

**Testes (CPU):** `python -m pytest tests -q`.

**Treino e avaliação (GPU, só via Slurm):** `sbatch` de um script em `*/slurm/`. Os scripts são idempotentes e reentrantes (marcadores `DONE`, `--signal=B:SIGUSR1@300`, `--requeue`).
Caminhos, partição e variáveis (`DATA_ROOT`, `RUNS_ROOT`, `IMAGE`) estão em [`common/slurm/b200_common.sh`](common/slurm/b200_common.sh) e podem ser sobrescritos por variável de ambiente.
O container é construído a partir de [`common/containers/flydet-train-b200.def`](common/containers/flydet-train-b200.def) (Ultralytics 8.4.142).
Métricas vão para o W&B (entity `pestline`; credenciais por arquivo `.env` fora do repositório).

**Do run ao artigo:** `rqN/slurm/*.sh` → W&B + `per_instance.csv` → `common/analysis/*.py` → `rqN/results/*.csv` → `docs/PAPER_FACTS.md` → `manuscript/sec/*.tex`.
Os geradores de cada CSV estão listados no README da RQ correspondente.

**Manuscrito:**

```bash
cd manuscript && mkdir -p _build
TECTONIC_CACHE_DIR=../_scratch/tectonic_cache tectonic -X compile main.tex --outdir _build
```

## Dados e código

O conjunto de dados pertence a um parceiro industrial e **não é distribuído**; o código será liberado e as imagens não.
Os CSVs em `*/results/` contêm as métricas por run, por seed e por instância usadas nas tabelas do artigo.

## Convenções

- Commits em português, um por etapa concluída; o manuscrito é commitado dentro de `manuscript/` e depois `git add manuscript` na raiz.
- Multi-seed (≥ 3) e teste pareado para qualquer afirmação comparativa; sempre marcar VAL vs TEST e single-seed vs multi-seed.
- O artigo anterior do grupo (PACBB 2026) é citado em terceira pessoa, sem reuso de texto ou figuras.
- Temporários só em `_scratch/` (ignorado pelo git).
