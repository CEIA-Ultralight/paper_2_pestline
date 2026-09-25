# paper_2_pestline — VISAPP 2027

Código, evidências e manuscrito do **paper 2** do projeto Pestline/fly-det:

> **Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps**
> Alvo: **VISAPP 2027** (Prague, 26–28/fev/2027), Area 2 — Recognition and Detection.
> Deadline regular paper: **22/out/2026** · notificação 4/dez · camera-ready 18/dez.
> Template SCITEPRESS, double-blind, submissão via PRIMORIS.

## Tese

Trabalho anterior ([PACBB 2026](docs/prior_work/pacbb2026_extract.txt)) levou a detecção single-stage em armadilhas
adesivas industriais a um teto via curadoria data-centric (STID: mAP 0,476 → 0,608) e concluiu que a confusão
entre espécies visualmente próximas seria *irredutível*. Mostramos que esse teto pertence ao **acoplamento
detecção+identificação**, não aos dados: estendendo o mesmo dataset com coletas dirigidas e desacoplando
os dois estágios (detector genérico → classificador fine-grained), recuperamos recall e F1 por espécie — incluindo
os pares tidos como irredutíveis — com um trade-off explícito em mAP; e mostramos que losses de tiny-object
validadas em imagens aéreas (NWD) **não transferem** para este regime.

| RQ | Pergunta | Pasta | Estado |
|---|---|---|---|
| **RQ1 — dados** | Estender o STID com coletas dirigidas (crowding / graduada / branco / copy-paste) melhora a detecção por espécie? | [`rq1_data/`](rq1_data/) | descritivo pronto; ablação STID-only vs full **pendente** |
| **RQ2 — arquitetura** | Desacoplar detecção de identificação (cascata) supera o single-stage em recall/F1 por espécie? A que custo? | [`rq2_cascade/`](rq2_cascade/) | resultado single-seed no TEST pronto; **multi-seed + McNemar pendentes** |
| **RQ3 — transferência** | NWD / RFLA / SAHI, validados em aerial, transferem para YOLO26 E2E em armadilhas? | [`rq3_tiny_object/`](rq3_tiny_object/) | sweep single-seed pronto; **3×3 seeds com seed corrigido pendente** |

## Layout

```
manuscript/        SUBMODULE → Overleaf (onde o paper é escrito)   git.overleaf.com/6ab5bbabb525d334d1b32b7d
baseline/          SUBMODULE → CEIA-Ultralight/fly-det @ b36580af  (pacote fly_det: YOLO26m baseline, SAHI, trainer)
common/            avaliação no TEST por espécie, export W&B, baseline E0, container Apptainer
rq1_data/          construção/curadoria do dataset (STID → DS-F2_v8.1.1), copy-paste
rq2_cascade/       crops, classificadores fine-grained, avaliação da cascata, políticas de fusão
rq3_tiny_object/   patches NWD/GCD/RFLA para Ultralytics, treino patched, SAHI/TTA/WBF
figures/           src/ (scripts que leem */results/*.csv) → out/ (PDFs referenciados em manuscript/fig)
tests/             smoke tests CPU (patches, cascata, harness de teste)
docs/              PAPER_FACTS.md (fonte da verdade numérica), mapa da literatura, reviews simuladas, paper anterior
.github/agents/    5 reviewers + area chair para rodadas de revisão simulada
```

Cada pasta de RQ tem um `README.md` com **o que já temos** e **o que falta**, e um `results/` onde entram
apenas CSV/JSON pequenos. **Regra: nenhum número entra no `.tex` sem arquivo correspondente em `*/results/` e
linha em `docs/PAPER_FACTS.md`.**

## Setup

```bash
git clone --recurse-submodules https://github.com/CEIA-Ultralight/paper_2_pestline
cd paper_2_pestline && git checkout dev
# manuscript/ pede token Overleaf pessoal (Account → Git integration); nunca commitar o token.
export UV_CACHE_DIR=/raid/$USER/.cache/uv        # nada de cache na home (regra do cluster)
uv sync                                          # instala fly_det (editável, do submodule) + deps
```

Treino/avaliação com GPU **só via Slurm** (`sbatch`, partição `h100n2`, Apptainer
`/raid/user_marcospaulo/containers/flydet-train.sif`, definição em `common/containers/flydet-train.def`).
W&B: entity `pestline`, credenciais em `/raid/user_marcospaulo/secrets/wandb.env` (fora do git).
Dataset: `DS-F2_v8.1.1` em `/raid/user_marcospaulo/fly-det/datasets/`, 1920×1080, **nunca reduzido**.

## Fluxo de um experimento

`rqN/slurm/*.sh` → `apptainer exec … python rqN/<script>.py` (importa `fly_det`) → W&B `pestline` →
`common/` exporta métricas → `rqN/results/*.csv` → `figures/src/*.py` → `manuscript/fig/` → `.tex`.

## Manuscrito

```bash
cd manuscript && git pull            # puxar edições feitas na web do Overleaf
# editar sec/*.tex ; compilar com pdflatex+bibtex (apalike)
git commit -am "..." && git push     # Overleaf; depois `git add manuscript` aqui para avançar o ponteiro
```

## Agentes (VS Code / Copilot — `.github/agents/`)

Abra o repo no VS Code e escolha o agente no seletor, ou use os prompts `/rq`, `/lit`, `/review-round`.
Regras globais herdadas por todos em `.github/copilot-instructions.md`.

| Agente | Papel | Escreve em |
|---|---|---|
| **orchestrator** (default) | decompõe a tarefa por RQ, delega, consolida, pede aprovação; mantém `rq*/README.md`, `docs/LEDGER.md`, `docs/DECISIONS.md` | READMEs, LEDGER, DECISIONS |
| **scholar** | literatura: busca, verifica, posiciona, BibTeX | `docs/related_work_map.md`, `manuscript/main.bib` |
| **experimenter** | código dos experimentos sobre `fly_det`; testes CPU | `rq*/*.py`, `common/`, `tests/` |
| **slurm-runner** | **único** que submete GPU (sbatch h100n2); registra job → run | `rq*/slurm/`, `rq*/results/runs.md` |
| **analyst** | W&B → CSV, média±std, McNemar; guardião do PAPER_FACTS | `rq*/results/*.csv`, `docs/PAPER_FACTS.md` |
| **illustrator** | figuras/tabelas geradas só a partir de CSV | `figures/`, `manuscript/{fig,tab}/` |
| **writer** | seções `.tex`, fixes aprovados, compilação | `manuscript/sec/`, `main.tex` |
| **fact-checker** | cruza `.tex` × PAPER_FACTS × CSV × `.bib` | `docs/reviews/round-NN/FACTCHECK.md` |
| **reviewer-{applied-detection, data-reproducibility, rigor-stats, domain-practitioner, writing-visapp}** | mesa redonda VISAPP (R1–R5), só ao fim de um ciclo | `docs/reviews/round-NN/R*.md`, `docs/reviews/personas/R*.md` |
| **area-chair** | meta-review, decisão, fixes com ID | `META.md`, `docs/LEDGER.md` |

Cadeia de custódia dos números: `slurm-runner` → `analyst` → `illustrator`/`writer` → `fact-checker`.
Memória compartilhada: `rq*/README.md` (backlog), `rq*/results/runs.md`, `docs/PAPER_FACTS.md`,
`docs/related_work_map.md`, `docs/LEDGER.md`, `docs/DECISIONS.md`, `docs/reviews/personas/`.

## Convenções

- Branch de trabalho: `dev`. `main` só recebe merge revisado.
- Ninguém dá push a partir de agentes; o autor humano envia.
- Multi-seed (≥3) e teste pareado (McNemar) para qualquer claim comparativo. Marcar sempre `VAL` vs `TEST` e
  `single-seed` vs `multi-seed`.
- Citar o paper anterior em 3ª pessoa; nunca "our previous work" / "same industrial partner" (double-blind).
