# paper_2_pestline — regras globais (herdadas por todos os agentes)

Repositório de **experimentos + manuscrito** do paper 2 do projeto Pestline (VISAPP 2027, deadline 22/out/2026).
Leia `README.md` (tese, RQs, layout) antes de agir. Responda em **português**.

## Cluster (violação = penalidade financeira)
- **GPU só via Slurm** (`sbatch`, partição `h100n2`, Apptainer `/raid/user_marcospaulo/containers/flydet-train.sif`). Nunca `python`/`torchrun` com GPU no login node.
- **Nada na home**: datasets, checkpoints, caches (uv, HF, torch, wandb) ficam em `/raid/user_marcospaulo/`.
- Jobs com `save_period`, `resume` e `--signal=B:SIGUSR1@300` (manutenções cancelam sem aviso).
- **Nunca reduzir a resolução do dataset** (1920×1080). `imgsz` alto e/ou tiling são treino, não alteração do dataset.

## W&B
- Entity `pestline`; credenciais via `set -a; source /raid/user_marcospaulo/secrets/wandb.env; set +a`. **Nunca** copiar a chave para o repo.

## Cadeia de custódia dos números
`slurm-runner` (job → run W&B em `rqN/results/runs.md`) → `analyst` (run → `rqN/results/*.csv` → `docs/PAPER_FACTS.md`) → `illustrator`/`writer` (CSV → figura/texto) → `fact-checker` (texto → CSV).
**Nenhum número entra em `manuscript/` sem CSV em `*/results/` e linha em `docs/PAPER_FACTS.md`.** Marcar sempre `VAL` vs `TEST` e `single-seed` vs `multi-seed`.

## Escrita e anonimato
- Double-blind: paper anterior (PACBB 2026) sempre em 3ª pessoa; nunca "our previous work" / "same industrial partner".
- Sem reuso de texto/figuras do PACBB (autoplágio INSTICC/LNCS).
- Métricas primárias do usuário: **recall, precisão, acurácia por espécie**; mAP é secundário.

## Git
- Trabalhar em `dev` (ou branch de tópico). **Agentes nunca fazem `git push`**; o autor humano envia.
- `manuscript/` e `baseline/` são submodules: editar `.tex` dentro de `manuscript/`, commitar lá, depois `git add manuscript` aqui.
- Arquivos temporários só dentro do workspace (`_scratch/`, ignorado).

## Diretórios de escrita por agente
| Agente | Escreve em |
|---|---|
| orchestrator | `rq*/README.md`, `docs/LEDGER.md`, `docs/DECISIONS.md` |
| scholar | `docs/related_work_map.md`, `manuscript/main.bib` |
| experimenter | `rq*/*.py`, `common/*.py`, `tests/` |
| slurm-runner | `rq*/slurm/`, `common/slurm/`, `rq*/results/runs.md` |
| analyst | `rq*/results/*.csv|json`, `docs/PAPER_FACTS.md` |
| illustrator | `figures/`, `manuscript/fig/`, `manuscript/tab/` |
| writer | `manuscript/sec/*.tex`, `manuscript/main.tex` |
| fact-checker, reviewers, area-chair | `docs/reviews/` apenas |
