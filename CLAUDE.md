# paper_2_pestline

Experimentos + manuscrito do paper 2 do Pestline (VISAPP 2027, **deadline 22/out/2026**; full paper = 12 pp. incluindo refs).
Detecção/identificação de moscas (YOLO26, 7 classes MD MV MC MF INS NOISE MAR) em imagens 1920×1080.
RQ1 dados (fora desta campanha), RQ2 cascata detector→classificador, RQ3 tiny-object. **Responder em português.**

## Estado e fontes da verdade
- Estado, jobs, resultados e plano RQ2: `docs/HANDOFF_CURSOR.md` (2026-09-30; confira `squeue`/`runs.md` antes de confiar nos jobs).
- Números: `docs/PAPER_FACTS.md` §9. Decisões já tomadas (não reabrir): `docs/DECISIONS.md`. Histórico: `docs/LEDGER.md`.
- Backlog por RQ: `rq*/README.md`. Literatura: `docs/related_work_map.md`.
- Papéis (scholar, experimenter, slurm-runner, analyst, illustrator, writer, fact-checker, R1–R5, area-chair): `.github/agents/*.agent.md`.
  Use o corpo do arquivo como prompt ao desempenhar o papel; nomes de modelo no frontmatter são do Copilot, ignore.
- `.github/copilot-instructions.md`: cadeia de custódia e diretórios de escrita por agente (corrigido para B200/`main`; em conflito, vale este arquivo).

## Cluster — regras OBRIGATÓRIAS (fonte: `docs/cluster/orientacoes_cluster_HPC.pdf`)
Violação = bloqueio automático de acesso + penalidade financeira.
- **Tudo via Slurm** (`sbatch`, monitorar com `squeue -u $USER`, cancelar com `scancel`). Proibido rodar GPU direto (`python`, `torchrun`, `docker`, `apptainer --nv`) no login node.
- Partição = a do nó onde estão os dados: aqui `b200n1` (dgx-B200-1; `/raid` é local ao nó). `sinfo -p b200n1` antes de submeter (`idle`/`mix` aceitam jobs; `alloc` não).
- **Armazenamento:** nada de dados/caches na home (só scripts, configs, chaves). Datasets, modelos, checkpoints (`/raid/$USER/checkpoints/`), containers (`/raid/$USER/containers/`) e caches (HF, pip, torch, uv, wandb) em `/raid/user_marcospaulo/`. Organizar por projeto.
- **Checkpoint obrigatório:** manutenções cancelam jobs sem aviso. Jobs reentrantes (`resume`, `save_period`, `#SBATCH --signal=B:SIGUSR1@300`, `--requeue`); salvar modelo, otimizador e epoch; persistir saída de cada fase em pipelines. Scripts em `common/slurm/` e `rq*/slurm/chain_*.sh` já seguem isso (idempotentes, marcadores `DONE`).
- GPUs são compartilhadas: máx. 2 jobs RUNNING por usuário (inclui outros projetos do autor); uso indevido = bloqueio.
- Temporários só em `_scratch/` (git-ignored), nunca `/tmp`. **Nunca reduzir a resolução do dataset** (1920×1080).
- W&B entity `pestline`; credenciais só via `set -a; source /raid/user_marcospaulo/secrets/wandb.env; set +a`. Nunca copiar a chave para o repo nem `--env`.

## Comandos
- Testes (CPU): `_scratch/venv_tests/bin/python -m pytest tests -q` (referência: 125 passed, 25 skipped).
- LaTeX: `cd manuscript && TECTONIC_CACHE_DIR=../_scratch/tectonic_cache tectonic -X compile main.tex --outdir _build`.
- Container/dataset/modelos/saídas: ver tabela em `docs/HANDOFF_CURSOR.md` §2 (campanha oficial v9: `DATA_ROOT=.../datasets/DS-F2_v9.0.5`, `RUNS_ROOT=/raid/user_marcospaulo/flydet_runs_v9/`, W&B `paper2-rq*-v9`).

## Convenções não óbvias
- **Cadeia de custódia:** nenhum número vai para `manuscript/` sem CSV em `*/results/` e linha em `docs/PAPER_FACTS.md`. Marcar `VAL` vs `TEST` e `single-seed` vs `multi-seed`. VAL foi usada para escolher `best.pt` → não é held-out.
- Métricas primárias: recall, precisão, acurácia por espécie; mAP é secundário.
- Double-blind: paper anterior (PACBB 2026) sempre em 3ª pessoa; sem reuso de texto/figuras dele.
- Não alterar afirmações da tese no `.tex` sem decisão do autor (há `\TODO{AUTHOR DECISION}` pendentes).
- Fluxo por item de RQ: experimenter (código + teste CPU) → slurm-runner → analyst → illustrator → writer → atualizar `rq*/README.md` e `LEDGER`.
  Planejar e obter aprovação do autor antes de qualquer `sbatch` ou edição de `.tex`. Mesa redonda: nenhum fix sem aprovação.

## Git
- **Nunca `git push`** (nem em `manuscript/`, que é o Overleaf). Commits locais em português, um por etapa, só quando o autor pedir ou o fluxo acima exigir.
- `manuscript/` e `baseline/` são submodules: commitar dentro de `manuscript/`, depois `git add manuscript` na raiz.
- Commits: o projeto autoriza um commit local por etapa concluída do fluxo; fora disso, só quando o autor pedir (isso prevalece sobre a regra global).
- Não apagar arquivos do autor sem pedir. Analyst: geradores de CSV ficam em `common/analysis/` (rastreados), não em `_scratch/`.
- **Autoria:** commits, PRs e documentos são do autor. Não incluir `Co-Authored-By`, "Generated with" nem qualquer menção a IA/Claude em mensagens de commit, README ou código (instrução do autor, 2026-10-02; vale mais que a atribuição padrão da ferramenta).
