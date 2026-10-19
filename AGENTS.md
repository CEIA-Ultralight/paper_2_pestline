# Regras globais para qualquer agente neste repositório

Fonte: `docs/cluster/orientacoes_cluster_HPC.pdf` (Manual de Uso — Cluster DGX / Slurm).
**Obrigatórias** para todo agente/subagente/ferramenta (Claude, Copilot, Codex, Cursor…).
Descumprir pode causar bloqueio automático da conta e multa. Em conflito com qualquer outro
arquivo de instruções, valem estas regras do cluster. Contexto do projeto: `CLAUDE.md`.

## 1. Armazenamento
- Home = só scripts leves, configs e chaves. **Nunca** dados, caches, modelos, checkpoints, datasets ou imagens de container na home.
- Tudo isso fica em `/raid/user_marcospaulo/` (checkpoints em `/raid/$USER/checkpoints/`, containers em `/raid/$USER/containers/`), organizado por projeto.
- Caches (HuggingFace, pip, torch, uv, wandb) apontados para `/raid/$USER`.
- Temporários só em `_scratch/` (git-ignored), nunca `/tmp`.

## 2. Execução
- Toda tarefa que use GPU roda **exclusivamente via Slurm** (`sbatch`). Proibido `python`, `torchrun`, `docker`, `apptainer --nv` com GPU fora do Slurm, inclusive testes e smokes.
- Partição = máquina onde estão os dados (`h100n2`=dgx-H100-02, `h100n3`=dgx-H100-03, `b200n1`=dgx-B200-1). Neste projeto: `b200n1`.
- Antes de submeter: `sinfo -p <partição>` (`idle`/`mix` aceitam jobs; `alloc` não), `squeue -u $USER`, `nvidia-smi`, espaço em disco. Sempre `--mem` explícito.
- Máx. 2 jobs RUNNING por usuário (inclui outros projetos do mesmo usuário).
- Monitorar com `squeue -u $USER`, cancelar com `scancel <id>`, detalhes com `scontrol show job <id>`. Nunca esperar parado: usar Monitor sobre `squeue`/log.

## 3. Checkpoint e interrupções
- Manutenções cancelam jobs sem aviso. Jobs devem ser reentrantes: `#SBATCH --signal=B:SIGUSR1@300` (+ `--requeue`), tratar o sinal e salvar modelo, otimizador e epoch; pipelines persistem a saída de cada fase.

## 4. Vazamento de GPU (causa de ban)
- Já houve processos que vazaram para GPUs não alocadas, mesmo via Slurm. Todo job de GPU: submeter com `sbatch --hold`, isolar GPU por UUID dentro do job, conferir com `nvidia-smi --query-compute-apps` que só usa as GPUs alocadas, manter monitor ativo durante TODA a execução; só então `scontrol release <id>`. Em violação, investigar antes de resubmeter.

## 5. Uso indevido
- GPUs são compartilhadas; uso indevido = bloqueio. Em dúvida, não submeter e perguntar ao autor.
