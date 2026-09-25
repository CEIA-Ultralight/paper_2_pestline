---
description: "Planeja ou executa um ciclo de uma questão de pesquisa (RQ1 dados / RQ2 cascata / RQ3 tiny-object) via orchestrator: backlog → experimenter → slurm-runner → analyst → illustrator → writer → README/LEDGER."
agent: orchestrator
argument-hint: "<rq: 1|2|3> <ação: plan|run|analyze|close> [detalhes]"
---

Trabalhe na **RQ${input:rq:3}** com a ação **${input:action:plan}**.

- `plan`: leia `rq${input:rq:3}_*/README.md` ("O que falta"), `docs/DECISIONS.md` e `docs/PAPER_FACTS.md`; proponha ao autor o plano (hipótese, jobs com GPU-horas, ordem de subagentes, o que vira TODO no paper). **Não execute nada antes da aprovação.**
- `run`: com plano aprovado, chame `experimenter` (código + testes CPU) e depois `slurm-runner` (submissão + `results/runs.md`). Devolva a tabela de jobs.
- `analyze`: com jobs concluídos, chame `analyst` (CSV + PAPER_FACTS), depois `illustrator` (figuras/tabelas) e `writer` (seção correspondente, TODOs removidos só com evidência).
- `close`: atualize `rq${input:rq:3}_*/README.md` (itens de "falta" → "temos", com origem), registre no `docs/LEDGER.md` e liste o que ficou pendente.

Detalhes adicionais do autor: ${input:details:nenhum}

Responda em português, curto: o que cada subagente fez, o que mudou, o que depende do autor.
