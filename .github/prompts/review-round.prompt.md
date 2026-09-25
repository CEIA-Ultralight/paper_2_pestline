---
description: "Roda a mesa redonda VISAPP 2027: fact-checker → 5 reviewers em paralelo (R1–R5) → area-chair; registra a rodada em docs/reviews/round-NN e docs/LEDGER.md. Só ao fim de um ciclo de experimentos."
agent: orchestrator
argument-hint: "Número da rodada (ex.: 01)"
---

Rode a rodada **${input:round:01}** da mesa redonda de reviewers do paper VISAPP 2027.

Passos, nesta ordem:

1. `git -C manuscript pull` e anote o hash `git -C manuscript log -1 --format=%h`. `mkdir -p docs/reviews/round-${input:round:01}`.
2. Invoque `fact-checker`: "Rodada ${input:round:01} — gravar docs/reviews/round-${input:round:01}/FACTCHECK.md". Se o veredito for `BLOQUEADO`, apresente as divergências ao autor e **pergunte** se prossegue mesmo assim.
3. Invoque **em paralelo** `reviewer-applied-detection`, `reviewer-data-reproducibility`, `reviewer-rigor-stats`, `reviewer-domain-practitioner`, `reviewer-writing-visapp`, cada um com:
   > "Rodada ${input:round:01}, manuscript @ {hash} — revisar manuscript/ (main.tex, sec/*.tex, main.bib), docs/PAPER_FACTS.md, docs/reviews/round-${input:round:01}/FACTCHECK.md, sua memória em docs/reviews/personas/R{k}.md e docs/LEDGER.md; gravar docs/reviews/round-${input:round:01}/R{k}-<persona>.md seguindo docs/reviews/REVIEW_TEMPLATE.md e atualizar sua memória."
4. Invoque `area-chair`: "Rodada ${input:round:01} — ler docs/reviews/round-${input:round:01}/, gravar META.md e atualizar docs/LEDGER.md."
5. Reporte ao autor, em português e curto: notas R1–R5, média, decisão do AC, progresso vs rodada anterior e os fixes priorizados com ID. **Não aplique nenhum fix sem aprovação**; aprovados vão para o `writer` e são marcados `fixed@hash` no LEDGER.
