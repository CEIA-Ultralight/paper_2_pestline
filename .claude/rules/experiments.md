---
paths:
  - "rq*/**"
  - "docs/PAPER_FACTS.md"
  - "docs/LEDGER.md"
  - "manuscript/**"
---
# Experimentos e rastreabilidade

- Todo job vira linha em `rqN/results/runs.md` (job Slurm → run W&B → commit) pelo slurm-runner; só então o analyst gera CSV.
- Mudar uma variável importante por vez; ablações single-seed são exploratórias e devem ser rotuladas assim.
- Não mudar o protocolo (splits, resolução, receita E0: 150 ép., batch 8, patience 30) sem registrar em `docs/DECISIONS.md` com data e motivo.
- Resultado negativo ou inconclusivo é resultado: reportar com Δ por seed e IC, sem reenquadrar para parecer positivo.
- Geradores de CSV/figura devem ficar em `common/` ou `figures/src/` (rastreados), não em `_scratch/`.
- Texto do manuscrito: cada número deve ser rastreável a `PAPER_FACTS.md`; divergência → o fact-checker/analyst corrige, não o writer.
