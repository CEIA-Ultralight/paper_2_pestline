# Plano do loop de melhoria do paper (início 2026-10-19)

Loop: revisor VISAPP 2027 (skill `revisor`, subagente independente) → análise → correção → recompilar → nova rodada.
Reviews salvas em `docs/reviews/round-NN/`.

## Meta (todas numa rodada limpa)
- Nota geral >= 4 (proxy 1–5; a escala oficial não é pública) com confiança >= 3.
- Zero problemas fatais e zero majors.
- Todo número do .tex rastreado a `docs/PAPER_FACTS.md`.
- <= 12 páginas incluindo referências (estado em 19/out: 16 pp.), double-blind limpo, `\TODO` resolvidos.
- Duas rodadas seguidas atingem a meta com revisores independentes.

## Limites
- Máx. 6 rodadas; corte duro em 21/out; plateau de 2 rodadas sem subir → parar e relatar.
- Sem GPU nem experimento novo; resultado negativo continua negativo; tese do .tex só se reduz/clareia.
- Sem push; commits locais em português, sem atribuição de IA (regra do projeto).
