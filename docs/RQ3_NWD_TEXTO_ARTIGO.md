# RQ3 — rascunho de texto para o artigo após o bug de unidade do NWD (2026-10-18)

> **Status: rascunho para decisão do autor.** Nada daqui foi aplicado ao `.tex` (regra do projeto: não alterar afirmações
> do manuscrito sem decisão do autor). Todo número citado está em `docs/PAPER_FACTS.md` §12, §12.1 e §12.2. Os marcados `[E15: …]`
> só existem depois dos CSVs do E15 (gerador `common/analysis/rq3_v9_dgx1_analysis.py`).
> Fatos de base: PAPER_FACTS §12.1. Histórico: LEDGER F01-06. Decisão: DECISIONS 2026-10-18.

## 1. O que precisa sair ou mudar no manuscrito, aconteça o que acontecer

| Onde | Hoje (provável) | Ação |
|---|---|---|
| Método: descrição do E10 | "NWD in the loss and in the assigner" | Se o E15 entrar: descrever o E15 (§2.A). Se não entrar: usar §2.B. |
| Mecanismo do colapso com w = 1 (E12) | "no shape/enclosure term, NWD saturates for well-localised boxes" | **Retirar.** É artefato de implementação. Ver §4. |
| Ablação B200 "NWD only in the loss" | "NWD-only loss" | **Retirar** ou renomear para "CIoU with halved weight". Não é NWD. |
| Ablação B200 "NWD only in the assigner" | — | Pode ficar. É a única ablação de NWD válida. |
| GCD, P2, SAHI, RFLA | — | Ficam como estão (não usam o NWD da loss). |
| Conclusão da RQ3 | "NWD does not transfer" | Depende do E15. Até lá, usar a frase segura (§3.C). |

## 2. Método (Sec. 4, parágrafo do NWD)

### A. Com o E15 (texto recomendado)

> We replace the IoU term of both the task-aligned assigner and the box-regression loss with a convex combination
> $S_w = (1-w)\,\mathrm{CIoU} + w\,\mathrm{NWD}$, $w = 0.5$, where
> $\mathrm{NWD}(a,b) = \exp\!\big(-W_2(a,b)/C\big)$ and $C = 44$ px is the mean $\sqrt{wh}$ of the training boxes.
> Because Ultralytics feeds the box loss with coordinates in feature-grid units (boxes divided by the level stride),
> we rescale predicted and target boxes by their stride before computing NWD, so that both call sites operate in
> pixels and $C$ keeps its physical meaning across P3–P5. CIoU is scale-invariant and is left unchanged.

Ponto para a resposta ao revisor (R1 W1): essa frase responde diretamente ao "coordinate frame" que ele apontou.

### B. Sem o E15 (só se a fila não terminar a tempo)

> In the variant evaluated here, NWD is computed in pixels inside the assigner; in the box loss the coordinates are in
> grid units, which saturates the NWD term ($\approx 0.99$ for all foreground pairs), so the loss-side mixture
> effectively halves the CIoU weight. We therefore report this variant as *NWD-based assignment*, not as an NWD loss.

## 3. Resultados (Sec. 5, RQ3)

### A. Se o E15 não melhorar sobre o E0 (Δ < std, McNemar não significativo) — ✅ **é o caso** (PAPER_FACTS §12.2, 2026-10-19)

> With NWD computed in pixels in both the assigner and the loss (three seeds), species identification on TEST does
> not improve over the CIoU baseline: macro-F1 78.8 ± 0.8 vs 78.8 ± 2.6 (Δ +0.1 ± 2.8; exact McNemar significant
> in 2/3 seeds, but in opposite directions; pooled species-accuracy Δ −0.4 pp, 95% bootstrap CI [−2.1, +1.4]),
> and small objects (< 32 px) do not benefit (Δ recall −1.8 ± 4.8). Fine localisation on VAL drops slightly
> (mAP50:95 −1.5 ± 1.0, 3/3 seeds). The earlier variant with a saturated loss term behaved the same way
> (macro-F1 79.3 ± 0.8, Δ +0.5 ± 3.1), so the null result does not depend on the implementation detail.

Nota: "McNemar significativo em 2/3" tem que vir com "sinais opostos" (s0 favorece E0, s2 favorece E15); sem isso a
frase sugere efeito. Alternativa mais curta: omitir o McNemar e citar só o IC do bootstrap.

### B. Se o E15 melhorar (Δ > std em 3/3 seeds e McNemar significativo em pelo menos 2/3) — ❌ não se aplica

> Computing NWD in pixels in the box loss matters: with the corrected implementation, macro-F1 on TEST rises to
> [E15: m ± s] (Δ [E15: Δ ± s] over CIoU, 3/3 seeds positive; McNemar significant in [E15: k]/3), whereas
> NWD-based assignment alone (saturated loss term) stayed within seed variance (Δ +0.5 ± 3.1).

Atenção: com B a tese da RQ3 ("tiny-object losses não transferem", DECISIONS 2026-09-25) **muda**. Isso é decisão do
autor e precisa entrar em `docs/DECISIONS.md` antes de reescrever abstract e Sec. 1.

### C. Frase segura (vale até sair o E15, e serve como fallback)

> None of the tested variants improves species identification beyond seed-to-seed variance.

## 4. E12 / E16 (NWD puro, w = 1)

- ~~Se o E16 não colapsar~~ — não se aplica.
- **O E16 colapsou** (PAPER_FACTS §12.2: best epoch 4, early stop na 34; VAL mAP50 78,2 / mAP50:95 48,9; TEST macro-F1 66,7, Δ −12,7 vs E0 s0):
  > Pure NWD ($w=1$, pixels, single seed, exploratory) fails to converge (best epoch 4; TEST macro-F1 66.7 vs 79.4
  > for CIoU with the same seed); far from the target the NWD gradient vanishes ($\exp(-W_2/C)\to 0$), so an
  > overlap-based term is still needed.
- O E12 antigo continua **fora** do artigo (outra campanha, com o bug). Mas o mecanismo "NWD satura para caixas bem
  localizadas / falta termo de forma" também sai: o colapso existe sem o bug e o mecanismo plausível é o gradiente ≈ 0
  longe do alvo (teste CPU da §12.1), não a saturação.

## 5. Limitações (Sec. 6)

> $C = 44$ px was measured on the previous dataset release and not re-estimated; NWD results come from a separate
> V100 campaign with its own CIoU baseline, and the pure-NWD run is single-seed.

## 6. Checklist antes de colar no `.tex`

- [x] CSVs do E15/E16 gerados em `rq3_tiny_object/results/` + linhas novas no PAPER_FACTS (§12.2) — 2026-10-19.
- [x] Escolher §3.A ou §3.B pelos números, sem reenquadrar (regra `.claude/rules/experiments.md`) → §3.A.
- [x] Se for §3.B: registrar a mudança de tese em `docs/DECISIONS.md` → não se aplica (tese mantida).
- [ ] Retirar o mecanismo do E12 e a ablação "NWD só loss" (§1).
- [ ] Fact-checker: todos os números do parágrafo batem com o PAPER_FACTS.
