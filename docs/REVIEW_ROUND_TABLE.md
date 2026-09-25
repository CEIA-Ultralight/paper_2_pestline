# Mesa redonda de reviewers — CVPR 2027

Seis subagentes em `.github/agents/` simulam o comitê de programa do CVPR. Rodar **a cada grande alteração** do paper para obter nota e lista de melhorias.

## Revisores

| Agente | Persona | Foco |
|---|---|---|
| `reviewer-tiny-object` | R1 — especialista em detecção de objetos pequenos | Novidade e correção de NWD/GCD/RFLA, baselines (SAHI, P2, TTA), comparação justa, ablações |
| `reviewer-data-centric` | R2 — data-centric AI / metodologia de datasets | Protocolo de coleta, splits sem leakage, copy-paste vs. sintético, viés, se o dataset pode ser contribuição |
| `reviewer-rigor-stats` | R3 — rigor experimental e estatística | Multi-seed, média±std, testes de significância, tamanho de efeito, cherry-picking, val vs. test |
| `reviewer-domain-finegrained` | R4 — classificação fine-grained / entomologia aplicada | Cascata detector→classificador, confusão MF↔BF, relevância prática, métricas por espécie (recall/precisão) |
| `reviewer-writing-format` | R5 — escrita, clareza e conformidade CVPR | 8 páginas, anonimato, figuras, claims vs. evidência, related work, limitações, checklist de formato |
| `area-chair` | AC — meta-revisor | Lê os 5 reviews, resolve conflitos, dá decisão (Accept/Borderline/Reject), prioriza os 5 fixes de maior impacto |

## Escala (formulário OpenReview CVPR 2025/2026 — verificar 2027)

| Rating | Significado |
|---|---|
| 5 | Strong Accept — top 5 %, oral/award candidate |
| 4 | Accept — contribuição clara, sólida |
| 3 | Borderline — pode ir para qualquer lado; rebuttal decide |
| 2 | Reject — problemas sérios de novidade/rigor |
| 1 | Strong Reject — falha fundamental ou violação de política |

Confidence 1–5 (5 = especialista absoluto, checou detalhes).

**Histórico de aceitação CVPR:** média ≥ 3.7 com nenhum review < 3 costuma ser aceito; qualquer 2 com AC não convencido → reject.

## Como rodar uma rodada

1. (Opcional) Compilar o PDF fora do cluster (Overleaf ou `latexmk -pdf main.tex` — **não há LaTeX no login node**) e copiar para `paper/cvpr2027/main.pdf`. Sem PDF, os revisores leem os `.tex` diretamente e o R5 estima a contagem de páginas.
2. Garantir que `paper/PAPER_FACTS.md` está atualizado (é a fonte da verdade que os revisores usam).
3. No chat (agente `main`), pedir:
   > "Rode a mesa redonda de reviewers na rodada N"
   ou usar o prompt `/review-round-table`.
4. O `main` invoca os 5 revisores **em paralelo** como subagentes (`reviewer-tiny-object`, `reviewer-data-centric`, `reviewer-rigor-stats`, `reviewer-domain-finegrained`, `reviewer-writing-format`), cada um lendo `paper/cvpr2027/sec/*.tex`, `main.tex`, `main.bib`, `paper/PAPER_FACTS.md` e (se existir) `sec/X_suppl.tex`; cada um grava `paper/reviews/round-NN/R{k}-<persona>.md`.
5. Em seguida invoca `area-chair`, que lê os 5 reviews e grava `paper/reviews/round-NN/META.md` + acrescenta uma linha à tabela abaixo.
6. Aplicar os fixes priorizados (atualizar `PAPER_FACTS.md` antes dos `.tex` quando envolver números) → commit local → nova rodada.

## Regras para os revisores

- Ler o **paper inteiro** antes de escrever. Citar seção/linha/tabela específica em cada weakness.
- Tom em 3ª pessoa ("the paper"), construtivo, sem sarcasmo.
- Alegar "já foi feito" **exige referência concreta** (autor, ano, venue).
- Não rejeitar por não bater SOTA em benchmark público; pesar novidade + impacto + rigor.
- Não pedir experimentos "substanciais" como condição; distinguir *must-fix* de *nice-to-have*.
- Verificar coerência **número a número** com `paper/PAPER_FACTS.md` (fonte da verdade dos resultados) — divergência = weakness grave.
- Pesar **positivamente** limitações honestas e resultados negativos reportados.
- Saída sempre no formato do formulário OpenReview (ver `paper/reviews/REVIEW_TEMPLATE.md`).

## Histórico de rodadas

| Rodada | Data | Versão (commit) | R1 | R2 | R3 | R4 | R5 | Média | Decisão AC | Top fix |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 2026-09-22 | 59d5da9 | 2 | 2 | 2 | 2 | 2 | 2.0 | Reject | Corrigir a cadeia de evidência RQ2: regra única de checkpoint (`best.pt`) em toda a Tab. 2 (+1,4 pp é máximo-sobre-épocas vs. `best.pt`), consertar seed em `train_nwd.py` (3×3 atual replica seed 0) e retirar "+1,4 pp" do abstract/Sec. 1/Sec. 6 até o multi-seed existir ([META](reviews/round-01/META.md)) |
