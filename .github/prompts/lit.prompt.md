---
description: "Pesquisa de literatura via scholar: busca, verifica e posiciona trabalhos relacionados, atualiza docs/related_work_map.md e gera BibTeX verificado para manuscript/main.bib."
agent: scholar
argument-hint: "Tema ou referência a verificar (ex.: 'cascata detector→classificador em insetos 2025–2026')"
---

Pesquise: **${input:topic}**

1. Busque em fontes primárias (arXiv, DOI, página da venue). Priorize 2024–2026 e trabalhos que o mapa (`docs/related_work_map.md`) ainda não lista.
2. Para cada trabalho relevante: venue/ano, claim central, setup, números-chave (métrica + split), limitação admitida, **diferença para nós** (1 linha), bloco do mapa onde entra (A–E).
3. Gere BibTeX verificado (ou `note={VERIFY}` se não conseguir confirmar) e atualize `docs/related_work_map.md`. Só edite `manuscript/main.bib` se o autor pedir explicitamente: ${input:bib:não}.
4. Devolva, em português: lista (claim → diferença), entradas BibTeX e uma frase sugerida de citação por trabalho (Intro / Related / Discussion). Nunca invente referência.
