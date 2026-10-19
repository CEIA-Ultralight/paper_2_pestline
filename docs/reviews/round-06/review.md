# Revisão independente, rodada 06 (VISAPP 2027, perfil `conferences/visapp.md`)

Paper: "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps" (manuscript_snapshot/, fonte .tex lida integralmente).
Perfil: parcialmente `[inferido]` (escala numérica 1-5 e layout PRIMORIS não públicos; critérios e perguntas fechadas oficiais).
Limite de leitura: o ambiente não tem poppler (pdftotext/pdftoppm), então NÃO vi o PDF renderizado. Legibilidade de figuras/tabelas, quebra de página e contagem real de páginas são "not verifiable" por mim; usei a informação do pedido (12 páginas). Números conferidos contra `docs/PAPER_FACTS.md` (§9-§13) e CSVs em `rq1_data/results/`.

## 1. Paper Summary
Estudo (em grande parte negativo) sobre o que limita a leitura de armadilhas adesivas em nível de espécie (4 moscas + 3 classes não-alvo, 1920x1080, objetos ~44 px). Três alavancas: (RQ1) dados, via escada de 5 conjuntos de treino cumulativos (acurácia de espécie 56,1% -> 82,2%) com controle de subamostra aleatória de mesmo tamanho; (RQ2) cascata detector->classificador (5 classificadores, 3 detectores, 15 configurações), nenhuma melhora o detector, e o oráculo com crops GT satura em ~83%; (RQ3) 8 variantes de tiny-object (NWD, RFLA, GCD, Wise-IoU, P2, 2560 px, YOLO11m) sem ganho, mais SAHI/TTA/WBF.

## 2. Strengths
- Honestidade metodológica rara: VAL declarada não held-out, ausência de hold-out intocado admitida, CIs pareados por imagem, McNemar, Holm, deltas por seed, "no gain detected" distinto de equivalência.
- Controle de mesmo tamanho (Sec. 5.1) que desfaz a leitura ingênua "coleta direcionada ajuda" e revela o resultado L0 (+19 pp).
- Oráculo + decomposição de erro (localização vs identificação) é um desenho útil; 97,7% de localização torna o diagnóstico convincente.
- Métricas primárias orientadas à tarefa (precisão/recall por espécie), mAP secundário.
- Números verificados batem com PAPER_FACTS (ver §6).

## 3. Weaknesses

### Fatais
Nenhum. (Nada que impeça a revisão; anonimato e limite de páginas aparentemente OK.)

### Majors
M1. **Contribuição positiva fraca / novidade.** O resultado central é negativo em 2 de 3 RQs e o terceiro (RQ1) é, pelo próprio controle, em grande parte "mais imagens ajudam". O valor para a conferência depende de a evidência negativa ser sólida, e ela é limitada por M2-M4. Abstract/Intro vendem "three levers" mas o paper não oferece nenhuma alavanca que funcione nem um método novo. O que mudaria minha nota: um positivo (p.ex. cascata com crop/entrada de maior resolução ou fine-tune conjunto que bata o detector) ou uma análise causal da confusão MD/MV/MC (re-anotação de especialista numa amostra).

M2. **A conclusão "cascata não ajuda" é condicionada a uma receita única e provavelmente subótima.** Sec. 4.2/5.2: detector congelado, margem de 75% por lado, crop 384x384 (objeto de ~44 px ampliado muito), fusão só "label-only" (e "product" idêntica no harness, artefato conhecido em PAPER_FACTS §10 não mencionado no texto), sem fusão com a distribuição real do detector ("not run", Sec. 5.2, a fusão aproximada usa distribuição inventada). Margem 75% injeta muito contexto/fundo de cola; nenhuma ablação de margem, resolução de crop ou de treinamento com negativos do detector (só BG, 8 classes, 3 seeds). O título e a conclusão ("Decoupling...", "identification is not moved") generalizam além de "com nossa receita". Abstract hedge, mas o título e a Sec. 6 não.

M3. **TEST pequeno (73 imagens, 1.210 instâncias, 7 placas aglomeradas, 6 só-INS) e dependências não tratadas.** Sem identificador de placa, vazamento intra-placa entre treino e VAL/TEST "não pode ser excluído" (Sec. 3.3, 6). Em fotos de placas adesivas, imagens da mesma placa/sessão são o risco principal; a verificação de near-duplicates (miniaturas >=0,98) não detecta mesma placa em outro ângulo/dia. Isso afeta todas as RQs (acurácias e ICs otimistas). Pares det x clf (9) cruzados não são independentes (reconhecido), e os ICs por imagem tratam as imagens como iid apesar de placas agrupadas.

M4. **RQ3 sem tuning e baixo poder.** Oito variantes com hiperparâmetros default, 3 seeds, IC de ~±1,7 pp e desvio entre seeds do E0 de 1,2 pp. O texto corretamente diz "no gain detected", mas o título da RQ3 ("Transfer of Tiny-Object Methods"), a conclusão e a Sec. 5.3 ("Why the aerial loss recipes do not transfer") oferecem explicações mecanísticas não testadas (stride-8, assignment). NWD+RFLA divergiu em 1 seed e parou cedo em outra: o -8,8 pp é um problema de implementação/estabilidade e não evidência sobre o método; incluí-lo na tabela principal com Holm p=0,002 induz leitura errada. C=44 px medido em versão anterior. Além disso P2/2560 pioram o recall <32 px (-5 a -7 pp), contra-intuitivo e sem investigação (batch 4, mesma schedule?). Não há baselines tiny-object não-YOLO.

M5. **Rastro dos números de escada/dataset incompleto e inconsistência interna na composição dos conjuntos.** Sec. 3.2 lista "97, 83, 70, 100 e 174 imagens por lote" mas L0 (v5.1, "earliest lot") tem 103 imagens, e L1 = L0 + 2 versões = 244 não fecha com 97+83+70 (=250) nem 103+83+70 (=256). L3 tem 1.738 imagens com 607 de copy-paste, i.e. 1.131 reais, contra 869 em L2 ("duas versões mais recentes sem copy-paste"): L2->L3 adiciona ~262 imagens não explicadas, além de copy-paste e remoção de ORG. Sec. 5.1 afirma que L2+ORG->L3 "mixes copy-paste with removing white-paper", o que subestima a diferença. Isso importa porque "quantidade vs. direcionamento" depende de composição exata. Leitor não consegue reconstruir.

M6. **Interpretação da curva de RQ1 e H1/H2.** (a) O controle só valida L1/L2; os ganhos "localizados" (objetos <32 px, placas só-INS) não foram testados contra subamostras (admitido), mas o abstract/conclusão ainda listam "largest gains on objects below 32 px and insect-only plates" como achado. (b) Subamostra aleatória de L3 reusa rótulos das versões finais e inclui copy-paste, então "L0 pior por qualidade" e "direcionamento" não se distinguem (admitido em parte). (c) Hardware misto (H100 vs L40S) nas células do controle e single-seed/célula de EQ244 com std 0,077: "indistinguível" é fraca porque o poder é baixo. (d) Confusões da Tabela 1 são estimativas pontuais agrupadas sem IC. (e) Sec. 5.1 e conclusão dizem "from L1 to L2 the rise is mostly quantity", mas L1->L2 não foi comparado diretamente e EQ244 varia 0,69-0,84.

### Minors
m1. Sec. 4.2: "ArcFace vs parts: mAP50 75.7 vs 76.4" está rotulado como mAP50 mas em PAPER_FACTS §2-§4 (75,68 / 75,5-76) esses valores são F1 de espécie de uma campanha single-seed legada (dataset anterior). Métrica/dataset possivelmente errados; número sem rastro direto para v9.
m2. Sec. 5.2: "(E0, mean over the nine pairs; single-seed classifiers)" contradiz "classifier seeds 42, 84, 126" (Sec. 4.2/5 intro). Os CIs do texto são BCa e batem com PAPER_FACTS §10.1, mas §10 também diz "percentil" em outro bloco; padronizar. Não há descrição do "sign-flip permutation" por imagem vs instância.
m3. Faltam citações para componentes usados: ConvNeXt, DINOv2, DINOv3, BioCLIP 2 (há bioclip2024; checar se é a v2), RT-DETR, YOLO11, Wise-IoU, ArcFace, SupCon, AdamW, McNemar/Holm. "GCD, arXiv:2510.27649" inline, sem entrada .bib. Citação de "Northcutt pervasive" para justificar 3,7% como "faixa de benchmarks curados" e frase "its confusions are the model's own, as ambiguity or label noise would produce" é circular (confident learning usa os mesmos classificadores) e inconclusiva.
m4. Sec. 5.3 afirma "a 44-px fly spans about five cells of stride-8"; 44/8 = 5,5, mas a afirmação sobre "AI-TOD objects of a few pixels" não está citada nem medida. Explicação mecanística sem teste (o texto admite).
m5. Latência: "9-11 ms" vs facts 8,6-11,6; ok em ordem de grandeza. Cascata 353-573 ms "depends on backbone" mas inclui I/O? Indicativo, ok. O "30-67x" bate.
m6. Abstract é longo (~230 palavras), cheio de ressalvas encadeadas ("so there we detect no advantage... but the earliest lot..."), difícil de ler; frase sobre RQ1 é ambígua.
m7. Definição: "acurácia de espécie" = recall micro (Sec. 5.3 admite igualdade); usar um nome só (recall) evita confusão com acurácia condicionada (acc|loc) e com acurácia do oráculo. Mesmos termos "accuracy" para 3 quantidades diferentes (56-82%, oráculo ~83%, acc among localized).
m8. Dataset: não há tabela de estatísticas por classe/versão/divisão; a Sec. 3.3 é curta. Um leitor externo não vê desbalanceamento (MF 416 vs MD 224) nem tamanho por classe, que são necessários para a discussão MD/MV/MC. Há "62 images re-annotated" sem rastro em PAPER_FACTS (pode estar nos CSVs de versões).
m9. Cobertura do TEST: 3.494 GT total vs 1.210 de espécie; a avaliação de INS/NOISE/MAR só entra como FP. Sem análise por placa/dia.
m10. Conclusão prática "collecting more annotated plates matters more than model complexity" excede o que foi testado (a "complexidade" testada foi cascata/NWD/P2 em um único detector; não se variou capacidade/modelo grande nem treino de longa duração).
m11. Reprodutibilidade: "code will be released, images will not" impede reproduzir RQ1; sem custo total de GPU, sem número de épocas efetivas por run; sem lista de hiperparâmetros da augmentação do detector (default Ultralytics: mosaic etc. afetam objetos pequenos).
m12. Related work: sem comparação com literatura recente de contagem/identificação em armadilhas de moscas além de laekeman2025; "we did not find..." duas vezes (sem multi-seed, sem coleta física) são afirmações de ausência fracas.

## 4. Questions for the authors
1. Existe como reconstruir identificador de placa/sessão (data de foto, nome de arquivo original, grade)? Um split por placa mudaria os números de RQ1/RQ2?
2. Qual a composição exata (nº de imagens reais, copy-paste, ORG) de L0, L1, L2, L2+ORG e L3? Por que 244 e 869 (e 1.738 = 2x869)?
3. A cascata melhora com margem de crop menor (p.ex. 25-50%), crop maior/nativo, ou com fusão usando o vetor de probabilidades real do detector? Há evidência de que o 83% de oráculo não é limitado pela recipe do crop?
4. O detector congelado já alcança acc|loc ~84% (0,822/0,977); o oráculo 83% não é, portanto, um teto independente do que o detector faz sozinho. Como isso muda a leitura "informação limitada nas imagens"?
5. NWD+RFLA: após correção da divergência (clip de grad, warmup), qual é o resultado? Com 3 seeds saudáveis a conclusão se mantém?
6. Um especialista re-anotou ao menos uma amostra (p.ex. 200 crops MD/MV/MC)? Isso separaria ambiguidade de ruído de rótulo.

## 5. Ratings per criterion (escala 1-5 do perfil)
| Critério | Nota | Rótulo | Justificativa |
|---|---|---|---|
| Relevance | 4 | Accept | Detecção/reconhecimento fino de pequenos objetos em aplicação real, no escopo do VISAPP |
| Originality | 2 | Reject | Sem método novo; contribuição é avaliação empírica negativa de técnicas conhecidas, com controle de dados interessante |
| Technical Quality | 3 | Borderline | Estatística cuidadosa e honesta, mas TEST pequeno, receita única de cascata, RQ3 sem tuning, possível vazamento por placa |
| Significance | 3 | Borderline | Útil como alerta prático e para o domínio; generalidade limitada (1 instalação, 1 split) |
| Presentation | 3 | Borderline | Bem estruturado e honesto; abstract/denso em ressalvas, composição de conjuntos inconsistente, citações faltando |

## 6. Overall rating and Justification
**Nota geral: 3 (Borderline)**, tendendo ao limite inferior. Sem fatais; seis majors endereçáveis, mas M2/M3 limitam o que a evidência negativa pode afirmar. Decidiria por aceite se os autores (a) corrigirem a composição dos conjuntos (M5), (b) mostrarem ao menos uma ablação de margem/entrada de crop e a fusão com scores reais do detector (M2), e (c) reformularem título/conclusão como "com esta receita". Pergunta decisiva ao rebuttal: Q1 (split por placa).

## 7. Reviewer confidence
**4/5** (domínio de detecção/cascatas e protocolo experimental; sem conhecimento direto do dataset; PDF renderizado não visto, só fonte .tex).

## 8. Closed questions (sim/parcial/não)
- Abstract and introduction adequate? **Parcial** (abstract denso demais; intro clara).
- More experimental results needed? **Sim** (ablação de crop/margem, fusão real, split por placa, RQ3 com tuning).
- Comparative evaluation needed? **Parcial** (baselines tiny-object e cascata fracos; sem SOTA recente de insetos).
- More critical discussion needed? **Não** (já é crítico; falta, porém, discussão de por que o oráculo ~ acc|loc do detector).
- Figures adequate? **Not verifiable** (só 1 figura, confusão L0 vs L3, 0,68 linewidth; não vi o render).
- Conclusions and future work convincing? **Parcial** (conclusão prática sobre "dados > complexidade" excede a evidência).
- References up-to-date and appropriate? **Parcial** (boas e recentes, mas faltam citações de componentes; ver m3).
- Formatting needs adjustment? **Parcial/not verifiable** (ver §10).
- English needs improvement? **Parcial** (correto, mas frases longas e encadeadas com muitas ressalvas, p.ex. abstract e contribuição 1).

## 9. Comments to the authors
Resumo em §3-§4. Prioridade: M5 (composição de L0-L3), M3/Q1 (placas), M2 (receita da cascata), reescrever abstract, título e conclusão prática para o escopo do que foi medido; remover ou reposicionar NWD+RFLA como falha de estabilidade; citar componentes.

## 10. Confidential comments to the Program Chair
Sem suspeita de plágio ou dual submission. Auto-citação do trabalho anterior (pacbb2026) em 3a pessoa, o que preserva a cegueira, mas a descrição ("prior work introduced a data-centric pipeline for this exact setting... 0.476 to 0.608 mAP") identifica um trabalho específico com o mesmo cenário/classes; se o anterior é público, a autoria é dedutível (risco leve de quebra de anonimato, decisão da chair). Meu veredito franco: borderline, com chance real de rejeição pela falta de contribuição positiva; aceitável como paper de "negative results" se M5/M2 forem resolvidos.

## 11. Additional comments (typos, figuras, redação)
- Sec. 4.2: "ArcFace ... mAP50 75.7" (ver m1). "Sec. 5.2: 'Background class' paragraph" dentro de "Limitations and open items" com `\paragraph` sem quebra: parágrafo colado.
- Sec. 5.4 (Error Analysis) começa com frase nominal sem verbo ("Confusion rates among TEST species ... ;").
- Contribuição 1 e conclusão: frases com 3-4 ressalvas encadeadas; dividir.
- Tabela 1 e 2 usam `resizebox`: conferir fonte final legível em coluna (not verifiable).
- Sec. 3.2: parênteses de "(maximum species instances per image: 42..., 97, 83, 70, 100 and 174 images per batch)" misturam duas grandezas; separar.

---

## Anexo A. Números sem rastro / com divergência (vs. docs/PAPER_FACTS.md e CSVs)
Conferidos e OK: escada de acurácia (0,561/0,738/0,797/0,811/0,822), recall de detecção, Deltas e ICs vs L3, FP em placas só-INS, EQ103/EQ244/EQ869, confusões MC->MV e MV->MC, todos os dados de RQ2 (79,5; deltas BCa; oráculo 82,7-83,8; 60,0/4,9/86,3/69,4/96,0/79,2; 127/3.470; 4,7%; fusão tardia; latência; confusão E0 vs CE), tabela RQ3 inteira (§13), SAHI/TTA/WBF, 73 imagens/1.210 instâncias e contagens por espécie.
Sem rastro em PAPER_FACTS (checar CSV/inclusão):
- Contagens de imagens por conjunto da escada (103/244/869/1.497/1.738) e 88.128 boxes de treino; 33,1k/69,3k/10,6k/4,4k boxes (talvez em `ablation_ladder*.csv`, mas não em PAPER_FACTS).
- Sec. 3.2: máximos por imagem (42/35/34/75/77), "97, 83, 70, 100, 174 imagens por lote", 8/20/40 imagens INS-only, "62 images re-annotated", "58 instances per image" (DS-ORG): ausentes em PAPER_FACTS (grep 0 para 174/628/62).
- ORG = 42% do L2+ORG (existe linha em PAPER_FACTS, mas é derivado; 628 imagens não registrado).
- "mAP50 75.7 vs 76.4 (ArcFace vs parts)" (Sec. 4.2): rotulado e datado incorretamente (ver m1).
- 42,4 MP / 5 um/pixel / 79-89% (laekeman2025): citação externa, ok, mas confira no paper citado.
- Inconsistência hardware: Sec. 4.1 diz B200 para RQ1/RQ2 e L40S para RQ3, mas PAPER_FACTS §11b/§6 indica H100 (EQ103/EQ869 s2) e L40S (EQ244, EQ869 s0-1) e §6 menciona H100 como padrão; limitações citam só "part of the equal-size runs".
- Dentro do texto: `Sec 5.1` diz "L2: 869 (two latest versions without copy-paste)" vs L3 1.738: não fecha (M5).

## Anexo B. Conferência de páginas, double-blind, TODO
- **Páginas:** pedido informa 12 páginas incluindo referências = limite Full (12). Não verifiquei o PDF; sem folga se houver ajustes. Contagem de caracteres (10k-50k sem espaços): fonte .tex tem ~46 kB de seções (incl. LaTeX); provavelmente dentro, not verifiable com precisão.
- **Double-blind:** autor "Anonymous Author(s)", afiliação/e-mail anônimos, Acknowledgements "Anonymous during review". Paper anterior (pacbb2026) citado em 3a pessoa. Código "will be released" sem URL. Nenhuma instituição/empresa/URL identificadora encontrada em main.tex e sec/*.tex (grep manual de TODO: apenas a definição do macro). Ressalva: ver nota confidencial sobre pacbb2026; verificar metadados do PDF e de figs/rq1_confusion.pdf (not verifiable).
- **TODO:** `\TODO` definido em main.tex mas nenhuma chamada em sec/*.tex (grep: 0 usos). OK para submissão. Comentário de rascunho em main.tex ("Working title - to be discussed with the team") e em sec/5 ("RQ1 from PAPER_FACTS...") ficam no fonte; remover antes de enviar o .tex (não aparecem no PDF).
- **IA (política oficial):** Acknowledgements não declara uso de IA. Se houve texto gerado/assistido por IA, a declaração é obrigatória pelo perfil. Autor decide (not verifiable por mim).
- **Template:** SCITEPRESS.sty/apalike usados; keywords e abstract ok.
