# LEDGER — histórico do paper 2 (VISAPP 2027)

Append-only. Escrito pelo `orchestrator` (ciclos de RQ) e pelo `area-chair` (rodadas de review).
Fixes têm ID `F{rodada}-{k}` e status `open` → `fixed@<hash manuscript>` | `wontfix (motivo)` | `superseded by F..`.

## Rodadas de review

| Rodada | Data | Manuscript | Venue-alvo | R1 | R2 | R3 | R4 | R5 | Média | Decisão | Fix nº 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 (legado CVPR) | 2026-09-22 | cvpr2027 draft | CVPR 2027 | 2 | 2 | 2 | 2 | 2 | 2,0 | Reject | multi-seed + seed bug; reviews em `reviews/round-01/` (personas antigas) |

## Fixes abertos

| ID | Fix | Origem | Status |
|---|---|---|---|
| F01-01 | Corrigir bug de seed em `train_nwd.py` e rerodar 3×3 E0 vs E10 | round-01 META (CVPR) | fixed (seed via `--yolo-config`; 3×3 B200 cadeia A 33321; manuscript a235828) |
| F01-02 | Multi-seed do classificador da cascata + McNemar no TEST | round-01 META | fixed (cadeias B 33322 / D 33364; manuscript a235828) |
| F01-03 | Tabela de mapeamento de classes STID ↔ DS-F2 (colisão de sigla MF) | round-01 R4/R5 | open |
| F01-04 | Não misturar mAP do `results.csv` com mAP do harness na mesma tabela | round-01 R3 | open |
| F01-05 | Remover "same setting/partner" e placeholders de anonimato | round-01 R5 | open (manuscrito reescrito para VISAPP — reverificar) |
| F01-06 | Unidade do NWD na loss (grid × px) — diagnóstico e rerun | round-01 R1 W1/Q1 + META item 1; reaberto por Isaac em 2026-10-16 | diagnosed 2026-10-18 (confirmado: loss em grid, NWD saturado; PAPER_FACTS §12.1); fix `FLYDET_NWD_LOSS_UNITS=px`; rerun E15 ×3 + E16 ×1 concluído 2026-10-18 (PAPER_FACTS §12.2: E15 sem ganho, E16 colapsa); texto do Método/E12 pendente de decisão do autor |

## Ciclos de RQ

| Data | RQ | O que fechou | Evidência (results/) | Registrado por |
|---|---|---|---|---|
| 2026-09-25 | — | Repo criado; tese e RQs definidas; abstract + Sec. 1–2 escritos (manuscript @ 2466fe0) | — | orchestrator (sessão inicial) |
| 2026-09-26 | RQ2, RQ3 | Inventário W&B (129 runs, 8 projetos); cadeia 33015 declarada **inválida** (seed bug confirmado: seeds idênticos; E0 s1 crashed; s2/E7/E14/TTA não rodaram); "+1,4 pp" NWD contradito sob protocolo único (E10 0,8544 < E0 0,8605 W&B). Analyst exportou single-seed → CSVs; PAPER_FACTS §1b/§6/§8. Writer escreveu Sec. 4 e 5 (RQ1 em aberto), tudo marcado single-seed/preliminary (manuscript @ 77459a1, 12 pp., tectonic). Infra: workspace em dgx-B200-1, dados/checkpoints/.sif no /raid de dgx-H100-02 (h100n2). | `rq3_tiny_object/results/{nwd_sweep,sahi_tta}_single_seed.csv`, `rq2_cascade/results/{cascade_test,cascade_val,classifiers}_single_seed.csv`, `*/results/runs.md` | orchestrator |
| 2026-09-30 | RQ2, RQ3 | Campanha B200 multi-seed (cadeias A 33321, B 33322, C 33363 [falhou só no WBF ensemble; resubmetida 33620 com E7 yolo26l], D 33364). RQ3: NWD/GCD/P2/RFLA sem ganho (E0 72,4±2,9 macro-F1 TEST). RQ2: cascata R +3,0 / P −2,2 / F1 +0,1; oráculo 79,6 % = teto; MD→MV não muda (CE) → **tese contradita**, `\TODO{AUTHOR DECISION}` no manuscrito. Writer: Sec. 4/5 com B200 (manuscript a235828, 16 pp). Scholar: revisão RQ2 + ~28 refs no main.bib. Agentes com modelo por tier. | `rq2_cascade/results/{cascade_*_seeds,mcnemar,confusion_single_vs_cascade,oracle_crops,latency,policy_comparison,ap_per_class}.csv`, `rq3_tiny_object/results/{nwd_3x3,tiny_variants_*,tiny_ablation_single_seed,recall_by_size,sahi_multiseed}.csv`, PAPER_FACTS §9 | orchestrator |
| 2026-10-06 | RQ2 | Campanha v9 (DS-F2_v9.0.5): cascata não melhora o detector em nenhuma das 5 backbones × 3 detectores (ΔF1 TEST −1,1 a −5,1; precisão cai, recall ≈ igual); abstenção por confiança nunca supera o detector; oráculo ≈ 83 % (teto), 69 % dos erros com confiança ≥ 0,8, acurácia de família MD/MV/MC 96 % vs espécie 81 %. Análises CPU geradas; BG multi-seed e curva de aprendizado em job 34026 (pendentes). Seção RQ2 escrita com `\TODO` nos pendentes. | `rq2_cascade/results/v9_*.csv`, PAPER_FACTS §10 | orchestrator |
| 2026-10-06 | RQ2 | v9: cascata não supera o detector em 5 backbones × 3 detectores (ΔF1 −1,1 a −5,1, TEST); oráculo ≈ 83%; 4,9% das instâncias erradas por todos os 15 clf, 69% dos erros com confiança ≥0,8, 79% dos erros dentro de MD/MV/MC; rótulos suspeitos 3,7%. BG multi-seed e curva de aprendizado em andamento (job 34026) | `rq2_cascade/results/v9_*.csv`; PAPER_FACTS §10 | claude (a pedido do autor) |
| 2026-10-08 | RQ2 | Fechamento v9: BG ×3 seeds (TEST ΔF1 −2,75, BCa [−4,78; −0,44], oráculo 77,6±2,8, sem ganho); curva de aprendizado 25/50/100 % (TEST 83,0/81,7/82,6, platô → compatível com ambiguidade, single-seed); subseção RQ2 completa no manuscrito | `rq2_cascade/results/v9_learning_curve.csv`, `v9_oracle_bg.csv`, `v9_bootstrap_delta_f1.csv` | orchestrator |
| 2026-10-14 | RQ1 | Escada de dados concluída (5 braços × 3 seeds, TEST v9.0.5): acurácia L0 0,561 → L1 0,738 → L2 0,797 → L2ORG 0,811 → L3 0,822; ΔL2ORG−L3 −0,012 [−0,029;+0,003] (IC inclui 0, não é equivalência); FP de espécie em placas só-INS: 66–87 (L0), 43–57 (L1), 0–3 (L2+); H2 não se sustenta integralmente (MV↔MC cai, MC→MV ~12% persiste). PAPER_FACTS §11; job 34630. |
| 2026-10-16 | RQ3 | E0 × NWD refeito no DS-F2_v9.0.5 (dgx1/V100, 3 seeds cada, campanha separada da B200): TEST macro-F1 78,8±2,6 → 79,3±0,8 (Δ +0,5±3,1; McNemar 0/3; Δacc +0,03 [−1,57;+1,68]); VAL mAP50:95 −1,7±0,4 (3/3 neg.); recall <32 px −1,7±1,1 → **NWD não transfere**, confirma §9.1 no dataset do paper. PAPER_FACTS §12. | `rq3_tiny_object/results/v9_dgx1_nwd_*.csv` | Lucas (dgx1) |
| 2026-10-19 | RQ3 | Campanha ovx (L40S, 3 seeds, TEST v9.0.5; E22 single-seed): nenhuma variante supera o E0 (acc 0,817±0,012): GCD −0,002, WIoU −0,003, NWD-assigner +0,003 (IC inclui 0); P2 −0,028, 2560 −0,038, NWD-loss −0,025; NWD+RFLA −0,088 [−0,127;−0,049] (s1 divergiu). Recall <32 px: nenhuma melhora. SAHI recall 86,7%; TTA +0,4 mAP50; WBF ensemble sem ganho. Detalhes `docs/PAPER_FACTS.md` §13, CSVs `v9_ovx_*`. |

## 2026-10-19 — Loop de revisão, rodada 1→2
- Rodada 1 (revisor VISAPP, nota 2): corrigidos dataset/TODOs, números (1.210/73), anonimato e afirmações absolutas, limitações ampliadas, 16→12 páginas (tabelas de mapeamento RQ1, lotes, confusão e inferência RQ3 viraram prosa).
- Pendente para o autor: conferir manualmente campos do main.bib (autores/volumes marcados antes como VERIFY, notas removidas da saída); pacbb2026 está como "Anonymous" na bib.

## 2026-10-19 — Loop de revisão, rodada 3
- Correções aplicadas: variantes RQ3 alinhadas à Tabela 2 (sem braço NWD loss+assigner), frase dos 44 px, "nunca excede", título da seção de dados, notas do bib, limitações ampliadas, "cinco classificadores (quatro backbones)".
- Revisor independente (Sonnet): nota 3/5, confiança 3, 0 fatais, 6 majors (vazamento/teste pequeno, confusão quantidade×qualidade na escada, teto de identificação interpretativo, cascata estreita, RQ3 sem tuning/seeds instáveis, novidade modesta/release indefinido). Revisão em `docs/reviews/round-03/review.md`.
- Rodadas 2 e 3 ambas 3/5 → platô de 2 rodadas; majors restantes exigem novos dados/experimentos (proibidos). Loop encerrado conforme `docs/PLANO_LOOP_REVISAO.md`.

## 2026-10-19 — Rodada 5 (loop retomado por diretriz do autor: só parar ao atingir a nota)
- Revisor independente (Sonnet): 2/5 (conf. 3), 0 fatais, 6 majors (`docs/reviews/round-05/review.md`). Majors dependentes de experimento: split por placa (sem ID de placa), origem dos rótulos/auditoria de erros confiantes (especialista), controle de mesmo tamanho completo (nó ovx fora do ar), variantes de cascata (margem, relabel só entre espécies, fusão com score), validação da explicação do RQ3.
- Texto corrigido: abstract enxuto, contradição "controle não rodado", "15 classificadores oráculo", ArcFace/SupCon, classificador principal (CE controle; parts escolhido na VAL).

## 2026-10-19 — Controle de mesmo tamanho RQ1 completo (H100)
- Job 34950 (h100n2, 4 treinos, ~55 min): EQ869_s2 0,806; EQ103 0,754±0,007 (3 seeds). Dados do DS-F2_v9.0.5 recarregados em `/raid/user_marcospaulo/datasets` (zip); GPU isolada por `common/gpu/` (jail+watch), sem violação.
- Resultado: a 244 e 869 imagens L1/L2 ≈ subamostras do L3 (quantidade); a 103 imagens, aleatório do L3 (0,754) ≫ L0 (0,561): o degrau L0 NÃO é só quantidade. O texto "mostly quantity" do manuscrito precisa ser restringido (L1/L2) — a atualizar em manuscript_snapshot/ (resultado negativo para a hipótese "só quantidade" mantido).
- Pendente: variantes da cascata (checkpoints E0/classificadores só na B200), NWD em entrada reduzida, AI-use e token Overleaf (autor).

## 2026-10-19 — Rodada 6 (revisor independente): 3/5, confiança 4, 0 fatais, 6 majors
- M1 sem contribuição positiva; M2 cascata com receita única (margem 75%, sem fusão com score real); M3 TEST pequeno/sem ID de placa; M4 RQ3 sem tuning e NWD+RFLA instável; M5 composição L0–L3; M6 leituras de RQ1.
- Feito: M5 (composição reconstruída, CSV + PAPER_FACTS §11c, texto corrigido; L3 = L2 + 869 copy-paste). Texto de RQ1 já restrito a L1/L2. 12 páginas.
- Pendente (precisa de experimento/autor): M2 (variantes da cascata; checkpoints E0 só na B200), M4 (reposicionar NWD+RFLA como instabilidade; experimento de entrada reduzida), M3 (limite), título/conclusão "com esta receita".
- Nota: o texto diz que o rótulo da espécie vem de etiqueta manuscrita na placa (v6.0 em diante) — confirmar com o autor.

## 2026-10-19 — Variantes da cascata (M2) sem GPU
- Checkpoints E0 e classificadores copiados da B200 (ssh, só leitura) para /raid/user_marcospaulo/flydet_runs_v9; jsonl da matriz também. `rq2_cascade_variants.py`: relabel só-espécies 4 NÃO ajuda (−1,4 pp); fusão tardia com score do detector dá +1,3 pp [+0,3;+2,3] no TEST, 9/9 pares (primeiro resultado positivo da cascata; w escolhido na VAL). Margem de crop (0%/25%) ainda por fazer (precisa retreinar classificador).

## 2026-10-19 — Ablação de margem do crop (M2) e controle de hardware
- Jobs 34953 (pad 0 e 0,25; 6 classificadores, cascata VAL+TEST) e 34965 (pad 0,75 reavaliado na H100), ambos na h100n2 sob jail + gpu_watch, sem violações. Cascata piora com menos contexto (−2,5/−2,8 pp vs −0,9 pp em 0,75); nenhuma margem supera o detector sozinho. A fusão tardia (+1,3 pp na B200) vira +0,3 pp [−0,1;+0,7] na H100: não é robusta. PAPER_FACTS §11e.

## 2026-10-20 — RQ3 cadeia G (h100n2, job 35018)
G0 (controle H100) 3 seeds, G1 3, G2 3, G3 2 (s2 cancelada por prazo). Avaliação no TEST completo via cópia PNG (HEIC convertidos; harness ignora .heic → 64 imagens). Nenhum NWD ajustado supera o controle; w=1,0 colapsa (53,9%). Ver PAPER_FACTS §14. Rodada 10 do revisor: 3/5, 0F/5M/11m.
| 2026-10-18 | RQ3 | **Bug de unidade do NWD na loss confirmado** (F01-06): o Ultralytics passa ao `BboxLoss.forward` caixas em grid (`target/stride`), então C = 44 virava 44 células e o NWD da loss saturava (≈0,99; log dgx1 `target_max_in=240`). E10 (§9.1 e §12) = NWD só no assigner + CIoU ×0,5; colapso do E12 = artefato; ablação "NWD só loss" = CIoU ×0,5. GCD/P2/SAHI/RFLA não afetados. Fix `FLYDET_NWD_LOSS_UNITS=px` + testes CPU; fila `E15_nwdfix_w0.5_s{0,1,2}` + `E16_nwdpure_w1.0_px_s0` iniciada às 09:04 (dgx1 GPU 0). Conclusão "NWD não transfere" **suspensa** até o E15. | PAPER_FACTS §12.1; `rq3_tiny_object/results/runs.md` | Lucas (dgx1) |
| 2026-10-19 | RQ3 | Rerun NWD com a loss em px concluído (dgx1/V100, mesma campanha e E0 da §12): **E15** (w=0,5, 3 seeds) TEST macro-F1 78,8±0,8 vs E0 78,8±2,6 (Δ +0,1±2,8; McNemar significativo em 2/3 seeds com **sinais opostos**; Δacc −0,36 [−2,14;+1,38]); VAL mAP50:95 −1,5±1,0 (3/3 neg.); <32 px −1,8±4,8 → "NWD não transfere" **confirmado sem o bug**. **E16** (NWD puro, s0) colapsa de novo (best ép. 4, macro-F1 66,7, Δ −12,7) → colapso do E12 não era só artefato. Gerador estendido para E15/E16; CSVs `v9_dgx1_nwd_*` regenerados (E0/E10 idênticos). | PAPER_FACTS §12.2; `rq3_tiny_object/results/v9_dgx1_nwd_*.csv` | Lucas (dgx1) |

## 2026-10-21 — Manuscrito §3 reorganizada por degraus L0–L3
Decisão do autor: nomes de lote/versão (v5.1…v9.0.5) saem do texto; §3 vira aquisição + Tabela 1 (L0, L1, L2, L2+ORG, L3) + classes/splits; RQ1 (§5) e §4 citam a tabela; linha de treino removida da tabela da RQ1. Contagens só-INS de v5.1/v6.0 verificadas (0) com `common/analysis/lot_profile.py` → `rq1_data/results/lot_profile.csv` (PAPER_FACTS §11c2). TODOs restantes na §3: protocolo de aquisição e procedimento de split (dependem da equipe). 12 páginas.

## 2026-10-21 — Revisão simulada (5/10): major points tratados no texto
M3 métrica única "species recall" (def. §4.4; "accuracy" só p/ recortes GT; tabela RQ3 "Rec."); M4 nota de limitação desatualizada corrigida (controle 3 seeds em 103/244/869) e leitura "consistent with quantity"; M5 runs com bug de unidade (CIoU×0,5, NWD†+RFLA) em bloco "diagnostic" e fora das contribuições/conclusão; cadeia G marcada como loss legada; M1 histórico de uso do TEST na §4.4 + "Threats to validity" na conclusão; M2 checagem de vazamento por placa via layout dos insetos (PAPER_FACTS §11c3, nenhuma placa compartilhada); M6 parágrafo de disponibilidade; intro com contribuição diagnóstica e contribuições enxutas; declaração de IA nos agradecimentos; figura/tabelas legíveis. 12 páginas, ~45,3 mil caracteres.
