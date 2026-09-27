# DECISIONS — decisões de escopo aprovadas pelo autor

Escrito pelo `orchestrator` **após** aprovação humana. Agentes não reabrem decisões listadas aqui; para revisar uma, o autor edita este arquivo.

| Data | Decisão | Motivo | Status |
|---|---|---|---|
| 2026-09-24 | Veículo-alvo: **VISAPP 2027**, regular paper, Area 2, deadline 22/out/2026 | A2 no Qualis CAPES 2025; prazo compatível com 4 pessoas; fit temático confirmado | ativa |
| 2026-09-25 | Tese: o teto do single-stage está no acoplamento detecção+identificação, não nos dados; cascata resolve, tiny-object losses não transferem | Confronta o claim de "irredutibilidade" do PACBB; único resultado positivo robusto é a cascata | ativa |
| 2026-09-25 | Ordem das RQs: **RQ1 dados → RQ2 cascata → RQ3 transferência (negativo)** | Continua o eixo de dados do PACBB e responde no eixo de modelos | ativa |
| 2026-09-25 | RQ3 é **estudo de transferência com resultado negativo**, não método proposto; o "+1,4 pp" single-seed não é claim | Artefato de checkpoint + bug de seed; variância entre seeds > efeito | ativa |
| 2026-09-25 | PACBB 2026 é citado em 3ª pessoa; **zero reuso** de texto/figuras | Double-blind + regras INSTICC/LNCS de autoplágio | ativa |
| 2026-09-25 | Ensemble de 3 classificadores escolhido após ver o TEST é **exploratório**, nunca resultado principal | Contaminação de seleção | ativa |
| 2026-09-25 | Manuscrito vive no Overleaf (submodule `manuscript/`); código no GitHub; baseline pinado como submodule `baseline/` | Escrita colaborativa + reprodutibilidade | ativa |
| 2026-09-26 | **Campanha RQ2+RQ3 migra para dgx-B200-1 (partição `b200n1`)**; dataset, crops e container copiados para `/raid/user_marcospaulo/` deste nó; novo container CUDA 12.8 (Blackwell). **Tudo é rerodado** no mesmo ambiente (E0/E10 3 seeds, classificadores CE e partes 3 seeds); resultados H100 anteriores ficam como preliminares e saem do paper | `/raid` é local por nó; fila h100n2 saturada; torch cu124 não roda em sm_100; misturar hardware/torch invalidaria a comparação | ativa |
| 2026-09-26 | Cadeia 33015 (E10y) é **inválida**: seeds idênticos (bug de seed), E0 s1 crashed, E0 s2/E7/E14/TTA não rodaram. Nenhum número dela entra no paper | Verificado no W&B (summaries byte a byte iguais) | ativa |
| 2026-09-26 | "+1,4 pp" do NWD **removido** de qualquer claim; sob protocolo único (W&B summary) E10 0,8544 < E0 0,8605; regra de checkpoint única = `best.pt` | R3 rodada 01; analyst 2026-09-26 | ativa |
| 2026-09-26 | Escopo GPU aprovado pelo autor ("rode tudo que achar necessário das RQ2 e 3"): P0 + P1 + RFLA 1 seed; 3 seeds de classificador (42/84/126); GCD +2 seeds; SAHI nos 3 E0; NWD loss-only / assigner-only 1 seed; depois mesa redonda | Plano do orquestrador 2026-09-26 | ativa |
| 2026-09-26 | W&B: entity `pestline`, **um projeto por RQ**: `paper2-rq2-cascade`, `paper2-rq3-tiny-object` (e `paper2-rq1-data` para a RQ1). `fly-species` fica como histórico | Pedido do autor | ativa |
| 2026-09-26 | Classificador de referência da cascata = ConvNeXt-T **CE** (controle); "partes" reportado como ablação | CE seed 42 F1 74,95 ≈ partes 75,35: o ganho vem de cascatear | ativa |
| — | RQ1 descritiva (Opção A) **ou** ablação STID-only vs full (Opção B)? | Depende do rastreio das coletas novas e da fila de GPU | **pendente** |
| — | Liberar o DS-F2 publicamente? | Parceiro industrial; define se dataset é contribuição | **pendente** |
| — | Título centrado na cascata ("Decoupling Detection from Identification…") ou no sistema? | Sugestão: cascata (diferencial vs Laekeman) | **pendente** |
