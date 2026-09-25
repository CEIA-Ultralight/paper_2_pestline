# Rodada noturna de arquiteturas — 2026-09-07

Autorizada pelo usuário: nova branch local, commits separados, **nenhum push**,
sem novas anotações. Base: `dev/marcos`, commit `b35ee7f`.

## Restrições e método

- Uma única instância / GPU de fly-det, Slurm `h100n2`, Apptainer. Não interferir
  nos jobs de outros projetos. Dados, caches, checkpoints e logs no RAID.
- Sete classes mantidas, incluindo INS/NOISE/MAR. Imagens originais intocadas.
  Classificadores usam resize **em memória** dos crops existentes pad75;
  RoI usa imagem 1920×1080 sem redução, padding até 1920×1088.
- Somente train/val existentes. Proveniência por placa/local/período ainda NÃO
  comprovada. Ausência de duplicatas byte a byte não comprova ausência de leakage.
- O test já foi consultado nas rodadas anteriores: não é mais um holdout cego.
  Esta busca não o acessa. Confirmação futura exige avaliação independente.
- Meta continua >0,95 em acurácia das espécies E mAP50. Aqui a seleção usa
  **macro-F1 das quatro espécies**, calculada na matriz completa de sete classes.
  Acurácia sobre exemplos cuja classe verdadeira é uma das quatro espécies
  contabiliza predição INS/MAR/NOISE como erro. Reportar também acurácia geral,
  recalls/F1 por classe e confusão completa. Não confundir estas medidas com mAP.

## Hipóteses e orçamento

| Variante | Mudança controlada | Hipótese | GPU-h de treino, máximo por chamada |
|---|---|---|---:|
| baseline | ConvNeXt-T, CE suave | Referência sob novo protocolo | 1,25 |
| supcon | Mesma rede, CE + SupCon, duas views | Separar espécies próximas no embedding | 1,25 |
| arcface | Cabeça cosseno com margem angular | Melhorar separação entre classes | 1,25 |
| parts | Quatro mapas de atenção e diversidade | Descobrir regiões discriminativas sem rótulos de partes | 1,25 |
| hires | Stem stride 4 → 2, pesos preservados | Reduzir perda de detalhe nas features | 1,25 |
| roi | RoIAlign P3/P4 E0 congelado + cabeça pequena | Classificar com features da imagem inteira | 1,0 + cache |

Máximo 24 épocas/paciência 5 para crops; 30/paciência 5 para RoI. Um orçamento
igual mede custo-benefício, **não** garante convergência equivalente. SupCon
duplica views e hires custa mais por batch. Sem alteração automática de labels,
split, backbone ou hiperparâmetros após examinar test.

A campanha pode dar **uma** extensão de até 1,5 h ao melhor candidato de crops
se superar baseline por pelo menos 0,005 macro-F1 e terminar por orçamento;
retoma a mesma run/checkpoint. Não extrapola para busca infinita. Limite Slurm
12 h para testes, cache, treinos, extensão e diagnósticos. O treinador de crops
descarta épocas incompletas no limite; RoI termina a época atual. Checkpoints
atômicos a cada época com estado completo e retomada. Sinal de manutenção
interrompe no próximo batch e preserva a última época completa; não há promessa
de retomada automática se o próprio cluster cancelar a alocação.

## Operação e artefatos

- Entradas em `fly-det/scripts/architecture_lab/`; launcher em
  `fly-det/slurm/architecture_night.sh`.
- A submissão fixa o commit e o job extrai um snapshot local, de modo que novas
  alterações no workspace não mudem código no meio de uma run.
- O orquestrador executa testes e smoke de cada arquitetura antes dos treinos.
  Falhas são registradas; variantes independentes continuam. Sinais interrompem
  a campanha. Um lock impede duas campanhas deste launcher simultâneas.
- W&B: `pestline/fly-species`, uma run por variante; retomada e diagnóstico na
  mesma run, sem copiar credenciais para argumentos, repo ou snapshot.
- Resumos/manifestos/predições/estados e relatório Markdown ficam na saída da
  campanha. Manifestos de crops identificam caminhos/classes/tamanhos, não
  são hashes do conteúdo dos pixels.
- Grad-CAM usa o logit predito sobre val, incluindo confusões MD/MV, MC/MV e
  MF/NOISE; é diagnóstico qualitativo, não segmentação nem prova causal.
- Latência CUDA FP32 p50/p95 mede somente o classificador, batch 1 e 32,
  excluindo leitura, preprocessamento, detecção, transporte e API. Um benchmark
  real ponta a ponta e avaliação detector→classificador ainda serão necessários.

## Limitações e pendências anteriores

E3/E4 tiveram discrepâncias entre avaliação original e reenvio de confusão
(.8792 vs .8155; .8549 vs .8609). Não usar esses reenvios como comparação
controlada e não sobrescrever silenciosamente histórico. A baseline desta
rodada tem manifests e pré-processamento persistidos. Os resultados anteriores
~.90 eram acurácia geral de crops GT, não acurácia exclusiva de moscas nem
melhoria comprovada do sistema detector→classificador.

RoI é uma prova de viabilidade **congelada e supervisionada por caixas GT**;
não é treino conjunto, não mede propostas do detector, e usa uma população
diferente dos crops filtrados por tamanho. Reportar separadamente. SAM ruim,
upscaling e saturação de algumas redes não demonstram um limite físico de .90.
O relatório automático desta rodada é parcial; a comparação final da API
continua pendente até validação ponta a ponta.

## Continuação automática autorizada — 2026-09-08

O usuário reiterou que quer acompanhamento e novos testes automáticos, não só
uma lista de ideias. A fase 1 continua com seu snapshot imutável. Uma fase 2,
em outro snapshot, aguarda o término do job 32005, sem sobrepor treinos.

- Limite adicional: **8 horas / uma GPU**, checkpoints e paciência mantidos.
- Combinações: `parts_supcon`, `hires_supcon`, `hires_parts`. Ordem definida por
  resultados válidos dos componentes na fase 1; todas são tentadas, salvo falha
  técnica ou orçamento. Isso é uma heurística de prioridade, não prova de ganho.
- Cada combinação: smoke + verificação de retomada; até 1 h de treino completo.
- Vencedor escolhido em val seed42, seguido de baseline e vencedor **frescos**
  seed84, até 1,25 h cada. Não carregar pesos/otimizadores da fase 1 usando código novo.
- Testes verificam preservação da inicialização dos modelos originais; não
  provam identidade numérica de toda trajetória de treinamento.
- W&B, confusões, Grad-CAM, latência e relatório de cobertura continuam. RoI é
  reportado separadamente e não participa da seleção de crops.
- Os budgets diferentes da fase 1/combos e a seleção repetida em val são
  confundidores; duas sementes não substituem confirmação estatística independente.
- Se o pai falhar parcialmente, aproveitar somente exports completos validados.
  Sem baseline válida, bloquear novos treinos e registrar a causa no relatório.

### Critério de conclusão honesta

Reportar ganho **exploratório consistente nas duas sementes testadas** somente
se o vencedor superar a baseline pareada por pelo menos 0,005 macro-F1 de
espécies em ambas. Sem reprodução, classificar como provisório/inconclusivo.
Sem ganho, concluir **não encontramos melhora nas hipóteses e custos testados**,
nunca “não é mais possível melhorar”. A meta conjunta >0,95 permanece não
comprovada até avaliar detecção mAP e classificação em confirmação independente.

A segunda etapa executa um conjunto finito de combinações e uma replicação;
não inventa nem implementa código novo autonomamente após seu término. O
relatório lista explicitamente hipóteses não testadas e limitações restantes,
incluindo novas arquiteturas do detector e avaliação ponta a ponta da API.