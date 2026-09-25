# YOLO E0 versus classificador em cascata — avaliação autorizada

Sem retreino: reutilizar E0 e checkpoints baseline/parts/arcface seed42.
Avaliar somente as 35 imagens originais de validação (1.649 objetos), sete
classes, mesmas caixas propostas em todos os pipelines. Os 1.645 crops GT das
rodadas anteriores não definem a população desta avaliação.

## Hipótese e controles

A reclassificação pode corrigir a identificação de objetos localizados pelo
YOLO, mas não pode recuperar objetos sem propostas adequadas. YOLO sozinho é
recalculado aqui para não misturar test, protocolos ou avaliações antigas.

- Predição única por imagem, `imgsz=1920`, `conf=0.001`, `iou=0.5`, `max_det=300`,
  FP32, sem TTA. Backend end-to-end YOLO26 pode ignorar o parâmetro de NMS;
  a característica real é registrada. O IoU de NMS não é o IoU de cálculo AP.
- Imagens originais 1920×1080 intactas. Crops das propostas usam padding de
  0,75 por lado, clipping e resize somente em memória para 384, com normalização
  salva no checkpoint. Não descartar propostas pequenas. Degeneradas mantêm
  a predição original e são contadas explicitamente.
- Carregar o código imutável dos classificadores compatível por SHA256.
- Não aplicar novo NMS após reclassificação. Remapear a ordem alfabética dos
  classificadores para a ordem YOLO; não remover MAR/NOISE/INS.
- Política principal: alterar classe, mantendo o score YOLO original. Política
  secundária: multiplicar pelo máximo softmax. Ambos são reportados; o score
  YOLO é específico de classe, não objectness, e produto não é posterior calibrado.
- AP com utilitário Ultralytics e matching verificado contra a implementação
  instalada. Reportar mAP50, mAP50:95, AP por classe e média das quatro espécies.
- Confusão com background em `conf>=0.25`, IoU>=0.5. Separar acurácia condicionada
  aos objetos pareados de acertos sobre todas as espécies GT, incluindo misses.
- Recall class-agnostic mede cobertura das propostas com esse matching; é um
  diagnóstico limitado pelo conjunto de propostas, não prova de teto físico.

## Custo e execução

Uma GPU em h100n2/Apptainer, até 30 minutos incluindo testes/smoke. Checkpoints
atômicos por imagem, retomada com mesmos hashes e identidade W&B, sinais de
manutenção. Snapshot do avaliador fixado no launcher; nenhum push Git.

Tempos medem inferência sobre imagem já decodificada: detector com seu pre/post
e, separadamente, crops/transforms/classificador. Excluem disco, rede e servidor
HTTP; portanto **não são benchmark completo da API em produção**. Há um registro
por imagem e warmup separado, com média/p50/p95 e VRAM agregada dos modelos.

W&B: nova run de avaliação `pestline/fly-species`, sem sobrescrever runs de treino.
Artefatos: predições de cada imagem, confusões, metadados, métricas e relatório.

## Limitações pré-declaradas

A validação já foi usada para seleção de checkpoints e arquiteturas; não é um
holdout cego. Predições reais diferem de caixas GT e o crop em memória não sofre
JPEG quality95 adicional como no treino. Não modificar crops/dataset para ocultar
essa diferença. Confirmação independente e análise de erro continuam necessárias.