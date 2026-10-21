# Avaliação consolidada TEST — plano aprovado em 2026-09-09

Objetivo: relatório de reunião sobre todos os modelos disponíveis, sem retreino.
Prioridade: recall, precisão e acurácia; F1, mAP e custo complementares.

## Protocolo congelado antes da execução

- TEST original: 55 imagens nativas 1920×1080 / 2864 objetos, sete classes.
- Espécies-alvo: MD/MV/MC/MF, preservando INS/NOISE/MAR na avaliação.
- Confusão com background: score >=0,25 e matching class-agnostic IoU>=0,5.
- TP exige espécie correta; micro/macro calculados por classe. Acurácia
  condicional exclui GT não pareados, ao contrário do recall. Não há TN de detecção.
- Sem otimizar scores, limiares, pesos ou modelos no TEST. Ambas as políticas
  pré-definidas são apresentadas: classe substituída/score original, e produto
  do score original pelo máximo softmax. Produto não é posterior calibrado.
- Detectores E0, P2 E1, E2 inteiro e E2 fatiado512/stride384/NMS0,5/cap300.
  Fatiamento manual reproduz a ideia SAHI, não afirma identidade bit a bit
  com a implementação Supervision histórica. Originais não são alterados.
- 18 configurações de classificação: seis legadas, dez arquiteturas/seeds,
  ensemble simples e ensemble+TTA. Todas compatíveis recebem as mesmas propostas
  E0, sem usar GT para filtrar propostas. SAM apenas no diagnóstico GT já limpo.
- GT-crops e GT-RoI em tabelas separadas do sistema completo. Crops pad15 têm2707
  exemplos; pad75/SAM2840; GT-RoI usa2864. Populações não são idênticas.
- mAP IoU0,50:0,95 usa matching e função AP Ultralytics verificados por testes.
- Métricas/perfis por classe, confusões, exemplos Grad-CAM para cabeças compatíveis,
  hashes e predições persistidos, nova run W&B pestline/fly-species.
- Tempos observacionais FP32 sobre imagens decodificadas, exclusões documentadas;
  não equivalem à latência HTTP de produção. RoI possui protocolo distinto.

## Execução e entrega

Uma H100 via Slurm h100n2/Apptainer, limite4h; nenhum outro projeto é alterado.
Checkpoint por imagem/lote, sinais, retomada e snapshots imutáveis. Testes + smoke
de uma imagem + retomada antes da avaliação completa. Falhas são registradas,
nunca ocultadas; interrupção preserva resultados parciais. Nenhum push.

Relatório Markdown final no repositório após leitura e conferência dos artefatos.
TEST já exposto anteriormente: avaliação exploratória consolidada, não conjunto
cego independente. Ganho em um limiar não prova ganho de mAP nem significância.
Nenhuma alegação de teto físico ou impossibilidade de melhorar além desta busca.