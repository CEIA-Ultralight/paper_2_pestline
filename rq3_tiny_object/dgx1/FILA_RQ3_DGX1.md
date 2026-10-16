# Fila de treinos da RQ3 na dgx1 (V100, GPU 0)

## Teste validado e como rodar a fila

O teste de 1 época (`smoke_E0`) deu **`n_gt=3494`**, o mesmo TEST da RQ1 e da RQ2 (73 imagens e todas as instâncias). O caminho inteiro funcionou de ponta a ponta: treino, checkpoint, avaliação no TEST e os 2 runs no W&B.

Antes disso, 83 imagens HEIC (66 no train, 8 no val e 9 no test) foram convertidas para PNG. Os originais estão em `DS-F2_v9.0.5/_heic_orig/`. Sem a conversão, o avaliador do TEST ignorava as 9 imagens HEIC do test e contava só 2.980 objetos.

**Avisos que podem ser ignorados:**

- `'half' is deprecated`: aviso interno da biblioteca.
- `NMS time limit exceeded`: acontece porque o modelo com 1 época gera caixas demais. Some quando o modelo treina.
- `Cache directory ... not writable`: o dataset é montado como somente leitura de propósito, para não ser alterado.

**Tempo medido:** cerca de 1:45 de treino + ~45 s de validação, ou seja, **~2,5 min por época**. Cada treino leva no máximo ~6 h (150 épocas) e costuma parar antes. Os **6 treinos devem levar ~1–1,5 dia**.

### Limpar o W&B

Em `pestline/paper2_rq3`, apagar os dois runs **antigos**, que tinham o TEST incompleto: **`kean52sk`** (`RQ3_Test`) e **`qlsyy0r4`** (`RQ3_Test_test_eval`). Ficam `oy267n1h` e `2tdjry06`.

### Soltar a fila

```bash
cd ~/pestline/paper_2_pestline
GPU=0 nohup bash rq3_tiny_object/dgx1/run_queue.sh \
  E0_s0 E10_nwd_w0.5_s0 E0_s1 E10_nwd_w0.5_s1 E0_s2 E10_nwd_w0.5_s2 \
  > ~/pestline/logs/fila_gpu0.log 2>&1 &
```

Depois disso dá para fechar o terminal.

### Acompanhar

```bash
tail -f ~/pestline/logs/fila_gpu0.log                 # Ctrl+C só sai da visualização
docker ps --filter name=rq3-                          # qual run está rodando
ls ~/pestline/flydet_runs_rq3_v9_dgx1/*/DONE          # quais terminaram
grep "n_gt=" ~/pestline/logs/E*.log                   # resultado no TEST de cada run (n_gt=3494)
```

### Se cair

Rodar **o mesmo comando** da fila de novo. Ele pula o que tem `DONE` e retoma o treino interrompido do `last.pt`. Para parar tudo:

```bash
pkill -f run_queue.sh; docker stop $(docker ps -q --filter name=rq3-)
```

## O que é cada treino da fila

A fila tem **6 treinos** do mesmo detector (YOLO26m), rodando um depois do outro na GPU 0. São **2 variantes × 3 seeds**:

| Ordem | Treino | Variante | Seed |
|---|---|---|---|
| 1 | `E0_s0` | baseline | 0 |
| 2 | `E10_nwd_w0.5_s0` | NWD | 0 |
| 3 | `E0_s1` | baseline | 1 |
| 4 | `E10_nwd_w0.5_s1` | NWD | 1 |
| 5 | `E0_s2` | baseline | 2 |
| 6 | `E10_nwd_w0.5_s2` | NWD | 2 |

### As duas variantes

- **E0 (baseline):** YOLO26m normal, sem nenhuma modificação. A loss de caixa é a CIoU padrão. É a referência da comparação.
- **E10 (NWD):** o mesmo YOLO26m com o patch da RQ3. A medida de sobreposição de caixas vira uma mistura de 50 % CIoU + 50 % **NWD** (*Normalized Wasserstein Distance*). O NWD foi criado para objetos minúsculos em imagens aéreas, e a constante C = 44 px corresponde ao tamanho médio dos insetos. O patch vale para a loss e para a etapa que decide quais previsões contam como positivas (`scope=both`).

**Todo o resto é idêntico nas duas:** dataset v9.0.5, 1920×1080, 150 épocas, batch 8, patience 30 e mesmos pesos iniciais. Assim, qualquer diferença vem só do NWD.

### Por que 3 seeds

A **seed** controla a aleatoriedade do treino: ordem das imagens, augmentations e inicialização da cabeça nova. Na B200 ficou demonstrado que dois treinos da mesma configuração com seeds diferentes já diferem de forma estatisticamente significativa. Então um treino só por variante não prova nada. Com 3 seeds de cada, dá para calcular **média ± desvio** e comparar NWD × baseline por seed.

### Por que intercalar

A ordem baseline, NWD, baseline, NWD… garante que, a cada 2 treinos, existe um **par completo** para comparar. Se o prazo apertar e a fila não terminar, os pares que já saíram continuam valendo.

### O que cada treino produz

- **Na pasta `~/pestline/flydet_runs_rq3_v9_dgx1/<treino>/`:** o `best.pt`, as curvas e a avaliação no TEST (`test_eval/metrics.json` e `per_instance.csv`).
- **No W&B (`pestline/paper2_rq3`):** 2 runs, `<treino>` e `<treino>_test_eval`.

### O que a RQ3 quer responder

A pergunta é se o NWD melhora o F1 por espécie no TEST além da variação natural entre seeds. A hipótese do paper é que **não**, ou seja, que essa técnica de imagens aéreas não se transfere para este caso. Na B200 isso se confirmou: Δ F1 = +0,8 ± 1,5, sem efeito. Este experimento repete a comparação no dataset v9.

## Quanto tempo demora

No teste de 1 época, cada época levou **~2,5 min** (1:45 de treino + ~45 s de validação).

| | Épocas | Tempo |
|---|---|---|
| **Um treino** | ~100–150 (para antes se não melhorar por 30 épocas) | **~4–6,5 h** |
| **Par E0 + NWD de uma seed** | | **~8–13 h** |
| **Fila inteira (6 treinos)** | | **~25–38 h** |

Com a fila iniciada por volta das 16h30 de 14/out/2026, a previsão é esta:

- **1º par** (`E0_s0` + `E10_nwd_w0.5_s0`): madrugada ou manhã de **15/out**.
- **Fila completa:** entre a noite de **15/out** e a manhã de **16/out**.

Isso deixa uns 6 dias até o prazo de 22/out. Dá para rodar depois o GCD ×3 (~15 h) e o P2 ×3. O P2 é mais pesado, então conte com uns 20–25 h para ele.

A estimativa pode mudar por três motivos:

- **Parada antecipada:** só dá para saber quando cada treino para depois que ele roda.
- **NWD:** o patch acrescenta um pouco de custo por iteração. Conte uns 5–10 % a mais.
- **Máquina compartilhada:** a GPU 0 é exclusiva, mas CPU e disco são divididos com outros usuários. Isso pode atrasar a leitura das imagens.

Para ver o tempo real de cada treino:

```bash
grep "Training finished" ~/pestline/logs/E*.log
```
