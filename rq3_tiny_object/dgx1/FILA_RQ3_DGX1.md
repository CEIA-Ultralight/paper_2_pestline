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

## Segunda fila: NWD corrigido (E15 + E16, 2026-10-18)

Os `E10_nwd_w0.5_s*` da primeira fila tinham o NWD **da loss** em unidades de grid (saturado, ≈0,99). Só o NWD do assigner estava correto (`docs/PAPER_FACTS.md` §12.1). A segunda fila refaz o NWD com a loss em pixels.

| Ordem | Run | O que é | Seed |
|---|---|---|---|
| 1 | `E15_nwdfix_w0.5_s0` | 0,5·CIoU + 0,5·NWD, loss em px | 0 |
| 2 | `E15_nwdfix_w0.5_s1` | idem | 1 |
| 3 | `E15_nwdfix_w0.5_s2` | idem | 2 |
| 4 | `E16_nwdpure_w1.0_px_s0` | NWD puro (w = 1,0), loss em px; exploratório | 0 |

**O que mudou no código**

- `patches/nwd_patch.py`:
  - `FLYDET_NWD_LOSS_UNITS=grid|px`. O default é `grid`, que reproduz os runs antigos.
  - `FLYDET_NWD_DIAG=N` imprime `[nwd_diag]` nas N primeiras chamadas da loss.
- `train_one.sh` aborta se um `E15_*` ou `E16_*` vier sem `px` ou com outro `w`.
- `wait_and_run.sh` espera a GPU ficar livre e depois chama o `run_queue.sh`. "Livre" quer dizer nenhum `run_queue.sh` rodando e nenhum container `rq3-*-gpu0`. Os containers de outros usuários com `-gpu0` no nome são ignorados.

### Soltar

```bash
cd ~/pestline/paper_2_pestline
GPU=0 nohup bash rq3_tiny_object/dgx1/wait_and_run.sh \
  E15_nwdfix_w0.5_s0 E15_nwdfix_w0.5_s1 E15_nwdfix_w0.5_s2 E16_nwdpure_w1.0_px_s0 \
  > ~/pestline/logs/fila_e15_e16.log 2>&1 &
```

### Conferir que a correção está ativa (≈15 min depois de começar)

```bash
grep -E "^Seed:|nwd_diag" ~/pestline/logs/E15_nwdfix_w0.5_s0.log | head -6
```

Tem que aparecer:

- `loss_units=px`;
- linhas `[nwd_diag]` do treino, além da primeira, que vem do precheck com caixas falsas (`target_max_in=41.0`);
- `target_max_in` até ~240, que é a unidade de grid que chega da loss;
- `target_max_px` até 1920;
- `nwd_mean` abaixo de ~0,95 e variando entre as linhas.

Resultado em 18/out às 09:04, no `E15_nwdfix_w0.5_s0`: `target_max_in=240.0`, `target_max_px=1920.0`, `nwd_mean` 0,79–0,90 e 23,9 GB de memória.

### Parar só esta fila

```bash
pkill -f wait_and_run.sh; pkill -f "run_queue.sh E1[56]"
for c in $(docker ps --format '{{.Names}}' | grep -E '^rq3-E1[56]_'); do docker stop "$c"; done
```

### Tempo

Os E0/E10 da primeira fila levaram 1,95–2,85 h cada, com early stop. A segunda fila deve levar **~10–12 h**, e o máximo, se nenhum treino parar cedo, é ~26 h.
