# RQ3 — Transferência: métodos de tiny-object detection em YOLO26 E2E sobre armadilhas

**Pergunta.** Losses/label assignment para tiny objects validados em imagens aéreas (NWD, GCD, RFLA) e inferência
fatiada (SAHI) transferem para insetos de ~44 px em armadilhas adesivas com um detector NMS-free end-to-end
(YOLO26m, Ultralytics 8.4.142)?

**Hipótese (a testar honestamente).** Não transferem: os ganhos reportados vêm de benchmarks aéreos
(AI-TOD, DOTA, VisDrone) com detectores anchor-based/two-stage; em YOLO26 E2E com objetos ~44 px o efeito
fica dentro da variância entre seeds. Resultado negativo é publicável como tal.

## O que já temos (VAL mAP50, `results.csv` Ultralytics, protocolo idêntico, single-seed)

| Exp | Variante | mAP50 | Job / onde |
|---|---|---:|---|
| E0 | baseline CIoU | 0,8565 | `../common/slurm/E0_yolo26m_baseline.sh` |
| E10 | NWD w=0,5, C=44 | 0,8706 (+1,4) | 32933 · `slurm/E10_yolo26m_nwd.sh` |
| E11 | NWD w=0,75 | 0,8642 | 32947 · `slurm/E10x_chain.sh` |
| E12 | NWD w=1,0 (sem CIoU) | 0,6719 — **colapso** | 32947 |
| E10b | GCD w=0,5 | 0,8619 (mAP50:95 **0,6261**, melhor de todos) | 32947 · `slurm/E10b_yolo26m_gcd.sh` |
| E13 | multi_scale 1920 | **CUDA OOM** | `slurm/E13_yolo26m_multiscale.sh` |
| SAHI | inferência fatiada (TEST F1) | 66,2 % vs 73,2 % baseline — **piorou** (FP nas bordas) | `slurm/E2_eval_sahi.sh`, `baseline/fly-det/fly_det/sahi_inference.py` |
| Treino em tiles 512 + aval. foto inteira | TEST F1 | 73,2 % — igual | `slurm/E2_yolo26m_sahi.sh` |
| TTA + WBF | TEST F1 | 75,4–75,6 %, 3,7× custo — sem ganho | `eval_wbf_tta.py`, `ensemble_tta.py`, `slurm/tta_wbf_eval.sh`, `slurm/E3_ensemble_tta.sh` |

Implementação: `patches/nwd_patch.py` — `(1−w)·CIoU + w·NWD` em `BboxLoss.forward` + `TaskAlignedAssigner.iou_calculation`,
C = 44 px (média √wh dos 59.346 boxes de treino); `patches/gcd_patch.py`; `patches/rfla_patch.py`
(`FLYDET_RFLA=1`, raio `max(2·stride, lado/2)`). `train_nwd.py` aplica o patch por env (`FLYDET_NWD_MODE`, `FLYDET_NWD_W`)
e chama o trainer do `fly_det`. Testes: `../tests/test_nwd_patch.py` (7 testes CPU).

## O que falta

- [ ] **⚠️ Corrigir o bug de seed em `train_nwd.py`**: a cadeia E10y (job 33015) treinou "seeds 0/1/2" todas com
      seed 0 — o `--seed` não chega ao Ultralytics. Corrigir e cobrir com teste antes de qualquer rerun.
- [ ] **Rerun 3×3**: E0 seeds {0,1,2} vs E10 (NWD w=0,5) seeds {0,1,2}, mesma receita (150 ép., batch 8,
      imgsz 1920, patience 30) → média ± std VAL e **uma** avaliação no TEST por variante. É o caminho crítico
      (6 treinos de ~6–10 h em H100). Orquestrador base: `slurm/E10y_full_chain.sh`.
- [ ] E14 (NWD + RFLA) e E7 (yolo26l + NWD) — rodaram no 33015? Conferir W&B; se sim, exportar; se não, decidir
      se entram (pelo menos 1 seed de RFLA para citar).
- [ ] Verificar se o "+1,4 pp" do E10 sobrevive: se |Δ| < std, reportar como **sem efeito** (tese do paper).
- [ ] Análise de por quê: distribuição de IoU/NWD nos positivos do assigner, curva de recall por faixa de tamanho
      (<32 / 32–48 / >48 px) E0 vs E10 → `results/recall_by_size.csv`.
- [ ] Exportar métricas do W&B → `results/nwd_3x3.csv` (run, seed, variante, mAP50, mAP50:95, recall/precisão
      por espécie VAL/TEST).
- [ ] Registrar o protocolo do SAHI (slice, overlap, merge) e o de TTA/WBF na Seção 4.
- [ ] Figura: `fig_nwd_seeds.pdf` (boxplot mAP50 por seed, E0 vs E10) e `fig_recall_by_size.pdf`.

## Saídas esperadas em `results/`

- `nwd_sweep_single_seed.csv` (o que já existe acima), `nwd_3x3.csv`, `recall_by_size.csv`, `sahi_tta.csv`.
