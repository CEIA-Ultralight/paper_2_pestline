# RQ3 — Escala: métodos tiny-object da literatura aérea em YOLO26 end-to-end

**Pergunta.** NWD, RFLA, GCD, Wise-IoU, cabeça P2, entrada em 2560 px, YOLO11m e inferência fatiada (SAHI/TTA/WBF) transferem das imagens aéreas para insetos de ≈ 44 px em armadilhas adesivas, com um detector NMS-free end-to-end?

## Resultado (TEST de 73 imagens, 3 seeds, controle E0 da mesma campanha)

Cada campanha roda em um hardware e tem o seu E0; os números nunca são misturados entre campanhas.

| Campanha | GPU | Conteúdo | CSVs | Fatos |
|---|---|---|---|---|
| (a) ovx | L40S | E0 + GCD, NWD+RFLA, P2, WIoU, 2560 px, NWD (assigner / assigner+loss), YOLO11m (1 seed), SAHI/TTA/WBF | `v9_ovx_*.csv` | `docs/PAPER_FACTS.md` §13 |
| (b) dgx1 | V100 | E0 × NWD com a loss em **pixels** (E15), NWD puro (E16, 1 seed) | `v9_dgx1_nwd_*.csv` | §12, §12.1, §12.2 |
| (c) cadeia G | H100 | NWD ajustado (w, C): G0 controle, G1 w0,5/C22, G2 w1,0/C44, G3 w0,5/C88 (2 seeds) † | `v9_g_tuned_*.csv` | §14 |

† Na campanha (a) e na cadeia G a loss do NWD usa unidades de grid (bug de unidade do Ultralytics, `docs/PAPER_FACTS.md` §12.1), o que satura o NWD na loss; só a campanha (b) usa a correção (`FLYDET_NWD_LOSS_UNITS=px`).

- Nenhuma variante melhora a acurácia de espécie do controle (diferenças de −3,8 a +0,3 pp); NWD+RFLA é −8,8 pp por seeds instáveis e NWD puro não converge.
- Nenhuma variante aumenta o recall abaixo de 32 px. SAHI, TTA e ensemble não superam o detector em resolução nativa.
- O ajuste foi feito só no NWD, então o resultado é "sem ganho detectado aqui", não prova de ausência de ganho.

## Implementação

`patches/nwd_patch.py` (NWD no assigner e na loss, `(1−w)·CIoU + w·NWD`, C em px), `patches/gcd_patch.py`, `patches/rfla_patch.py`; `train_nwd.py` aplica os patches por variáveis de ambiente (`FLYDET_NWD_MODE`, `FLYDET_NWD_W`, `FLYDET_NWD_LOSS_UNITS`, `FLYDET_RFLA`, `FLYDET_SEED`) e chama o trainer do `fly_det`. `ensemble_tta.py` e `eval_wbf_tta.py` avaliam TTA/WBF.

| Etapa | Arquivo |
|---|---|
| Campanha (a) L40S | `ovx/ovx_queue.sbatch`, `ovx/ovx_infer.sbatch`, `slurm/chain_E_rq3_v9.sh`, `slurm/chain_F_rq3_extras_v9.sh` |
| Campanha (b) V100 | `dgx1/` (Dockerfile, fila, `FILA_RQ3_DGX1.md`) |
| Cadeia G (H100) | `slurm/chain_G_rq3_tuned_h100.sh`, `slurm/eval_G_png.sh` |
| Cadeias B200 e variantes antigas | `slurm/chain_A_detectors.sh`, `chain_C_extra.sh`, `E*.sh` (histórico) |
| Análise | `common/analysis/rq3_v9_analysis.py`, `rq3_v9_dgx1_analysis.py`, `rq3_g_tuned_analysis.py`, `rq3_ovx_infer_csv.py` |
| Testes (CPU) | `tests/test_nwd_patch.py`, `test_wiou_patch.py`, `test_train_nwd_seed.py`, `test_eval_wbf_tta.py` |

## `results/`

Os arquivos `v9_*` são os vigentes. Os demais (`nwd_sweep_single_seed.csv`, `nwd_3x3.csv`, `sahi_*`, `tiny_*`, `recall_by_size.csv`, `tta_vanilla_multiseed_PENDENTE_validacao.csv`) vêm das campanhas preliminares em DS-F2_v8.1.1 (H100/B200) e são mantidos como histórico; **não entram no artigo**. `runs.md` liga job Slurm → run W&B (`paper2-rq3-tiny-object`).

## Limitações declaradas

Hiperparâmetros padrão (só o NWD foi ajustado: três configurações), três seeds, um conjunto de teste de 73 imagens com IC de ≈ ±2 pp; efeitos menores que isso não são resolvidos. YOLO11m e NWD puro têm uma seed. O mAP@50:95 não foi reportado para todas as variantes.
