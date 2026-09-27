# runs — rq3_tiny_object

Registro job → run W&B (escrito pelo `slurm-runner`; coluna "usado em" pelo `analyst`).

| Data | Job | Script | Commit (repo / baseline) | Seed | Variante | Run W&B (pestline/<projeto>/<nome>) | Status | Usado em |
|---|---|---|---|---|---|---|---|---|
| 2026-09-06 | n/d | `common/slurm/E0_yolo26m_baseline.sh` | n/d | default (0) | E0 yolo26m baseline, imgsz 1920, 150 ép. (early stop 111) | pestline/fly-species/E0_yolo26m_baseline (y0joklcn) | finished, 0,72 h | `nwd_sweep_single_seed.csv`, `sahi_tta_single_seed.csv` (linha E0 TEST via test-campaign) |
| 2026-09-06 | n/d | `rq3_tiny_object/slurm/E2_yolo26m_sahi.sh` | n/d | default (0) | E2 tiles 512 (SAHI), 150 ép. (56 logadas) | pestline/fly-species/E2_yolo26m_sahi (ojzrkngx) | finished, 0,91 h | `nwd_sweep_single_seed.csv`, `sahi_tta_single_seed.csv` |
| 2026-09-06 | n/d | `rq3_tiny_object/slurm/E2_eval_sahi.sh` | n/d | — | avaliação legado E2 plain/sliced (protocolo antigo, não comparável) | pestline/fly-species/E2_eval_sahi (v40xeurj) | finished | `sahi_tta_single_seed.csv` (linhas `E2_eval_sahi`, protocolo legado) |
| 2026-09-07 | n/d | `rq3_tiny_object/slurm/E3_ensemble_tta.sh` | n/d | — | ensemble de 3 classificadores + TTA (avaliação, não treino) | pestline/fly-species/E3_ensemble_tta (wq75vi0t) | finished | `sahi_tta_single_seed.csv` (E3_ensemble / E3_ensemble_tta via test-campaign), `rq2_cascade/results/classifiers_single_seed.csv` |
| 2026-09-21 | 32933 | `rq3_tiny_object/slurm/E10_yolo26m_nwd.sh` | 29c37e8 / n/d | default (0) | E10 NWD w=0,5 C=44, 138 ép. | pestline/fly-species/E10_yolo26m_nwd (ro6yymqk) | finished, 1,31 h | `nwd_sweep_single_seed.csv` |
| 2026-09-22 | 32947 (E10x) | `rq3_tiny_object/slurm/E10x_chain.sh` | n/d | default (0) | E11 NWD w=0,75 | pestline/fly-species/E10_w075 (4r8wbsby) | finished, 0,87 h | `nwd_sweep_single_seed.csv` |
| 2026-09-22 | 32947 (E10x) | `rq3_tiny_object/slurm/E10x_chain.sh` | n/d | default (0) | E12 NWD w=1,0 (sem CIoU) — colapso, 33 steps | pestline/fly-species/E10_w100 (ziqlwumu) | finished, 0,26 h | `nwd_sweep_single_seed.csv` |
| 2026-09-22 | 32947 (E10x) | `rq3_tiny_object/slurm/E10x_chain.sh` (≈ `E10b_yolo26m_gcd.sh`) | n/d | default (0) | E10b GCD w=0,5 | pestline/fly-species/E10b_yolo26m_gcd (tvpy6wmz) | finished, 1,05 h | `nwd_sweep_single_seed.csv` |
| 2026-09-22 | 32947 (E10x) | `rq3_tiny_object/slurm/E10x_chain.sh` (≈ `E13_yolo26m_multiscale.sh`) | n/d | default (0) | E13 multi_scale rect=false | pestline/fly-species/E13_yolo26m_multiscale (mbp9634d) | **failed** (CUDA OOM, 127 s) — abortou a cadeia; E7 e `tta_wbf_eval.sh` não rodaram | `nwd_sweep_single_seed.csv` (linha E13) |
| 2026-09-22 | 33015 (E10y) | `rq3_tiny_object/slurm/E10y_full_chain.sh` | n/d | declarada 0 / efetiva 0 | E10 NWD seed 0 | pestline/fly-species/E10_nwd_seed0 (c9o5qs8b) | finished, 0,96 h — summary idêntico ao E10_yolo26m_nwd | `nwd_sweep_single_seed.csv` (inválido como multi-seed) |
| 2026-09-22 | 33015 (E10y) | `rq3_tiny_object/slurm/E10y_full_chain.sh` | n/d | declarada 1 / **efetiva 0 (bug)** | E10 NWD seed 1 | pestline/fly-species/E10_nwd_seed1 (mtipm02r) | finished, 0,96 h — idêntico ao seed0 | `nwd_sweep_single_seed.csv` (inválido) |
| 2026-09-22 | 33015 (E10y) | `rq3_tiny_object/slurm/E10y_full_chain.sh` | n/d | declarada 2 / **efetiva 0 (bug)** | E10 NWD seed 2 | pestline/fly-species/E10_nwd_seed2 (gt2rg88m) | finished, 0,92 h — idêntico ao seed0 | `nwd_sweep_single_seed.csv` (inválido) |
| 2026-09-22 | 33015 (E10y) | `rq3_tiny_object/slurm/E10y_full_chain.sh` | n/d | declarada 0 / efetiva 0 | E0 baseline seed 0 | pestline/fly-species/E0_baseline_seed0 (yw5svpnc) | finished, 0,74 h — idêntico ao E0_yolo26m_baseline | `nwd_sweep_single_seed.csv` (inválido) |
| 2026-09-22 | 33015 (E10y) | `rq3_tiny_object/slurm/E10y_full_chain.sh` | n/d | declarada 1 / **efetiva 0 (bug)** | E0 baseline seed 1 | pestline/fly-species/E0_baseline_seed1 (z4i201go) | **crashed** step 85 (0,58 h) — abortou a cadeia | `nwd_sweep_single_seed.csv` |
| 2026-09-22 | 33015 (E10y) | `rq3_tiny_object/slurm/E10y_full_chain.sh` | n/d | 2 | E0 baseline seed 2 | — (sem run) | **não rodou** | `nwd_sweep_single_seed.csv` (linha "não rodou") |
| 2026-09-22 | 32947 / 33015 | `E10x_chain.sh` / `E10y_full_chain.sh` | n/d | — | E7 yolo26l + NWD | — (sem run) | **não rodou** | `nwd_sweep_single_seed.csv` (linha "não rodou") |
| 2026-09-22 | 33015 | `E10y_full_chain.sh` | n/d | — | E14 NWD + RFLA | — (sem run) | **não rodou** | `nwd_sweep_single_seed.csv` (linha "não rodou") |
| 2026-09-22 | 32947 / 33015 | `rq3_tiny_object/slurm/tta_wbf_eval.sh` | n/d | — | TTA + WBF dos detectores (VAL) | — (sem run) | **não rodou** | `sahi_tta_single_seed.csv`, `nwd_sweep_single_seed.csv` (linhas "não rodou") |

Notas (analyst, 2026-09-26): jobs de 2026-09-06/07 anteriores ao registro Slurm → "n/d"; commits não registrados no W&B. Seeds "efetiva 0" inferidas por summary byte-idêntico (mAP50 0,854434 etc.), ver `docs/PAPER_FACTS.md` §1. Durações = `_runtime` do W&B.
