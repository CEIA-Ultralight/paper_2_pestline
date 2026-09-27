# runs — rq2_cascade

Registro job → run W&B (escrito pelo `slurm-runner`; coluna "usado em" pelo `analyst`).

| Data | Job | Script | Commit (repo / baseline) | Seed | Variante | Run W&B (pestline/<projeto>/<nome>) | Status | Usado em |
|---|---|---|---|---|---|---|---|---|
| 2026-09-06 | n/d | `rq2_cascade/slurm/E1_yolo26m_p2.sh` | n/d | default (0) | E1 yolo26m + cabeça P2, 200 ép. | pestline/fly-species/E1_yolo26m_p2 (shp7xlkc) | finished, 2,02 h | `rq3_tiny_object/results/nwd_sweep_single_seed.csv`, `sahi_tta_single_seed.csv` (linha E1 TEST) |
| 2026-09-07 | n/d | `rq2_cascade/slurm/E3_swin_classifier.sh` | n/d | n/d | Swin-T 224, 40 ép. | pestline/fly-species/E3_swin (ettsirmw) | finished, 0,89 h | `classifiers_single_seed.csv` |
| 2026-09-07 | n/d | `rq2_cascade/slurm/E3b_convnext.sh` | n/d | n/d | ConvNeXt-T 384 CE, 50 ép. | pestline/fly-species/E3b_convnext (l2ys9lsg) | finished, 5,74 h | `classifiers_single_seed.csv` |
| 2026-09-07 | n/d | `rq2_cascade/slurm/E3c_focal.sh` | n/d | n/d | ConvNeXt-T 384 focal + class weights (31 ép. de 60) | pestline/fly-species/E3c_focal (453bepak) | finished, 3,55 h | `classifiers_single_seed.csv` |
| 2026-09-07 | n/d | `rq2_cascade/slurm/E3d_convnext_s.sh` | n/d | n/d | ConvNeXt-S 384 focal+cw (18 ép. de 60) | pestline/fly-species/E3d_convnext_s (iegv21jl) | finished, 2,84 h | `classifiers_single_seed.csv` |
| 2026-09-07 | n/d | `rq2_cascade/slurm/E4_convnext_sam.sh` (+ `E4_sam_clean.sh`) | n/d | n/d | ConvNeXt-T 384 em crops limpos por SAM (49 ép.) | pestline/fly-species/E4_convnext_sam (wrjfm0w0) | finished, 5,61 h | `classifiers_single_seed.csv` |
| 2026-09-07 | n/d | `rq2_cascade/slurm/E5_convnext_512.sh` | n/d | n/d | ConvNeXt-T 512 focal+cw (18 ép.) | pestline/fly-species/E5_convnext_512 (skfyw3du) | finished, 3,80 h | `classifiers_single_seed.csv` |
| 2026-09-08 | 32004 | `rq2_cascade/slurm/architecture_tracking_probe.sh` | n/d | — | smoke test de tracking W&B | pestline/fly-species/SMOKE_architecture_tracking_32004 (6fd6393a) | finished | — |
| 2026-09-08 | 32005 | `rq2_cascade/slurm/architecture_night.sh` | n/d | 42 | A1 baseline ConvNeXt-T 384 CE (18 ép., best 16) | pestline/fly-species/A1_baseline_architecture-night-v1-20260907 (2361b96d) | finished, 1,27 h | `classifiers_single_seed.csv`, `cascade_test_single_seed.csv` (controle CE), `cascade_val_single_seed.csv` |
| 2026-09-08 | 32005 | `rq2_cascade/slurm/architecture_night.sh` | n/d | 42 | A1 SupCon (6 ép., best 1) | pestline/fly-species/A1_supcon_architecture-night-v1-20260907 (8a5a552d) | finished, 0,67 h | `classifiers_single_seed.csv` |
| 2026-09-08 | 32005 | `rq2_cascade/slurm/architecture_night.sh` | n/d | 42 | A1 ArcFace (18 ép., best 15) | pestline/fly-species/A1_arcface_architecture-night-v1-20260907 (40be9d77) | finished, 1,27 h | `classifiers_single_seed.csv`, `cascade_val_single_seed.csv` |
| 2026-09-08 | 32005 | `rq2_cascade/slurm/architecture_night.sh` | n/d | 42 | **A1 atenção a partes (PRINCIPAL)** (22 ép., best 17) | pestline/fly-species/A1_parts_architecture-night-v1-20260907 (55af7b92) | finished, 1,55 h | `cascade_test_single_seed.csv` (linha principal), `cascade_val_single_seed.csv`, `classifiers_single_seed.csv` |
| 2026-09-08 | 32005 | `rq2_cascade/slurm/architecture_night.sh` | n/d | 42 | A1 hires stem/2 (10 ép., best 9) | pestline/fly-species/A1_hires_architecture-night-v1-20260907 (b7ddf7f8) | finished, 1,26 h | `classifiers_single_seed.csv` |
| 2026-09-08 | 32005 | `rq2_cascade/slurm/architecture_night.sh` | n/d | 42 | A1 RoI head sobre YOLO E0 congelado (9 ép., best 4) | pestline/fly-species/A1_roi_architecture-night-v1-20260907 (b357fdffeb4d) | finished | `classifiers_single_seed.csv` (só GT-crop; sem cascata) |
| 2026-09-09 | 32009 | `rq2_cascade/slurm/followup.sh` | n/d | 42 | A2 partes + SupCon | pestline/fly-species/A2_parts_supcon_seed42_architecture-followup-v1-20260908 (10d71da6) | finished, 1,02 h | `classifiers_single_seed.csv` |
| 2026-09-09 | 32009 | `rq2_cascade/slurm/followup.sh` | n/d | 42 | A2 hires + partes | pestline/fly-species/A2_hires_parts_seed42_architecture-followup-v1-20260908 (f9ae47a1) | finished, 1,01 h | `classifiers_single_seed.csv` |
| 2026-09-09 | 32009 | `rq2_cascade/slurm/followup.sh` | n/d | 42 | A2 hires + SupCon | pestline/fly-species/A2_hires_supcon_seed42_architecture-followup-v1-20260908 (e12bc766) | finished, 1,01 h | `classifiers_single_seed.csv` |
| 2026-09-09 | 32009 | `rq2_cascade/slurm/followup.sh` | n/d | **84** | A2 baseline CE seed 84 (12 ép., best 7) | pestline/fly-species/A2_baseline_seed84_architecture-followup-v1-20260908 (4f49c176) | finished, 0,79 h | `cascade_test_single_seed.csv` (seed 84), `classifiers_single_seed.csv` |
| 2026-09-09 | 32009 | `rq2_cascade/slurm/followup.sh` | n/d | **84** | A2 partes seed 84 (8 ép., best 3) | pestline/fly-species/A2_parts_seed84_architecture-followup-v1-20260908 (9ccb38a3) | finished, 0,53 h | `cascade_test_single_seed.csv` (seed 84), `classifiers_single_seed.csv` |
| 2026-09-09 | 32088 | `rq2_cascade/slurm/eval_yolo_cascade.sh` | n/d | det. default / clf. 42 | VAL: E0 vs E0+{CE, partes, ArcFace, …} × {label_only, product}, 617 GT | pestline/fly-species/EVAL_E0_cascade_val_20260909T134157Z (114dbfba) | finished | `cascade_val_single_seed.csv` (PAPER_FACTS §2) |
| 2026-09-09 | n/d | `common/slurm/eval_test_campaign.sh` | n/d | det. default / clf. 42 e 84 | TEST (55 imgs, 918 GT): E0/E1/E2/E2_sliced + cascatas A1/A2/E3–E5 + ensembles | pestline/fly-species/test-campaign-036ef018 (036ef018cfcf494d8e15eb49e95acc1e) | finished, 0,30 h | `cascade_test_single_seed.csv`, `classifiers_single_seed.csv`, `rq3_tiny_object/results/sahi_tta_single_seed.csv` (PAPER_FACTS §3, §4) |

Notas (analyst, 2026-09-26): jobs 32005/32009 inferidos dos diagnósticos de latência dos runs A1/A2 (`slurm_job_id`); demais "n/d" não estão no W&B nem em `docs/`. Seeds E3–E5 não registradas. Todos os runs são **single-seed**; 42 vs 84 são dois runs independentes, não média.

| Data | Job | Script | Commit (repo / baseline) | Seed | Variante | Run W&B (pestline/<projeto>/<nome>) | Status | Usado em |
|---|---|---|---|---|---|---|---|---|
| 2026-09-26 | 33320 | `common/slurm/smoke_b200.sh` → `rq2_cascade/slurm/train_classifier_b200.sh` | 9297f13+infra | 42 | smoke: ConvNeXt-T CE 1 ép. (256/128 crops, sem W&B) + cascata VAL 5 imgs | — | SMOKE_OK | — (não usar) |
| 2026-09-26 | 33322 (cadeia B) | `rq2_cascade/slurm/chain_B_classifiers.sh` → `train_classifier_b200.sh` | 9297f13+infra / b36580a | 42 | ConvNeXt-T CE (baseline), 24 ép., patience 5, 384 px, VAL | pestline/paper2-rq2-cascade/clf_convnext_t_ce_s42 | submetido/rodando | |
| 2026-09-26 | 33322 (cadeia B) | idem | idem | 84 | ConvNeXt-T CE | pestline/paper2-rq2-cascade/clf_convnext_t_ce_s84 | submetido (fila da cadeia) | |
| 2026-09-26 | 33322 (cadeia B) | idem | idem | 126 | ConvNeXt-T CE | pestline/paper2-rq2-cascade/clf_convnext_t_ce_s126 | submetido (fila da cadeia) | |
| 2026-09-26 | 33322 (cadeia B) | idem | idem | 42 | ConvNeXt-T partes | pestline/paper2-rq2-cascade/clf_convnext_t_parts_s42 | submetido (fila da cadeia) | |
| 2026-09-26 | 33322 (cadeia B) | idem | idem | 84 | ConvNeXt-T partes | pestline/paper2-rq2-cascade/clf_convnext_t_parts_s84 | submetido (fila da cadeia) | |
| 2026-09-26 | 33322 (cadeia B) | idem | idem | 126 | ConvNeXt-T partes | pestline/paper2-rq2-cascade/clf_convnext_t_parts_s126 | submetido (fila da cadeia) | |

Notas (slurm-runner, 2026-09-26): partição **b200n1**, QOS `onejob` (máx. 2 jobs). Log `/raid/user_marcospaulo/slurm_logs/p2-chainB-classifiers_33322.out`; saídas `/raid/user_marcospaulo/flydet_runs/clf_convnext_t_*/{best.pt,last.pt,summary.json}`. `architecture_lab` grava `status` ∈ {completed, early_stop, budget, interrupted, failed}; a cadeia considera concluído `exit_code=0` + `export_complete=true` (o check antigo por `"succeeded"` nunca casava — corrigido). Se `status=budget` (MAX_HOURS=1,25 h) aparecer, o analyst deve tratar como treino truncado. Cadeia D (`chain_D_cascade_cross.sh`) depende de `CHAIN_A_DONE` + `CHAIN_B_DONE`.
