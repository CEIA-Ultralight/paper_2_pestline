# Plano RQ3 — execução no ovx-l40s-01 (2026-10-17)

Autorizado pelo autor em 2026-10-17 ("ir rodando tudo e colocando monitor"). Fonte da fila: `docs/PROMPT_AGENTE_OUTRA_MAQUINA.md`.
Infra: `rq3_tiny_object/ovx/ovx_queue.sbatch` (Slurm `ovx01`, L40S 46 GB, container `~/pestline_rq3/containers/pestline-rq3.sif`,
torch 2.6.0+cu124, ultralytics 8.4.142). Saídas em `~/pestline_rq3/{runs,logs}`; W&B `pestline/paper2_rq3`.
Receita idêntica ao E0 (yolo26m, 150 ép., batch 8, imgsz 1920, patience 30). Dataset DS-F2_v9.0.5, TEST = 73 imgs / n_gt=3494.

## Regras desta execução
- Comparar cada variante só com o E0 **desta máquina** (E0 s0–s2 são rodados aqui; pesos da B200 não existem aqui).
- Máx. 2 filas simultâneas minhas; um Monitor por job (fim + falha + log parado).
- Requeue (rc 75): ressubmeter o mesmo comando. OOM: `BATCH` menor em TODAS as seeds da variante e registrar.
- Sem `git push`; commits locais em português, sem menção a IA.

## Ordem
0. Smoke `smoke_E0` (1 época): confirma `n_gt=3494` e pipeline.
1. Fila A (bloco A): `E0_s{0,1,2}` → `E11_gcd_w0.5_s{0,1,2}` → `E14_nwd_rfla_s{0,1,2}` → `E1_p2_s{0,1,2}`.
2. Fila B (bloco B/C): `E20_wiou_s{0,1,2}` → `E21_imgsz2560_s{0,1,2}` (batch 4) → `E22_yolo11m_s0` → `E15_nwd_loss_s{0,1,2}` → `E16_nwd_assigner_s{0,1,2}`.
   - E10 NWD (v9) vem da dgx1 (V100): não é comparável com E0 daqui; se sobrar GPU, rodar `E10_nwd_w0.5_s{0,1,2}` aqui.
3. Inferência (após E0 s0–s2): SAHI (tile 512, overlap 0,25), TTA vanilla/aug e WBF dos 3 E0, split TEST
   (`rq3_tiny_object/slurm/chain_E_rq3_v9.sh` passo SAHI; `rq3_tiny_object/eval_wbf_tta.py`).
4. Análise: `common/analysis/rq3_v9_analysis.py <RUNS_ROOT> rq3_tiny_object/results [0,1,2]` (prefixo `v9_ovx_`), recall por faixa de
   tamanho, Δ pareado vs E0 com IC95 bootstrap, Holm, TOST ±1 pp. Tudo `TEST`, `multi-seed`/`single-seed` explícitos.
5. Registro: `rq3_tiny_object/results/runs.md`, `docs/PAPER_FACTS.md` (nova seção), `docs/LEDGER.md`, `rq3_tiny_object/README.md`,
   LOG em `docs/PLANO_FINAL_2026-10-16.md`. `manuscript/` só após decisão do autor.

## Critério de término (meta)
RQ3 encerrada = runs do bloco A (e o que couber do B/C) concluídos, SAHI/TTA/WBF rodados, CSVs em `rq3_tiny_object/results/`,
PAPER_FACTS/LEDGER/runs.md/README da RQ3 escritos. O que não couber antes de 22/out fica rotulado `single-seed` ou "não rodou".
