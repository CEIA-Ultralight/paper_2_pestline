# HANDOFF RQ3 — cadeia H (variantes NWD/GCD/RFLA com a loss em pixels)

Escrito em 2026-10-27 para o agente que continuar em outra máquina. **Prazo do paper: 29/10/2026** (limite 50k caracteres; 12 pp.).
Responder ao autor em português. Regras de cluster/Git: `CLAUDE.md` e `AGENTS.md` da raiz (tudo via Slurm; nunca `git push`; commits sem menção a IA e sem `Co-Authored-By`).

## 1. Por que esta cadeia existe

O Ultralytics 8.4.142 passa as caixas ao assigner em **pixels**, mas à `BboxLoss.forward` em **unidades de grid** (`target/stride`).
O patch NWD usava C=44 nos dois pontos, então na loss C=44 virava 44 células (352/704/1408 px em P3/P4/P5) e o NWD saturava (~0,99). Detalhes: `docs/PAPER_FACTS.md` §12.1.

Resultado: precisam ser refeitos (o paper tem `TODO` para eles em `manuscript/sec/5_experiments.tex`, seção RQ3):

| Afetado | Onde está no paper | O que falta |
|---|---|---|
| NWD só na loss; NWD + RFLA (campanha a, L40S) | tabela (a), linha TODO | H2, H3 |
| Campanha (c) inteira (H100, cadeia G: C22, C88, w=1,0) | tabela (c), TODO | H4, H5, H6 (+ H1 como referência) |
| GCD da campanha (a) (clamp de 1 célula em grid) | linha GCD da tabela (a) | H7 |

**Válidos e não afetados** (não refazer): E0 (qualquer campanha), NWD só no assigner, WIoU v3, P2, imgsz 2560, YOLO11m, e a campanha (b) V100 (E15 px −0,4 pp [−2,1; +1,4]; E16 NWD puro −13,3 pp). O **G0** (E0 na h100n2, 3 seeds, sem patch) é o controle da cadeia H: 82,2 ± 1,9 (84,4 / 80,7 / 81,6).

## 2. Estado em 2026-10-27

Commits locais já feitos (ver `git log`): blindagem da unidade (`9a56106`) e geradores de análise. **Nenhum treino da cadeia H rodou**: o `sbatch` do slogin-01 falhou desde 26/10 ~15h com `I/O error writing script/environment to file` (disco `/` do login node em 99%, `/var/spool/slurmctld`). Se o Slurm da máquina nova funcionar, comece direto do §4.

O que já está pronto e testado em CPU:
- `rq3_tiny_object/patches/nwd_patch.py`: sem default silencioso de `FLYDET_NWD_LOSS_UNITS` (scope loss/both aborta se não definido); `FLYDET_REQUIRE_PX=1` aborta se a loss não estiver em px; diagnóstico gravado em `nwd_diag.log` no WORK.
- `common/slurm/train_detector_b200.sh`: precheck numérico da loss em px nos strides 8/16/32 e **gate antes do `DONE`** (config/diag em px; TEST = 73 imagens e 1210 instâncias de espécie). Testado: aceita o G0 e rejeita G1/G3 (cadeia G).
- `rq3_tiny_object/slurm/chain_H_rq3_px_h100.sh`: braços H1–H7 por `H_ARMS`, seeds por `H_SEEDS`.
- `common/analysis/rq3_h_px_analysis.py` (cadeia H vs G0), `common/analysis/rq3_ovx_holm_valid.py` (Holm de (a)).
- `tests/test_nwd_patch.py`: 17 passam (fail-fast sem unidade, REQUIRE_PX rejeita grid, GCD em px).
- `docs/PAPER_FACTS.md` §15 com o diagnóstico e a tabela de Holm recalculada.

## 3. Runs a executar (21 no total; yolo26m, receita E0: 150 ép., batch 8, imgsz 1920, patience 30; seeds 0/1/2)

| Braço | Config (`FLYDET_*`) | Nome do run | Prioridade |
|---|---|---|---|
| H2 | NWD w0,5 C44 scope=loss | `H2_nwdloss_px_w0.5_C44_s{0,1,2}` | 1 |
| H3 | NWD w0,5 C44 both + `FLYDET_RFLA=1` | `H3_nwd_rfla_px_w0.5_C44_s*` | 1 |
| H1 | NWD w0,5 C44 both | `H1_nwd_px_w0.5_C44_s*` | 2 |
| H4 | NWD w0,5 C22 both | `H4_nwd_px_w0.5_C22_s*` | 2 |
| H5 | NWD w0,5 C88 both | `H5_nwd_px_w0.5_C88_s*` | 2 |
| H6 | NWD w1,0 C44 both (puro) | `H6_nwd_px_w1.0_C44_s*` | 3 |
| H7 | GCD w0,5 both | `H7_gcd_px_w0.5_s*` | 4 (opcional) |

Custo medido no G: 0,6–0,9 h de treino por run + ~5 min de avaliação ≈ 19 GPU-h no total. Divisão sugerida em 2 cadeias (limite de 2 jobs RUNNING por usuário):
- H-a: `H_ARMS="H2 H1 H5 H7"` (12 runs)
- H-b: `H_ARMS="H3 H4 H6"` (9 runs)

Se faltar tempo, corte H7 e H6 (e diga no texto que não foram refeitos). **Regras pré-registradas:** sem retry nem mudança de receita em divergência/colapso (`nan`, early stop precoce); o `best.pt` é o anterior à divergência e o run é reportado como negativo/instável, com Δ por seed. C=44 fica fixo (o mesmo da campanha b) para mudar uma variável por vez.

## 4. Como rodar (obrigatório, regras do cluster)

Dados e container devem existir na máquina com GPU (partição = máquina dos dados; aqui `h100n2`, `/raid/user_marcospaulo`):
- `DATA_ROOT=/raid/user_marcospaulo/datasets/DS-F2_v9.0.5` e **`DS-F2_v9.0.5_evalpng`** (73 imagens em `images/test`; a cadeia aborta se não forem 73 — sem isso o harness avalia 64 por ignorar `.heic`).
- `IMAGE=/raid/user_marcospaulo/containers/flydet-train.sif` (ultralytics 8.4.142; o precheck exige exatamente essa versão).
- Controle: `/raid/user_marcospaulo/flydet_runs_v9/G0_e0_h100_s{0,1,2}` com `DONE` e `test_eval_png/`.
- Saída em `RUNS_ROOT=/raid/user_marcospaulo/flydet_runs_v9`. W&B: entity `pestline`, projeto `paper2-ds-f2-v9`, credencial só via `set -a; source /raid/user_marcospaulo/secrets/wandb.env; set +a`.

Passos:
1. `sinfo -p h100n2`, `squeue -u $USER`, `nvidia-smi`, `df -h /raid`. Máx. 2 jobs RUNNING.
2. **Smoke** (2 épocas, não suja o `DONE` real): `H_ARMS="H2 H3 H7" H_SEEDS=0 EPOCHS=2 RUNS_ROOT=/raid/user_marcospaulo/flydet_runs_v9_smoke WANDB_MODE=offline WANDB_RUN_GROUP=smoke sbatch --hold --job-name=p2-smokeH --time=02:00:00 rq3_tiny_object/slurm/chain_H_rq3_px_h100.sh`.
   Critério: log com `PRECHECK_OK`, `[precheck] loss NWD/GCD em pixels: ok (strides 8/16/32)`, `[nwd_diag] ... loss_units=px` e `GATE_OK`, sem crash. Se `sbatch` exportar mal o ambiente, use `--export=ALL,...`.
3. **Cadeias reais**, uma por vez, cada uma com `sbatch --hold` + `H_ARMS=...`, depois: `nohup python common/gpu/gpu_watch.py --jobs <id> &` no mesmo nó, **Monitor** no log (cobrir `GATE_OK`, `DONE`, `CHAIN_H_DONE`, estados terminais do Slurm, `Traceback|Error|OOM|CUDA out of memory|ABORT|nan|loss_units=grid`, log parado), e só então `scontrol release <id>`. `loss_units=grid` no log = `scancel` imediato.
4. Cada run concluído vira linha em `rq3_tiny_object/results/runs.md` (job Slurm → run W&B → commit) e no `docs/LEDGER.md`. Em OOM: `BATCH=4` e anotar. Em requeue (rc 75), ressubmeter o mesmo comando (a cadeia é idempotente).
5. O job do GPU jail/audit é `common/gpu/slurm/gpu_env.sh` (já sourced pela cadeia). Não rode nada com GPU fora do Slurm, nem testes.

Testes de CPU antes de submeter (no ambiente atual não existe `_scratch/venv_tests`): `apptainer exec --cleanenv <sif> /opt/venv/bin/python -m pytest tests/test_nwd_patch.py -q` (com pytest instalado via `--target`). Referência: 17 passed em `test_nwd_patch.py`; o conjunto completo tem 13 falhas de ambiente preexistentes (`test_eval_test_campaign`, `test_eval_cascade_matrix`, `test_policy_study`) sem relação com o patch.

## 5. Análise (CPU) quando as runs terminarem

1. `python3 common/analysis/rq3_h_px_analysis.py /raid/user_marcospaulo/flydet_runs_v9` → `rq3_tiny_object/results/v9_h_px_{runs,summary,paired,size_bins}.csv`. O gerador recusa runs sem `DONE`, com seed efetiva ≠ declarada, loss fora de px ou TEST ≠ 1210; Holm entre os braços H.
2. Checar: seeds com métricas distintas; best epoch registrada (fitness = mAP50-95 do VAL); nada de TEST para escolher modelo.
3. `docs/PAPER_FACTS.md` §15: preencher "Resultados: pendente" com tabela (marcar TEST, multi-seed, H100) e atualizar a nota do §14 (cadeia G inválida para a loss; só o G0 vale). Registrar no `docs/LEDGER.md`.
4. A Holm de (a) com comparações válidas já foi calculada (`rq3_tiny_object/results/v9_ovx_paired_valid.csv`): imgsz 2560 Holm 0,041 (m=5), P2 0,111; WIoU, NWD-assigner e YOLO11m 1,00. Excluídos: E14, E15_nwd_loss, E11_gcd.

## 6. Manuscrito (só com números no PAPER_FACTS; sem alterar a tese sem decisão do autor)

`manuscript/` é um submodule ligado ao Overleaf: commitar dentro de `manuscript/` e depois `git add manuscript` na raiz; **nunca `git push`**. Pontos a atualizar em `manuscript/sec/5_experiments.tex` (RQ3), `4_method.tex` (§4.3), `1_intro.tex` e `6_conclusion.tex` quando houver resultados:
- Tabela (a): trocar os `TODO` de p ajustado pelos valores de `v9_ovx_paired_valid.csv` (m=5); substituir a linha TODO de loss-side NWD / NWD+RFLA pelas linhas H2 e H3 (campanha H, H100), com nota apontando para (c).
- Tabela (b): apagar o TODO "rerun the earlier NWD assigner-plus-loss variant"; esse rerun já é a linha "pixel-space".
- Tabela (c): reescrever com G0 e H1–H6 (H7 se rodou), contra o G0. Remover o TODO e os trechos "remain TODO".
- §4.3: descrever o RFLA como "RFLA-inspired circular candidate pool" (raio max(2·16, lado/2) px com `stride_val`=16 para todos os níveis; sem campo receptivo gaussiano/KLD) e dizer que o NWD puro passou a ter 3 seeds; ajustar "All use three seeds except YOLO11m".
- Parágrafo final da RQ3 e conclusão: só mudam com os números novos. Tom: resultado nulo/negativo é resultado; reportar Δ por seed e IC; "no difference detected" não é equivalência.
- Compilar: `cd manuscript && TECTONIC_CACHE_DIR=../_scratch/tectonic_cache tectonic -X compile main.tex --outdir _build` (12 páginas incluindo refs).

## 7. Armadilhas conhecidas

- Nunca omitir `FLYDET_NWD_LOSS_UNITS`; a cadeia H já exporta `px` + `FLYDET_REQUIRE_PX=1`.
- `REEVAL_RUNS` é esvaziado na cadeia H (o default reavalia G0_s0/G0_s1 e gasta GPU).
- Não misturar hardware numa mesma tabela: cada campanha só se compara com o seu E0.
- `RUNS_ROOT` do smoke é separado (`flydet_runs_v9_smoke`); não deixar marcadores `DONE` de smoke no real.
- Bug antigo de seed (cadeia 33015): conferir `args.yaml` (`seed` efetivo) e que as métricas diferem entre seeds.
