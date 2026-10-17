# Prompt para o agente da outra máquina (RQ3 do paper 2 — Pestline)

> Cole o bloco abaixo como primeira mensagem do agente (Claude Code) na máquina com GPU. Responder em português.

```
Você é o agente executor da RQ3 (tiny-object) do paper 2 do Pestline (VISAPP 2027, deadline 22/out/2026; hoje é 17/out). O autor cancelou a fila na B200 (não andava) e quer rodar tudo aqui. Rode TUDO que está abaixo, em loop autônomo, e só pare quando terminar ou travar de verdade. Responda em português, conciso e técnico.

## 0. Primeiro, oriente-se (não pule)
1. `git clone`/`git pull origin main` do repo `https://github.com/CEIA-Ultralight/paper_2_pestline` (já tem tudo; não há arquivos "só locais"). Leia, nesta ordem: `CLAUDE.md`, `docs/PLANO_FINAL_2026-10-16.md` (plano e ordem da fila), `rq3_tiny_object/dgx1/FILA_RQ3_DGX1.md` + `rq3_tiny_object/dgx1/run_queue.sh` + `train_one.sh` (infra Docker/V100 já validada pelo lcpacheco), `docs/PAPER_FACTS.md` (seção "dgx1"/RQ3 v9) e `rq3_tiny_object/results/runs.md`.
2. As regras de cluster do CLAUDE.md (Slurm/B200, "nunca git push", `_scratch/`) são da B200. Aqui adapte ao ambiente real (confira com `nvidia-smi`, `docker`, quantas GPUs, espaço em disco). Mantenha o que vale em qualquer máquina: dataset 1920×1080 SEM reduzir resolução, receita idêntica (YOLO26m, 150 épocas, batch 8, imgsz 1920, patience 30, Ultralytics 8.4.142), TEST nunca usado para seleção, 73 imagens/`n_gt=3494` no TEST (o dataset exige as 83 imagens HEIC já convertidas para PNG; confira com o smoke test).
3. Dataset: DS-F2_v9.0.5. Se não estiver na máquina, PARE e avise o autor (não invente outro dataset). Chave W&B só via arquivo de segredos fora do repo (entity `pestline`, projeto `paper2_rq3`, tag da máquina); nunca em commit/log.

## 1. O que JÁ foi feito (não repita)
- Nesta infra (V100, lcpacheco): E0 e E10 (NWD w=0.5) × seeds 0,1,2, v9.0.5 → CSVs `rq3_tiny_object/results/v9_dgx1_nwd_*.csv`. Resultado: NWD ≈ baseline (acc 0.817 vs 0.817; F1 0.793 vs 0.788; MV recall +4.6 pp).
- B200: E0 s0,1,2 e E7 yolo26l s0 existem só lá (acc E0 TEST 0.822±0.016). Os pesos NÃO estão nesta máquina → compare cada variante com o E0 DESTA máquina (mesma GPU/stack), nunca misture números de máquinas diferentes na mesma média.

## 2. O que RODAR (ordem de prioridade; seeds 0,1,2; mesmas seeds do E0)
`run_queue.sh` já tem as especificações (atualizadas neste commit). Uso: `GPU=<n> nohup bash rq3_tiny_object/dgx1/run_queue.sh RUN1 RUN2 ... > logs/fila.log 2>&1 &`. Reentrante: pula o que tem `DONE`, retoma de `last.pt`. Se houver 2+ GPUs livres, uma fila por GPU (nunca ocupe GPU que outro usuário usa).
Bloco A (essencial): 
 1. `E11_gcd_w0.5_s{0,1,2}` (GCD)
 2. `E14_nwd_rfla_s{0,1,2}` (NWD+RFLA)
 3. `E1_p2_s{0,1,2}` (head P2; se OOM, `BATCH=4` em TODOS os runs P2 e registre)
 4. SAHI (tile 512, overlap 0.25, `python -m fly_det.sahi_inference`, comando em `rq3_tiny_object/slurm/chain_E_rq3_v9.sh`, passo "SAHI") e TTA (`rq3_tiny_object/eval_wbf_tta.py`, vanilla/aug) nos `best.pt` dos 3 E0 desta máquina, split TEST; depois ensemble WBF dos 3 (`--wbf-only`). Só inferência (rápido).
Bloco B: 5. `E20_wiou_s{0,1,2}` (Wise-IoU v3, patch novo em `rq3_tiny_object/patches/nwd_patch.py`, testes `tests/test_wiou_patch.py`); 6. `E21_imgsz2560_s{0,1,2}` (batch 4; se OOM em V100 32 GB, batch 2 e registre; se inviável, pule e registre); 7. `E22_yolo11m_s0` (precisa `yolo11m.pt` em MODELS_DIR; baixe de https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt).
Bloco C (se sobrar tempo): 8. `E15_nwd_loss_s{0,1,2}` e `E16_nwd_assigner_s{0,1,2}`; 9. `E7_yolo26l_s{0,1,2}`.
Antes da fila: rode os testes CPU (`pytest tests -q`) e o `smoke_E0` (1 época) para confirmar `n_gt=3494`. Tempo medido na V100: ~2,5 min/época (~4–6,5 h por treino); se a máquina for B200/A100/H100 será bem mais rápido. Priorize concluir os pares/seeds completos de cada variante do bloco A antes de passar ao B; se o tempo acabar, o que está incompleto (ex.: 1 seed) deve ser rotulado `single-seed`.

## 3. Análise e registro (cadeia de custódia — obrigatório)
- Para cada run concluído: linha em `rq3_tiny_object/results/runs.md` (run → job/log → W&B).
- Gere CSVs com `common/analysis/rq3_v9_dgx1_analysis.py` (do lcpacheco; leia o uso) e/ou `common/analysis/rq3_v9_analysis.py <RUNS_ROOT> rq3_tiny_object/results [0,1,2]` (acc por espécie, recall por faixa de tamanho <32/32–48/≥48 px, delta pareado vs E0 com IC95 bootstrap por imagem, Holm, TOST ±1 pp). Ajuste só o mapeamento de nomes de pasta se necessário; mantenha `split=TEST`, `n_seeds` e `multi-seed`/`single-seed` nas tabelas. Prefixe os arquivos com a máquina (ex.: `v9_<maquina>_...csv`).
- Acrescente linhas em `docs/PAPER_FACTS.md` (nova seção, VAL vs TEST, multi-seed vs single-seed) e `docs/LEDGER.md`. NÃO edite `manuscript/` (o autor/o outro agente integra). Resultado negativo/inconclusivo é resultado: reporte Δ por seed e IC sem reenquadrar. Métricas primárias: recall, precisão, acurácia por espécie; mAP é secundário.

## 4. Git
- Commits locais em português, um por etapa (ex.: "RQ3 v9: GCD ×3 seeds — CSVs e PAPER_FACTS"). SEM `Co-Authored-By`, SEM "Generated with", SEM qualquer menção a IA/Claude em commits, READMEs, código ou documentos (instrução do autor).
- Faça `git pull --rebase origin main` antes de cada push e `git push origin main` ao fim de cada etapa concluída (o autor autorizou push desta campanha; nunca `--force`; não mexa em `manuscript/` nem em `baseline/`).
- Não apague arquivos do autor. Temporários em `_scratch/` (git-ignored).

## 5. Ao terminar (ou travar)
Relate em português: o que rodou (runs e seeds), resultados principais (média±desvio e Δ pareado com IC95 por variante), o que falhou/foi pulado e por quê, e o que ficou pendente. Atualize o "LOG DE EXECUÇÃO" no fim de `docs/PLANO_FINAL_2026-10-16.md`.
```
