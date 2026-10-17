# Plano final de experimentos restantes (2026-10-16) — autorizado pelo autor: "rodar tudo" (A+B+C)

Deadline do paper: 22/out/2026. Dataset: DS-F2_v9.0.5, B200 (`RUNS_ROOT=/raid/user_marcospaulo/flydet_runs_v9`).
Regras: 1 job RUNNING por vez; tudo via Slurm; sem push; commits locais em português SEM Co-Authored-By/menção a IA;
cadeia de custódia (CSV em `*/results/` + linha em PAPER_FACTS + runs.md) antes de qualquer número no `.tex`.

## Diagnóstico
- A campanha RQ3 antiga (E10, GCD, P2, RFLA, SAHI, TTA) foi na **v8.1.1**. Na v9.0.5 existem só E0 s0/1/2 e E7 yolo26l s0.
- Amigos (lcpacheco, V100, W&B `pestline/paper2_rq3`, 14 runs): só E0 vs NWD w=0,5, 3 seeds, v9.0.5. Resultado: acc 0,817±0,009 vs 0,817±0,006;
  F1 0,788±0,026 vs 0,793±0,008; recall MV +4,6 pp (0,868→0,914). Tratar como **replicação independente** (V100), não misturar com B200. Split do eval não confirmado.
- 1 treino YOLO26m@1920 ≈ 0,6 h na B200 (3 seeds ≈ 1,8 h).

## Bloco A — essencial (~13 GPU-h)
- A1 RQ3 treino v9, 3 seeds: NWD E10 (w0.5,C=44,scope=both), GCD E11, RFLA E14 (NWD+RFLA), P2 E1 (yolo26m-p2; batch 4 se OOM).
- A2 Inferência sobre os 3 E0: SAHI (tile 512, overlap 0.25), TTA vanilla/aug, ensemble WBF (corrigir IndexError em `eval_wbf_tta.py` antes).
- A3 Estatística CPU (RQ1/2/3): bootstrap pareado por imagem IC95%, Holm, TOST margem ±1 pp F1. Usar `common/stats.py`.
- A4 RQ2 ruído de rótulo: confident learning out-of-fold nos crops GT (~1–2 GPU-h); re-anotação cega ~150 crops MD/MV/MC + κ (depende do autor).
- A5 RQ3 análise: recall por faixa de tamanho (<32/32–48/>48 px), CSVs com split e seeds.

## Bloco B — recomendado (~7 GPU-h)
- B1 resolução 2560 (batch 4), 3 seeds (autor autorizou: "rodar tudo").
- B2 uma perda IoU moderna (Wise-IoU v3), 3 seeds.
- B3 RQ2 detector de outra família (YOLO11m ou YOLOv12m), 1 seed.
- B4 RQ1 curva de aprendizado (lei de potência) em L0..L3, CPU.
- B5 RQ1 controle aleatório×dirigido com mesmo nº de imagens (L2 de tamanho igual), 3 seeds.

## Bloco C — opcional (autorizado)
- C1 scope NWD só loss / só assigner, 3 seeds (~3,6 h).
- C2 E7 yolo26l seeds 1 e 2 (~2 h).
- C3 SimD, D-FINE/DEIM, super-resolução, TransFG: só citar no Related Work. E13 multiscale@1920: cortado (OOM; B1 cobre).

## Ordem na fila (1 slot)
NWD → GCD → RFLA → P2 → SAHI/TTA/WBF → B2 → B1 → B3 → B5 → C1 → C2. CPU em paralelo: A3, A5, B4, gerador de CSVs (`common/analysis/`).

## Cronograma
16: fix WBF + `chain_E_rq3_v9.sh` + teste CPU + sbatch. 17–18: GPU; eu faço A3/A5/B4. 18–19: A4 (autor re-anota). 19: CSVs, PAPER_FACTS §12, runs.md, DECISIONS, LEDGER.
20–21: manuscrito (tabela multi-seed RQ3, 3 `\TODO` do abstract/intro/conclusão, Related Work). 22: revisão e envio.

## Literatura (scholar; só abstracts lidos; arXiv conferido pela API)
- Estatística: Bouthillier+ MLSys21 arXiv:2103.03098; Picard arXiv:2109.08203; Agarwal+ NeurIPS21 arXiv:2108.13264; Dietterich 1998 (NÃO verificado).
- Curva de aprendizado: Viering&Loog arXiv:2103.10948; Mahmood+ arXiv:2207.01725; Sun+ ICCV17 arXiv:1707.02968.
- Ruído: Zhu+ DN-TOD arXiv:2401.08056; Ryoo+ UNA arXiv:2312.13822.
- Detectores: YOLOv12 arXiv:2502.12524; RT-DETR 2304.08069; RT-DETRv2 2407.17140; D-FINE 2410.13842 (ICLR25 não confirmado); DEIM 2412.04234.
- Perdas IoU: Wise-IoU 2301.10051; Focaler-IoU 2401.10525 (overlap com 2312.17663); Inner-IoU 2311.02877; SIoU 2205.12740; DotD (CVPRW21, doi:10.1109/CVPRW53098.2021.00130); SimD 2407.02394.
- Domínio: YOLO-YSTs (Agronomy 2025), PCSNet-YOLOv12 (Agriculture; atenção Transformer piorou objetos pequenos), Yolo-pest (Sci. Rep. 2025), SRNet-YOLO (Front. Plant Sci. 2024). Verificar todas antes do .bib.

## Ainda aberto para o autor
Push (manuscript depois raiz); re-anotação A4; ver `\TODO{AUTHOR DECISION}` do .tex; split do eval dos amigos.

## LOG DE EXECUÇÃO
- 2026-10-16: working tree tinha 10 arquivos com versões antigas (mtime idêntico 14/out 13:58; regrediam alertas/retry/projeto W&B v9). Restaurar via `git checkout` foi negado pelo classificador; backup do diff em `_scratch/wt_modificacoes_20261014.patch`. Para não tocar nos arquivos do autor, os jobs rodam do snapshot limpo `git archive HEAD` em `_scratch/repo_head` (REPO=essa pasta). Novos scripts: criar em `rq3_tiny_object/slurm/` e copiar para o snapshot.
- Bug do WBF (IndexError) já estava corrigido no HEAD (`eval_wbf_tta.py` evaluate()).
- 2026-10-16: submetido job **34860** `chain_E_rq3_v9.sh` (NWD, GCD, RFLA ×3 seeds → SAHI/TTA/WBF → P2 ×3). Marcador final `$RUNS_ROOT/CHAIN_E_DONE`. Ressubmeter o mesmo comando se rc 75.
  Comando: `cd _scratch/repo_head && sbatch --export=ALL,REPO=/raid/user_marcospaulo/paper_2_pestline/_scratch/repo_head rq3_tiny_object/slurm/chain_E_rq3_v9.sh`
- Pendente: chain_F (WIoU, 2560, YOLO11/12, scopes NWD, E7 s1/s2, controle L2 igual), CPU (A3, A4, A5, B4), CSVs, manuscrito.
- 2026-10-16 (tarde): patch Wise-IoU v3 adicionado como `FLYDET_NWD_MODE=wiou` em `rq3_tiny_object/patches/nwd_patch.py` (scope forçado a loss) + `tests/test_wiou_patch.py` (CPU ok; única falha restante, `test_train_bg_metrics`, vem do working tree antigo de `rq2_cascade/architecture_lab/train.py`, não do snapshot). `yolo11m.pt` baixado em `/raid/user_marcospaulo/models`.
- Criado `rq3_tiny_object/slurm/chain_F_rq3_extras_v9.sh` (E20 WIoU, E21 2560 b4, E22 YOLO11m s0, E7 s1/s2, E15/E16 scopes NWD). Submeter SÓ depois de CHAIN_E_DONE (1 slot). Comando: `cd _scratch/repo_head && sbatch --export=ALL,REPO=/raid/user_marcospaulo/paper_2_pestline/_scratch/repo_head rq3_tiny_object/slurm/chain_F_rq3_extras_v9.sh`
- RISCO DE PRAZO: Slurm estima início do job 34860 em 2026-10-19 06:21 (fila b200n1 cheia de outros usuários); chain_E ≈ 14 GPU-h + chain_F ≈ 20 GPU-h. Pode não caber antes de 22/out. Prioridade se faltar tempo: E10 (NWD) > E0 SAHI/TTA > E11/E14 > P2; chain_F é opcional.
- 2026-10-16: B4 feito (CPU): `common/analysis/rq1_learning_curve.py` → `rq1_data/results/learning_curve.csv` (acc(N)=a−b·N^−c em L0..L3, TEST, 3 seeds; assíntota a≈0,823, c≈1,28; ganho previsto de dobrar L3 ≈ +0,4 pp → saturação; exploratório, 4 pontos/3 parâmetros). A3/A5 RQ3: gerador pronto `common/analysis/rq3_v9_analysis.py <RUNS_V9> rq3_tiny_object/results [0,1,2]` (testado em E0/E7; gera rq3_v9_{runs,summary,size_bins,paired}.csv com Holm e TOST ±1 pp). Rodar quando chain_E/F produzirem runs. SAHI/TTA/WBF têm formato próprio (ler metrics.json das pastas sahi_E0_s*, ttawbf_*), analisar depois.
- Próximo: aguardar job 34860 (início estimado 19/out). A cada checagem: `squeue -u $USER`; ao CHAIN_E_DONE submeter chain_F.
- 2026-10-17: fila na B200 não andou (job 34860 pendente por prioridade, início estimado só em 19/out) → **job cancelado** a pedido do autor, que roda tudo em outra máquina. Preparado `docs/PROMPT_AGENTE_OUTRA_MAQUINA.md` e `rq3_tiny_object/dgx1/run_queue.sh` estendido (E14 s0-2, E15/E16, E20 WIoU, E21 2560, E22 YOLO11m, E7 s0-2). Já existiam no remoto (lcpacheco, V100): E0 e E10 NWD ×3 seeds em `v9_dgx1_nwd_*`. Arquivos antigos do working tree (10 arquivos de 14/out) NÃO foram commitados (regrediriam o HEAD); idem os 5 zips DS-F2_v*.zip.
