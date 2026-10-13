# RQ1 — Comparação das versões do DS-F2 (v4, v5.3.5, v6.1.1, v7.1.0, v8.1.1, v9.0.5)

Gerado em 2026-10-13. Geradores: `common/analysis/rq1_version_inventory.py`, `rq1_version_diff.py`, `rq1_origin_profile.py`.
CSVs: `rq1_versions_{splits,origin,classes,overlap,leakage,test_content_overlap,diff,vs_v9}.csv`, `rq1_profile_*.csv`.
Dados extraídos em `/raid/user_marcospaulo/datasets/rq1_versions/` (zips originais na raiz do repo, não rastreados).

## Tamanho por versão (imagens train/val/test; instâncias totais)
| Versão | train | val | test | classes | resolução | origem do conteúdo |
|---|---|---|---|---|---|---|
| v4 | 82 | 14 | 41 | 6 (sem MAR) | **1280×720** | DS_F2-v3 (placas antigas, ~310 inst/img, √wh≈18 px) |
| v5.3.5 | 1528 | 10 | 29 | 7 | 1920×1080 | placas v5.1 (97 imgs, 5 coletores) + DS-ORG |
| v6.1.1 | 1823 | 18 | 38 | 7 | 1920×1080 | + lote v6.0 (83 imgs) |
| v7.1.0 | 2152 | 25 | 45 | 7 | 1920×1080 | + lote v7 (70 imgs) |
| v8.1.1 | 1259 | 35 | 55 | 7 | 1920×1080 | + lote v8 (100 imgs); train reamostrado (menos aug de imagem) |
| v9.0.5 | 1738 | 52 | 73 | 7 | 1920×1080 | placas v5.1–v9 (lote v9 +174 imgs); **sem DS-ORG**; + aug de instância |

## Achados verificados (por hash/nome/audit)
1. **Linhagem incremental**: lotes de placas v5.1 → v6.0 → v7 → v8 → v9 (audits `audit-v6..v9.jsonl`); splits val/test crescem por lote
   (test: 29 → +9 → +7 → +10 → +18 = 73 em v9). Rótulos das imagens em comum entre v8.1.1 e v9.0.5: **idênticos** (267/267 originais comuns).
2. **Sem vazamento test→train** em nenhuma versão (nome-base e md5), nem de test de uma versão no train de outra.
3. **v4 não tem nenhuma imagem em comum** com v5+ (0 hashes), é 1280×720 e 6 classes → é outra coleta/etapa, não subconjunto.
4. **DS-ORG** (imagens orgânicas: moscas sobre papel branco, uma espécie por foto, ~58 inst/img) existe em v5.3.5–v8.1.1 e **foi removido em v9.0.5**.
5. **Aumento**: sufixo `-uuid` = aug de imagem (v5.3.5–v9); `-instance-uuid` = aug de instância/copy-paste, **só em v9.0.5** (desc.txt: "instance data-augmentation sobre v9.0.1").
6. v6.1.1 → v7.1.0: 62 imagens em comum com número de caixas alterado (reanotação) — único salto com mudança de rótulo.
7. Test v8.1.1 (55) vs v9.0.5 (73): 46 imagens em comum por md5 (as 9 do lote v6.0 não casam por hash — provável reexportação PNG; checar por nome).

## NÃO verificado
- Qual lote corresponde a crowding / graduada / branco / copy-paste (README RQ1). Nenhuma imagem de train/val/test tem rótulo vazio em nenhuma versão → "papel em branco = negativo puro" não aparece como imagem sem caixas.
- Se v4 = STID do PACBB (STID-SOP ≈ 900 imgs ≠ 137 do v4).

## Confirmação visual (2026-10-13, leitura direta de overlays, sem GPU/modelo)
- **v4** = coleta do paper anterior (confirmado pelo autor): 1280×720, placas com muito INS/NOISE, sem MAR.
- **MAR = marcações impressas da placa** (logo, "PLACA ADESIVA", legenda de ordens, ID manuscrito); caixas laranja. Aparece em todas as placas a partir de v5.
- **Placas de espécie única** (v6.0 em diante, ID manuscrito com letra = espécie): D→MD, V→MV, C→MC, F→MF, G→INS (outros insetos). Lotes v7/v8/v9 têm placas só de INS (8/20/40 imagens sem MD–MF).
- **v5.1 e parte do v6.0** são placas de espécies **mistas**.
- **Crowding da mesma espécie**: confirmado — placas F (MF) com até 75–77 moscas por imagem em v8 (F16, F4) e v9 (F39, F40, F20); densidade máxima de espécies por lote: v5.1=42, v6.0=35, v7=34, v8=75, v9=77.
- Dois tipos físicos de placa: impressa lisa e coberta de cola (textura), ambos em v8/v9.
- **DS-ORG** = moscas sobre papel branco, uma espécie por foto (MD/MV/MC/MF; sem INS/MAR), objetos grandes; presente v5.3.5–v8.1.1, ausente em v9.0.5.
- Não encontrei placa "graduada (escala)" nem "papel em branco (negativo puro)" nas amostras vistas → depende do autor dizer qual lote/pasta é cada uma.

## Verificação das 4 coletas dirigidas (2026-10-13, imagens brutas por lote + contagem de rótulos)
- **Papel branco = DS-ORG** (confirmado visualmente: moscas sobre folha branca, 1 espécie por foto; ~290 imgs originais em v5.3.5, 212 em v8.1.1). Ausente em v9.0.5.
- **Placa graduada**: a grade impressa (8×4 células) aparece em TODAS as placas de v4, v5.1, v6.0, v7, v8 e v9 (lisas e com cola). Não é um lote isolado nem separa lotes. Tipo físico: v4 só com cola; lisa+cola desde v5.1.
- **Negativo puro**: nenhuma imagem sem mosca/INS em nenhuma versão (0 em v5.3.5/v8.1.1/v9.0.5). Placas só de INS: v7 8, v8 20, v9 40. Existem placas quase vazias (poucos pontos de sujeira), mas anotadas como NOISE/INS.
- Crowding: ver seção anterior (F até 77 moscas, v8/v9). Copy-paste: só v9.0.5 (`-instance-`).
