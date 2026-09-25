# Meta-Review — Round 01

**Paper version:** 59d5da9 · **Date:** 2026-09-22

## Ratings
| R1 | R2 | R3 | R4 | R5 | Mean | Min |
|---|---|---|---|---|---|---|
| 2 (conf 4) | 2 (conf 4) | 2 (conf 5) | 2 (conf 4) | 2 (conf 4) | 2.0 | 2 |

## Decision
**Reject** — Unanimous 2/5, all provisional because ~55 % of the manuscript (Sec. 3, 4, 5, 7, supplementary) is `\TODO{}` skeleton, Tab. 1 is empty and the teaser is the template placeholder. Beyond incompleteness, two reviewers found flaws in the evidence that would stand even if the text were finished: R3 verified on the raw `results.csv`/`args.yaml` that (a) the headline +1.4 pp RQ2 gain compares E10's max-over-137-epochs against E0's `best.pt` — under a consistent rule E10 is ≤ E0 — and (b) the running 3×3-seed campaign is training seed 0 three times, so no detector variance exists; R3 also showed the RQ3 F1/precision claims reverse sign on the second classifier seed (cascade F1 70.1 % < detector-only 73.2 %). R1, R4 and R5 independently found the abstract's "disproving the irreducibility claim" contradicted by the paper's own per-species numbers (flesh-fly recall falls 85.3→81.2 %). The written sections (Sec. 0–2, 6) are of Borderline-to-Accept quality and every transcribed number matches `PAPER_FACTS.md`; the problem is that the current evidence does not support the two quantitative headline claims, and the first-listed contribution (RQ1) has no evidence at all.

## Consensus strengths
- **Sec. 6 (Limitations) is unusually candid** — single-seed status, "consolidated, not blind" test set, ensemble demoted to exploratory, two evaluators never mixed, latency as upper bound (R1 S1, R2 S3, R3 S1, R4 S5, R5 S1). All five weigh this positively.
- **RQ3 "re-label without re-rank → accuracy up / mAP down" is a useful methodological observation** for anyone bolting a classifier onto a detector and evaluating with COCO AP (R1 S2, R3 S3, R4 S1, R5 S4). R4 W6 qualifies the mechanism (see Disagreements).
- **Every number that appears in the text matches `PAPER_FACTS.md`** (R1, R2, R3, R4, R5 §8). Discrepancies are of *status* (asserted vs. unconfirmed) or between `PAPER_FACTS.md` and raw artefacts (R3), not transcription.
- **Related work is accurate and the prior PACBB paper is cited in third person with correctly reproduced numbers** (R1 S5, R2 S4, R5 S3); no reused text/tables/figures, no dual-submission overlap (R5).
- **Negative results are numerous and framed with mechanisms** (R1 S4, R3 S4, R4 S4).
- **The RQ1 design principle (failure-targeted acquisition with per-failure ablation) is the right one** (R2 S1, R5 S2) — if executed.

## Consensus weaknesses
- **[must-fix] RQ1 is presented as a result (title, abstract, contribution 1) with zero evidence; Tab. 1 empty; unknown whether the new plates are already inside DS-F2_v8.1.1** (R1 W2, R2 W1, R5 W2). If the plates are already in E0/E10's training set, Tab. 1 must be a leave-one-out ablation and "Base" is not the current baseline (R2 Q1, R1 Q4).
- **[must-fix] Abstract overclaims "disproving the irreducibility claim" and "at fixed precision"** (R1 W5, R4 W1, R5 W1, R3 W3). Flesh-fly (MC) recall drops; the gain comes from MD and MF; pairwise MC↔MV confusion counts are a TODO.
- **[must-fix] +1.4 pp RQ2 headline is single-seed and (per R3) a checkpoint-selection artefact; GCD "best mAP50:95" is +0.6 pp single-seed with E0/E10 cells blank** (R1 W3/Minor 6, R3 W1/Minor 4, R5 §8-iii). Contradicts Sec. 6 ¶3's own "we do not interpret 1–2 pp".
- **[must-fix] Species-code collision with PACBB (old MF = flesh fly; new MF = fruit fly); `X_suppl.tex` and `PAPER_FACTS` §5 already use the old codes; taxonomic rank undefined; no class-mapping table** (R2 W5, R4 W2/W7, R5 W4).
- **[must-fix] Status misstatements in Sec. 1 P4-4**: E13 multi-scale was CUDA-OOM, not "failed to beat"; "TTA/WBF" conflates the finished classifier ensemble with the pending detector job 33015 (R1 W6, R2 §8-d, R4 Minor 3–4, R5 W3).
- **[must-fix] Split rule asserted ("no plate appears in two splits") but marked "confirm" in `PAPER_FACTS`; project notes suggest a version-based split** (R1 W6, R2 W3, R4 Q8, R5 Q5).
- **[must-fix] Dataset-as-contribution undecided; abstract hedges "and data, if approved"** (R2 W4, R5 W8). CVPR policy: decide now — release commitment or reframe contribution 1 as protocol + code.
- **[must-fix, policy] Format/anonymity**: 48+ red `\TODO` markers, placeholder teaser, `tab:rq3` overfull by 59 pt, `note = {VERIFY …}` rendered in references, XeTeX/tectonic build loses italics (must be pdfLaTeX), "the client" and "the same setting" phrasing next to a full-author-list self-citation (R1, R2, R4, R5 §8).
- YOLO26 and SAM used but never cited (R1 Minor 5, R2 §8, R5 Minor 5).
- Non-target classes (INS/NOISE/MAR ≈ 63 % of boxes) — matching/denominator rules for P/R undefined; 4-class vs 7-class mAP mixed in Tab. 3 (R2 W6, R4 Minor 1, R5 Minor 3).
- No confidence intervals / MDD despite known $n$ (R3 W5); test set consulted 38 + 19 times while Sec. 5.1 TODO plans "touched once" (R3 W4, R4 Minor 2).

## Disagreements and adjudication

1. **RQ3 mechanism: "part attention transfers to 40 px crops" (R4 S1, Sec. 2 ¶3) vs. "part attention is indistinguishable from plain-CE ConvNeXt-T" (R3 W6: F1 75.4 vs 75.0/75.7, recall identical to the decimal).** Ruling for R3: the gain is attributable to *cascading*, not to part attention; add the plain-CE cascade row to Tab. 3 and reword to "did not hurt". Statistical validity trumps enthusiasm.
2. **Selection protocol "clean and stated" (R4 S2) vs. ≥ 12 classifier variants compared on VAL within noise and TEST consulted 38 + 19 times (R3 W4).** Ruling: both are factually right — selection *was* made on VAL — but R3's consequence holds: state the number of configurations tried, drop "held-out"/"touched once", report a bootstrap CI, and protect TEST for the RQ1 ablation, which is the last blind result available.
3. **"Re-label without re-rank" explains the mAP drop (R1 S2, R3 S3, R5 S4) vs. incomplete — the product-score policy, which *does* re-rank, also loses AP (R4 W6; `PAPER_FACTS` §2: 77.28 vs 81.18 VAL mAP50).** Ruling for R4: R4 cites a specific number the others did not weigh. Keep the mechanism but complete it (low-confidence background proposals relabelled into species classes lengthen the PR tail) and support it with a per-class AP breakdown or a restricted-reclassification ablation.
4. **RQ2 gain undermined by NWD units mismatch in the loss (R1 W1, inferential, confidence 4) vs. by checkpoint-selection artefact + replicated seeds (R3 W1–W2, verified on artefacts, confidence 5).** Ruling: not contradictory — both must be answered — but R3's finding is decisive on its own and cheaper to verify (Q1: epoch/mAP under both rules). R1's W1 is a strongly supported inference, resolvable by one diagnostic print; if confirmed, the $w{=}1$ collapse has the simpler explanation R1 gives (box loss ≈ 0) and the mechanism sentence in Sec. 4.2 must change. The AC did not re-open the raw CSVs (outside `paper/`); R3's claim is specific and falsifiable and is adopted pending the authors' answer to R3 Q1–Q2.
5. **Cascade must be re-run on E10 (R4 W3) vs. E10 `best.pt` is below E0 under a consistent rule (R3 W1).** Ruling: deferred — until the seeds are fixed there is no evidence E10 is the "strongest detector"; run E10+cascade only if the multi-seed comparison shows E10 ≥ E0.
6. **Limitations "adequate: yes" (R5) vs. "partial" (R1–R4).** Ruling: partial — Sec. 6 ¶3 omits that the seed-84 cascade falls *below* the detector-only baseline (R3), and asserts an unverified split rule.
7. **E12 "early stopping at epoch 10" (text) — R1 flagged inconsistency with `patience 30`; R3 resolved it from `results.csv` (32 epochs, best fitness epoch 2, max mAP50 epoch 10).** No conflict; adopt R3's wording.

## Top-5 fixes (ordered by impact on acceptance probability)

1. **[must-fix] Fix the RQ2 evidence chain before any RQ2 number is written** (R3 W1–W2, R1 W1/W3, R5 §8-iii). Define one checkpoint rule (`best.pt`) in Sec. 5.1 and apply it to every row of Tab. 2; correct `PAPER_FACTS.md` §1 (0.8706 is max-over-epochs, `best.pt` is 0.8484 @ ep 107); repair `train_nwd.py` so `seed=` reaches the trainer (verify `args.yaml` differ and `results.csv` are not identical); rerun 3×3; remove "+1.4 pp" from abstract/Sec. 1/Sec. 6 until then. Add loss-only / assigner-only NWD runs and one diagnostic of the NWD coordinate frame (R1 Q1–Q2) so the mechanism paragraph in Sec. 4.2 is correct.
2. **[must-fix] Rewrite RQ3 claims to what survives two classifier seeds** (R3 W3/W6, R1 W5, R4 W1/W6, R5 W1). Tab. 3: add seed 84 columns, the plain-CE cascade row, and a McNemar $p$ on the 918 paired instances; abstract/Sec. 1: replace "disproving" with the exact MC↔MV before/after confusion counts, state recall gain as a range (+2.2 to +4.4 pp), drop "at fixed precision" and the F1 claim; complete the mAP-down mechanism with per-class AP (R4 W6).
3. **[must-fix] Resolve RQ1: run Tab. 1 or demote it** (R2 W1–W3, R1 W2, R5 W2/W8). First answer R2 Q1 (are crowding/graduated/blank plates already in the 1,259 training images? counts per split?), then design Tab. 1 as add-in or leave-one-out accordingly, with corrected seeding and TEST untouched; verify and state the split rule by plate ID; decide dataset release vs. "protocol + code" and make title/abstract/contribution 1/Sec. 3.1/Sec. 7 consistent. If the ablation cannot be run, remove RQ1 from title and abstract.
4. **[must-fix] Class taxonomy and code mapping** (R4 W2/W7, R2 W5, R5 W4). Add a table in Sec. 3.1: code (MD/MV/MC/MF) → common name → taxon/rank → PACBB code; never use FF/HF/BF; fix `X_suppl.tex` "MF↔BF" and `PAPER_FACTS` §5; define P/R matching rules for INS/NOISE/MAR in Sec. 5.1; state annotator provenance and rank in Sec. 6 ¶1.
5. **[must-fix, policy] Desk-reject hygiene and status corrections** (R5 W2–W7, R1 W6, R4 Minor 3–4/§8). Remove all `\TODO`, replace placeholder teaser, fix `tab:rq3` overflow (59 pt), delete the `VERIFY` bib note, compile with pdfLaTeX and re-measure pages, cite YOLO26 and SAM, drop "the client"/"the same setting", and correct Sec. 1 P4-4: E13 = OOM/not run; "TTA/WBF" = classifier ensemble only until job 33015 lands; E12 = best fitness ep 2, patience stop ep 32.

## Nice-to-have (deferred)
- $C$ sensitivity ($C\in\{22,44,88\}$) and Wise-IoU/Shape-IoU baselines for RQ2 (R1 W3); per-level positive counts P3/P4/P5 (R1 Q6); per-class AP50 for E0 vs E10 (R1 Minor 7).
- State once that objects are relatively tiny but absolutely small/medium (AI-TOD bands) and use it as the shared mechanism for the P2/SAHI/512-crop negatives (R1 W4).
- Controlled real-copy-paste vs. rendered-paste-on-real-plates comparison at matched volume; cite Kisantal et al. 2019 (R2 W2).
- Fixed-size native-resolution crop window or box-size auxiliary input in the classifier; state train-on-GT / infer-on-proposals mismatch and 4- vs 7-class head (R4 W4–W5).
- Cascade on E10 proposals, conditional on E10 ≥ E0 after seeds (R4 W3).
- 95 % bootstrap CIs in Tab. 2–3; effect-size tiering of negative results vs. seed spread; cite Bouthillier et al. 2021 and Dodge et al. 2019 (R3 W5, Minor 1/7).
- Sticky-trap cascade prior art: Zhong et al. 2018, Rustia et al. 2021, Ding & Taylor 2016, Høye et al. 2021 (R4 Minor 5); report or disclaim the DINOv2 variant (R4 Minor 6).
- Deployment scenario for the 18× latency (offline batch vs. edge) (R4 Minor 7); production-threshold latency column (R1, R3).
- Datasheet in the supplementary; per-class counts per split; temporal coverage in Sec. 6 (R2 W3, Minor 4/6).
- Shorter title; consistent UK/US spelling; Tab. 3 Δ rounding and bolding; section order 5.2–5.4 aligned to RQ order (R5 Minor 1–2, 4, 10, 14).

## Estimated post-fix rating
If all five fixes are applied well — and the corrected multi-seed comparison still shows E10 ≥ E0, the cascade recall gain holds across ≥ 3 classifier seeds with a paired test, and Tab. 1 is filled — **3.4** mean (Borderline-Accept). If the RQ2 gain does not survive consistent seeding, the paper remains viable as a data + cascade paper with an honest negative on NWD-in-E2E-YOLO, at ≈ **3.0**.
