# VISAPP 2027 review, round 08

Conference: VISAPP 2027 (Prague). Paper: "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps".
Profile: partly [inferido] (numeric scale 1-5 and exact PRIMORIS layout are inferred; criteria and closed questions are official).

**Reading limitation (stated explicitly):** the PDF could not be rendered or text-extracted here (no poppler/pypdf; Read failed). The review is based on the LaTeX sources in `manuscript_snapshot/` (all sections, tables, bib). Figure 1 (confusion matrices) was NOT viewed; page count (12) and layout/float placement are "not verifiable from the manuscript" in my reading, and I take the stated 12 pages at face value. No `\TODO` calls appear in the section files (only the macro definition in main.tex, unused); author block is anonymous.

## 1. Paper Summary
Species-level fly detection on industrial sticky traps (1920x1080, mean object ~44 px, 7 classes) with YOLO26m. Three levers are tested on DS-F2 v9.0.5 (73-image test split): (RQ1) a ladder of five cumulative training sets (accuracy 56.1% to 82.2%, with an equal-size random-subsample control); (RQ2) a frozen-detector + crop-classifier cascade (5 classifiers, 3 detectors) that never beats the detector, plus a ground-truth-crop oracle saturating near 83%; (RQ3) eight aerial tiny-object variants (NWD, RFLA, GCD, WIoU, P2, 2560 px, YOLO11m), SAHI/TTA/WBF, none helping. The paper is framed as a set of negative/mixed results.

## 2. Strengths
- Unusually honest framing: association vs. causation in RQ1, "no gain detected" vs. equivalence in RQ3, explicit statement that no never-consulted hold-out exists, VAL-selected checkpoints flagged everywhere.
- Per-species precision/recall/count error as primary metrics (Table 2) fits the monitoring use case; mAP is secondary.
- Multi-seed (3) reporting with paired deltas and image-level bootstrap; equal-size control for RQ1 is a good design choice and its L0 result is reported even though it complicates the story.
- Ground-truth-crop oracle cleanly separates localization from identification; localization is 97.7%.
- Relevant application domain, real industrial data at native resolution.

## 3. Weaknesses

### Fatal
None. The paper is internally coherent and does not claim what it cannot show; but see major M1-M3, which together cap the contribution.

### Major
- **M1. Tiny test set, no independent hold-out, possible plate leakage (Sec. 3.3, 4.5, Conclusion).** 73 test images (e.g. 7 crowded, 6 insect-only plates), 52 VAL, no plate identifier, so same-plate images may sit on both sides of the split; TEST was consulted in earlier campaigns; 4.7% of TEST crops have near-duplicates in training. Nearly every headline (all RQ2 deltas, RQ3 non-differences, RQ1 L2 vs L3 -2.5 pp) is within what 73 correlated images can resolve. The authors say so, but the conclusions are still worded as findings ("does not beat in any of 15 configurations").
- **M2. The central negative claims are conditional on a single, untuned recipe (Sec. 4.2, 5.2, 5.3).** Cascade: frozen detector, label-only fusion, 75% margin, no joint training, no tuning of fusion beyond a VAL-weighted late fusion; sign is consistent but 11 of 15 intervals include zero and the 15 configurations are not independent (shared detectors/seeds). RQ3: default hyperparameters, C=44 fixed from an older data version, P2/2560 changes batch/schedule dynamics; "no transfer" cannot be separated from "not tuned". The abstract and title-level message ("decoupling") promise more than a recipe-specific null can give. A stronger paper would show at least one tuned arm per lever or a power/MDE statement for the 73-image split.
- **M3. Ceiling "explanation" is untested (Sec. 5.2).** The paper offers "limited information in the pixels" as consistent but cannot separate it from label noise, crop recipe or object size; there is no expert re-annotation, no inter-annotator agreement, no larger crop/context ablation beyond margin, and confident learning only flags 3.7% as "suspect". Since the intro frames the work as testing a prior claim of "irreducibility", the study ends without discriminating evidence and, per the authors, "the comparison is indicative, not like-for-like" with that earlier study (different data/classes). Contribution 3 ("diagnosis of the identification ceiling") is thus overstated; it is a characterization of errors.
- **M4. RQ1 conclusions about targeting are inconclusive and partly contradictory (Sec. 5.1, Abstract, Contribution 1, Conclusion).** Random subsamples match L1/L2 (so no targeting effect seen), yet a random 103-image subsample is 19 pp better than L0, so the "lot" dimension (labels/version/domain) dominates and is not controlled; the L1 control has std 0.077 (range 0.69-0.84) so it cannot exclude a sizeable targeting effect, and the small-object/crowded/INS-only gains were not tested against the control. The practitioner take-away "collecting more annotated plates matters more than model complexity" is only weakly supported (no model-complexity axis was varied against data in a common budget) and sits uneasily with "whether targeting adds beyond quantity remains open".
- **M5. Limited state-of-the-art comparison and metric/harness concerns (Sec. 4, 5).** Single detector family (YOLO26 plus one RT-DETR, one-seed YOLO26l, one-seed YOLO11m); no comparison with a species-level sticky-trap pipeline from the literature on the same data (not feasible for lab-based work, but then it should be argued), and the only species-level recognition alternatives are classifiers on crops. Species accuracy is defined as micro recall at a fixed score 0.25, which makes RQ1/RQ3 headline numbers insensitive to precision; precision is only shown in Table 2 for E0 vs. cascade (and the cascade loses precision for MF by 8 pp while gaining recall). RQ3 SAHI/TTA/WBF are scored under a different evaluator (mAP@50, conf 0.0) not comparable with the species numbers, and the tile-trained alternative was not run.

### Minor
- m1. Sec. 5.1: L2 vs L3 CI is [-4.8,-0.1] (excludes zero) but the text says "at most small"/Conclusion says "indistinguishable"; Table 1 CI for L2 reads [-4.8,-0.1], consistent, but the abstract's "match the 244- and 869-image rungs" is a statement about means only.
- m2. RQ3 uses L40S while RQ1/RQ2 use B200, and the H100 control shows a late-fusion delta flipping from +1.3 to +0.3 pp ("hardware-dependent noise"); this hardware sensitivity (FP32, different GPU) is itself unexplained and undermines cross-table comparisons (E0 81.7 on L40S vs 82.2 L3 on B200). Different non-determinism sources should be named.
- m3. Sec. 5.2: "the sign is the same in all 5x3 configurations" mixes a one-seed YOLO26l; the 15 count should say which are seed-replicated. Late fusion "+0.3 [-0.1,+0.7]" is called "only matches the detector", fine, but the weight tuned on VAL was then evaluated where VAL had also selected best.pt.
- m4. Sec. 5.3: p-values for P2/2560/NWD-loss (0.14, 0.057, 0.084) are described as "do not survive Holm" yet the table bolds only NWD+RFLA; two of three intervals exclude zero in the negative direction. Holm correction "over eight comparisons" excludes YOLO11m, which is also tabulated.
- m5. Sec. 5.3 "Why the aerial loss recipes do not transfer": explanations (stride-8 cells, dynamic assignment, SAHI use case) are speculative and carry no ablation; the arithmetic "44 px spans about five cells" is fine but "AI-TOD objects of a few pixels after downscaling" is uncited.
- m6. Sec. 5.2 error-analysis paragraph and the confusion rates (e.g. MD->MV 26.3%, MD->non-species 7.0 to 15.0%) lack a table; the section is a dense list of numbers, hard to verify. The paragraph numbered "Error Analysis" starts with a fragment ("Confusion rates among TEST species ...").
- m7. Sec. 5.2 "Limitations and open items" contains an unmarked \paragraph{Background class} jammed mid-paragraph; BG result (-2.8 pp) is mixed into limitations.
- m8. Sec. 4.2: ArcFace/SupCon "ranked below parts in single-seed VAL cascade check (75.7 vs 76.4)", a 0.7 pp single-seed difference used to drop heads; and the headline CE-vs-parts selection is on 52 VAL images.
- m9. Anonymity: the authors cite their own earlier work in the third person with bib author "Anonymous" (fine), but the text relies heavily on it ("we test its claim..."), plus the dataset name/lineage and "Code will be released" are fine. A GitHub URL appears only for Ultralytics. No identifying info found. The `arXiv:2510.27649` GCD reference is given inline rather than as a bib entry.
- m10. Reference currency: bib has 74 entries; the detection baselines are limited to YOLO and RT-DETR; no recent small-object or fine-grained insect cascade papers beyond those cited (e.g. foundation-model fine-grained classification is covered only by DINOv2/3 and BioCLIP 2). I name no missing work.
- m11. Language: the paper contains long, cumulative sentences (abstract, Contribution 1 and 4, conclusion) that are hard to parse; the abstract is dense (~230 words) with parenthetical qualifiers. The sentence "from L1 to L2 the rise is mostly quantity, but not for L0" appears three times.
- m12. Table 1 and 3 use `\resizebox` and 2.5pt tabcolsep with footnotesize; legibility at column width not verifiable without the PDF. Figure 1 not viewed.
- m13. Dataset descriptions (Sec. 3.2) are heavy on version bookkeeping (v5.1...v9, 62 re-annotated images, DS-ORG) that a reader cannot reproduce; a single summary table of lots (images, boxes, instances per species, collection type) would help.

## 4. Questions for the authors
1. Can you give a plate/session identifier (even coarse) and re-split by plate to quantify leakage? What would the L3 accuracy and the cascade deltas be on a plate-disjoint split?
2. What is the minimum detectable effect of the 73-image TEST at your seed variance? State it for the cascade and RQ3 claims.
3. Is there any tuned cascade variant (joint training, detector-class-distribution fusion, hard-example mining of detector proposals) that can win? If not run, why does the title promise "decoupling" as a conclusion?
4. Can two annotators relabel the 127 confident-learning flags and the MD/MV/MC test crops to give agreement and a noise-corrected ceiling?
5. Why does hardware (L40S/H100/B200) change a delta from +1.3 to +0.3 pp; is evaluation nondeterministic (e.g. TF32 in training)?
6. For RQ1, can you add a size control for L0 (e.g. random 103 from the same earliest versions) to disentangle label quality from version?

## 5. Ratings per criterion (1-5)
| Criterion | Score | Label | Justification |
|---|---|---|---|
| Relevance | 4 | Accept | Applied recognition/detection on real industrial imagery fits VISAPP Recognition and Detection. |
| Originality | 3 | Borderline | Mostly application of known methods; value is in controlled negative results on a new domain. |
| Technical Quality | 3 | Borderline | Careful stats and honesty, but tiny non-independent test, untuned recipes, no resolution of the ceiling cause. |
| Significance | 3 | Borderline | Useful negative evidence for practitioners, but generality (one facility, one recipe) is limited. |
| Presentation | 3 | Borderline | Structured and honest, but dense, number-heavy prose; some figures/tables hard to verify. |

## 6. Overall rating and Justification
**3 - Borderline.** The paper is honest and carefully calibrated, and its negative cascade/transfer findings are worth telling. However, with a 73-image test split that may share plates with training, one facility, untuned recipes and an unresolved ceiling diagnosis, the evidence supports "no gain observed in this setting" rather than the decoupling/ceiling narrative. I would move to 4 if the authors showed a plate-disjoint split (or leakage bound), a power/MDE statement, and at least one expert relabel of the confusable pairs; I would argue to reject if leakage turned out to be material.

## 7. Reviewer confidence
3/5 (solid on detection/fine-grained recognition methodology; not an entomology specialist; PDF not rendered, review from LaTeX source).

## 8. Closed questions
- Abstract and introduction adequate? Partial (dense, qualifier-laden; the intro is clear on RQs).
- More experimental results needed? Yes (plate-disjoint split, tuned variants, expert relabel).
- A comparative evaluation needed? Partial (more detector families/tuned cascades; literature pipelines not comparable).
- Needs a more critical discussion? Partial (limitations are good; causal explanations in 5.3 are speculative).
- Figures adequate? Not verifiable (Figure 1 not viewed; tables dense).
- Conclusions and future work convincing? Partial (practitioner take-away over-reaches M4).
- References up-to-date and appropriate? Yes, broadly (74 entries, many 2023-2026); minor inline arXiv for GCD.
- Formatting needs adjustment? Not verifiable (page count taken as 12 per the task; `\resizebox` tables).
- English needs improvement? Partial (long cumulative sentences, repeated phrases).

## 9. Comments to the authors
Please tone the headline claims to the evidence: state minimum detectable effects, make clear that all RQ2/RQ3 nulls refer to one recipe, and move "ceiling diagnosis" to "error characterization" unless relabelling is added. Add a lot-summary table, an error-analysis table, and drop duplicated sentences about L0. Resolve the L2 vs L3 wording against its CI.

## 10. Confidential comments to the Program Chair
No plagiarism or dual-submission evidence; the earlier PACBB paper is cited in the third person and is clearly a different study (data version, classes, models), though the dataset lineage is shared (check overlap with the earlier submission for substantial dataset/text reuse). No anonymity breach found in the sources. Decision: borderline, leaning weak accept only if the rebuttal addresses split leakage and test size; otherwise weak reject.

## 11. Additional comments
- Fix the stray "\paragraph{Background class}" inside the limitations paragraph; add a verb to the first sentence of Error Analysis.
- Add GCD as a proper bib entry.
- Anonymity, ≤12 pages, no \TODO: no violations found in sources (page count and rendering not verified in the PDF).
- Numbers spot-checked for internal consistency: Table 2 F1 means (79.5 vs 76.9) match the text; Table 1 CIs match the text; the 0.752/0.754 control values and 127/3,470 are traced only partially to PAPER_FACTS (127/3470 found; others not located by my grep, so treat as not independently verified).

## Summary table
| Item | Value |
|---|---|
| Relevance | 4 |
| Originality | 3 |
| Technical Quality | 3 |
| Significance | 3 |
| Presentation | 3 |
| **Overall** | **3 (Borderline)** |
| Confidence | 3/5 |
| Fatal / Major / Minor | 0 / 5 / 13 |
