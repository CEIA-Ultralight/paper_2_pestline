# VISAPP 2027 review (benevolent-typical-reviewer posture)

**Paper:** Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps (12 pp., anonymous)
**Profile:** visapp.md, partially `[inferido]`: numeric scale 1-5, labels, and the layout of the Strengths/Weaknesses/Questions sections are inferred. The five criteria and the closed questions are official.
**Read:** all 12 pages via pypdf text extraction. Figure 1 is read from its extracted matrix values only, not visually.

## 1. Paper Summary
The paper studies what limits species-level reading of industrial sticky traps (7 classes, 1920x1080 images, mean object about 44 px). It tests three levers with a YOLO26m detector.
1. **Data:** a ladder of five cumulative training sets, with an equal-size random-subsample control.
2. **Identification:** a detect-then-classify cascade with five classifiers on four backbones (ConvNeXt-T, DINOv2, DINOv3, BioCLIP 2), and a ground-truth-crop oracle.
3. **Scale:** eight tiny-object variants from the aerial literature (NWD, RFLA, GCD, Wise-IoU, P2 head, 2560 px, YOLO11m), plus SAHI, TTA and WBF.

Main findings:
- Species accuracy rises from 56% to 82% along the ladder. Random subsamples of the final set match the L1 and L2 rungs, but not L0.
- The cascade never beats the detector (-1.1 to -5.1 pp macro-F1).
- Oracle accuracy saturates near 83%.
- No tiny-object variant helps.

The paper is explicit that most results are negative and that the causes cannot be separated with the available data.

## 2. Strengths
- **Honesty.** Negative results are reported plainly (cascade, RQ3). Claims are hedged ("association, not proof"; "no gain detected, not equivalence"), and the paper says that no never-consulted hold-out exists.
- **Statistical care.** Three seeds, paired deltas, BCa and bootstrap intervals over images, McNemar tests with Holm correction, and an explicit remark that McNemar understates uncertainty. This is above the norm for applied VISAPP papers.
- **Cascade study (RQ2).** It is thorough: 5 classifiers, 3 detectors, relabelling-only, late fusion, margin variants, a background-class head, abstention, and an oracle that separates localization from identification. The error decomposition (Sec. 5.4) is informative.
- **Equal-size control (RQ1).** It prevents a naive "targeted collection works" story and surfaces the L0 anomaly.
- **Realistic setting and honest reproducibility.** Production-resolution field imagery with a fine-grained species task, which is under-studied. Checkpoint selection on VAL is declared.
- **Practical takeaway.** The paper tells practitioners that more annotated plates helped most, and that the MD/MV/MC confusion may be a labeling or imaging problem.

## 3. Weaknesses

### Fatal
None.

### Major
1. **Small, single-facility test set (Sec. 4.4, Limitations).** 73 images, 1,210 instances, intervals about ±2 pp. Many of the reported differences (L2 vs L3: -2.5 [-4.8, -0.1]; margins; most RQ3 variants) are at or below resolution. The paper acknowledges this. It still matters for the strength of the "no effect" conclusions, which are best read as "undetected".
2. **Possible train/test leakage by plate (Sec. 3.3, 5.2).** There is no plate identifier. 4.7% of TEST crops are near-duplicates of training crops. Because of this, the absolute 82-83% levels and the oracle ceiling could be inflated and are not clean generalization numbers. The authors declare this honestly. A same-plate split rule, even a post-hoc one from file metadata, would materially strengthen the paper. I treat it as major because it touches every absolute number. The relative comparisons are less affected.
3. **The identification ceiling is not explained (Sec. 5.2, contribution 3).** Ambiguity, label noise and crop recipe are not separated, and no expert re-annotation was done. The paper says this itself. The title and abstract nevertheless frame the finding as "decoupling does not help". A reader may take the "ceiling" as intrinsic. The wording is mostly careful, so I count this as a framing point rather than a flaw.

### Minor
1. **Hyperparameter tuning in RQ3 (Sec. 5.3, Limitations).** Default hyperparameters and no per-variant tuning, so this is "no gain detected here". NWD+RFLA is unstable (one early stop, one divergence) and was not retried, which makes the 8.8 pp drop uninformative about the method. Declared.
2. **Control comparability across campaigns (Sec. 4.1, 5.3, Limitations).** RQ3 ran on an L40S, RQ1/RQ2 on a B200, and some margin and fusion runs on an H100. E0 therefore appears with different values (79.5% macro-F1; 82.2% and 81.7% accuracy). The paper separates the tables, but the reader has to track which control applies. The late fusion changing from +1.3 pp (B200) to +0.3 pp (H100) is interpreted as hardware noise, which is not verifiable from the manuscript.
3. **Equal-size control and L0 (Sec. 5.1).** The control covers only L1 and L2 sizes and shows large variance (0.69-0.84 at 244 images). The localized gains (small objects, insect-only plates) are not tested against it, and L2+ORG to L3 confounds copy-paste with removal of the white-paper photographs. The paper states all three. The cleaner design is listed as future work.
4. **Confusion rates without intervals (Sec. 5.1, Table 1).** The MC<->MV rates are pooled point estimates. The text says so. An interval would be cheap to add, since the data exist.
5. **Cascade baseline scope (Sec. 4.2).** The cascade uses a frozen detector, label-only fusion, and a fixed 75% margin. Fusion with the detector's class distribution was not run. The authors qualify the claim ("with our recipe"). Sec. 4.2 says ArcFace and SupCon were dropped on a single-seed VAL check (75.7 vs 76.4 mAP50). That is thin justification for a 0.7 pp gap, but not important.
6. **SAHI comparison (Sec. 4.3, 5.3).** It is measured under a different evaluator (mAP@50, confidence 0.0), and no tile-trained model was run. The paper notes this. The resized-frame baseline (35.5% recall) is acknowledged as unfair. The text is clear, but the sentence structure makes the section harder to follow.
7. **Novelty positioning (Sec. 1, 2).** The contribution is empirical (a controlled negative study plus a dataset-ladder analysis), not methodological. For VISAPP's applied scope I judge this acceptable. The paper should say plainly what a practitioner takes away beyond "the cascade did not help here".
8. **Data and reproducibility.** Images will not be released and the code is promised only. The ladder cannot be reproduced independently. Declared.
9. **Dependence on prior work.** The paper leans on an earlier pipeline paper cited in third person ("Anonymous, 2026"), including the "irreducible" claim. Anonymity is handled well, and the paper admits the comparison is indicative, not like-for-like. A short self-contained description of the old data would help.
10. **Presentation.**
   - Contribution 1 (p. 2) is one long, hard-to-parse sentence ("from L1 to L2 the rise is mostly quantity, but not for L0").
   - The abstract and conclusion are dense with parenthetical qualifiers.
   - Table 1 is wide, and Figure 1 shows a confusion matrix with raw counts that are hard to compare in one column. Visual legibility of the figure is not verifiable from the text extraction.
   - Several citations are bare arXiv preprints (GCD arXiv:2510.27649 is cited in text only, Brock et al. 2026, Chachula 2022, Cheng 2018, Du 2025).
11. **Formatting.** The paper is at 12 pages including references. It appears within the limit, but the character count is not verifiable from the manuscript.
12. **AI declaration.** The Acknowledgements read "Anonymous during review". If AI tools were used for text, VISAPP requires declaring it there. Not verifiable from the manuscript.

## 4. Questions for the authors
1. Can the split be re-derived by plate from file names or timestamps, so that a plate-disjoint evaluation is possible? Do the L3 results hold on it?
2. Which of the 127 confident-learning flagged crops fall in the MD/MV/MC family, and would even a small expert review (say 100 crops) change the oracle accuracy or the cascade deltas?
3. Does the cascade sign change if the classifier is trained jointly, or fused with the detector's full class distribution rather than label-only?
4. For L0, is the 19 pp gap to a random 103-image subsample explained by the species-mixed plates or by label differences in the earlier version? Can a per-plate-type breakdown be given?
5. Would a modest per-variant tuning of NWD (C, w) change the RQ3 conclusion at an affordable cost?

## 5. Ratings per criterion (1-5)
| Criterion | Score | Label | Justification |
|---|---|---|---|
| Relevance | 5 | Strong Accept | On-topic for Recognition and Detection and applications; a real industrial setting. |
| Originality | 3 | Borderline | Empirical and negative; the data-lever and cascade-vs-single-stage comparison on trap imagery is new in this setting, but there is no new method. |
| Technical Quality | 4 | Accept | Careful statistics and controls with declared gaps (small test, plate leakage, one facility). |
| Significance | 3 | Borderline | Useful negative results and design guidance for practitioners; generalization untested. |
| Presentation | 3 | Borderline | Complete and honest but dense; some sentences and one table are heavy. |

## 6. Overall rating and justification
**Overall: 4 (Accept)**, at the low end.

The paper is solid, honest, and useful to the community: it reports multi-seed negative results with proper statistics and clearly marks what cannot be concluded. The main weaknesses (a 73-image test split, possible plate leakage, no explanation of the ceiling, no new method) are all declared by the authors and are limits on the strength of the conclusions, not invalidating errors. I would accept it for VISAPP and ask for the minor fixes, especially a clearer abstract and contribution 1.

## 7. Reviewer confidence
**3 of 5.** I am familiar with detection and tiny-object work, but not an expert on entomological monitoring. I could not visually inspect the figures.

## 8. Closed questions
- The abstract and introduction are adequate? **Partial.** They are accurate and hedged, but dense.
- More experimental results are needed? **Partial.** A plate-disjoint check and intervals for the confusion rates would help; nothing else is essential.
- A comparative evaluation is needed? **No.** The cascade and tiny-object comparisons are extensive.
- The paper needs a more critical discussion? **No.** It is already critical of itself.
- The figures are adequate? **Partial.** Figure 1 is a confusion matrix of raw counts, and Table 1 is wide. Not verified visually.
- The conclusions and future work are convincing? **Yes.** They are cautious and match the evidence.
- The references are up-to-date and appropriate? **Partial.** Mostly yes, but several are bare arXiv preprints.
- The formatting needs adjustment? **No** (not fully verifiable).
- The English needs improvement? **Partial.** Some long sentences (contribution 1) need splitting.

## 9. Comments to the authors
Thank you for a candid paper. Reporting negative results with three seeds, paired intervals and an explicit "no hold-out" statement is exactly what applied vision needs. Suggestions:
- Split the long sentences in the abstract and contribution 1.
- Add intervals to the Table 1 confusion rates.
- If any metadata allow a plate-level split, add one even post hoc.
- State once, early, which E0 control belongs to which table.
- Add a short, self-contained description of the earlier data so the "irreducible" comparison does not depend on the prior paper.
- Consider declaring AI tool use in the Acknowledgements if applicable.

## 10. Confidential comments to the Program Chair
No suspected plagiarism, dual submission, or anonymity breach: the earlier work is cited in third person as "Anonymous (2026)". The overlap with that paper is in the dataset lineage only; the experiments are new. I would accept; a shepherd check on the minor presentation fixes is enough.

## 11. Additional comments
- Page 2 abstract: "no advantage of targeting over quantity" is correct for L1/L2 but easy to over-read, since L0 contradicts it. The qualifier is present, so keep it.
- Sec. 5.1 says "H2 identification does not move" as the heading, and then "only partly supported". Rename the heading.
- The "15 configurations" (5 classifiers x 3 detectors) is used in abstract and conclusion, while YOLO26l has one seed, which the text does note in Sec. 5.2.
- Typographic: "V AL" spacing in the extracted text is a rendering artifact in my extraction and not verifiable as a manuscript problem.

## Summary table
| Item | Value |
|---|---|
| Overall | **4 (Accept)** |
| Confidence | 3/5 |
| Fatal / Major / Minor | 0 / 3 / 12 |
