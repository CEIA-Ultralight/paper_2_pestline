# Review, VISAPP 2027 (Regular, Full paper)

**Paper:** "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps"
**Profile:** `visapp.md`. The criteria and the closed questions are [oficial]. The numeric scale, the form layout and the confidence scale are [inferido] (1-5 convention).
**How the PDF was read:** the Read tool with pages failed (pdftoppm is not installed). I read the full text of all 12 pages with pypdf, including tables and references. I could NOT render the pages, so I did not see the figures as images. Figure 1 is judged only from its extracted numbers. The legibility of the figures at one-column width and the layout are not verifiable from my reading.

## Compliance checks
- **Pages:** 12 pages, with the references ending on p. 12. This is at the Full limit. Non-whitespace characters in the extraction are about 45k, inside the 10k-50k range (the extraction includes figure text).
- **Anonymity:** author, affiliation and e-mail are anonymised, and the acknowledgements say "Anonymous during review". The prior paper is cited in the third person as "Anonymous, 2026".
  - The reference gives the full title and venue (PACBB, LNCS), and the text describes the prior pipeline (0.476 to 0.608 mAP). This is easily traceable, so the chair should check it.
  - Portuguese species names (mosca doméstica, etc.) appear in 3.3 and hint at the origin. This is minor.
  - I found no URLs, repository links or names.
- **TODO / placeholders:** none found in the text.
- **AI declaration:** there is none in the acknowledgements. Not verifiable whether it is required.
- **Numeric consistency:** I checked several figures and most close.
  - Table 2: the F1 values from P and R close, and the macro-F1 values (79.5 and 76.9) match the text.
  - Table 1: the deltas and CIs match section 5.1, and the size-bin weights reproduce the species accuracy.
  - Table 3: the size counts sum to 1,210.
  - Inconsistencies are listed under M5 and m1-m4.

## 1. Paper Summary
The paper studies species-level detection of four fly species (plus INS, NOISE and MAR) on 1920x1080 images of industrial sticky traps (mean object about 44 px). It tests three "levers" with a YOLO26m detector, on a single-facility dataset with a 52-image VAL split and a 73-image TEST split:
- **Data (RQ1):** a ladder of five cumulative training sets, with species accuracy rising from 56.1% to 82.2%.
- **Identification (RQ2):** a frozen detector followed by a crop classifier (ConvNeXt-T, DINOv2, DINOv3, BioCLIP 2). It never beats the detector in 15 configurations. A ground-truth-crop oracle saturates near 83%.
- **Scale (RQ3):** eight tiny-object variants from the aerial literature (NWD, RFLA, GCD, Wise-IoU, P2, 2560 px, YOLO11m) bring no gain. Sliced inference, TTA and ensembling also give none, or only a marginal one.

The paper is framed as an honest, largely negative result with explicit caveats.

## 2. Strengths
- **Calibrated, honest writing.** The text separates association from causation and "no gain detected" from "equivalence". It states that VAL is not held out. It reports the unfavourable result that a random 103-image subsample beats L0 by 19 pp. It reports the NWD+RFLA divergence without discarding the seeds.
- **Metric choices.** Per-species precision, recall and F1 at a fixed operating point are primary, with mAP secondary. The cascade comparison uses paired CIs over images, McNemar tests and Holm correction. A per-species table with count error (Table 2) is included.
- **Oracle analysis.** The ground-truth-crop oracle, applied to 15 classifiers, usefully separates localization from identification. It includes the error overlap (4.9% wrong for all), the confidence of errors and the confident-learning flags.
- **Practical relevance.** There is a real industrial dataset at native resolution, and a negative transfer result that could save practitioners effort.
- **Reproducibility details.** Seeds, thresholds, matching rules and software versions are given.

## 3. Weaknesses

### Fatal
None. The paper is internally honest, and its central claims are mostly hedged to match the evidence.

### Major
- **M1. The statistical power of the test split (73 images, 1,210 instances from 73 plates) is too low for the claims, and plate-level leakage cannot be excluded (3.3, 5.2, Limitations).** The dataset has no plate identifier, so train/test overlap at the plate level is possible.
  - A crop-level near-duplicate check flags 4.7% of TEST crops (5.2).
  - This could inflate all absolute numbers (82%, the 83% oracle). It could also affect the quantity-versus-quality inference of RQ1.
  - The paper discloses it, but it is a structural weakness. The "ceiling at 83%" and "no cascade gain" results both live inside the noise floor (about ±2 pp).
  - Most single-configuration CIs for the cascade include zero (5.2). The headline "does not improve in any of 15 configurations" rests on the sign pattern, with correlated runs (the nine detector-classifier pairs are crossed, not independent).
  - What would change my view: a plate-grouped split, or a leakage-free held-out set, even if small.
- **M2. The equal-size control does not support the "targeting vs. quantity" reading (abstract, 5.1, contribution 1).**
  - The subsamples are drawn from L3, which already contains the targeted lots (v8/v9 crowded plates, insect-only plates) and about 50% copy-paste images. The L2 arm (869 images) has no copy-paste.
  - The control therefore shows that old lots are about as good as a random draw from the targeted pool. It does not show that targeted collection is no better than untargeted data of the same size, and no untargeted pool exists.
  - The variance at 244 images is huge (0.752 ± 0.077; range 0.69-0.84). "Indistinguishable" is a statement of low power, not equivalence. The abstract's "we detect no advantage of targeting over quantity" is technically hedged but reads as stronger.
  - The localized gains (small objects, insect-only plates) are not tested against the control (acknowledged in 5.1).
  - What would change my view: a quantity-matched untargeted arm, or removing copy-paste from the subsamples.
- **M3. RQ1 is a version-ladder confound, so "the data lever" is not isolated.** Size, annotation quality, class set, plate protocol (mixed vs species-pure, from v6.0 on) and labels co-vary along L0 to L3. The authors say so.
  - The refutation of the prior "irreducible" claim (contribution 3: MV→MC 14.8% to 5.4%) rests on L0, a 103-image set of mixed-species plates. A random 103-image subsample of the final data is 19 pp better.
  - The reduction in confusion is therefore largely a small-data/label-regime effect, and not a clean "data effect" in the sense of the paper. The confusion rates are pooled point estimates without intervals (5.1).
- **M4. The identification "ceiling" is under-supported as a diagnosis, and the cascade design is narrow (4.2, 5.2).**
  - All classifiers see 384 px resizes of crops cut from the same 1920x1080 pixels. The ceiling is shared across backbones, but the information bottleneck is not demonstrated.
  - Four alternatives are listed as inseparable: ambiguous pixels, label noise, crop recipe and object size. No expert relabelling, inter-annotator agreement or higher-resolution crop check is available. "Saturates" is not a bound either: the 15-model majority vote reaches 86.3% against a mean of 83.4%.
  - The negative cascade result is for one recipe: frozen detector, 75% margin, label-only fusion, no joint training and no detector-score calibration. Only one detector family (YOLO26) gets three seeds with a classifier. Soft fusion with the detector's full class distribution was not run (5.2).
  - A negative result is useful, but the conclusions ("investigate the confusion as a labeling and imaging problem") go beyond what the experiments establish. The abstract hedges this.
- **M5. Inconsistency and conflation in the per-class and per-size reporting.**
  - Table 1 "Recall <32 / 32-48 / >=48" are species-correct recalls (a weighted average of the three bins reproduces the species accuracy of about 82%), not detection recall (97.7% overall). Section 5.1 and the heading "H1: detection improves" present "recall below 32 px 44% to 81%" as detection repair. These are identification-plus-localization gains. This confuses the central H1/H2 split (detection vs identification).
  - Section 5.4: "MF ... errors are non-species and missed objects (5.3%, identical in both)". Table 2 gives MF recall of 86.0% (E0) and 87.1% (cascade), so about 14% are not correct, and the two systems differ. Figure 1 (L3) shows 9% MF to INS/NOISE/MAR among localized objects. The 5.3% cannot be reconciled with these.
  - The text for the cascade omits that MF precision falls 82.9 to 74.9 and MF F1 falls 3.9 pp (Table 2), which is the largest drop after MD. It is presented as "MF unchanged".
  - Section 4.4 says paired bootstrap plus McNemar with Holm. Section 5.2 says BCa bootstrap with a sign-flip permutation test (Du, 2025). Which procedure underlies which number is unclear.
- **M6. Limited baselines and scope for a "comparative evaluation".**
  - There is no comparison with any published sticky-trap species detector or with other modern detector families beyond RT-DETR-l (cascade only) and YOLO11m (one seed, RQ3).
  - Tiny-object variants use default hyperparameters. NWD's C was measured on an earlier data version. Several confounds sit in the variants: batch size 4 for 2560 px, and different GPUs (B200, L40S, H100).
  - A FP32 fusion result that changes by +1.0 pp between GPUs is called "hardware-dependent noise" (5.2). This means run-to-run noise exceeds some claimed effects, and CIs over images do not cover it.
  - For RQ3 the paper correctly says "no gain detected here". Its conclusion that aerial methods "do not transfer" (RQ3 framing, contribution 4) is stronger than three seeds with default hyperparameters allow.

### Minor
- **m1.** Contribution 2 says "five classifier backbones" while the introduction and 5.2 say five classifiers on four backbones (CE and parts share ConvNeXt-T). Make this consistent.
- **m2.** Count of "eight variants": Table 3 has 8 non-control rows including YOLO11m (one seed). The text says "eight tiny-object variants ... three seeds each" and "none of eight variants ... with three seeds".
- **m3.** Cost "30-67x" does not match the quoted 9-11 ms vs 353-573 ms (about 32-64x). Check the source.
- **m4.** Section 5.2: "Why: a ceiling at identification" mixes TEST and VAL (87-88% on VAL) and several one-seed numbers (the 25/50/100% oracle runs are single seed, 83.0/81.7/82.6, a ±1 pp spread that is itself not interpretable).
- **m5.** Related work: Gutierrez et al. (2019) is cited as a sticky-trap whitefly benchmark, but the reference title is about tomato plants and scouting robots. Verify the citation. "GCD (arXiv:2510.27649)" is cited inline and absent from the references. "Insect-1M" in the text and "InsectFoundation" in the reference list should be unified. Several key references are arXiv preprints (Brock 2026, Cheng 2018, Chachuła 2022, Du 2025).
- **m6.** The novelty claim "physical targeted collection ... we did not find in the insect-monitoring literature" is a statement about absence, supported only by "we reviewed". Soften it or support it with a search description.
- **m7.** Section 4.2: ArcFace and SupCon heads are single-seed and dismissed with a single-seed VAL cascade check (75.7 vs 76.4). The comparison is not informative.
- **m8.** The hypotheses H1 and H2 in 5.1 read as pre-registered, but the paper elsewhere says earlier campaigns consulted TEST. State clearly whether they were formulated before the results.
- **m9.** The "species accuracy" metric is micro recall over species instances, not classification accuracy. Naming it "accuracy" is confusing next to the oracle accuracy and the "among localized" accuracy.
- **m10.** The abstract is dense (many numbers and qualifiers). The introduction repeats results at length and the contributions list is long (4 items with sub-results), which makes the paper hard to scan.
- **m11.** Portuguese terms and class codes (MD/MV/MC/MF) are used throughout with English gloss only at first mention. A small table of species would help.
- **m12.** Tables 1 and 3 are very wide for a one-column layout; their legibility cannot be verified from my reading. Figure 1 shows percentages in small cells; the same applies.

## 4. Questions for the authors
1. Can a plate-grouped re-split (even by acquisition session or date) be done, and how much do the main numbers change?
2. Does an untargeted, quantity-matched baseline exist? Can you rerun the equal-size control without copy-paste images and restricted to real images, to match L2?
3. Why does Table 1 report species-correct recall by size next to "detection recall" 97.7%? What is the class-agnostic recall below 32 px?
4. Which statistical procedure (BCa bootstrap with sign-flip, or percentile bootstrap with McNemar) produced each interval and p-value in 5.2 and Table 3? Is the 10,000-resample setting the same?
5. How can MF errors be 5.3% (5.4) when Table 2 shows MF recall of 86-87% and Figure 1 shows 9% to non-species?
6. What do experts say about 100 randomly drawn MD/MV/MC crops? Even a small relabelling study would separate ambiguity from label noise.
7. Would a cascade with joint fine-tuning or a score-calibrated fusion change the sign for any detector?

## 5. Ratings per criterion (1-5)

| Criterion | Score | Label | Justification |
|---|---|---|---|
| Relevance | 4 | Accept | Applied detection and fine-grained recognition of tiny objects fit VISAPP's Recognition and Detection area. |
| Originality | 3 | Borderline | Mostly a careful negative-result study, with the physical-collection idea and the oracle analysis as the distinct pieces; the methods themselves are known. |
| Technical Quality | 3 | Borderline | Good statistical hygiene, but 73-image test split, possible plate leakage, confounded ladder and a weak equal-size control. |
| Significance | 3 | Borderline | Useful to practitioners and honest, but single facility, the central question (what limits identification) is left open. |
| Presentation | 3 | Borderline | Clear and calibrated, but dense, repetitive in the introduction, and some inconsistencies (M5, m1-m3). |

## 6. Overall rating and justification
**Overall: 3 (Borderline).** The paper makes an honest, mostly well-hedged contribution: a data ladder, a controlled cascade-vs-detector comparison, an oracle ceiling analysis and a negative transfer result. Its value is as a careful negative-result and diagnostic study on real industrial data. Against that, the evidence base is small (73 test images, one facility, three seeds, possible plate leakage). The RQ1 control cannot support its interpretation (M2, M3), the "ceiling" is not shown to be informational (M4), and a few reporting inconsistencies need fixing (M5). I would lean to accept if the authors fix M5 and m1-m3 and tone down the quantity-vs-targeting and "ceiling" readings. A positive rebuttal on the leakage question (M1) would move me to 4. A weak rebuttal leaves it at 3 or below.

## 7. Reviewer confidence
**4 of 5.** I am familiar with detection, tiny-object methods and evaluation protocols. I did not see the rendered figures, and I cannot judge the entomological ambiguity of the species.

## 8. Closed questions
- The abstract and introduction are adequate? **Partial.** They are accurate and hedged but overloaded with numbers.
- More experimental results are needed? **Yes.** A plate-grouped split or leakage control, an untargeted equal-size arm, and some expert relabelling.
- A comparative evaluation is needed? **Partial.** More detector families and the published sticky-trap detectors, plus cascade variants (joint training, calibrated fusion).
- The paper needs a more critical discussion? **Partial.** Limitations are good; discuss the confounds of the ladder (M3) and the meaning of "ceiling" (M4).
- The figures are adequate? **Not verifiable** (not rendered). Figure 1 needs checking for one-column legibility; tables 1 and 3 are wide.
- The conclusions and future work are convincing? **Partial.** Conclusions are hedged, but the practitioner recommendation goes beyond the evidence.
- The references are up-to-date and appropriate? **Partial.** Mostly recent and relevant; check Gutierrez et al. (2019), the missing GCD entry and the reliance on preprints.
- The formatting needs adjustment? **Partial.** The paper is at the 12-page limit; wide tables.
- The English needs improvement? **Partial.** Good, but long sentences and heavy qualifiers in the abstract and introduction.

## 9. Comments to the authors
Thank you for a candid paper. The calibrated language, the explicit statement that VAL is not held out and the reporting of unfavourable results are exactly what this literature needs.
- The priorities are M1 and M2: a leakage control, and an equal-size baseline that is both real-only and untargeted.
- Next is M5: reconcile the MF numbers, separate detection recall from species-correct recall in Table 1, and state which statistical procedure underlies each interval.
- Then soften the "ceiling" and "targeting vs quantity" wording to match the evidence, and fix the minor points.
- Shortening the introduction's results recap would free room for the expert-relabelling discussion.

## 10. Confidential comments to the Program Chair
- **Anonymity:** the prior paper (PACBB 2026) is cited as "Anonymous" with full title and venue and described in detail. A reviewer could identify the group quickly. This looks borderline rather than a violation.
- **Overlap and self-plagiarism:** the dataset lineage overlaps with the earlier paper, and the paper explicitly tests its "irreducible" claim. I saw no evidence of text reuse, but overlap with the prior PACBB paper should be checked with the INSTICC tools.
- **AI declaration:** no declaration appears in the acknowledgements. I cannot say whether one is required.
- **Decision:** borderline, leaning accept if the authors respond on M1, M2 and M5.

## 11. Additional comments
- Typos and format: none found in the extraction (a few split words from the PDF text layer only). Render the PDF and check Figure 1 and Tables 1 and 3 at print size.
- In 5.3 the paragraph on sliced inference compares mAP@50 under a different evaluator. Consider moving it to an appendix or dropping the 512 px resize baseline, which the text itself calls unfair.
- Unify the terminology: "accuracy", "recall" and "species-correct" refer to related but different quantities.

## Overall
**Overall rating: 3 (Borderline). Confidence: 4.** Fatal: 0. Major: 6. Minor: 12.
