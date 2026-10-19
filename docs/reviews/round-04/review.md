# Review: VISAPP 2027 (regular paper, double-blind)

Paper: "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps"
Profile: visapp.md (criteria and closed questions official; numeric scale and form layout inferred). Numbers spot-checked against docs/PAPER_FACTS.md (examples: 127/3,470 suspect crops, latencies 353-573 ms, NWD+RFLA 0.729 +/- 0.075, delta -0.088 [-0.127,-0.049], p=0.002): consistent. Pages viewed as images: p07 (Table 1, Fig. 1 start). Tables 1 and 2 are very small in single column.

## 1. Paper Summary
Applied study of species-level fly detection on 1920x1080 industrial sticky-trap images (7 classes). Three questions: (RQ1) a ladder of five cumulative training sets, species accuracy 56.1% to 82.2% (3 seeds), framed as an association; (RQ2) a frozen YOLO26 detector plus crop classifier (ConvNeXt-T, DINOv2/v3, BioCLIP 2) never beats the detector alone (15 configurations, macro-F1 -1.1 to -5.1 pp), with a ground-truth-crop oracle saturating at ~83%; (RQ3) aerial tiny-object recipes (NWD, RFLA, GCD, Wise-IoU, P2, 2560 px, SAHI, TTA, WBF) give no gain over a control. The paper is candid about confounds and negative results.

## 2. Strengths
- Unusually honest: confounds (size/quality), absence of a held-out set, single-facility data, default hyperparameters and unstable seeds are all stated.
- Per-species precision/recall at a fixed operating point is the right primary metric for monitoring; the protocol (matching, thresholds) is precisely specified and mAP is secondary.
- Multi-seed reporting, paired bootstrap, Holm correction, same-campaign control for RQ3.
- The GT-crop oracle plus error decomposition (not localized / wrong species / correct) cleanly separates localization from identification.
- Negative results on a real industrial regime are useful to practitioners; the analysis of why NWD/SAHI do not transfer (native resolution, task-aligned assigner) is plausible and labelled as untested.

## 3. Weaknesses

### Fatal
None. The weaknesses below are serious but the paper's hedged claims keep it above the fatal threshold.

### Major
M1. Evaluation set is too small and possibly leaky, and there is no held-out data. TEST = 73 images (7 crowded, 6 insect-only plates), VAL = 52. No plate ID, so images from the same plate may straddle splits (Sec. 3.3, Limitations); 4.7% of TEST crops reach thumbnail similarity >=0.98 with a training crop (Sec. 5.2). Model/head choices were made on VAL, TEST was consulted in earlier campaigns, and the paper itself says "no never-consulted hold-out exists". Every headline number (82.2%, ~83% ceiling, -1.1 to -5.1 pp) inherits this. A plate-level re-split (or at least a documented plate grouping by the handwritten labels/capture dates) is needed to make the claims credible.

M2. The cascade is a weak instance, so "decoupling does not help" and "the ceiling is identification" are over-generalized. Stage 1 is frozen and never retrained; fusion is label-only (hard replacement); crops use a single 75% margin at 384 px; the seven-way head must also re-decide NOISE/INS/MAR, which the error analysis (Sec. 5.4: MD->non-species +8.0 pp) shows is the actual source of loss. The BG variant and a late fusion using the detector's true class scores were not run (the latter explicitly, Sec. 5.2), and the fusion test uses an approximated detector distribution. The abstract/title frame this as a lever result; it is a result about one recipe. A cascade that only reclassifies among the four species (keeping the detector's non-species decision) would be the fair test and is missing.

M3. The "limited information in the images" explanation for the ~83% ceiling is not supported beyond non-exclusion. Label noise, annotation ambiguity and crop recipe are acknowledged as inseparable, no expert re-annotation of even a small subset was done, and the learning curve is single-seed and flat within noise. Since the 15 classifiers share the same labels and crops, agreement of errors (4.9% wrong for all; 60% right for all) is equally what label noise produces. This is the paper's central claim contra the prior work ("irreducible"), yet the paper's own data (MV->MC 14.8% to 5.4%) show the confusion is movable. The framing sets up a straw man (the earlier claim, on different data/classes, is "not like-for-like" by the authors' own admission), and then refutes it only partially (H2 "only partly supported").

M4. RQ1 cannot support its causal reading and the practitioner recommendation. The ladder confounds quantity, quality, domain (white-paper photographs are 42% of L2+ORG and then removed), and copy-paste; L0 has only 103 images, so 56% vs 82% is largely "more data". The equal-size control is the key missing experiment and is deferred to future work, yet the abstract says gains are "most visible where the collections aim" and the conclusion recommends prioritizing species-pure/crowded/insect-only plates "over model complexity". The latter comparison (data vs. model complexity) is not tested anywhere (model complexity was never varied against data on the ladder). Insect-only plates: 6 TEST plates, 0 vs 75.7 FP is descriptive. H1/H2 are described as pre-registered, but the "H2: identification does not move" hypothesis is then partly refuted; the framing of the hypotheses is not neutral.

M5. RQ3 conclusion "no gain" has little power and uses an untuned design. Three seeds, seed std 1.2-3.4 pp, CI half-width ~1.7 pp: only differences of >2-3 pp are detectable, which the authors concede. Default hyperparameters, C fixed at 44 px measured on an older data version, NWD weight w=0.5 untuned, RQ3 trained on a different GPU (L40S) with its own control, one seed for YOLO11m, no tile-trained SAHI model (SAHI is evaluated with a model trained at 1920 and compared on a non-species evaluator, so it is uninformative about SAHI). The title/abstract statements ("what does not transfer") exceed "no difference detected with n=3". Also, the paper itself argues the recipes are mismatched to this regime (native resolution, task-aligned assigner), which makes the negative result unsurprising and of narrower interest than presented; more informative would be a small-object detector with static assignment or a non-NMS-free baseline to verify the proposed mechanism.

M6. Novelty and breadth are thin for a regular full paper. The methodological components are all off-the-shelf (YOLO26, ConvNeXt/DINO/BioCLIP, NWD/RFLA/SAHI); the contribution is empirical, on one proprietary dataset that will not be released (code only), so the key RQ1 result is not reproducible. No comparison with other detectors beyond YOLO26l/RT-DETR-l/YOLO11m for the cascade and no non-YOLO fine-grained detection approach; no comparison to a one-stage detector with a stronger classification head or class-balanced losses, which would be the natural single-stage competitor to the cascade. The three RQs read as three loosely coupled mini-studies, and the novelty claim "physical targeted collection ... not found in the literature" is not a tested contribution (see M4).

### Minor
m1. Table 1 and Table 2 are set in tiny type and are barely legible in one column (p. 7); Table 1 mixes detection and confusion rows with no visual grouping beyond italics. Fig. 1 is a pair of confusion matrices with small cell text; combine or enlarge. Captions are long but acceptable.
m2. Inconsistent controls: L3 in Table 1 is 82.2+/-1.6 while E0 in Table 2 is 81.7+/-1.2 (explained as a different campaign/GPU, but the reader sees two E0s on the same TEST; better to note that GPU nondeterminism alone yields 0.5 pp).
m3. Statistical reporting is heterogeneous: Sec. 4.4 describes McNemar + image-level bootstrap, Sec. 5.2 uses BCa bootstrap + sign-flip permutation (a 2025 arXiv preprint), Table 2 uses Holm over eight comparisons, Table 1 has no multiplicity correction. The 15-configuration claim ("sign is the same in all") is not independent (shared test images, crossed seeds, nine pairs from 3x3).
m4. "Species accuracy" is described as equal to micro recall (Sec. 5.3), but the paper's headline metric (accuracy) should be named recall consistently, and precision (important for monitoring FP counts) is hardly reported in the main text.
m5. Page/format: the extracted PDF has a 13th page carrying two lines of the reference list, which breaks the 12-page limit for a Full paper; verify against the SCITEPRESS template. Abstract (~290 words) is long and dense in numbers.
m6. Anonymity: self-citation is third person (acceptable), but "Code will be released", "mosca doméstica", the DS-F2 name and the described single industrial facility may identify the group; the "partner facility" is not named (good). Acknowledgements say "Anonymous during review" (fine). No AI-use statement is visible; VISAPP requires a declaration if AI-generated text was used (not verifiable).
m7. Reference quality: several key citations are arXiv preprints (Brock et al. 2026 arXiv:2607.18700, Du 2025, Chachula 2022, Cheng 2018, GCD arXiv:2510.27649 cited as a bare arXiv number, Jain 2024 workshop paper); Beery 2019 is an arXiv paper for a central cascade claim. Some tiny-object work after 2022 (e.g., newer small-object detectors and the YOLO-family tiny variants) is missing; the "Insect-1M" label in text vs. "Insect-Foundation" in the reference list is inconsistent. Heavy reliance on Ultralytics release notes ("Jocher et al., 2026") as a citation for the architecture.
m8. Sec. 3.2 (iv) "Graduated plates: the printed grid is on all plates; what changed is the quality..." is awkward and not tied to any measurable difference; dataset statistics (per-class training counts, plate counts per version) are missing, so the class imbalance behind per-species metrics cannot be assessed.
m9. The statement "FP on insect-only plates vanish" (abstract) rests on six plates and an FP definition that includes INS-matched species predictions; give counts and per-plate variance.
m10. Latency (30-67x) is indicative only and from a shared node; with 353-573 ms per image for 1920x1080 traps this is arguably still acceptable for monitoring, so the cost argument is weak; report per-instance cost and throughput.
m11. Writing: dense and hedged to the point of repetition (the "association", "no detectable gain", "cannot be separated" caveats appear in abstract, intro, results, conclusion, and limitations); consider trimming to free space for the missing experiments. Minor typos/typography: stray space before colon ("metrics :", Sec. 5.1), "(which (Anonymous, 2026) called irreducible", "pclf pw det" formula is not readable in extracted text (check math rendering).

## 4. Questions for the authors
1. Can you re-split by plate (or by capture date/handwritten label) and report the cascade-vs-detector and ladder results on a plate-disjoint test set?
2. What happens if the cascade only relabels among the four species and leaves INS/NOISE/MAR decisions to the detector? Does the -2.7 pp loss disappear?
3. Does an expert re-annotation of even 100-200 MD/MV/MC crops change the oracle ceiling? Without it, how do you justify "limited information in the images" over label noise?
4. Can you subsample L3 to the size of L1/L2 to give an equal-size control for at least one rung?
5. Why was SAHI evaluated on a model trained at 1920 rather than a tile-trained one, and why on a different evaluator?
6. How sensitive are NWD/GCD results to C and w, given the stated mismatch with the original setting?

## 5. Ratings per criterion (1-5; 3 = Borderline)
| Criterion | Score | Label | Justification |
|---|---|---|---|
| Relevance | 4 | Accept | Detection/recognition application in VISAPP scope (Recognition and Detection). |
| Originality | 2 | Reject | Off-the-shelf components; the novelty is empirical and the data collection idea is not tested against a control. |
| Technical Quality | 2 | Reject | Tiny TEST, possible plate leakage, no hold-out, weak cascade variant, underpowered RQ3 (M1-M5). |
| Significance | 3 | Borderline | Useful negative evidence for practitioners, but generality beyond one facility/proprietary data is untested. |
| Presentation | 3 | Borderline | Clear structure and honest framing, but dense hedging, tiny tables, possible page overflow. |

## 6. Overall rating and justification
Overall: 3 (Borderline), leaning towards reject at the lower end of this band. The paper is carefully hedged and honest, and the oracle/decomposition analysis is good practice. But the central conclusions (identification is the ceiling; aerial recipes do not transfer; data collections are the lever) each depend on a 73-image possibly leaky test set, a single cascade recipe, three seeds with default hyperparameters, and an RQ1 confound whose controlling experiment is left as future work. What would move me to Accept: plate-disjoint evaluation, an equal-size control, a species-only relabelling cascade, and some expert relabelling of the confusable pairs. Accept is plausible after rebuttal only if the authors can supply at least M1 and M2 evidence.

## 7. Reviewer confidence
3 of 5 (competent in detection and fine-grained recognition; no access to the data, and the entomological ambiguity claim cannot be verified from the manuscript).

## 8. Closed questions
- The abstract and introduction are adequate? Partial. Clear motivation, but the abstract is overlong and number-dense, and the framing against the earlier "irreducible" claim is contrived.
- More experimental results are needed? Yes: plate-disjoint test, equal-size control, species-only cascade, tile-trained SAHI, expert relabelling subset, hyperparameter sensitivity for NWD.
- A comparative evaluation is needed? Partial. More single-stage competitors with improved classification heads / class-balancing, and other fine-grained insect pipelines, would strengthen the comparison.
- The paper needs a more critical discussion? Partial. Limitations are exhaustive, but the discussion does not draw the consequence that the headline claims are therefore tentative; the conclusions still make recommendations.
- The figures are adequate? Partial. Only one figure (two confusion matrices); the tables, which carry the results, are too small in single column; no qualitative examples of the confusable MD/MV/MC crops or of failure cases, which would be essential to judge the "limited information" claim.
- The conclusions and future work are convincing? Partial. The hedged conclusions follow the data; the practitioner recommendation (data over model complexity) is not supported by an experiment.
- The references are up-to-date and appropriate? Partial. Reasonably recent and relevant but over-reliant on preprints, one prior self-citation central to the argument, and some small-object detection literature is missing.
- The formatting needs adjustment? Yes: verify the 12-page limit (a 13th page appears to hold reference lines), enlarge tables.
- The English needs improvement? Partial. Understandable and mostly correct but repetitive hedging and some awkward constructions; a light edit would help.

## 9. Comments to the authors
Your honesty about confounds is welcome, and the oracle-based decomposition is a good way to separate localization from identification. To make the paper publishable at its stated ambition: (1) give a plate-disjoint evaluation; (2) add the equal-size control or drop causal language and the "data over model complexity" recommendation; (3) test a cascade that cannot move proposals to non-species classes and uses the detector's true class scores; (4) provide at least a small expert audit of MD/MV/MC labels, otherwise soften the "limited information" interpretation; (5) tile-train for SAHI and sweep C/w for NWD before concluding on transfer; (6) shorten the hedging and enlarge Tables 1-2; add qualitative crops.

## 10. Confidential comments to the Program Chair
No evidence of plagiarism or dual submission; the paper builds on an earlier PACBB-style paper by presumably the same group (cited in third person, "Anonymous"). The overlap should be checked by the chair for substantial reuse (the dataset lineage is shared but the manuscript states no image overlap and new classes). Anonymity appears preserved. Possible 13-page overflow should be verified. My honest recommendation: borderline, a rebuttal that supplies a plate-disjoint evaluation would tip it; otherwise reject, because the central claims rest on a very small, possibly leaky test set.

## 11. Additional comments
See minors m1-m11. Also check the rendering of the late-fusion formula (p_clf p_det^w) and the stray spaces before colons.
