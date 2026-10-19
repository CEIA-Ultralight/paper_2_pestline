# VISAPP 2027 review (independent, round 03)

Conference: VISAPP 2027, regular paper (double-blind). Paper: "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps".
Profile: criteria and closed questions [official]; numeric scale and form layout [inferred] (1-5 convention). Read: text of pp. 1-12 plus the page image of p. 6 (Fig. 1 and Tables 1-2 checked from the extracted text only; I did not visually verify figure legibility). Numbers cross-checked against docs/PAPER_FACTS.md sections 10-13; the ones I checked match (Table 2 values, cascade deltas, oracle agreement 60.0/4.9/86.3, confident errors 69.4, label issues 127/3470).

## 1. Paper Summary
Species-level detection of four fly species (plus INS/NOISE/MAR) on 1920x1080 industrial sticky-trap images (DS-F2, 73 test images, 1,210 species instances). Three levers: (i) a five-step cumulative data ladder (targeted physical collections), species accuracy 56.1 -> 82.2% with YOLO26m; (ii) a frozen-detector + crop-classifier cascade with five backbones, which never beats the single stage (macro-F1 -1.1 to -5.1 pp), with a ground-truth-crop ceiling near 83%; (iii) eight aerial tiny-object recipes (NWD, RFLA, GCD, WIoU, P2, 2560 px, SAHI/TTA/WBF), none better than control. The conclusion: spend effort on data, not on architecture.

## 2. Strengths
- Honest, well-framed negative results with multi-seed (3), paired bootstrap, Holm correction, and explicit statements of "no difference detected, not equivalence".
- Strong protocol hygiene: fixed operating point, same harness for all pipelines, frozen fusion policy, explicit disclosure that VAL selected best.pt and that no untouched hold-out exists.
- Oracle (GT-crop) decomposition cleanly separates localization from identification; localization is 97.7%, so the diagnosis is credible.
- Concrete practitioner message; sizeable realistic dataset at native resolution; numbers traceable to the project fact sheet.
- Limitations section is candid.

## 3. Weaknesses
### Fatal
None. The conclusions are mostly appropriately hedged, but the issues below keep the paper at the borderline.

### Major
M1. **Test set is small and possibly leaky; no plate identifier.** 73 test images / 1,210 instances (Sec. 3.3, 4.4, Limitations). Images of the same plate may sit on both sides of the split, and 4.7% of TEST crops reach thumbnail similarity >= 0.98 to a training crop (Sec. 5.2). Since each batch comes from the same facility and lab plates, in-distribution leakage could inflate every number, and can distort the comparison between arms (e.g. the ladder: later batches are closer to the test distribution). "Species accuracy 82%" is therefore not a trustworthy absolute number. A plate-level or batch-level split (e.g. hold out the v9 batch) would change my assessment.

M2. **The data ladder confounds quantity, quality, domain and label changes, so "targeted collections drive the gain" is not established.** Sec. 5.1 itself says an equal-size control was not run; arms differ in size by 17x (103 -> 1,738 images), L0 comes from the earliest lot with a different protocol, 62 images were re-annotated, white-paper images make 42% of L2+ORG, and copy-paste enters only in L3. The monotonic rise could be explained by "more images of the test distribution". H1's "gain follows the targets" is descriptive (the size-bin and INS-only results are consistent with the claim but the other bins also rise). The abstract/intro/conclusion nonetheless state "data is a strong lever" and give advice ("species-pure, crowded and insect-only plates before model complexity") that the design cannot attribute to species-purity/crowding/insect-only specifically. Missing: equal-size subsample control and per-collection ablation (leave-one-collection-out from L3).

M3. **The "identification ceiling" claim is interpretive, not demonstrated; the ceiling is also measured under the same selection regime.** The ~83% GT-crop accuracy is flat across backbones, but all share one recipe (crop expansion 75%, 384 px, same augmentations, same labels), one dataset and one set of labels. Flat learning curve (single seed, 83.0/81.7/82.6) and 4.9% "wrong for all" are compatible with label noise, with a crop recipe that discards context (75% context at 384 px from a ~44 px object means heavily upsampled crops), or with limited resolution of the object itself (a 44 px fly), not only with "information in the species". No expert relabelling, no human-accuracy reference, no resolution/crop-size ablation, and the oracle accuracy is non-monotonic with size (85.7% <24 px vs 69.4% 24-32 px; confounded with class). The abstract and conclusion ("points to the information available") go further than a discussion point should; the intro's claim to refute the prior work's "irreducible" confusion is also weakened by the acknowledged non-like-for-like comparison (different data, resolution, classes) and by the fact that confusion fell only partly (MC->MV 11.7%).

M4. **Cascade evaluation is narrow and the negative result may be partly an artifact of the design.** Late fusion uses an approximated detector distribution (Sec. 5.2, "this is an approximation"; true scores not run); the classifier is not trained on detector proposals in the main configuration (the BG variant is tried, but with one head/seed set), the 7-way head re-labels FPs (MF precision -9 pp), detector stays frozen, and there is no cascade that uses the detector's own class probability properly, nor a joint/hard-example-trained classifier, nor a classifier with larger context. Hence "decoupling does not improve" is established for this specific recipe only. The statistical support is also weak per configuration: most individual intervals include zero (e.g. parts, BioCLIP 2, DINOv2, DINOv3), McNemar p varies 0.005-0.51 across seeds, and the "sign is the same in all 15 configurations" argument ignores that the nine detector-classifier pairs are not independent (3 detector seeds) and that YOLO26l has a single seed. The abstract's "does not improve in any of 15 configurations" is accurate but the stronger reading (decoupling is useless) is not supported; also the 5 classifier head comparison (ArcFace, SupCon) is reported as "trained once and not reported", which is unreported work.

M5. **RQ3 conclusions are under-powered and un-tuned; mixed claims.** Table 2: seed std of 1-3 pp, CIs of about +-1.7 pp, three seeds, defaults without per-variant tuning (C=44 px measured on an earlier data version, w fixed at 0.5), and a single control from a different GPU/campaign. NWD+RFLA (-8.8 pp) is driven by one early-stopped and one diverged (nan) seed: reported in the abstract as "8.8 points worse" without the caveat that this is an instability, not evidence about NWD+RFLA. The explanation ("why the aerial recipes do not transfer") is explicitly untested (acknowledged). SAHI is evaluated with a detector trained at 1920 and 512 px tiles, a mismatch (a tile-trained model was not run), and under a different evaluator (mAP@50, conf 0.0), so the comparison with the species harness is indirect. The title-level message "what does not transfer" is thus "default-hyperparameter variants show no detected gain". The abstract does say "no gain detectable with three seeds", which is appropriate, but the contributions list reads "negative transfer result".

M6. **Novelty is modest and the contribution type is unclear for a VISAPP audience.** Every component (YOLO fine-tuning, crop classifier on foundation backbones, NWD/RFLA/SAHI) is standard; the dataset is not released ("release of data and code is undecided"), so neither the central data result nor the negative results can be reproduced or built upon; and the data ladder (the main positive result) is not replicable without the data. For a paper whose value is empirical evidence, an undecided release is a serious weakness for reproducibility and significance. Open-sourcing the test annotations (even without images) or the exact configs would help.

### Minor
m1. Consistency: the abstract and conclusion say "four backbones" saturate near 83%, while the intro/Sec. 5.2 use five stage-2 backbones and "fifteen classifiers" (82.7-83.8% for all). Harmonise.
m2. Table 1 E0/L3 accuracy 82.2 +- 1.6 (B200) vs Table 2 E0 81.7 +- 1.2 (L40S) for supposedly the same recipe/data; the 0.5 pp gap is within noise but should be stated at the first point where both appear (it is only implied in Sec. 5.3).
m3. Species-false-positive rows (75.7, 48.3, 0, 1.7, 0) in Table 1 have no dispersion and rely on 6 plates; the text quotes ranges (66-87, 43-57) that Table 1 does not show.
m4. Recall <32 px: Table 1 uses "recall" (class-aware?), caption says "accuracy" for the first row and "recall" for others; clarify that all are species-correct recall, and whether 44.2% is class-agnostic.
m5. The MD recall (0.64-0.69, std up to 0.11) is the weakest species and drives macro-F1; it deserves a per-species table in the main paper (per-species P/R only appear in prose, even though per-species metrics are declared "primary").
m6. Latency (9-11 ms vs 353-573 ms) is "indicative only"; 30-67x is quoted as a headline number in contributions. Crop classification on 300 proposals at conf floor 0.001 is a design choice that inflates cascade cost; a top-k proposal cap would be fairer.
m7. Statistics: McNemar on instances and bootstrap over images are both used, plus Holm over 36 pairs, plus a BCa sign-flip test citing a 2025 arXiv preprint (Du, 2025); the pipeline of tests is heavy relative to 73 images. Report one primary test per claim.
m8. Related work: several references are preprints (Brock et al., 2026; Chachula et al., 2022; Cheng et al., 2018; Du, 2025) and Insect-1M is cited as "Nguyen et al., 2024" though the reference is titled Insect-Foundation. No comparison with recent fine-grained detection or hierarchical/taxonomic approaches, nor with other tiny-object detectors beyond the aerial set (e.g. dedicated small-object YOLO variants or high-res tiling training). The related-work claim "we did not find ... in the insect-monitoring literature" (physical targeted collection) is an absence claim and should be softened.
m9. Anonymity: appropriate (prior work in third person as "Anonymous (2026)", acknowledgements anonymised). The Portuguese class mnemonics and "handwritten label" are harmless; the self-citation is central to the framing, so reviewers can infer common authorship; acceptable under the rules. Ensure no repository link or GitHub reference to Ultralytics fork remains.
m10. Presentation: Sections 4-5 are very dense (long single-paragraph protocol text; "Cost." sentence ends with a semicolon, p. 6; "MC->MV ... 11.7%,;" typo in 5.1 with ",;"; "Threats and what the result does not show" heading is awkward). Table 1 spills text size; Fig. 1 shows two 4x5 confusion matrices that are legible only if enlarged, and the right-hand INS/NOISE/MAR block is unlabelled in the cells. Hypothesis statement H2 is labelled "partly supported", but the section title still says "identification does not move".
m11. AI-use declaration: the acknowledgements say only "Anonymous during review"; the VISAPP guidelines require disclosure of AI-generated text. State whether it applies.
m12. The cascade is described as "30-67x the inference time" but the Intro says "at 30-67x" without GPU/CPU context. Also the title promises "species-level monitoring" while the data come from one facility and the evaluation from one test split.
m13. Section 5.4 Error analysis is a single paragraph of numbers with no table; the key trade (MD->non-species +8.0 pp) deserves a table.

## 4. Questions for the authors
1. Can you give a plate-/session-level split (or at least report results on the latest batch v9, held out from training) to rule out same-plate leakage? What fraction of test images share a session/day with training images?
2. Can you run the equal-size subsample control (e.g. L3 subsampled to 244, 869 images) and a leave-one-collection-out ablation from L3, to separate quantity from the targeted collections?
3. Did you test a classifier at larger crop size or with a smaller context margin than 75%/384 px? Does the 83% ceiling move?
4. Would an expert relabelling of even a 200-instance subset of the MD/MV/MC confusions be feasible before camera-ready, to separate ambiguity from noise?
5. For NWD+RFLA, what does it give if the diverged/early-stopped seeds are rerun (or with lower LR/warm-up)? How should the reader read the 8.8 pp figure?
6. Will test annotations, configs and the dataset be released? If not, what exactly can others reproduce?
7. Why was a tile-trained SAHI model and a fusion with true detector scores not run, given that the paper's conclusion about both is negative?

## 5. Ratings per criterion (1-5)
| Criterion | Score | Label | Justification |
|---|---|---|---|
| Relevance | 4 | Accept | Recognition/Detection with an application-driven small-object, fine-grained setting fits VISAPP. |
| Originality | 2 | Reject-leaning | Standard methods; novelty is empirical (negative transfer, physical collections) and not reproducible. |
| Technical Quality | 3 | Borderline | Rigorous statistics and honest reporting, but small test, leakage risk, confounded ladder, narrow cascade design, untuned variants. |
| Significance | 3 | Borderline | Useful for practitioners, but findings are specific to one facility and unreleased data. |
| Presentation | 3 | Borderline | Clear structure and good hedging; very dense, a few typos/inconsistencies, results in prose rather than tables. |

## 6. Overall rating and justification
**Overall: 3 (Borderline).** A carefully reported, honest empirical study whose most interesting claims (data collections are the lever; identification has an information ceiling; aerial recipes do not transfer) are each weakened by design limits that the authors themselves list: a 73-image test with possible plate leakage, a confounded ladder, a single-recipe cascade, and untuned, unstable tiny-object variants. I would accept if the authors added the equal-size/leave-one-out control and a leakage-resistant split, and would reject if the reading of the ceiling and the practitioner advice stay as stated. A rebuttal answering Q1 and Q2 would decide it.

## 7. Reviewer confidence
**3/5.** I know detection and evaluation methodology well, fine-grained insect recognition less so; I could not verify the data, code or dataset (not released).

## 8. Closed questions
- The abstract and introduction are adequate? **Yes (partly).** Clear and hedged, but the abstract says "four backbones" (vs five) and its causal phrasing ("pointing to the information available") is stronger than the evidence.
- More experimental results are needed? **Yes.** Equal-size/leave-one-out ladder controls, leakage-robust split, larger-crop/other cascade designs, tuned tiny-object variants.
- A comparative evaluation is needed? **Partly.** Missing other small-object/high-res strategies and a non-YOLO strong baseline across the whole study (RT-DETR only for RQ2; YOLO11m single seed in RQ3).
- The paper needs a more critical discussion? **Partly.** Limitations are good, but the conclusions and abstract do not carry them through.
- The figures are adequate? **Partly.** Only one figure (confusion matrices); no figure for the ladder or the cascade pipeline; the tables carry the load.
- The conclusions and future work are convincing? **Partly.** Data conclusion plausible but confounded; ceiling and non-transfer conclusions are hypotheses.
- The references are up-to-date and appropriate? **Partly.** Mostly appropriate and recent; several preprints; light on recent fine-grained/small-object detection work.
- The formatting needs adjustment? **Minor.** 12 pages including references as required; dense text blocks and a few typographical glitches.
- The English needs improvement? **Minor.** Understandable, a few awkward phrasings and typos (",;").

## 9. Comments to the authors
See Weaknesses and Questions. The most valuable improvement for the camera-ready is to soften the practitioner advice and the "ceiling" interpretation to match the evidence, to move per-species precision/recall (declared primary) into a main-text table, to harmonise the backbone counts, and to resolve the release decision.

## 10. Confidential comments to the Program Chair
No anonymity breach, plagiarism or dual-submission concern found; the prior work is cited as "Anonymous (2026)" in the third person. The paper is borderline: honest and careful, but with limited novelty and unresolved leakage/confound issues. Page limit appears respected (12 pages including references). I did not check the SCITEPRESS template files, and the AI-use declaration is not visible.

## 11. Additional comments
Typos: "11.7%,;" (Sec. 5.1); stray semicolon at the end of "Cost." (Sec. 4.4); "Threats and what the result does not show" heading; "Aﬀiliation" ligature okay. Fig. 1 right-hand block label should say "INS/NOISE/MAR" in the caption and the axes consistently.
