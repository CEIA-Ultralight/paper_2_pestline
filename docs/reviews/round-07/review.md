# Review - VISAPP 2027 (round 07)

**Paper:** "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps" (anonymous submission, 12 pp. incl. references).
**Profile:** visapp.md; criteria and closed questions are [oficial]. The numeric scale and the exact form layout are [inferido] (1-5 convention).
**Reading note:** the PDF was read in full via text extraction (pdftoppm was unavailable). Figure 1 (confusion matrices) was read from its embedded numbers, not visually inspected. Layout and legibility of the figure are therefore not verifiable from my side.

## 1. Paper Summary
The paper asks what limits species-level reading of sticky traps for four fly species (plus INS, NOISE and MAR). It studies three levers on the authors' DS-F2 dataset (1,738 training, 52 validation and 73 test images at 1920x1080):
- **RQ1, data:** a ladder of five cumulative training sets, with an equal-size random-subsample control. Species accuracy of a YOLO26m detector rises from 56.1% to 82.2%. Random subsamples of the final set match the 244- and 869-image rungs, but not the 103-image rung.
- **RQ2, identification:** a frozen-detector, detect-then-classify cascade with five crop classifiers, three detectors and a ground-truth-crop oracle. The cascade never beats the detector. Oracle accuracy saturates near 83%, and the errors are confident and concentrated in the MD/MV/MC family.
- **RQ3, tiny-object transfer:** eight aerial-imagery variants (NWD, RFLA, GCD, Wise-IoU, P2, 2560 px, YOLO11m) plus SAHI, TTA and WBF. None gives a detectable gain.

The contribution is largely a set of carefully hedged negative results.

## 2. Strengths
- Honest, well-hedged reporting. Examples: VAL is declared non-held-out (Sec. 4.1, 4.4); "no gain detected" is distinguished from equivalence (Sec. 5.3); the author-flagged plate-leakage risk, the unstable NWD+RFLA seeds and the confounds of the ladder are all stated.
- The oracle design is informative. The sign of the cascade effect is consistent in all 15 configurations. A background-class head, abstention, soft fusion, crop-margin and training-fraction controls rule out several cheap explanations (Sec. 5.2).
- Multi-seed reporting with paired, image-level bootstrap intervals. The equal-size control is a good idea, and it tempers the data claim (Sec. 5.1).
- Species-level, real-field, full-resolution data is a real application niche. The evaluation protocol is explicit (fixed threshold 0.25, IoU 0.5, greedy matching, FP rules).
- Implicit but valuable observation: the detector's own accuracy among localized objects (82.2/97.7, about 84%) already equals the oracle ceiling (about 83%). This explains the cascade failure well, but it is never stated.

## 3. Weaknesses

### Fatal
None. The weaknesses below are serious but do not invalidate the paper.

### Major
- **M1. Test set far too small and possibly leaky (Sec. 3.3, 4.4, 6).**
  - TEST has 73 images and VAL has 52. The dataset has no plate identifier, so the authors cannot exclude that images of the same plate fall on both sides of the split.
  - 4.7% of TEST crops are near-duplicates of some training crop (Sec. 5.2).
  - TEST was consulted in earlier campaigns, and "no never-consulted hold-out exists" (Sec. 4.4).
  - Almost every conclusion is a difference of 1-3 pp whose intervals touch zero. The "saturation at 83%" and the "no cascade gain" are statements about this one split and one facility.
  - What would change my view: a plate-grouped re-split or a fresh, never-used test set from a second facility or session.
- **M2. The RQ1 data conclusion is weakly identified and partly contradictory (Sec. 5.1).**
  - The ladder confounds size, quality, version and domain. The equal-size control has high variance (L1-size subsamples span 0.69-0.84), L2-size subsamples use only two seeds in Sec. 5.1 but "three seeds each" in the result text and abstract (inconsistent), and the L0 control contradicts the "mostly quantity" reading (19 pp gap).
  - The "localized gains" (small objects, crowded plates, insect-only plates) are explicitly not tested against the control.
  - The paper therefore supports neither "targeting matters" nor "quantity matters" cleanly.
  - The headline "56% to 82%" in the abstract and Contribution 1 is largely a size curve. Framing "targeted physical collections" as the contribution (Sec. 1, RQ1, Sec. 3.2) is stronger than the evidence.
  - H1/H2 are stated as hypotheses but were evidently written after the data were seen, and H2 is "only partly supported".
- **M3. The RQ2 causal diagnosis is not established (Sec. 5.2, Conclusions).**
  - The paper concedes that ambiguity, label noise and the crop recipe cannot be separated, yet the title and the "identification ceiling" framing (Contribution 3, "limited information in the images") suggest an inherent limit.
  - The decisive experiments are missing: expert relabelling of the confusable pairs, a human-performance estimate, and classifiers trained on higher-context or native-pixel crops. Crops of about 110 px are upsampled to 384, so a resolution or crop-recipe explanation is plausible.
  - The cascade failure is established only for one recipe: frozen detector, label-only fusion, no joint training, and a 7-way head that must reject false positives. The paper itself calls its fusion (soft, late) exploratory, and the late-fusion gain flips from +1.3 pp (B200) to +0.3 pp (H100), which is explained away as "hardware-dependent noise" without evidence.
  - The generalization of "does not improve in any of 15 configurations" is therefore overstated in the abstract.
- **M4. RQ3 is an under-tuned null result (Sec. 4.3, 5.3).**
  - Default hyperparameters were used (for example C=44 px fixed from an earlier data version, w=0.5). The L40S campaign has its own baseline. YOLO11m has one seed. SAHI is evaluated with a non-tile-trained model on a different evaluator (mAP at confidence 0.0), and the comparison with the resized frame is called unfair by the authors themselves.
  - NWD+RFLA diverged (nan), and this seed instability is a possible implementation or learning-rate issue that was "not retried", so it is uninformative.
  - Per-seed standard deviations (up to 3.4 pp) and CI half-widths of about 1.7 pp are as large as plausible effects.
  - The explanations of "why aerial recipes do not transfer" are explicitly untested.
  - The contribution claims a "negative transfer result" (Contribution 4) which the evidence does not rise to. Honestly framed, it is "no detectable effect at this power".
- **M5. Metrics and reporting do not match the stated primary metrics (Sec. 4.4, 5, Tables).**
  - The paper declares per-species precision, recall and F1 primary. The main tables report only accuracy (equal to micro recall), and per-species precision appears nowhere in the paper.
  - For monitoring, the operational quantity is per-plate count error per species, which is not reported.
  - There is no table for the central RQ2 results. The cascade deltas, oracle accuracies, 15-classifier agreement and error shifts are scattered in prose (Sec. 5.2, 5.4), with no per-species recall table for E0 versus cascade, so numbers cannot be cross-checked against a table.
  - Several numbers are non-comparable across sections: macro-F1 (79.5), accuracy (82.2 and 81.7 for different E0 runs on different GPUs), mAP@50 (84.2 under a different evaluator).
- **M6. Statistical protocol is not decision-complete (Sec. 4.4, 5).**
  - The 9-pair detector x classifier cross is not independent, and the cross is treated as 9 replicates when summarizing.
  - Per-pair McNemar tests are acknowledged to understate uncertainty. They are then "not summarized" because they vary from 0.005 to 0.51.
  - YOLO26l has a single seed, so the "15 configurations" are not all multi-seed.
  - Selection on VAL (52 images, 856 instances) of the head, fusion weight and checkpoint is a noisy basis for the choices on which the cascade conclusion rests.

### Minor
- **m1.** Abstract is long and heavily hedged, and it reads as a list of caveats. Phrases such as "so there we detect no advantage of targeting over quantity, but the earliest lot is 19 pp worse" are hard to parse. The title promises "decoupling" as a method, but the paper reports that decoupling does not help.
- **m2.** Missing citations for components used: ConvNeXt, DINOv2, DINOv3, BioCLIP 2, RT-DETR, Wise-IoU v3, YOLO11, GCD (only an arXiv id in text), McNemar and Holm. "Insect-1M" is cited as Nguyen et al. but the reference is titled InsectFoundation. Several references are arXiv preprints, and I cannot verify the 2026-2027 ones (Brock et al., arXiv:2607.18700; Du, arXiv:2511.19794).
- **m3.** Related work claims to be the first at several points ("we did not find ..."). The non-existence claims on physical targeted collection and on sticky-trap multi-seed evaluation are not backed by a stated search protocol. Reviews of the cascade literature on insects (for example fine-grained insect classification on crops) are thin.
- **m4.** Sec. 3.2 "four verified ways" is not verifiable from the manuscript. Item (iv), graduated plates, ends up as "the presence of the grid has not changed", so it is not a collection effect. The dataset description does not state how many plates exist or how many images are in each class per split, so a reader cannot judge dependence.
- **m5.** L2 +/- ORG removal at L3 mixes copy-paste with removing white-paper photographs (acknowledged), so the L2+ORG to L3 step is uninterpretable. The Table 1 CI for L2 (-4.8, -0.1) excludes zero, but the text says "at most small", yet "monotonic".
- **m6.** Confusion rates in Fig. 1 and Table 1 are pooled point estimates "without intervals" (stated), though conclusions (MV->MC 14.8 to 5.4) hang on them. Class counts per cell are about 60-100 per row in a pool of three seeds, which is not 3 independent samples.
- **m7.** Sec. 4.3 says RQ3 used L40S while the cascade used B200, with different baselines (81.7 vs 82.2 vs 79.5 F1). The reader must track four "E0" numbers.
- **m8.** Cost is quoted as 30-67x on a shared node and is "indicative only"; CPU or edge deployment cost is not considered although the motivation is deployment ("edge deployment" is future work).
- **m9.** "Conclusions" practical advice ("collecting more annotated plates matters more than model complexity") rests on M2 and contradicts the paper's own statement that targeting versus quantity remains open.
- **m10.** The wording "The answer is mixed and we treat it as a result" and several first-person hedges are informal. English is generally good but the text is dense, with long parenthetical sentences (for example Sec. 5.1 "Threats").
- **m11.** Anonymization is largely clean (Anonymous author, affiliation, e-mail; the prior work is cited in third person as "Anonymous, 2026"; no URLs; acknowledgements "Anonymous during review"). However, the prior paper is described in unusual detail (v4 data, 0.476 to 0.608 mAP, 6 classes, PACBB venue appears in the reference), which makes it identifiable to readers who know the venue. The self-reference carries a lot of weight in Sec. 1 and 2.4, and the "irreducible" claim attributed to it is the paper's motivating framing, which is also a risk for double-blind. There is no AI-use declaration in Acknowledgements (required by the profile if AI-generated text was used; not verifiable).
- **m12.** Figure 1 is the only figure. There are no qualitative examples of crops, failure cases or the plates, so readers cannot judge the "limited information" claim visually. The dataset section has no example images. (Figure legibility itself: not verifiable.)

## 4. Questions for the authors
1. Can you re-split by plate (or at least by acquisition session) and report whether the ladder and the cascade conclusions persist? What is the number of distinct plates in TEST?
2. What is the human accuracy (expert) on a subset of MD/MV/MC crops at the same resolution, and what do experts say about the 127 flagged crops? Is "limited information" distinguishable from label noise even in principle with your data?
3. Is the cascade failure a consequence of the frozen detector and 7-way head, or would a classifier trained on detector proposals with a score-aware fusion or jointly fine-tuned detector help? Why was the product/soft fusion with the full class distribution not run, given that it is the standard cascade design?
4. For RQ1, can you run the equal-size control at the L0 size with three more seeds and at L2 with three seeds, and test the localized gains (small objects, insect-only plates) against it?
5. For RQ3, did you tune the learning rate or the NWD constant C for any variant, and why was the NWD+RFLA divergence not retried with a different seed or LR?
6. What is the per-species precision and the per-plate count error of E0 on TEST?

## 5. Ratings per criterion (1-5)
| Criterion | Rating | Label | Justification |
|---|---|---|---|
| Relevance | 4 | Accept | Detection and recognition of tiny, fine-grained objects in a real application fits VISAPP's Recognition and Detection area well. |
| Originality | 2 | Reject | The methods are all existing (YOLO, NWD/RFLA, standard cascades, crop classifiers); the novelty lies in a domain study and null results, with only modest data-collection novelty. |
| Technical Quality | 3 | Borderline | Careful, honest protocol and multi-seed statistics, but a 73-image test with possible plate leakage, no hold-out and confounded RQ1 control limit what is established. |
| Significance | 2 | Reject | Null results on a single facility and split, with untested explanations and no positive recipe; useful as a cautionary note but of limited impact. |
| Presentation | 3 | Borderline | Clear structure and protocol, but central RQ2 results are in prose with no table, the abstract is over-hedged, citations are missing and visuals are sparse. |

## 6. Overall rating and Justification
**Overall: 3 (Borderline).**
The paper is honest, mostly well-executed within its limits, and in scope. It reports a useful empirical observation: the detector's species accuracy among localized objects already equals the ceiling of five classifiers on ground-truth crops. However:
- It has no positive method.
- All headline conclusions rest on a 73-image test split that may leak by plate and that has been consulted before.
- The RQ1 data claim and the RQ3 transfer claim are weakly identified (M2, M4).
- The RQ2 explanation remains speculative (M3).
- Key results lack tables and precision/count metrics (M5).

I would accept if the authors can show robustness under a plate-grouped split or an independent test set and present the RQ2 results in a table with precision. I would reject if the leakage cannot be excluded. The decisive question to the authors is Q1.

## 7. Reviewer confidence
**4 / 5.** I am familiar with small-object detection and cascade evaluation. I could not verify the underlying data, the figure rendering, or the 2026-2027 arXiv references.

## 8. Closed questions
- The abstract and introduction are adequate? **Partial.** Informative, but the abstract is over-hedged and long, and the intro lists three levers and four contributions that mix findings and caveats.
- More experimental results are needed? **Yes.** Plate-grouped or independent test, expert relabelling, stronger controls for RQ1, tuned RQ3 variants, joint or score-aware cascades.
- A comparative evaluation is needed? **Partial.** Detectors and classifiers are compared; no comparison to other published insect or sticky-trap pipelines, nor to a larger detector family or tile-trained SAHI.
- The paper needs a more critical discussion? **Partial.** Limitations are honest, but conclusions (practitioner advice, "identification ceiling") go beyond the evidence.
- The figures are adequate? **No/partial.** One figure only; no qualitative examples; no table for RQ2 (figure legibility not verifiable).
- The conclusions and future work are convincing? **Partial.** The conclusions are properly hedged but the practical recommendation conflicts with the open targeting question.
- The references are up-to-date and appropriate? **Partial.** Mostly recent, but missing citations for all model components and several 2025-2027 preprints cannot be verified.
- The formatting needs adjustment? **No.** 12 pages including references, SCITEPRESS-style layout; no TODO marker appears in the PDF (a `\TODO` macro is defined in main.tex but not used in the body). Character count is within the 10k-50k range by my estimate (not exactly verified).
- The English needs improvement? **Partial.** Correct but dense and hedge-heavy; the abstract needs simplification.

## 9. Comments to the authors
See sections 3-4. Priorities: (1) establish or bound plate-level leakage; (2) move RQ2 numbers into a table (with precision and per-plate counts); (3) soften title, abstract and Contribution 3/4 wording to what is shown (single facility, one recipe, default hyperparameters); (4) rerun the equal-size controls with matched seeds; (5) add missing citations; (6) add qualitative figures of crops and failure cases.

## 10. Confidential comments to the Program Chair
- No plagiarism or dual-submission concern from the manuscript itself. The paper builds explicitly on an earlier same-group paper (PACBB) that is cited in the third person; the overlap appears to be limited to the problem setting and part of the data lineage. Please verify that the earlier paper's data (v4) and text do not overlap substantially.
- Anonymity is largely respected, but the earlier-work description is detailed enough to be identifying to readers who know the venue.
- Borderline; I lean slightly toward weak accept only if the plate-leakage question is answered credibly in the rebuttal, otherwise weak reject.

## 11. Additional comments
- Sec. 3.3 and 4.4 repeat the split description verbatim; consolidate.
- "Detector V AL" in the extracted text is a typesetting artifact of small caps: check the "VAL" spacing in the PDF.
- Table 1 header mixes "(H1: expected to improve)" with results; consider a separate caption sentence.
- Table 2 reports p values with Holm correction over eight comparisons, but Table 2 has nine rows (YOLO11m is included in the "eight" count in the text): clarify.
- "Insect-1M" is cited in related work whereas the reference lists InsectFoundation.
- Define "MAR" in Sec. 3.3 (it appears without expansion).

## Self-check
Every major has a location and evidence. The overall rating (3) is consistent with zero fatals and six majors, none of which falsifies the main findings but several of which limit them. Items I could not verify (figure rendering, dataset, 2026-27 arXiv references, exact character count, AI declaration) are marked as such.
