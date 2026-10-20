# Review: VISAPP 2027 (Regular paper, Full, 12 pp.)

**Paper:** "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps" (anonymous submission)
**Profile:** `visapp.md`. Criteria and closed questions are [oficial]. The numeric scale (1-5), the layout of the PRIMORIS form and the reviewer-confidence scale are [inferido]; they are not public.
**Basis of the review:** the full text of the 12 pages was extracted with pypdf and read, including tables, captions, limitations and references. Figure 1 was read only through its extracted cell values and caption; I could not inspect the rendering, so legibility is not verifiable from my reading. The numbers were cross-checked against `docs/PAPER_FACTS.md`, sections 10 and 11. They agree, with the exceptions noted below.

---

## 1. Paper Summary

The paper studies species-level detection of four fly species (MD, MV, MC, MF) plus three non-target classes (INS, NOISE, MAR) on 1920x1080 photographs of industrial sticky traps. The mean object is about 44 px. It asks which of three "levers" limits performance.

- **RQ1 (data):** a ladder of five cumulative training sets (103 to 1,738 images) with a YOLO26m detector. Species accuracy on a 73-image TEST split rises from 56.1% to 82.2% (3 seeds). Random subsamples of the final set match the 244- and 869-image rungs. A random 103-image subsample is 19 pp better than the earliest lot.
- **RQ2 (identification):** a frozen-detector, detect-then-classify cascade with five crop classifiers (ConvNeXt-T CE and parts head, DINOv2, DINOv3, BioCLIP 2) and three detectors. It never beats the detector alone (macro-F1 -1.1 to -5.1 pp, 15 configurations). Classifiers applied to ground-truth crops (the "oracle") reach 82.7-83.8% species accuracy. The authors read this as an identification ceiling, concentrated in the MD/MV/MC family.
- **RQ3 (transfer):** eight aerial tiny-object variants (NWD in loss or assigner, NWD+RFLA, GCD, Wise-IoU v3, P2 head, 2560 px, YOLO11m), plus sliced inference, TTA and WBF. None gives a detectable gain over a same-campaign control. NWD+RFLA is 8.8 pp worse.

The paper is candid about its limits: single facility, 73 test images, VAL not held out, no expert relabelling, undocumented plate-level split.

## 2. Strengths

1. **Honest reporting of negative results.** The paper reports "no gain detected", not "no gain". It labels VAL as not held-out, states the fusion policy was frozen before TEST, and discusses seed instability (NWD+RFLA diverging). This is uncommon and valuable.
2. **Multi-seed designs with paired tests and bootstrap over images.** This is rare in the sticky-trap literature. Reporting L3 vs L0/L1 with image-level CIs (Table 1) and per-seed deltas (Table 3) is good practice.
3. **The equal-size random-subsample control (Sec. 5.1).** It stops the data ladder from being read as proof of targeting. The finding that targeted sets are indistinguishable from random subsets at 244 and 869 images is informative.
4. **Per-species, fixed-operating-point metrics (Sec. 4.4)** with mAP as secondary. This fits a monitoring application. The decomposition into "not localized / wrong species / correct" is useful.
5. **Real, native-resolution industrial data** (1920x1080, never downscaled). The tiny-object question is posed for a modern NMS-free detector, which is a legitimate gap.
6. **The quantitative cost (30-67x latency).** The practical consequence is clear.

## 3. Weaknesses

### Fatal
None. The central empirical statements are mostly hedged and mostly traceable to the tables.

### Major

**M1. The "identification ceiling" is a ceiling of crop classifiers, and the data in the paper contradict reading it as an information limit.**
- Sec. 5.2 reports that crop classifiers on perfect GT boxes reach 82.7-83.8%.
- Table 1 gives the detector's own species accuracy as 82.2% overall, with detection recall of 97.7%. Among localized objects, the detector therefore identifies correctly in about 84.1% of cases (82.2/97.7). That is at or above every crop classifier's oracle accuracy, and the detector is evaluated on proposals, not on perfect crops.
- So the detector, which sees full-image context, already sits on the "ceiling". It follows that a cascade whose stage 2 is trained only on GT crops cannot beat it. The negative result then says "a crop-only ConvNeXt/DINO/BioCLIP at 384 px does not out-identify YOLO26m", not "identification is saturated".
- The text never compares oracle accuracy to the detector's accuracy among localized objects, although Sec. 4.4 defines that metric.
- Consequences for the claims:
  - Contribution 3 and the abstract ("ground-truth-crop accuracy saturates near 83% for five classifiers") do not show a ceiling for the task. They show a plateau for one family of inputs (a 2.5x context crop upsampled from about 110 px to 384).
  - The paper says residual errors are "shared by all classifiers" (Intro; Conclusions). Its own numbers say 4.9% of instances are wrong for all 15 classifiers while individual error is about 16.6%. Only about 30% of errors are shared. Majority vote reaches 86.3% against a mean of 83.4%, and 35.1% of instances are mixed. That is not saturation: ensembling beats the "ceiling" by about 3 pp.
  - The 15 oracle classifiers are 5 types x 3 seeds. Seeds of the same architecture are not independent evidence of a ceiling.
- Stage 2 is trained on annotated boxes and applied to detector proposals (jittered boxes, false positives). The only mitigation is the BG head. Training stage 2 on detector proposals, or with box jitter, is not tried, and the train/inference distribution mismatch is not discussed.
- The 25/50/100% data curve is flat (83.0/81.7/82.6, one seed, differences within seed noise), which is compatible with a plateau of this model family, not with ambiguity specifically.
- *What would change my view:* (a) report the detector's accuracy-among-localized next to the oracle in one table; (b) train stage 2 on proposals; (c) recompute the "ceiling" claims and the "shared by all classifiers" language from the agreement statistics. If crop classifiers are no better than the detector among localized objects, say so and drop the ceiling framing.

**M2. RQ1 cannot attribute the 56% to 82% rise to the data strategy, and the controls are themselves confounded.**
- The abstract and Intro lead with "species accuracy rises from 56% to 82%", but the ladder varies quantity, quality, species mix, domain, and the inclusion of copy-paste and white-paper photographs together. The authors acknowledge this.
- The equal-size control draws random subsets from L3, which is the most test-like distribution. VAL and TEST "grew with the batches" (Sec. 3.1) and are drawn from the same recent batches. L0 is the earliest lot (mixed-species plates, v5.1, different acquisition). A random L3 subsample being 19 pp above L0 is therefore at least as consistent with temporal and domain shift between older lots and the test distribution as with "label quality or version".
- Half of L3 is copy-paste (869 of 1,738), so random L3 subsamples contain synthetic images in unknown proportion. Equal-size does not mean equal composition. The paper does not report the composition of the subsamples.
- The "equal-size control" says nothing about the localized claims (small objects, insect-only plates, crowded plates), which the paper itself says were not tested against the control (end of Sec. 5.1). Yet the Conclusions say "largest gains on objects below 32 px and insect-only plates". The insect-only effect (66-87 FP to 0) coincides with adding INS-only plates, and is a deterministic consequence of having no such training examples, not evidence about "targeted collection".
- Plate-level leakage cannot be excluded (Sec. 3.3 and Limitations: no plate identifier, split rule undocumented). RQ1 is the worst-affected RQ: species-pure crowded plates (up to 77 specimens per image, 373 training images from v9) are exactly the setting where adjacent photographs of the same plate could straddle splits. The copy-paste source crops are described as "real annotated crops"; the paper does not state that they are restricted to training images. Leakage-check evidence is not in the manuscript beyond a 4.7% crop-similarity figure.
- The conclusion "more annotated plates helped most" is not supported by the control either. At 103 images the control reverses and says the ladder is not simply quantity.
- *What would change my view:* a plate-id (or acquisition-date) split audit; a control built from each rung's version mix rather than L3; confirmation that copy-paste sources and subsamples exclude VAL/TEST images.

**M3. The RQ3 test has little power to detect what tiny-object methods are designed to fix, and several conclusions go beyond it.**
- All RQ3 comparisons use species accuracy at IoU>=0.5 and score>=0.25 on 1,210 instances. The paper itself states E0 localizes 97.7% of instances, so a box-similarity loss (NWD) "has little to repair" at that threshold. NWD and RFLA exist to improve IoU-sensitive assignment and regression at tiny scale, which would appear at higher IoU (mAP@50:95, or recall at IoU 0.75). mAP@50:95 is not reported for any RQ3 variant in the manuscript, only for E0 and TTA. The evaluation design is thus nearly insensitive to the hypothesised mechanism.
- "Recall by object size" in Table 3 is species-correct recall. It is identical in definition to species accuracy, and the sizes sum to 1,210. The text then argues that "the smallest objects are limited by identification" (end of Sec. 5.3). That requires a class-agnostic recall by size for E0, which is not shown (class-agnostic recall is only reported overall: 97.7%). Table 1 also labels "Recall <32 px" ambiguously under "Detection", while the 44% to 81% values are probably species-correct recall. In that case "recall on objects below 32 px goes from 44% to 81%" conflates detection and identification, which is the very distinction the paper is built on.
- The "why aerial recipes do not transfer" paragraph is speculation (the authors say "untested explanations"). The 44 px = about five stride-8 cells argument is plausible but unchecked.
- Settings are confounded: imgsz 2560 uses batch 4 vs 8 for E0; P2 changes the architecture and the training dynamics. The comparison is "default recipe" for variants, whereas tuning was done only for NWD (and it collapsed at w=1.0). Hardware changes the control: the same late fusion gives +1.3 pp on B200 and +0.3 pp on H100, and "detector alone differs by 0.6 pp", so hardware-level noise in the same range as the effects under discussion.
- Sliced inference is evaluated only as inference-time tiling of a model trained on full frames, with a different evaluator (conf 0.0, mAP@50), and the whole-frame-at-512 baseline is acknowledged as unfair. "A tile-trained model was not run" removes the main variant of SAHI (fine-tuning). The abstract still lists "sliced inference" under methods that "give no gain".
- The abstract and Contribution 4 say "negative transfer result"; the data show "no detectable gain at +/-1.7 pp for 3 seeds with seed std up to 3.4 pp". The hedge is present in the text, but the contribution list and title of the section read stronger.

**M4. Statistical resolution and the "never improves in any of 15 configurations" claim.**
- The test set is 73 images with 1,210 instances clustered within images (and probably within plates, see M2). Image-level bootstrap is used, but not plate-level. For ConvNeXt-T parts, DINOv2, DINOv3 and BioCLIP 2 the 95% intervals on E0 include zero (e.g. -1.4 [-3.7, +1.3]; -1.1 [-2.7, +0.8]). The abstract and Contribution 2 state flatly that the cascade "does not beat the detector in any of 15 configurations" and "never improves". That is an observation of a consistent sign, not a demonstrated harm. The configurations share the same 73 images, the same detector weights across classifiers, and non-independent seed pairs (nine crossed pairs), so counting "15 of 15 same sign" overstates independent evidence.
- Several statistical procedures are stacked (McNemar exact, image-level percentile/BCa bootstrap, sign-flip permutation, Holm over 9 and over 36 pairs), and the paper says McNemar p "varies between seeds (0.005-0.51) so we do not summarize it". Holm is said to be applied "over the eight comparisons" in Sec. 5.3 though Table 3 lists nine rows. Table 3 shows P2 with CI [-5.0, -0.4] and Holm p=0.14, imgsz 2560 [-6.1, -1.1] p=0.057 and NWD-loss [-4.5, -0.5] p=0.084: the test behind the p values (per-seed McNemar combined how?) is not specified for Table 3.
- Model selection touched VAL for: detector checkpoint, classifier checkpoint, the choice of "parts" over CE, fusion weight, and "frozen before TEST" policies; TEST "was consulted in earlier campaigns" and an earlier ensemble was built after seeing TEST. The paper discloses this, but it means the adaptive-analysis risk is non-zero for TEST-level claims (the 75% crop margin, imgsz 384 and the tiny-object variants were chosen in earlier campaigns; not verifiable from the manuscript).

**M5. External validity and reproducibility: one facility, one non-released dataset, no second benchmark.**
- Every conclusion rests on one facility, one camera protocol, and one 73-image test split. The images "will not" be released. The central empirical object (the ladder) therefore cannot be reproduced independently, as the authors acknowledge. For a paper whose contribution is largely a diagnostic about *what limits performance*, there is no public or second-domain dataset on which the negative cascade/NWD/SAHI results can be checked (a trap-based public detection set, or another trap colour/setting; the paper cites such an issue, Ong and Høye 2025, but does not test it).
- The tested cascade is one narrow design: frozen detector, label-only (or product) replacement, one crop margin family, GT-crop training. It is not joint training, nor hierarchical (family then species), nor higher-resolution re-imaging. The abstract hedges "with our recipe", but the title, the framing of the Intro ("what limits ... which lever moves that limit") and the Conclusions ("Identification did not move with the classifier") generalize beyond that recipe.
- Reproducibility details present: Ultralytics version, optimiser settings, seeds, GPU. Missing from the manuscript: the augmentation settings used for the detector (mosaic and copy-paste off or on?), the exact copy-paste procedure, how the 62 re-annotated images were handled across rungs, and the number and class balance of crops used per classifier.

### Minor

1. **Numbers and consistency.**
   - Contribution 1 is hard to parse: "from L1 to L2 the rise is mostly quantity, but not for L0" should be rewritten.
   - Sec. 4.2 says only CE and parts heads are replicated over three seeds, but Sec. 5.2 reports nine-pair crossings for DINOv2, DINOv3 and BioCLIP 2 and "15 oracle classifiers (five types, three seeds each)". The seed protocol for the foundation-model heads is not described.
   - The cost sentence quotes "30-67x" in Contribution 2; the Sec. 5.2 ranges (9-11 ms vs 353-573 ms) give about 32-64x, so the two do not map one-to-one.
   - Table 2 and Sec. 5.4 attribute the fall of MF precision (82.9 to 74.9) to "other species relabelled MF", while the text says MF is never confused with another species. Presumably the mass comes from NOISE/INS or from other species relabelled MF; the sentence should say which.
2. **Figure 1 and the MD to MV error.** The text focuses on MC<->MV, but Figure 1 shows MD to MV at about 27% in both L0 and L3, the single largest error and unchanged by data. The data-effect story ("part of what the earlier pipeline treated as a fixed limit moves with the training data") rests on MV to MC (14.8% to 5.4%) while MC to MV (18.5% to 11.7%) and MD to MV barely move. These rates are pooled point estimates without intervals, which the paper notes.
3. **Hypotheses H1/H2.** They are stated as predicted outcomes, yet H2 is "only partly supported". Describe them as exploratory expectations if they were not pre-registered.
4. **Missing citations.** DINOv2, DINOv3, BioCLIP 2, YOLO11, ArcFace, SupCon and the GCD method (cited by arXiv number in running text) have no bibliography entries; Insect-1M and the cited reference title (InsectFoundation) differ. Reference currency for fine-grained recognition, vision foundation models and hierarchical or family-level classification is thin; the related work on fine-grained recognition is a single sentence on open-set insect recognition.
5. **Novelty claims "we did not find".** Several (physical targeted collection, multi-seed sticky-trap studies, untested transfer to insect traps) are scoped as "in the literature we reviewed". That is acceptable, but the contribution should not lean on absence-of-evidence. The closest methodology (Jain et al. 2024) is discussed but not compared experimentally.
6. **Metrics.** The same quantity is called "species accuracy", "micro recall" and "recall" in different places (Sec. 5.3: "equal to micro recall"). Use one term. Latency is "indicative only" and varies between jobs; give per-image variance or a median.
7. **Operating point.** Everything is at score>=0.25 and IoU>=0.5. A precision-recall curve or an operating-point sweep would show whether the cascade's lower precision (74.9 vs 82.9 for MF) is threshold-specific. The 0.25 threshold was fixed once, not tuned per pipeline, which is fair, but the cascade "label-only" keeps the detector's score, which is not calibrated for the classifier's new label.
8. **Counts.** Training and TEST statistics are given in several places (Secs. 3.3, 4.4); state them once. The "7 crowded / 6 insect-only plates" bins are descriptive with very small n; Table 1 should say so in the rows.
9. **Copy-paste detail.** Table 1 mixes in "L2+ORG to L3 mixes copy-paste with removing the white-paper photographs (42% of L2+ORG)". The L3 vs L2+ORG comparison therefore cannot be interpreted; consider a rung L2+copy-paste.
10. **Presentation.** The abstract is dense and carries 15+ numeric and methodological qualifiers; the Intro repeats the abstract nearly in full. Section 5.2 has several unnumbered sub-topics (BG head, learning curve, fusion, margin, cost) packed in single paragraphs; a table would help (margins, fusion variants, hardware). "V AL" spacing artifacts appear in the extracted text (probably a font/macro issue). The text refers to hardware changes (B200, H100, L40S) across the same table family; add a short table of which experiment ran where.
11. **Ethics and AI declaration.** Acknowledgements say only "Anonymous during review". The VISAPP policy requires disclosure of AI-generated text; I cannot tell from the manuscript whether any applies (not verifiable).

## 4. Questions for the authors

1. What is the detector's species accuracy among localized objects (82.2/97.7 = about 84%) compared with the oracle accuracy of 82.7-83.8%? If the detector is already as accurate on proposals as crop classifiers are on perfect crops, in what sense is 83% an identification ceiling rather than a classifier-family plateau?
2. Were any stage-2 classifiers trained on detector proposals (with IoU-jittered boxes and the detector's false positives) other than the BG head? Would a stage 2 trained on E0's own out-of-fold proposals change the cascade delta?
3. Of the 35.1% mixed-agreement instances, how many does the majority vote fix? Why does the 86.3% majority vote not undermine the saturation claim?
4. How were the VAL/TEST images assigned relative to plates and acquisition sessions? Can you provide a plate or session identifier (even coarse) and re-run the bootstrap clustered by plate? Were copy-paste source crops ever taken from VAL/TEST images?
5. What is the composition (original vs copy-paste, per version) of the random equal-size subsamples? Is a version-matched control possible, to separate "older lots" from "less data"?
6. Can you report class-agnostic recall by size and mAP@50:95 (or recall at IoU 0.75) for every RQ3 variant, to test whether NWD/RFLA/P2 improve localization quality that species accuracy at IoU 0.5 cannot see?
7. Which hyperparameters (crop margin 75%, imgsz 384, parts head, w and C of NWD) were fixed after looking at TEST in earlier campaigns?
8. Can the cascade and RQ3 findings be checked on any public dataset (even a different trap type)?

## 5. Ratings per criterion (scale 1-5 [inferido]; 1 Strong Reject, 2 Reject, 3 Borderline, 4 Accept, 5 Strong Accept)

| Criterion | Rating | Justification |
|---|---|---|
| Relevance | 4 (Accept) | Application of detection and fine-grained recognition to a real monitoring problem, squarely in VISAPP's Recognition and Detection area. |
| Originality | 3 (Borderline) | Question and diagnostics are useful, but the methods are standard (YOLO, ConvNeXt/DINO crops, NWD/SAHI); novelty is mostly the combination and negative results. |
| Technical Quality | 3 (Borderline) | Careful and honest, but the central "ceiling" inference is undermined by the detector's own accuracy, and RQ1/RQ3 designs have confounds (M1-M3). |
| Significance | 2 (Reject) | One facility, 73 test images, private data, narrow cascade recipe; a "no effect detected" outcome with limited power gives practitioners little to act on. |
| Presentation | 3 (Borderline) | Clear structure and honest hedging, but dense, repetitive, with several muddled sentences and missing citations. |

## 6. Overall rating and Justification

**Overall: 3 (Borderline).**

This is a conscientious, well-hedged empirical study, and I value that it reports negative results with seeds and paired intervals. I would not argue for rejection on rigour alone. However, the most quotable conclusion, that identification is capped near 83% by information in the images, is not supported, and its own numbers (detector 84% among localized; majority vote 86.3%; 4.9% shared errors) point to a classifier-specific plateau. RQ1's headline rise is largely confounded, and RQ3's evaluation is insensitive to the mechanism being tested. On a single private dataset with a 73-image TEST, these weaknesses limit significance.

I would move to **4 (Accept)** if the authors, in a rebuttal or revision: (1) compare the oracle with the detector's accuracy among localized objects and reword the ceiling claims accordingly; (2) show that the plate split and copy-paste sources do not leak; (3) add IoU-sensitive and class-agnostic size-stratified metrics for RQ3. I would move to **2** if the plate-level leakage cannot be excluded for the RQ1 rungs.

Counts: **fatal 0, major 5, minor 11.**

## 7. Reviewer confidence

**4 / 5** (confident; I am familiar with the detection and fine-grained recognition literature but not a specialist in entomological monitoring). Items I could not verify from the manuscript: Figure 1 rendering, plate-level split, copy-paste source provenance, AI-use declaration, tuning history of earlier campaigns.

## 8. Closed questions

| Question | Answer |
|---|---|
| The abstract and introduction are adequate? | Partial. Adequate scope, but overly dense; the claims on a "ceiling" and "shared errors" go beyond the evidence. |
| More experimental results are needed? | Yes. Detector vs oracle comparison; proposals-trained stage 2; IoU-sensitive/class-agnostic RQ3 metrics; version-matched RQ1 control; a second dataset. |
| A comparative evaluation is needed? | Partial. Other cascade designs (joint training, family-level) and a public benchmark; the YOLO family comparison is otherwise adequate. |
| The paper needs a more critical discussion? | Partial. Limitations are candid; the interpretation of the oracle needs a critical revision (M1). |
| The figures are adequate? | Partial / not fully verifiable. Figure 1 values extracted but rendering not inspected; only one figure for a 12-page results paper; tables carry most results. |
| The conclusions and future work are convincing? | Partial. Practitioner takeaways are sensible but rest on confounded and underpowered evidence. |
| The references are up-to-date and appropriate? | Partial. Recent for detection; missing citations for models used (DINOv2/v3, BioCLIP 2, GCD, YOLO11, ArcFace, SupCon); thin on fine-grained recognition. |
| The formatting needs adjustment? | Minor. Within 12 pages, template looks right; spacing artifacts ("V AL") and dense paragraphs. Anonymity looks preserved. |
| The English needs improvement? | Minor. Generally clear; several overlong, hedge-stacked sentences (Contribution 1, abstract). |

## 9. Comments to the authors

Thank you for a careful and candid study. The most valuable parts are the equal-size control and the explicit refusal to turn "no detected effect" into "no effect". My main requests, in order of importance:

1. Put the detector's accuracy among localized objects next to the oracle accuracy, and rewrite "identification ceiling" and "errors shared by all classifiers" in line with the 4.9% / 16.6% / 86.3% figures. As written, the cascade result may be a consequence of the stage-2 design (GT-crop training, crop-only input) rather than of the data.
2. Document how VAL/TEST relate to plates and acquisition sessions, and show the copy-paste sources cannot come from VAL/TEST. Until then, RQ1 and the 73-image TEST should be presented as potentially optimistic.
3. For RQ3, add IoU-sensitive metrics and class-agnostic recall by size for all variants; otherwise the claim that box-similarity methods "have little to repair" is self-fulfilling. Be explicit that sliced inference without tile training is only a partial test.
4. Replace the "15 of 15" counting argument with an analysis that accounts for shared test images and detectors, and reconcile Holm family sizes (eight vs nine rows).
5. Tighten the abstract and Contribution 1; add the missing citations; define "recall" and "accuracy" once.

## 10. Confidential comments to the Program Chair

- No sign of plagiarism. The submission builds heavily on a prior published study by presumably the same group (cited as Anonymous 2026, PACBB, in the third person) and on a shared dataset lineage. I see no text overlap in what I can verify; the chair may want the standard overlap check to confirm the new data and experiments are substantially distinct. The paper explicitly frames its relation to the prior work as "indicative, not like-for-like".
- Anonymity looks preserved in the PDF (no names, institutions, URLs; Portuguese species glosses are the only weak cue).
- No AI-use statement appears in the Acknowledgements. I cannot tell whether one is required.
- My recommendation is borderline: I would not oppose acceptance if a rebuttal resolves M1 and the plate-split question, and I would not defend it otherwise.

## 11. Additional comments (typos, figures, wording)

- "V AL" appears with a space throughout the extracted text; check the macro or font.
- Abstract: "give no gain detectable with three seeds and default hyperparameters" conflicts with the later NWD grid; say "default hyperparameters, except a small NWD grid".
- Sec. 5.3: "Holm-corrected over the eight comparisons" while Table 3 lists nine rows; align.
- Sec. 5.4 opening sentence is a long fragment ("Confusion rates among TEST species ... ; MV to non-species 4.7% to 1.5% ...") and should be a proper sentence.
- Table 1: indicate that "Species FP, INS-only plates" is a mean over six plates and per-seed values are in the text; label "Recall" rows as species-correct or class-agnostic.
- Provide a one-glance table: experiment, GPU, seeds, split, and metric, to simplify tracking B200 / L40S / H100 differences.

---

## Autochecagem

- Every weakness cites a section, table or figure.
- The overall rating 3 is consistent with 0 fatal and 5 major weaknesses, with Significance 2 and Technical Quality 3.
- Ratings are on the 1-5 profile scale; the scale and the form layout are [inferido].
- Unverifiable items are marked "not verifiable" (Figure 1 rendering, plate split, copy-paste provenance, AI use).
- All form sections of the profile are present (summary, strengths, weaknesses, questions, ratings, overall, confidence, closed questions, comments to authors, confidential comments, additional comments).
