# Review, VISAPP 2027 (regular paper, double-blind)

**Paper:** "Decoupling Detection from Identification for Species-Level Fly Monitoring on Industrial Sticky Traps"
**Profile:** visapp.md (edition 2027). The 5 criteria and the closed questions are [oficial]. The numeric scale and the form layout are [inferido].
**Read:** all 12 pages (text and rendered pages; Figure 1 and Tables 1-2 inspected). Traceability was checked only against docs/PAPER_FACTS.md §9, §10, §11 and §13.

## 1. Paper Summary
The paper studies species-level detection of flies (MD, MV, MC, MF plus INS/NOISE/MAR) on 1920x1080 industrial sticky-trap images (mean object size about 44 px). It reports three sets of experiments on DS-F2_v9.0.5 (1,738 train, 52 val, 73 test images):
- **RQ1:** a ladder of five cumulative training sets (L0-L3, three seeds). Species accuracy rises from 56.1% to 82.2%, with the largest gains on objects below 32 px and on insect-only plates. Look-alike confusion falls but persists.
- **RQ2:** a frozen YOLO26/RT-DETR detector followed by a crop classifier (ConvNeXt-T, DINOv2, DINOv3, BioCLIP 2). The cascade lowers macro-F1 in all 15 configurations. Ground-truth-crop accuracy saturates near 83%, which the authors read as an information ceiling (not an architectural one).
- **RQ3:** eight tiny-object variants (NWD, RFLA, GCD, Wise-IoU, P2, 2560 px, YOLO11m), plus SAHI/TTA/WBF, give no gain over a same-campaign control. NWD+RFLA is 8.8 pp worse.

The message is that data helps, architecture does not, and aerial tiny-object recipes do not transfer.

## 2. Strengths
- The work is honest and well calibrated for an applied paper. VAL-selected checkpoints are flagged as not held-out, there is a "what we cannot say" attitude, and equivalence is not claimed from wide CIs.
- The protocol is a real strength. Primary metrics are per-species and applied at a fixed operating point. Paired bootstrap over images, McNemar, Holm correction, seed crossing, and effective-seed verification are all present, which is rare in the insect-monitoring literature.
- The ground-truth-crop oracle, the 15-classifier agreement analysis (4.9% wrong for all) and the learning curve are a clean way to separate localization from identification. The result is informative: the E0 detector localizes 97.7% of instances.
- Negative results with variance are useful to practitioners. The NWD scope ablation (loss vs assigner) is a good mechanistic probe.
- The real industrial field data at native resolution (no downscaling) is valuable and uncommon.
- The limitations section is substantive.

## 3. Weaknesses

### FATAL
None. The conclusions are mostly hedged and the numbers I could trace match PAPER_FACTS. However, W-M1 and W-M2 together come close to a fatal flaw for the "data" claim.

### MAJOR
**M1. No leakage guarantee and no held-out set (Sec. 3.3, 4.4, 5.2, 6).**
- The dataset has no plate identifier. Plates are species-pure, so species correlates with plate, session, specimen age and glue appearance.
- Test and validation images may come from the same plates as training images.
- 4.7% of TEST crops have thumbnail similarity ≥0.98 to a training crop.
- The classifier crops are expanded by 75% per side, so they contain a lot of background that could act as a shortcut.
- TEST was consulted in earlier campaigns and VAL selects every checkpoint.
- With 73 test images, the 56%→82% ladder and the 83% "ceiling" could be partly plate memorization. Both "data helps" and "information ceiling" are therefore weakened.
- *Fix:* reconstruct plate IDs (file timestamps, the handwritten plate labels, EXIF), or at minimum re-split by session/batch and re-run E0 and the oracle. Report the oracle on a plate-disjoint subset. Otherwise downgrade the claims to "within-lineage".

**M2. The RQ1 attribution to "targeted physical collections" is not isolated (Sec. 5.1, abstract, contribution 1).**
- Quantity (103→1,738 images, 4.4k→88k boxes) and quality co-vary. There is no equal-size control, which the authors admit.
- There is also no digital-synthesis baseline (copy-paste exists only in L3), nor a comparison against simply adding mixed-species plates.
- The gain on objects below 32 px (+36 pp) is accompanied by gains in all other bins (+17, +13 pp). So "concentrated where the collections aim" is weak.
- The insect-only-plate result is expected by construction. L0 contains no INS-only plates and L2 adds them, and the test plates are from the same collection style, so this is trivial rather than a finding.
- The ladder is also a mix of different dataset versions (L0 = v5.1 and so on).
- *Fix:* add an equal-size subsample control (e.g. L3 subsampled to 244 and 869 images with the same plate-type mix). Rephrase the claim as "more and better data", not "targeted collections", unless the control supports it.

**M3. The cascade comparison may be a weak strawman and the RQ2 evidence is prose-only (Sec. 4.2, 5.2).**
- The detector is frozen. The classifier is trained on GT crops and applied to detector crops.
- The label-only fusion discards the detector's class evidence. The partial fix (late fusion) uses an approximated detector distribution ("rest spread uniformly") and is only an approximation.
- Cost is inflated by classifying all proposals down to confidence 0.001 (30-67x), whereas only proposals ≥0.25 matter for the metric.
- Four heads (ArcFace, SupCon and others) were trained once and not reported (Sec. 4.2). That is selective reporting.
- The central RQ2 results (five backbones x three detectors, oracle, 15-classifier agreement, BG head, abstention, fusion, error decomposition) are in text only. No table shows per-species precision/recall/F1, although the abstract and Sec. 4.4 declare these as primary metrics.
- Per-species precision and recall appear nowhere in the paper. Only the Table 1 "accuracy" (= micro recall) and macro-F1 deltas are given.
- *Fix:* add a compact table (E0, each backbone: per-species P/R/F1, macro-F1, ΔF1 with CI, oracle accuracy, latency). Report the cost at a score ≥0.25 gate. Either report or drop the unreported heads. Train the classifier on detector-proposal crops (not GT-only) as a more realistic variant.

**M4. The "negative transfer" in RQ3 is underpowered, untuned and partly confounded (Sec. 4.3, 5.3, abstract).**
- Each CI is about ±1.7 pp or wider with 3 seeds and 73 images, so gains below about 2 pp cannot be detected. The conclusion is "no detectable gain", which the paper admits, but the title-level framing ("what does not transfer") overstates it.
- No variant was tuned. C=44 px was measured on an earlier data version, w=0.5 is fixed, and RFLA and Wise-IoU are used with defaults.
- P2 and 2560 use batch 4 or a different head, and 2560 is upsampling of 1920 images (no new information). This confounds the comparison with E0 at batch 8.
- The "8.8 pp worse" headline (abstract) is driven by a diverged seed (NaN) and an early-stopped seed that were not retried. This is a training failure, not evidence about the method. The ±7.5 std shows it.
- SAHI was tested with a model trained at full frame and no tile-trained model. Sec. 4.3 and 5.3 state that the "relevant baseline" is E0 at native resolution, so this is not a fair test of sliced inference.
- *Fix:* retry the diverged runs (with or without a lower LR/warm-up), or report the failure separately and exclude NWD+RFLA from the headline. Add at least a small tuning of C and w, and a tile-trained SAHI model. Add a power statement (minimum detectable effect).

**M5. The headline NWD variant is missing from Table 2, and several counts are inconsistent (Sec. 1, 4.3, 5.3, Table 2).**
- The text and contribution 4 list "NWD in loss and assigner" as the main variant, and Sec. 4.3 describes it first. Table 2 does not contain it.
- The table lists 8 variants: GCD, Wise-IoU, NWD-assigner, NWD-loss, P2, 2560, NWD+RFLA and YOLO11m. The combined-NWD result (E10) lives in a different campaign (V100, PAPER_FACTS §12), which the paper never reports.
- "Three seeds per variant" (contribution 4) is false for YOLO11m (one seed).
- Holm is "over eight comparisons", while the text says "none of eight variants beats the control". The count of variants therefore depends on what is counted.
- *Fix:* either add E10 on the L40S control or state explicitly that it was run elsewhere. Reconcile the number of variants everywhere (abstract, intro, Sec. 5.3, conclusion).

**M6. The tiny-object premise is shaky and the RQ3 framing contradicts itself (abstract, Sec. 1, 5.3).**
- The introduction says the mean object (≈44 px) is "below the 32 px threshold". 44 > 32. Only 51% of species instances are below 32 px (n=613 of 1,210).
- Sec. 5.3 then argues that a 44 px fly spans about five cells at stride 8 and is therefore not tiny for a detector at native resolution. This undermines the premise of RQ3 ("tiny-object regime").
- The tiny-object framing in the title, keywords and the RQ3 motivation is partly a mismatch with the authors' own analysis.
- *Fix:* correct the factual error, report the size distribution (median, histogram), and reframe RQ3 as "small-object recipes on mid-scale objects at native resolution", or restrict the claims to the sub-32 px subset.

**M7. Significance and external validity are limited (whole paper).**
- The data are from one facility, one test split (73 images, only 7 crowded and 6 INS-only plates), and one detector family for most experiments.
- The dataset and code are not stated to be released, which makes this hard to reproduce or build on. Nothing in the manuscript says whether DS-F2 will be available.
- The conclusions about "practitioners of trap-based monitoring" generalize beyond this single facility and plate-type composition.
- *Fix:* state the data and code release plan (anonymized), temper the generalization in the conclusion, and, if possible, add an external test set (for example the 1280x720 v4 data, or another trap colour as in Ong and Høye).

### MINOR
- **m1.** Sec. 5.1 H2 versus the abstract/intro. The intro says "Identification is not moved" (p. 2), while Sec. 5.1 shows MV→MC falling from 14.8% to 5.4%, and the text says "H2 is only partly supported". The abstract and conclusion nuance this, but the intro sentence and the "ladder H2: expected not to move" label in Table 1 read as a pre-registered null that was falsified. Rewrite consistently. Confusion rates (Fig. 1, Table 1) are point estimates without intervals.
- **m2.** "Five backbones" (abstract, contribution 2). The five stage-2 configurations are ConvNeXt-T CE, ConvNeXt-T parts, DINOv2, DINOv3 and BioCLIP 2, which is four backbones with two heads for one of them. Define "15 configurations" and "15 classifiers" precisely.
- **m3.** Sec. 4.2 and 4.4 say "parts" was chosen as the main classifier on VAL, but Sec. 5.2 leads with CE and highlights its significance. Explain this and avoid highlighting the one significant result (CE, p = 0.013) after seeing it.
- **m4.** Fusion text (Sec. 5.2) says fusion "never exceeds the detector's F1", but w=2 gives 79.6 vs 79.5 for E0 (79.56 vs 79.54 in PAPER_FACTS). Rewrite as "matches".
- **m5.** Batch counts in Sec. 3.2 ("97, 83, 70, 100 and 174 images per batch" = 524) do not obviously reconcile with the ladder sizes (L0 = 103, L1 = 244, L2 = 869) or the 1,738 total. Clarify what these counts refer to.
- **m6.** Section title "FROM STID TO DS-F2" (p. 4) uses an acronym that is never defined and that points to the prior work. This is an anonymity risk and a clarity problem.
- **m7.** The inference time is reported only on a shared node (the authors say "indicative only") and the sentence in Sec. 4.4 ends in a dangling semicolon ("indicative only;").
- **m8.** Typos: "11.7%,;" (p. 8); "Table 2: ∆is" and "∆between" missing spaces (p. 9); "Intervals ... (73 images)" fragmentary.
- **m9.** Figure 1 is half a column wide with illegible cell numbers at print size; it should be full width, with CIs. It also shows the MF row with 46% of L0 going to non-species, which the text does not discuss. Tables 1 and 2 are dense but readable. There is no figure for the cascade results.
- **m10.** The GCD variant is cited as "arXiv:2510.27649" in the text and has no entry in the reference list. The Chen et al. reference lists "et al." and leaks internal notes (see checks below).
- **m11.** The NWD constant C=44 px is "measured on an earlier data version". Recompute it on v9.0.5 (cheap) to remove the caveat.
- **m12.** The ladder uses seeds pooled for bootstrap CIs (Table 1). Per-seed tests are in the facts file, but the paper does not say how seed variance enters the CI.
- **m13.** The related work is adequate, but there is no comparison with recent fine-grained or part-based small-object classifiers, and there are only a few 2025-26 references on insect detection. The Brock et al. (2026) and Du (2025) citations are preprints, and some key ones (Wang 2021 NWD, Beery 2019) are arXiv versions.
- **m14.** Language: mostly good. Several very long sentences and parentheticals in Sec. 5.2 would be better as a table.

## 4. Questions for the authors
1. Can you reconstruct plate or session identifiers and show that TEST/VAL are plate-disjoint from training? What does E0/oracle accuracy look like on a plate-disjoint subset?
2. What does an equal-size subsample of L3 (244 and 869 images) give, compared with L1 and L2?
3. How does the cascade behave if the classifier is trained on detector-proposal crops, with the real detector scores in the fusion, and if only proposals ≥0.25 are classified (cost)?
4. Where is NWD in loss+assigner (the main variant) on the L40S control? Do NWD+RFLA results change if the diverged seed is retried?
5. What are the results of the unreported heads (ArcFace, SupCon)? Why were they dropped?
6. Do the 56%→82% ladder gains hold for MD, the weakest species (recall 0.64-0.69 and flat from L1)?
7. Will DS-F2 and code be released?

## 5. Ratings per criterion (1-5)
| Criterion | Score | Label | Justification |
|---|---|---|---|
| Relevance | 4 | Accept | Fine-grained, tiny-object detection and cascades on real trap imagery fit the Recognition and Detection area. |
| Originality | 3 | Borderline | An empirical study with negative results and one dataset; no new method. The "physical collection" idea is interesting, but its isolation (M2) is missing. |
| Technical Quality | 3 | Borderline | Good statistics, but leakage and plate confounds (M1), the untuned and unequal RQ3 variants (M4), and the cascade design (M3) limit the conclusions. |
| Significance | 3 | Borderline | Useful for trap-monitoring practitioners, but a single facility and a 73-image test set limit generalization (M7). |
| Presentation | 3 | Borderline | Honest and clear writing, but the RQ2 core is prose-only, Figure 1 is too small, there is a factual slip (44 < 32), and internal notes are in the references. |

## 6. Overall rating and justification
**Overall: 3 (Borderline)**, leaning weak accept only if the rebuttal can address M1, M2 and M5; the author target of 4 is not reached by this version.
Justification: the paper is careful and honest, and the protocol is better than most in this literature. But the three headline claims each rest on one weak point: a possible plate leak with a tiny test set (data, identification ceiling), a confounded ladder (data attribution), and untuned or partly failed variants (RQ3 negatives). None is fatal on its own. Together they leave the conclusions as "indicative" rather than established. The changes that would move the score to 4 are: a plate-disjoint re-evaluation, an equal-size control, an RQ2 table with per-species P/R/F1, and a complete Table 2 (with NWD both and retried or separately reported failures).

## 7. Reviewer confidence
**3/5.** I know detection and fine-grained evaluation well, but I cannot verify the dataset or code from the manuscript, and I am not a specialist in entomology or trap monitoring.

## 8. Closed questions
- The abstract and introduction are adequate? **Partial** (the 44 px "below 32 px" slip; "Identification is not moved" versus the H2 result; five backbones vs four).
- More experimental results are needed? **Yes** (plate-disjoint evaluation, equal-size control, retried failed runs, tuning, tile-trained SAHI).
- A comparative evaluation is needed? **Partial** (no strong non-YOLO baselines beyond RT-DETR in the cascade; no external dataset).
- The paper needs a more critical discussion? **Partial** (the limitations are honest, but the shortcut/plate-confound and the cascade-design caveats are under-discussed).
- The figures are adequate? **No** (Figure 1 is small and has no CIs; there is no figure or table for the RQ2 results).
- The conclusions and future work are convincing? **Partial** (the cascade and data conclusions are plausible; "irreducible is a data effect" and "practical guidance" go beyond one facility).
- The references are up-to-date and appropriate? **Partial** (many 2025-26 preprints, a missing GCD entry, an internal note in the references).
- The formatting needs adjustment? **Yes, minor** (Figure 1, a dangling sentence, typos, references with internal notes).
- The English needs improvement? **Minor** (mostly fine; long sentences).

## 9. Comments to the authors
The paper is strongest where it is most modest. I would lead with the protocol and the oracle result, move the RQ2 evidence into a table, fix the factual slip about 44 px vs 32 px, and demote the claims that depend on the unresolved plate leakage. Treating the NWD+RFLA divergence as a training failure rather than a finding will make the RQ3 section more credible. A plate-disjoint check would be the single most valuable addition.

## 10. Confidential comments to the Program Chair
- No plagiarism concern. The paper builds heavily on an earlier, anonymized study ("Anonymous, 2026"), and the dataset lineage and "irreducible" claim come from it. The overlap with the prior paper (dataset, setting) should be checked against the camera-ready version. The text explicitly distinguishes new data and experiments, so I do not see substantial overlap.
- Anonymity: no names or URLs; the undefined acronym "STID" and the Portuguese common names (mosca doméstica, etc.) are mild hints. The prior work is cited in the third person.
- Honest decision: Borderline. I would accept it if the plate-leak and the equal-size control were resolved in the rebuttal, and reject it if the authors cannot show a plate-disjoint evaluation.

## 11. Additional comments / checks
- **Numbers without traceability.** I found no number that I could not trace. Spot checks against PAPER_FACTS: Table 1 and the Δ values (§11); Table 2 and the size bins (§13); macro-F1 79.5 / 76.9 / fusion values and the abstention bound (§10); the RT-DETR mean 74.6 (72.9/77.4/73.6); the 1,210 test instances and the per-class counts. Two cases are not traceable from the manuscript alone: (a) the "4.7% of TEST crops ≥0.98" and the 127/3,470 confident-learning counts (they sit in §10 but the text does not name the CSV), (b) the batch counts in Sec. 3.2 (m5). The oracle, learning-curve, BG and 15-classifier statistics were not verified here (only PAPER_FACTS §10 headings were checked), so I mark them "not verifiable from the manuscript".
- **Anonymity.** No name, institution, URL or acknowledgement leak. Residual hints: "STID" (undefined), Portuguese names, and heavy dependence on "Anonymous, 2026". Acceptable, but remove "STID".
- **TODO and internal notes.** No TODO/AUTHOR DECISION markers in the body. The references contain internal notes that must be removed: Chen et al. ("Title and venue checked against arXiv; full author list not transcribed.") and Jain et al. ("title/authors/venue checked against arXiv metadata; previous entry mixed in the ECCV 2024 AMI-dataset paper"). The "Preprint." tags on Brock, Chachuła, Cheng 2018 and Du are acceptable but should be updated if published.
- **Page and size limits.** 12 pages including references (Full paper limit satisfied). About 46,000 non-space characters in the extracted text, within the 10,000-50,000 range, but close to the upper limit, so re-count after any additions (any new table must replace text).
- **AI declaration.** Not present in the Acknowledgements ("Anonymous during review"). VISAPP requires declaring AI-generated text. Add it if applicable.

## 12. Recommendation
**Borderline (overall 3/5), leaning reject as submitted but salvageable.** If the rebuttal can deliver a plate-disjoint evaluation and an equal-size control, I would move to weak accept (4). Without those, the claims should be reduced to "indicative within the dataset lineage" and the score stays at 3.
