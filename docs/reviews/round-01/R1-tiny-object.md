# Review — CVPR 2027 (OpenReview form)

**Reviewer:** R1 — small/tiny object detection (label assignment, tiny-box regression losses, multi-scale features, slicing inference, AI-TOD/VisDrone/TinyPerson literature)
**Round:** 01 · **Paper version:** commit `59d5da9` (2026-09-22; `main.pdf` 6 pages, tectonic)

> **Scope note.** At this version, Sec. 0 (abstract), 1, 2 and 6 are written; Sec. 3, 4, 5 and 7 and the supplementary are skeletons made of `\TODO{}` markers with two partially filled tables (Tab. 2 = `tab:rq2`, Tab. 3 = `tab:rq3`). The review below evaluates what exists, checks every number against `paper/PAPER_FACTS.md`, and lists explicitly what is missing for the paper to be reviewable as a submission. The rating is therefore *provisional* and mostly reflects the framing, the methodological correctness concerns visible in the code/formulas, and the state of the evidence.

## 1. Paper Summary

The paper continues a data-centric pipeline for per-species fly counting on industrial sticky traps (four target species, ≈44 px objects in native 1920×1080 frames that must not be resized). It makes three claims: (RQ1) three failure-targeted acquisition protocols plus a real-instance copy-paste augmentation improve the detector more than volume-driven synthetic data; (RQ2) a hybrid regression/assignment similarity $(1-w)\,\mathrm{CIoU}+w\,\mathrm{NWD}$ applied inside both heads of an end-to-end (NMS-free) YOLO26 gives +1.4 pp VAL mAP50 (single seed), pure NWD collapses, and a GCD variant gives the best mAP50:95; (RQ3) a detector→crop→ConvNeXt-T part-attention classifier cascade raises species recall on the test set by +4.4 pp at unchanged precision while lowering 7-class mAP50 by 4.7 pp, which the authors explain as "re-labelling without re-ranking". A list of negative results (SAHI, P2 head, multi-scale at 1920, SAM background removal, larger backbones/crops, TTA/WBF) is promised with one-line mechanisms. At this version the evidence for RQ1 is entirely absent, the RQ2 table is half-filled and single-seed, and RQ3 is the only result with a complete table (Tab. 3).

## 2. Strengths

**S1 — Unusually honest reporting of the evidence status (Sec. 6, Sec. 1 P4).** The limitations section already states that the +1.4 pp detector gain is single-seed, that classifier-stage seed variance is "several points" (F1 70.1–72.6 % on a second seed vs. the reported one), that the test set is "consolidated, not blind", that the ensemble was chosen after looking at test and is exploratory only, and that the two evaluators (Ultralytics validator vs. cascade harness: 0.857 vs. 0.812 VAL mAP50 on the same E0) must not be mixed in a table. This is exactly the discipline the tiny-object literature usually lacks, and I weigh it positively.

**S2 — The RQ3 "accuracy up / mAP down" observation is a genuinely useful methodological point (Sec. 1 P4-3, Sec. 2 ¶3, Tab. 3).** The explanation (the cascade changes labels but keeps detector scores, so the operating-point decision improves while the PR curve degrades) is correct and relevant to any practitioner who bolts a classifier onto a detector and then evaluates with COCO-style AP. Tab. 3 matches `PAPER_FACTS.md` §3 line by line (75.7→80.1, 70.9→71.2, 73.2→75.3, 77.4→81.8, 76.8→72.1, 25→463 ms). The 18× latency is reported honestly and Sec. 6 further clarifies it is an upper bound (all 4,197 low-confidence proposals re-classified, FP32).

**S3 — The RQ2 design touches an under-explored corner (Sec. 2 ¶1, Sec. 4.2).** Most NWD/RFLA/GCD results are on anchor-based or two-stage mmdetection baselines on AI-TOD/DOTA; putting the metric into a TaskAlignedAssigner-based, NMS-free YOLO with a one2one head is not something I have seen characterised, and the $w$-sweep (0.672 / 0.864 / 0.871 vs. 0.857) is the right first experiment. The authors correctly refrain from calling the metric new and frame the contribution as a controlled study.

**S4 — Negative results are numerous and, once explained, valuable (Sec. 1 P4-4, `PAPER_FACTS` §4).** P2 head −5.7 F1, SAHI −7.0 F1, SAM cleaning −7 pp, ConvNeXt-S and 512-px crops below ConvNeXt-T/384 — each contradicts a common "default" recommendation for small objects, and the paper promises a mechanism for each.

**S5 — Related work is accurate on my area.** The description of NWD (fixed $C$, $\exp(-\sqrt{W_2^2}/C)$), Dot Distance, GCD (local normalisation) and RFLA (Gaussian receptive field assignment) is correct; the AI-TOD motivation (IoU sensitivity to 1-px shifts) is correctly stated; SAHI and WBF are correctly positioned as inference-time. Bibliography entries were verified against DOIs by the authors.

## 3. Weaknesses

**Major (must fix before submission)**

1. **The coordinate frame of the NWD term in the loss is very likely not pixels, so "C = 44 px" (Sec. 4.2, Eq. 1–2; Sec. 1 P4-2) does not describe what was trained.** In Ultralytics `v8DetectionLoss.__call__` (8.3.x–8.4.x, inherited by `E2EDetectLoss`), the assigner is called with `pred_bboxes.detach() * stride_tensor` (pixels), but immediately before `self.bbox_loss(...)` the code does `target_bboxes /= stride_tensor`, and `pred_bboxes` from `bbox_decode` are in feature-grid units. The project's own `nwd_patch.py` confirms this: the non-DFL branch of the patched `BboxLoss.forward` multiplies `target_ltrb` by `stride` and divides by `imgsz` precisely because the inputs are grid units. Consequently, inside the loss a 40-px fly at P3 is a 5×5 box and a 2-px shift is $W_2\approx0.25$ grid units → $\mathrm{NWD}=\exp(-0.25/44)\approx0.994$ for essentially every foreground pair (and even flatter at P4/P5). The NWD term in the loss is therefore saturated: the effective E10 loss is $\approx 0.5\,\mathrm{CIoU}+\mathrm{const}$, i.e. a *halved CIoU weight*, and only the **assigner** sees a properly scaled NWD. This has three consequences the authors must address: (a) the method description must state the coordinate frame in both call sites and, if the mismatch is confirmed, either fix it (multiply $C$ by the per-level stride, or compute NWD in pixels) and retrain, or re-describe the method as "NWD in the assigner + down-weighted CIoU in the loss"; (b) the promised mechanistic explanation of the $w{=}1$ collapse (Sec. 4.2 TODO: "no aspect/enclosure term, saturates for well-localised boxes") has a much simpler competing explanation — at $w{=}1$ the box loss is $\approx(1-0.99)\cdot\text{weight}\approx0$, so regression receives no gradient at all; (c) the loss-only vs. assigner-only decomposition (see W3) becomes mandatory to attribute the +1.4 pp. I flag this as *likely* rather than certain because I could not run the container; a one-line print of `target_bboxes.max()` inside `BboxLoss.forward` settles it.

2. **RQ1 has no evidence at all, yet the abstract, title and contribution list already present it as a result (Sec. 0; Sec. 1 P4-1; Sec. 3; Tab. 1).** The abstract says the acquisitions are "each mapped to a documented failure and ablated individually"; Tab. 1 (`tab:rq1`) is empty and Sec. 6 says the ablation "is not yet run". `PAPER_FACTS.md` §0 further records that the authors do not yet know whether the new plates are even inside DS-F2_v8.1.1 or whether any model was trained with/without them. Until the ablation exists, RQ1 is a plan, not a contribution; the title ("From Better Data…"), abstract and Sec. 1 must be rewritten to match, or the experiment must be run. As written, the paper's first-listed contribution is unsupported.

3. **The RQ2 baseline set is too thin to attribute the gain (Tab. 2, Sec. 5.2).** For a paper whose detector contribution is "NWD/GCD inside an end-to-end YOLO", the minimal fair ablation in this area is: (i) NWD in the loss only, (ii) NWD in the assigner only, (iii) both (E10); (iv) $C$ sensitivity (at least $C\in\{22,44,88\}$, since $C$ is the only free hyper-parameter and the AI-TOD papers show it matters); (v) a CIoU baseline with the *box-loss gain halved* — given W1, this may reproduce E10 without any Gaussian metric. Stronger tiny-object regression baselines that cost nothing at inference (Wise-IoU v3, Tong et al., arXiv 2023; Shape-IoU, Zhang & Zhang, 2023) are also absent. I do **not** demand all of these as an acceptance condition, but (i)–(iii) and one $C$ point are needed to make any claim about NWD specifically. Note also that Tab. 2 is single-seed for every row and E0/E10 mAP50:95 are blank, so the "GCD gives the best mAP50:95 of all variants" statement (Sec. 1 P4-2, Sec. 4.2) currently compares 0.626 against nothing.

4. **"Tiny" is used loosely; by the benchmark taxonomy the objects here are small-to-medium, and this matters for interpreting the negative results (Sec. 1 P1, Sec. 2 ¶1, Sec. 5.5).** AI-TOD defines very tiny 2–8 px, tiny 8–16 px, small 16–32 px, medium 32–64 px. With mean $\sqrt{wh}=44.4$ px and median 39.2 px at native resolution, the objects are *relatively* tiny (0.1 % of the frame) but *absolutely* small/medium — a fly covers ≈5×5 cells at P3. This is exactly why an extra P2 level (designed for <16 px) and SAHI slicing (designed for objects that vanish after downscaling) cannot help here: there is no resolution deficit to recover, only extra false positives at tile borders and more low-level noise. The paper should say this once, precisely, in Sec. 1 and use it as the shared mechanism for the P2/SAHI/512-crop negatives instead of three separate one-liners. It also calibrates expectations for NWD gains, which in the literature are largest in the ≤16 px regime.

5. **Overclaim on the "irreducible boundary" (Sec. 0 last sentence: "disproving the irreducibility claim").** The only per-species evidence (`PAPER_FACTS` §3) shows flesh fly (MC) recall *falling* 85.3→81.2 % while precision rises, and blow fly (MV) recall 83.6→86.9 %; the pairwise MC↔MV confusion counts are a TODO (Sec. 1 P4-3). Net movement of a confusable pair in opposite directions is not a disproof of irreducibility; it is a shift of the operating point. Either report the before/after confusion counts and a paired statistic, or weaken to "revisits" (as Sec. 2 ¶4 already does correctly).

6. **Number/protocol statements not backed by `PAPER_FACTS.md`.**
   - Sec. 6 ¶2 states as fact "no plate appears in two splits", while `PAPER_FACTS` §0 and §7-3 mark the split rule as "confirm". This must be verified, not asserted.
   - Sec. 1 P4-2 and Sec. 4.2 say E12 "collapses … with early stopping at epoch 10", but the shared recipe is `patience 30`; early stopping cannot trigger at epoch 10 under that recipe. Either the best epoch was 10 and training stopped ≈40, or the recipe differed. Clarify.
   - Sec. 1 P4-4 lists "multi-scale training at 1920 px" among things that "fail to beat the simple baseline"; per `PAPER_FACTS` §1/§4 E13 was a CUDA OOM and produced no result. It should be reported as "not feasible on 80 GB without gradient accumulation" (as Sec. 6 does), not as a negative result.
   - Sec. 1 P4-4 also lists "TTA/WBF ensembles" as failing; `PAPER_FACTS` §4 has only the *classifier* ensemble + TTA (F1 75.4–75.6 %); the *detector* TTA/WBF evaluation is still running (job 33015). Make clear which one is meant.

**Minor**

1. **RFLA naming (Sec. 4.2 "Receptive-field-aware assignment (RFLA)", Tab. 2 row E14).** `rfla_patch.py` replaces `select_candidates_in_gts` with a radius-based candidate expansion (radius $\max(2\cdot\text{stride},\,\text{side}/2)$ for boxes with side $<2\cdot$stride). That is closer to center sampling (FCOS/ATSS) than to RFLA, which ranks anchors by a KL/Wasserstein distance between the Gaussian receptive field and the Gaussian GT and uses a hierarchical label assignment (HLA). Call it "RFLA-inspired candidate expansion" and say what it is; do not call it RFLA (Xu et al., ECCV 2022).
2. **GCD formula (Sec. 4.2 paragraph, `gcd_patch.py`).** The implemented variant is NWD with $C_\text{local}=\sqrt{(w_1h_1+w_2h_2)/2}$; the docstring itself calls it "same structure as NWD, constant → local scale". State whether this is the exact GCD of Guan et al. (GRSL 2025) or a simplification, and cite accordingly. Also check the bib entry: DOI `LGRS.2025.3531970` (early 2025) vs. the arXiv id `2510.27649` quoted in the code (Oct 2025) cannot both be the same paper.
3. **Alignment metric exponent (Sec. 4.1 TODO).** In TaskAlignedAssigner the overlap enters as $u^{\beta}$ with $\beta{=}6$; substituting $S_w$ for IoU compresses the ranking differently from substituting it in the loss. One sentence on this is needed when Sec. 4.1 is written.
4. **Two evaluators, two deltas.** The Ultralytics validator gives E10−E0 = +1.4 pp; the cascade harness (repo records) gives 0.854 vs. 0.812 = +4.2 pp on the same checkpoints. If both are reported anywhere, explain why the delta triples across evaluators (conf/NMS/matching), otherwise readers will assume cherry-picking.
5. **Missing citations in text.** `ultralytics2026yolo26` and `kirillov2023sam` are in `main.bib` but never cited; YOLO26 (Sec. 1, 4.1) and SAM (Sec. 1 P4-4, Sec. 4.4) should be.
6. **Sec. 6 ¶3 "we do not interpret differences of 1–2 pp"** — but the headline detector gain is +1.4 pp single-seed. The sentence is correct and admirable; the abstract/intro should apply the same rule until the 3×3 seeds are in.
7. **Per-class AP for the detector (RQ2) is entirely absent.** For the tiny-object story, per-species AP50 of E0 vs. E10/E10b (especially MF, the smallest species) is more informative than aggregate mAP50; it is in `results.csv` at zero cost.

## 4. Questions for the authors

Q1. Inside `BboxLoss.forward`, what is the numeric range of `target_bboxes` (pixels or grid units)? If grid units, what is the mean and standard deviation of the NWD term over foreground pairs at epoch 1 and at the best epoch? (This resolves W1 with no retraining.)

Q2. Can you provide loss-only and assigner-only NWD runs (single seed is fine) so the +1.4 pp can be attributed?

Q3. For E12 ($w{=}1$): with `patience 30`, how did training stop at epoch 10? What were box/cls loss curves — did the box loss go to ≈0 immediately (consistent with saturation) or diverge?

Q4. Are the crowding/graduated/blank plates inside DS-F2_v8.1.1 already (i.e., are E0/E10 already trained on them)? If yes, the RQ1 ablation must *remove* them, not add them, and Tab. 1's "Base" row is not the current baseline.

Q5. For SAHI: tile size, overlap, and whether NMS or WBF was used for merging; for the P2 head: was the P5 head kept (four heads) and was the batch/imgsz identical? Strawman settings would invalidate the negative result.

Q6. Per-level positive-sample counts (P3/P4/P5) for E0 vs. E10 vs. E14: does the hybrid assigner actually move positives towards P3, as the tiny-object argument implies?

Q7. What are E0 and E10 mAP50:95 (needed to substantiate "GCD best mAP50:95 of all")?

## 5. Preliminary Rating

**2** — Reject (provisional; the manuscript is incomplete)

## 6. Justification of Rating

The written parts (Sec. 0–2, 6) are clear, honest, and correctly positioned in the tiny-object literature; the RQ3 result is complete, matches the ground-truth facts, and carries a useful methodological message (S1–S2). However, the paper's first contribution (RQ1) has no evidence, the second (RQ2) rests on a single seed, an ablation table with half its cells empty, and — most seriously for my area — a plausible units mismatch that would mean the loss-side NWD never acted and the reported mechanism for the $w{=}1$ collapse is wrong (W1–W3). The abstract's "disproving irreducibility" overclaims relative to the per-species numbers (W5), and a few statements are ahead of what `PAPER_FACTS.md` supports (W6). None of this is unfixable: W1 is settled by one diagnostic print and, if confirmed, one retrain with a stride-scaled $C$; W3 needs three single-seed runs; W2 depends on whether the acquisition data exist. If those land and the 3×3-seed comparison holds, the paper moves to Borderline/Accept territory on the strength of S1–S4. At the current state I cannot recommend acceptance.

## 7. Confidence

**4** — I am confident in the domain assessment and in the code-level reading of the Ultralytics loss path; I did not execute the training container, so the units mismatch (W1) is a strongly supported inference rather than an observed fact.

## 8. Checks

- Limitations discussed adequately: **partial** — Sec. 6 is the best-written section, but it contains four `\TODO{}` (annotators, data release, split rule, GPU-hours, seeds) and asserts the split disjointness that `PAPER_FACTS` marks as unconfirmed.
- Data/code assets properly cited: **partial** — YOLO26 and SAM used but not cited; dataset name/release status pending; prior PACBB paper cited correctly in third person.
- Anonymity / formatting violations noticed: teaser is a placeholder (`example-image-golden` ×3); `main.log` shows `Font shape TU/ptm undefined` → the tectonic build is falling back from Times, which violates the CVPR font requirement (compile with pdfLaTeX or provide the Times font); two overfull hboxes (14 pt and 59 pt); `\TODO{}` markers render in red in the PDF (48 occurrences across `sec/*.tex`). No author-identifying strings found.
- Numbers consistent with `PAPER_FACTS.md`: **yes for every figure that appears** (0.857/0.871/0.864/0.672/0.862/0.626; 0.476→0.608; 44.4/39.2; 617/918/2,864/4,197; Tab. 3 all cells; 416.5/22.9 and 463/25 ms). Discrepancies are of *status*, not value: (a) split disjointness asserted vs. unconfirmed; (b) E13 OOM presented as a negative result; (c) "early stop at epoch 10" vs. patience 30; (d) detector TTA/WBF listed as failed while still running; (e) RQ1 ablation presented as done in the abstract while absent.
- Ethics flag: **none** — no people, no personal data; industrial monitoring use.

## 9. Additional comments (typos, figures, wording)

- Missing for reviewability (list to close before Round 02): Sec. 3 (all subsections, Tab. 1), Sec. 4.1/4.3/4.4 text, Sec. 5.1 setup, Sec. 5.2 filled Tab. 2 + paired test, Sec. 5.3 per-species table and paradox paragraph, Sec. 5.4, Sec. 5.5 negative-results table, Sec. 7, teaser figure, all of the supplementary (per-seed table, SAHI/P2/SAM/TTA settings, confusion matrices, GPU-hours).
- Sec. 0: "at fixed precision" → "at essentially unchanged precision (70.9→71.2 %)" to match Sec. 1.
- Sec. 1 P1: give the AI-TOD size bands once and state the objects are relative-tiny / absolute-small (see W4).
- Sec. 2 ¶1: "These methods were validated on aerial imagery with anchor-based or two-stage detectors" — RFLA also reports anchor-free (FCOS) results; soften to "predominantly".
- Sec. 4.2 Eq. 1: define $W_2^2=\|c_p-c_g\|_2^2+\|\tfrac{1}{2}(w_p,h_p)-\tfrac{1}{2}(w_g,h_g)\|_2^2$ explicitly (this is what the code computes) and state the units.
- Tab. 2 caption: mark which rows come from `results.csv` best-epoch and specify that "best epoch" is selected on VAL mAP50 (which optimistically biases VAL numbers — state it).
- Tab. 3: add a row "Proposals re-classified per image" so the 18× is interpretable; consider a second latency column at a production threshold.
- `main.bib`: key `wang2022nwd` has Xu as first author (correct for the ISPRS 2022 paper); consider renaming the key to avoid confusion with the 2021 arXiv NWD preprint by Wang et al.
- Sec. 6 ¶4: "cascade latency … an upper bound on cost, not a production figure" — good; repeat this caveat in the Tab. 3 caption.
- Typo-level: "colour" / "standardised" (UK) vs. "normalization" in the bib and "localized" in Sec. 4.2 TODO — pick one variant.
