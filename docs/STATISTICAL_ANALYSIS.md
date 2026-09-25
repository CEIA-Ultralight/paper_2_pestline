# Statistical Background for the Annotation Agreement Evaluation

## Purpose

This document describes the statistical foundation of the annotation-review evaluation workflow used for the sampled object review process. It focuses on:

- how the sample is constructed,
- what population-level quantities are being estimated,
- what agreement metrics are reported,
- what each metric measures,
- how each metric is defined mathematically,
- how confidence intervals are obtained,
- and what limitations apply to the current review protocol.

---

## 1. Study design

### 1.1 Population of interest

The population is the set of all annotated objects in the chosen dataset scope:

- one subset only: `train`, `val`, or `test`, or
- the union of all subsets.

Each object is an individual statistical unit. An object is defined by a fixed bounding box and an original class label.

Because the current review protocol does **not** ask annotators to change bounding boxes, the evaluation concerns **classification agreement conditional on fixed localization**, not localization quality.

### 1.2 Stratification by class

Sampling is performed at the **object level**, stratified by the original class. For each class c:

- $N_c$ = total number of objects of class c in the population,
- $n_c$ = number of sampled objects of class c.

This design is appropriate because the main substantive question is not only overall agreement, but also **which classes are more ambiguous or more frequently relabeled**.

### 1.3 Finite population correction sample sizing

Per-class sample size is based on estimation of a proportion with finite population correction (FPC). For confidence level $1-\alpha$, margin of error \(E\), and assumed proportion \(p\), the initial infinite-population sample size is:

$n_{0,c} = \frac{z^2 p(1-p)}{E^2}$

where \(z\) is the standard normal critical value corresponding to the chosen confidence level.

Applying finite population correction for class \(c\):

$n_c = \frac{n_{0,c}}{1 + \frac{n_{0,c}-1}{N_c}}$

In practice:

- $p = 0.5$ is often used as the worst-case assumption,
- $E$ controls desired precision,
- small classes are capped by their actual population size \(N_c\),
- optional engineering constraints such as a maximum number of sampled objects per image may also be applied.

---

## 2. What is being estimated

The review exercise aims to estimate, for the full population:

1. **Agreement with the original label**
2. **Relabel rate**
3. **Inter-annotator agreement**
4. **Consensus agreement with the original**
5. **Class-specific confusion patterns**
6. **Per-object disagreement intensity**

Because the sampling is stratified by original class, overall population quantities should be estimated using **class-weighted aggregation**, not naive averaging over the sampled objects alone.

---

## 3. Agreement with the original label

Let:

- $i = 1, \dots, m$ index sampled objects,
- $y_i$ be the original class label of object \(i\),
- $r_{ij}$ be the recorded label given by annotator \(j\) for object \(i\),
- \(J\) be the number of annotators.

### 3.1 Object-level agreement with the original

For one object \(i\), the mean agreement with the original is:

$A_i = \frac{1}{J}\sum_{j=1}^{J} \mathbf{1}(r_{ij} = y_i)$

where $\mathbf{1}(\cdot)$ is the indicator function.

Interpretation:

- $A_i = 1$: all annotators kept the original class,
- $A_i = 0$: all annotators changed it,
- intermediate values indicate partial agreement.

### 3.2 Per-class agreement

For a class \(c\), let $S_c$ be the sampled objects whose original class is \(c\). Then:

$A_c = \frac{1}{|S_c|}\sum_{i \in S_c} A_i$

This measures how often, on average, annotators retain the original label for objects originally labeled as class \(c\).

### 3.3 Weighted overall agreement

Let $N_c$ be the total population count for class \(c\), and let

$N = \sum_c N_c$

Then the class-weighted overall agreement estimator is:

$A = \sum_c \frac{N_c}{N} A_c$

This is the correct overall estimator for a stratified design when the target is agreement in the original population.

### 3.4 Interpretation

This metric measures **recorded agreement with the original annotation**.

It is intuitive and operationally useful, but it is **not chance-corrected**. It should not be interpreted as a formal inter-rater reliability coefficient.

---

## 4. Relabel rate

Relabel rate is simply the complement of agreement with the original.

### 4.1 Object-level relabel rate

$R_i = \frac{1}{J}\sum_{j=1}^{J} \mathbf{1}(r_{ij} \neq y_i)$

Since every annotator either agrees or disagrees with the original:

$R_i = 1 - A_i$

### 4.2 Per-class relabel rate

$R_c = \frac{1}{|S_c|}\sum_{i \in S_c} R_i = 1 - A_c$

### 4.3 Weighted overall relabel rate

$R = \sum_c \frac{N_c}{N} R_c = 1 - A$

### 4.4 Interpretation

This metric answers a particularly practical question:

> How often do annotators change the original label?

It is often easier to communicate than agreement because it directly quantifies annotation instability.

---

## 5. Consensus vs original

A consensus label is derived from the set of annotator labels for each object.

### 5.1 Majority-vote consensus

For object \(i\), define the consensus class:

$\hat{y}_i^{cons} = \text{mode} (r_{i1}, r_{i2}, \dots, r_{iJ})$

If there is a tie, a deterministic tie-breaking rule is used. In the current implementation, the smallest class id among tied classes is selected.

### 5.2 Consensus agreement with original

For object \(i\):

$C_i = \mathbf{1}(\hat{y}_i^{cons} = y_i)$

Per-class consensus agreement:

$C_c = \frac{1}{|S_c|}\sum_{i \in S_c} C_i$

Weighted overall consensus agreement:

$C = \sum_c \frac{N_c}{N} C_c$

### 5.3 Interpretation

Consensus-vs-original is more stringent than raw agreement because it asks whether the **panel as a whole** would preserve the original class.

This is often a useful summary for dataset auditing, because it better approximates what would happen if reviewer judgments were aggregated.

---

## 6. Confusion matrices

For each annotator \(j\), a confusion matrix compares original labels to reviewed labels.

For classes \(a\) and \(b\), the entry is:

$M^{(j)}_{ab} = \{ i : y_i = a,\ r_{ij} = b \}$

Similarly, one can define a confusion matrix between the original label and the consensus label:

$M^{cons}_{ab} = \{ i : y_i = a,\ \hat{y}_i^{cons} = b \}$

### Interpretation

Confusion matrices do not produce a single scalar score, but they are among the most informative outputs because they reveal:

- which classes are stable,
- which classes are commonly confused,
- whether some relabelings are directional,
- and which original classes tend to migrate to the same alternative class.

For the intended use case, confusion matrices are essential for diagnosing label ambiguity.

---

## 7. Inter-annotator agreement

Agreement with the original label is not the same as agreement **among annotators**. Inter-annotator metrics evaluate how consistently the annotators label the same objects.

Two chance-corrected coefficients are used:

- **Fleiss' kappa**
- **Krippendorff's alpha (nominal)**

Under the current implementation, where the rating matrix is complete and the scale is nominal, these two coefficients are expected to be numerically identical or extremely close.

### 7.1 Rating matrix

Suppose there are:

- \(m\) sampled objects,
- \(J\) annotators,
- \(K\) possible classes.

For object \(i\), let $n_{ik}$ be the number of annotators assigning class \(k\).

---

## 8. Fleiss' kappa

### 8.1 What it measures

Fleiss' kappa measures the extent to which annotators agree **beyond what would be expected by chance**, based on the overall class frequencies in the ratings.

### 8.2 Per-item agreement

Observed agreement for item \(i\):

$P_i = \frac{1}{J(J-1)} \sum_{k=1}^{K} n_{ik}(n_{ik}-1)$

This is the proportion of annotator pairs that agree on object \(i\).

### 8.3 Mean observed agreement

$\bar{P} = \frac{1}{m}\sum_{i=1}^{m} P_i$

### 8.4 Expected agreement by chance

Let the marginal proportion of ratings assigned to class \(k\) be:

$p_k = \frac{1}{mJ}\sum_{i=1}^{m} n_{ik}$

Then the expected agreement by chance is:

$P_e = \sum_{k=1}^{K} p_k^2$

### 8.5 Fleiss' kappa

$\kappa = \frac{\bar{P} - P_e}{1 - P_e}$

### 8.6 Interpretation

- \(\kappa = 1\): perfect agreement
- \(\kappa = 0\): agreement no better than chance
- \(\kappa < 0\): agreement worse than chance

### 8.7 Caveat

Kappa is known to be sensitive to:

- class prevalence imbalance,
- marginal asymmetry,
- very dominant classes.

Thus, a moderate kappa does not always imply poor practical agreement, and a high raw agreement does not always imply a high kappa.

---

## 9. Krippendorff's alpha (nominal)

### 9.1 What it measures

Krippendorff's alpha also measures agreement beyond chance, but it is framed in terms of **disagreement** rather than agreement.

For nominal data, two labels either agree or disagree completely.

### 9.2 Disagreement function

For classes \(a\) and \(b\):

$
\delta(a,b) =
\begin{cases}
0, & a=b \\
1, & a\neq b
\end{cases}
$

### 9.3 Observed disagreement

Observed disagreement is the average pairwise disagreement among annotators across items:

$D_o = \text{mean pairwise disagreement within items}$

### 9.4 Expected disagreement

Expected disagreement is computed from the overall class proportions:

$D_e = 1 - \sum_{k=1}^{K} p_k^2$

### 9.5 Krippendorff's alpha

$\alpha = 1 - \frac{D_o}{D_e}$

### 9.6 Interpretation

- $\alpha = 1$: perfect agreement
- $\alpha = 0$: chance-level agreement
- $\alpha < 0$: systematic disagreement

### 9.7 Why alpha and kappa coincide here

In the current implementation:

- the data are nominal,
- the ratings matrix is complete,
- and the same marginal proportions are used.

For this case:

$D_o = 1 - \bar{P}
\quad\text{and}\quad
D_e = 1 - P_e
$

Therefore:

$
\alpha = 1 - \frac{1-\bar{P}}{1-P_e}
= \frac{\bar{P} - P_e}{1 - P_e}
= \kappa
$

So identical values for Fleiss' kappa and nominal Krippendorff's alpha are mathematically expected under this setup.

---

## 10. Per-object disagreement entropy

### 10.1 What it measures

Entropy summarizes how dispersed the annotator labels are for one object. It does not compare to the original label and is not chance-corrected. Instead, it measures **label uncertainty among reviewers**.

### 10.2 Shannon entropy

For object \(i\), let \(q_{ik}\) be the proportion of annotators assigning class \(k\). Then the entropy is:

$H_i = -\sum_{k=1}^{K} q_{ik}\log_2 q_{ik}$

### 10.3 Normalized entropy

To keep the value in a convenient range, entropy can be normalized by its maximum over the observed categories:

$H_i^{(norm)} = \frac{H_i}{\log_2(K_i)}$

where \(K_i\) is the number of distinct classes actually used by annotators on object \(i\).

Thus:

- $H_i^{(norm)} = 0$: full agreement
- $H_i^{(norm)} = 1$: maximal disagreement over the observed labels

### 10.4 Interpretation

Entropy is especially useful for identifying:

- intrinsically ambiguous objects,
- classes that split reviewers in multiple directions,
- difficult examples even when the majority still agrees with the original.

---

## 11. Confidence intervals via bootstrap

### 11.1 Why bootstrap is used

The reviewed sample is only one realization of a sampling process. Agreement metrics therefore have sampling variability.

Bootstrap provides an empirical approximation of the sampling distribution without requiring closed-form variance formulas for every metric.

### 11.2 Stratified bootstrap

Because the sample itself was obtained by class-stratified sampling, the bootstrap is performed **within class**.

For each bootstrap iteration:

1. For each original class \(c\), resample the sampled objects of class \(c\) **with replacement**.
2. Keep the same number of sampled objects for that class as in the observed reviewed sample.
3. Recompute the metric on the resampled data.
4. Repeat this process many times.

If the number of bootstrap replications is \(B\), the resulting set of bootstrap statistics is:

$T^{*(1)}, T^{*(2)}, \dots, T^{*(B)}$

### 11.3 Percentile confidence intervals

A two-sided 95% percentile bootstrap interval is obtained from:

- the 2.5th percentile of the bootstrap distribution,
- the 97.5th percentile of the bootstrap distribution.

### 11.4 Interpretation of the `--bootstrap` parameter

If the script is run with:

```bash
--bootstrap 2000
```

then \(B = 2000\) stratified bootstrap resamples are generated for each metric that receives a confidence interval.

- Smaller values: faster, rougher intervals
- Larger values: slower, more stable intervals

The bootstrap count does **not** change the point estimate itself. It only affects the stability of the estimated confidence interval.

---

## 12. Important limitation of the current review protocol

The current review files are prefilled with the original class label. Therefore, if an annotator leaves a file unchanged, the data cannot distinguish between:

1. the annotator actively reviewed the object and agreed with the original label, and
2. the annotator never actually reviewed the object.

This means the analysis supports only the following interpretation:

> The metrics reflect the annotators' **recorded labels**, not verified review completion.

### Consequences

If some files were skipped unintentionally or intentionally and left unchanged:

- agreement with the original may be biased upward,
- consensus-vs-original may be biased upward,
- inter-annotator agreement may be biased upward.

### Recommended future remedy

A future version of the review protocol should include an explicit review marker, such as:

- `reviewed=1`,
- a timestamp written on save,
- or a blank initial label that must be actively filled.

Without such a marker, true completion rate is unidentifiable from the files alone.

---

## 13. Recommended interpretation workflow

A sound interpretation workflow is:

1. Look at **per-class agreement vs original**
   - Which classes are stable?
   - Which classes are frequently relabeled?

2. Look at **per-class relabel rate**
   - Which classes are annotation-risk classes?

3. Look at **confusion matrices**
   - Which specific class pairs are most often confused?

4. Look at **consensus-vs-original**
   - Would the panel collectively keep the label?

5. Look at **inter-annotator agreement**
   - Are annotators consistent with one another overall?

6. Look at **disagreement entropy**
   - Which objects or classes generate dispersed reviewer opinions?

No single metric is sufficient on its own. The combination of these outputs is what gives the evaluation its diagnostic power.

---

## 14. Summary of reported metrics

### Agreement with original
Measures how often reviewers keep the original label.

### Relabel rate
Measures how often reviewers change the original label.

### Consensus-vs-original
Measures whether the reviewer panel, aggregated by majority vote, would preserve the original label.

### Confusion matrices
Show the direction and magnitude of class-to-class relabeling patterns.

### Fleiss' kappa
Chance-corrected multi-rater agreement coefficient for nominal categories.

### Krippendorff's alpha (nominal)
Another chance-corrected inter-rater agreement coefficient, formulated via disagreement; identical to Fleiss' kappa under the current complete nominal setup.

### Disagreement entropy
Measures dispersion of reviewer labels per object; useful for ambiguity analysis.

### Bootstrap confidence intervals
Quantify sampling uncertainty while respecting the class-stratified design.

---

## 15. Final note

This evaluation framework is statistically appropriate for auditing **classification consistency** in a stratified sample of fixed object annotations. It is especially useful for:

- dataset quality assessment,
- identifying confusing classes,
- prioritizing relabeling efforts,
- and measuring the stability of the annotation taxonomy.

Its main current limitation is not the statistical methodology, but the inability to verify whether an unchanged prefilled review file was actually reviewed.
