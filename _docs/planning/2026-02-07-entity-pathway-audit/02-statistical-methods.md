# Statistical Methods Audit

**Files:** bayesian.py, agreement.py, clustering.py, distances.py, comparison.py

---

## 1. Bayesian Modeling (bayesian.py)

### C4: Log-Likelihood Not Captured in LOO-CV (Lines 1191-1203)

**Bug:** `pm.compute_log_likelihood()` returns a modified InferenceData object, but the return value is discarded:

```python
# CURRENT (broken):
pm.compute_log_likelihood(full_trace)  # Return value discarded!

# CORRECT:
full_trace = pm.compute_log_likelihood(full_trace)
```

The same bug appears for the reduced model trace. Without capturing the return value, the trace doesn't have the `log_likelihood` group, and subsequent `az.loo()` calls will fail.

### C5: Bayes Factor Computation Is Fundamentally Incorrect (Lines 1326-1331)

The Savage-Dickey density ratio requires:
```
BF_01 = p(delta=0 | data, H1) / p(delta=0 | H1)
```

The **prior density** at delta=0 is computed by:
1. Sampling `sigma_group` from the prior (HalfNormal)
2. Computing `p(delta=0 | sigma_group)` for each sample
3. Averaging across samples

This correctly marginalizes out sigma_group from the prior. However, the **posterior density** at delta=0 is computed using a KDE on the group contrast samples:

```python
kde = gaussian_kde(contrast_samples)
posterior_at_0 = float(kde.evaluate(0.0)[0])
```

This is the marginal posterior density at 0, which is NOT the same as integrating out sigma_group from the posterior. The correct approach would be:

1. For each posterior draw, extract both `group_effect` AND `sigma_group`
2. Compute `p(delta=0 | sigma_group_draw)` using the Normal density
3. Average across posterior draws

The KDE approach would only be correct if the contrast samples directly represent the parameter of interest, but here they represent a derived quantity (difference of group effects), not the group_effect parameter that sigma_group conditions on.

**Impact:** Bayes Factors computed by this code are unreliable and should not be used for inference.

### H6: Participant ID Type Mismatch (Line 175)

```python
# Line 159: participant_map uses str keys
participant_map = {pid: i for i, pid in enumerate(sorted(str(x) for x in ...))}

# Line 175: participant_to_group uses ORIGINAL type keys
participant_to_group[pid] = gid  # pid is original type (could be int)

# Line 352: lookup uses str key from participant_map
pidx = self.data["participant_map"].get(pid)  # pid from participant_to_group (original type)
```

If participant IDs are integers in the input DataFrame, `participant_map` has string keys like `"1"`, `"2"`, but `participant_to_group` has integer keys like `1`, `2`. The lookup at line 352 will fail to find the key.

### H7: MCMC Convergence Criteria

Current thresholds: Rhat < 1.01, ESS >= 1000, divergences == 0.

- **Rhat < 1.01** is the minimum recommended by Vehtari et al. (2021), but with only 4 chains and 2000 draws, this may not be sufficiently conservative. Consider also checking ESS/n_draws ratio.
- **ESS >= 1000** is reasonable for summary statistics but may be insufficient for tail quantiles (HDI bounds). Consider requiring `ess_tail >= 400` separately.
- **Zero divergences** is correct and appropriately strict.

### M7: ICC Variance Approximation (Lines 820-821)

```python
var_run = (np.pi ** 2) / (3.0 * kappa_vals)
```

This approximates the within-run variance on the logit scale as `pi^2 / (3 * kappa)`. The `pi^2/3` term comes from the logistic distribution variance, which is a standard approximation for Beta distributions. However, this is only accurate when the Beta distribution is symmetric (near p=0.5). For strongly skewed Betas (high or low mu_obs), the approximation deteriorates. Consider using the exact Beta variance formula conditioned on the posterior draws of mu_obs and kappa.

### M8: Shrinkage Calculation Edge Case (Lines 919-921)

```python
if abs(raw - grand_mean) > 1.0:
    shrinkage_pct = round((raw - post_mean) / (raw - grand_mean) * 100, 2)
else:
    shrinkage_pct = 0.0  # Should be NaN, not 0
```

When raw mean is near grand mean, shrinkage percentage is mathematically undefined (0/0). Reporting 0.0 implies "no shrinkage occurred" when the truth is "shrinkage is not measurable." Should use `float('nan')` or `None`.

### Beta Distribution Transformation (Lines 147-152)

The Smithson & Verkuilen (2006) squeeze transformation is correctly implemented:
```python
y = (x * (N-1) + 0.5) / (N * 100)
```
This is appropriate for mapping bounded [0,100] data to (0,1) for Beta regression. No issues found.

### Non-Centered Parameterization (Lines 406-410)

The documentation claims "non-centered throughout" but the implementation uses a mixed parameterization: entity effects are centered (include mu_pop) while participant effects are non-centered (via z_participant). This is actually fine and may improve sampling, but the documentation should be corrected.

---

## 2. Agreement Analysis (agreement.py)

### H8: Expected Disagreement with Missing Data (Line 157)

```python
expected_disagreement = 2 * np.var(all_values, ddof=0)
```

For interval-scale data, `E[(X_i - X_j)^2] = 2 * Var(X)` when computed over the full population. This is mathematically correct for the MCAR (Missing Completely At Random) case. However, Krippendorff's original formulation computes expected disagreement from the **coincidence matrix marginals**, which properly accounts for structured missingness patterns.

If missingness is not MCAR (e.g., one rater systematically skips certain entities), this formula will underestimate expected disagreement, biasing Alpha upward (making agreement appear better than it is). For research applications, the coincidence-matrix-marginal formulation should be used instead.

### M9: Bootstrap Exchangeability Assumption (Lines 194-197)

The bootstrap resamples units (entity-participant pairs) with replacement, assuming exchangeability. If entities have different intrinsic variability (e.g., "water" is easier to score consistently than "governance"), the bootstrap doesn't account for this heterogeneity. A stratified bootstrap by entity type would be more appropriate.

### Interpretation Scale

The alpha interpretation thresholds are:
- < 0.0: "poor"
- 0.0-0.33: "fair"
- 0.33-0.67: "moderate"
- 0.67-1.0: "good"

These are reasonable but more conservative than Krippendorff's own recommendations (he suggests alpha >= 0.667 as minimum for tentative conclusions). The code's "good" threshold at 0.67 aligns with this. However, for LLM-generated scores (which tend to have lower intrinsic variability than human raters), these thresholds may need recalibration.

### Coincidence Matrix Construction

The code generates ordered pairs (i,j) where i != j, creating both (a,b) and (b,a). Since `(a-b)^2 = (b-a)^2`, this doubles computation without changing the result. Functionally correct but inefficient.

---

## 3. Clustering (clustering.py)

### C6: Wrong Input Type to compute_pairwise_distances (Line 381)

```python
distance_matrix = compute_pairwise_distances(
    centroids, metric=metric, aggregate=None  # centroids is numpy array!
)
```

`compute_pairwise_distances()` expects `Dict[str, np.ndarray]` (mapping participant IDs to score vectors), but receives a raw numpy array. This will crash at `scores.keys()` since numpy arrays don't have `.keys()`.

Additionally, the return value is a tuple `(matrix, ids)` but only assigned to a single variable.

### H9: Ward Linkage with Non-Euclidean Distances (Lines 145-150)

Ward's method minimizes within-cluster variance, which requires Euclidean geometry. The code allows any distance metric to be used with Ward linkage without validation. Using Ward with cosine or Aitchison distances produces statistically invalid dendrograms.

**Fix:** Add a check:
```python
if linkage_method == "ward" and metric != "euclidean":
    raise ValueError("Ward linkage requires Euclidean distances")
```

### H10: K-Means Ignores Metric Parameter (Lines 366-382)

K-means always operates in Euclidean space (`KMeans.fit_predict(score_matrix)`), regardless of the `metric` parameter. If a user specifies `metric="cosine"`, K-means still uses Euclidean distances internally. This is inconsistent with hierarchical clustering and HDBSCAN, which do respect the metric parameter.

**Options:**
1. Document that K-means only supports Euclidean
2. Implement spherical K-means for cosine distance
3. Use a metric-agnostic algorithm (e.g., K-medoids/PAM) when metric != "euclidean"

### PCA Implementation

The PCA implementation is correct. When `use_clr=True` and `standardize=True`, both transforms are applied sequentially (CLR first, then z-score). This is legitimate but the interaction should be documented, as the CLR transform already centers the data (subtracts geometric mean on log scale), and subsequent z-scoring changes the geometry further.

### Silhouette Analysis for Optimal k

Silhouette score is an appropriate internal validation metric for determining optimal k. The implementation correctly uses `metric="precomputed"` with distance matrices for hierarchical clustering and raw scores for K-means. The elbow method is provided as an additional heuristic.

---

## 4. Distance Metrics (distances.py)

### CLR/ILR Transforms

Both are correctly implemented:
- CLR: `log(x / geometric_mean(x))`
- ILR: Helmert sub-composition basis

Zero values are handled by clipping to epsilon (1e-10), which is standard practice but can amplify noise for near-zero scores.

### Aitchison Distance

Correctly implemented as the Euclidean distance in CLR-transformed space. However, this metric is **only valid for compositional data** (parts summing to a constant). The docstring correctly warns about this, but there's no runtime validation that input data actually meets the compositional constraint.

### Wasserstein Distance

The sliced Wasserstein approximation uses a hardcoded seed (42) for random projections. This ensures reproducibility but means:
1. Every call uses the same projection directions
2. There's no way to estimate the approximation error
3. Results may be biased if the chosen projections are not representative

**Fix:** Accept `random_seed` as a parameter with default 42.

### Cosine Distance

Correctly implemented as `1 - cosine_similarity`. Handles the zero-vector edge case (returns 1.0).

---

## 5. Comparison Module (comparison.py)

### M11: Unweighted Group Centroids (Lines 662-668)

```python
group_centroid = np.mean(member_centroids, axis=0)
```

If participants have different numbers of entities, each participant contributes equally to the group centroid regardless of their entity count. A participant with 2 entities has the same influence as one with 50 entities. This may or may not be desirable depending on the research question:

- **Current behavior:** Each participant is one observation (defensible if participants are the unit of analysis)
- **Alternative:** Weight by entity count (defensible if entities are the unit of analysis)

This should be documented and configurable.

### Permutation Test

The permutation test is correctly implemented as a one-tailed test (`p = mean(null >= observed)`). The one-tailed direction is appropriate here because we expect between-group distances to be *larger* than within-group distances if groups differ. The effect size metric (ratio of between-group to within-group mean distance) is non-standard but interpretable; it should be documented as such.

### Distance Matrix Properties

All distance matrices are properly symmetric (computed as upper triangle then mirrored). Diagonal entries are correctly set to 0.
