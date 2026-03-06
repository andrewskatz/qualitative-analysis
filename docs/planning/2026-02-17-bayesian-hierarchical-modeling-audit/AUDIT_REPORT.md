# Bayesian Hierarchical Modeling — Implementation Audit

**Date:** 2026-02-17  
**Scope:** `qualitative_analysis/entity/bayesian.py` (2076 lines), `tests/test_bayesian.py` (687 lines), CLI integration in `entity_cli.py`  
**Objective:** Identify logical flaws, statistical issues, implementation bugs, and areas for improvement in the Bayesian hierarchical model used for entity scoring and group comparison.

---

## Executive Summary

The implementation is well-structured and demonstrates strong familiarity with Bayesian methodology. It includes appropriate features: non-centered parameterization, ROPE analysis, ICC decomposition, prior/posterior predictive checks, LOO-CV model comparison, and Savage-Dickey Bayes Factors. However, this audit identifies **5 high-severity issues**, **6 medium-severity issues**, and **6 low-severity items**. The most critical findings involve incorrect variance decomposition for ICC, a misspecified posterior predictive inverse transform, and inconsistent HDI labeling.

---

## Table of Contents

1. [High-Severity Findings](#1-high-severity-findings)
2. [Medium-Severity Findings](#2-medium-severity-findings)
3. [Low-Severity / Style Findings](#3-low-severity--style-findings)
4. [Test Coverage Gaps](#4-test-coverage-gaps)
5. [Recommendations Summary](#5-recommendations-summary)

---

## 1. High-Severity Findings

### H1. ICC Run-Variance Formula is Incorrect

**File:** `bayesian.py`, lines 819–821  
**Code:**
```python
var_run = (np.pi ** 2) / (3.0 * kappa_vals)
```

**Problem:** The code comment says *"On logit scale, approximate as π²/(3·κ)"*. The standard logistic-distribution variance **π²/3** is the residual variance of a logistic regression (i.e., the variance of the logistic distribution used in the probit/logit link). For a Beta distribution with concentration κ, the variance on the (0,1) scale is `μ(1−μ)/(κ+1)`. Converting this to the logit scale via the delta method gives approximately `1/(μ(1−μ)(κ+1))`, which for μ≈0.5 simplifies to `4/(κ+1)`. The formula `π²/(3κ)` conflates two different quantities:

1. **π²/3** — the inherent variance of the logistic link function (a constant, not observation-level noise).
2. **Observation-level variance** — which depends on κ and the mean.

Dividing π²/3 by κ has no standard derivation. As κ grows, this goes to 0, but the logistic-link variance is a structural constant, not reducible by data precision.

**Impact:** ICC values for "run" variance are biased. When κ is large, `var_run` shrinks toward zero and inflates the relative contribution of other variance components. When κ is small, it inflates run-level variance.

**Recommendation:** Use the delta-method approximation for the Beta-to-logit transformation:
```python
# Approximate logit-scale run variance via delta method
# Var_logit ≈ 1 / (mu * (1-mu) * (kappa + 1))
# Using the observation-level mu from the model, or pooling across observations
var_run = 1.0 / (kappa_vals + 1)  # Simplified for mu ≈ 0.5
```
Or, since the entire linear predictor is on the logit scale, directly decompose variance of `eta_ep` from the posterior samples instead of mixing parametric approximations.

---

### H2. Group Contrast Ignores Entity-Level Variation

**File:** `bayesian.py`, lines 753–761  
**Code:**
```python
prob_a = 1 / (1 + np.exp(-(mu_pop + eff_a))) * 100
prob_b = 1 / (1 + np.exp(-(mu_pop + eff_b))) * 100
delta = prob_a - prob_b
```

**Problem:** The group contrast is computed as `invlogit(mu_pop + group_effect_A) − invlogit(mu_pop + group_effect_B)`. This computes the difference between **marginal group means** ignoring entity-level and participant-level effects. While this is a defensible quantity to report, it is **not** the "expected difference in scores" that a user would observe in practice, because the nonlinearity of the inverse-logit transformation means:

- `E[invlogit(X)] ≠ invlogit(E[X])` (Jensen's inequality)
- The actual expected score for a group member involves integrating over entity and participant effects

This means the reported group difference is computed at the **center of the logit-scale distribution** (essentially, for the "average entity scored by the average participant"), not the average across entities and participants.

**Impact:** For moderately wide entity distributions (σ_entity > 0.5), the marginal group means can differ substantially from the ones reported. The magnitude of group contrasts is exaggerated when the sigmoid is evaluated at its steepest region.

**Recommendation:** Document clearly that the contrast is a "typical-entity, typical-participant" comparison. For a more representative marginal comparison, integrate over entity and participant effects:
```python
# Per posterior sample:
#   for each entity e, participant p in group g:
#     score = invlogit(theta[e] + group_eff[g] + z_participant * sigma_participant)
#   then average across e, p
```
Or add this as an alternative output (`marginal_contrast` vs. `conditional_contrast`).

---

### H3. HDI Labels Say 3%/97% but Use 94% HDI

**File:** `bayesian.py`, lines 764 (`hdi_prob=0.94`), 700–706, 830–833  

**Problem:** Throughout the code, `az.hdi(..., hdi_prob=0.94)` is called, which returns a **94% HDI** (i.e., the central 94% credible interval). However, the result keys are labeled `"hdi_3%"` and `"hdi_97%"`. A 94% HDI has its boundaries at approximately the 3rd and 97th percentiles only if the posterior is symmetric and unimodal. The labels should be `"hdi_lower"` and `"hdi_upper"` (or `"hdi_3%"` / `"hdi_97%"` only if using `np.percentile`).

In `summarize_posteriors()` (line 697–706), **percentiles** are used instead of `az.hdi`:
```python
"hdi_3%": round(float(np.percentile(samples, 3)), 4),
"hdi_97%": round(float(np.percentile(samples, 97)), 4),
```
These are **equal-tailed intervals (ETI)**, not HDI. They are labeled as HDI but computed as percentiles. These are different for skewed posteriors (e.g., HalfNormal variance parameters).

**Impact:** Misidentification of interval type. For right-skewed posteriors (σ parameters), the HDI is narrower and asymmetric compared to the ETI. Reporting ETI as HDI overstates uncertainty on one side.

**Recommendation:** 
1. Either use `az.hdi()` consistently and label as `"hdi_lower"` / `"hdi_upper"` (with documented HDI probability).
2. Or use percentiles consistently and label as `"eti_3%"` / `"eti_97%"`.
3. Pick one and apply it everywhere.

---

### H4. Posterior Predictive Inverse Transform is Wrong

**File:** `bayesian.py`, lines 1806–1807  
**Code:**
```python
rep_scores = (flat[r_idx] * n_dim * 100 - 0.5) / max(n_dim - 1, 1)
```

**Problem:** The Smithson & Verkuilen (2006) squeeze applied during data preparation is:
```python
y = (score / 100 * (N - 1) + 0.5) / N
```
The correct inverse is:
```python
score = (y * N - 0.5) / (N - 1) * 100
```

The code has `y * n_dim * 100` instead of `(y * n_dim - 0.5) / (n_dim - 1) * 100`. The multiplication by 100 is inside the parentheses rather than outside, and the order of operations is wrong.

**Impact:** Posterior predictive check plots display replicated data on an incorrect scale, making visual calibration against observed data unreliable.

**Recommendation:** Fix the inverse transform:
```python
rep_scores = (flat[r_idx] * n_dim - 0.5) / max(n_dim - 1, 1) * 100
```

---

### H5. CLI LOO-CV Uses Undefined Variable `chains`

**File:** `entity_cli.py`, line 1819  
**Code:**
```python
loo_result = bayesian_model.compare_models(
    dim,
    chains=chains,       # <-- 'chains' is not defined; should be 'args.chains'
    draws=args.draws,
    tune=args.tune,
    sampler=args.sampler,
)
```

**Problem:** The variable `chains` is not defined in the local scope. `args.chains` is the correct reference. This will raise a `NameError` at runtime when `--loo-compare` is used.

**Impact:** The `--loo-compare` feature is broken and will crash.

**Recommendation:** Change `chains=chains` to `chains=args.chains`.

---

## 2. Medium-Severity Findings

### M1. Shrinkage Computed from Point Estimate, Not Full Posterior

**File:** `bayesian.py`, lines 889–898

**Problem:** The posterior entity means are computed from `theta_mean = theta_samples.mean(axis=(0, 1))`, then transformed via `invlogit(theta_mean)`. This is a point estimate transformation. Due to Jensen's inequality, `invlogit(E[θ]) ≠ E[invlogit(θ)]`. The correct posterior mean on the probability scale is:
```python
posterior_prob = np.mean(1 / (1 + np.exp(-theta_samples)), axis=(0, 1)) * 100
```

Similarly, the posterior SD uses the delta method approximation instead of computing the actual posterior standard deviation of `invlogit(θ)` samples.

**Impact:** Mild bias in reported posterior means and SDs for entities with large θ uncertainty.

**Recommendation:** Compute `invlogit` on each posterior sample first, then take the mean/SD.

---

### M2. Shrinkage Formula Can Produce Values > 100% or Negative

**File:** `bayesian.py`, lines 915–923

**Problem:** The shrinkage formula `(raw - posterior) / (raw - grand_mean) * 100` can yield values outside [0, 100]. If the posterior estimate is on the opposite side of the grand mean from the raw estimate (overshoot), shrinkage exceeds 100%. If the posterior moves *away* from the grand mean, shrinkage is negative. No clamping or flagging is applied.

**Impact:** Confusing output for users, and values >100% or <0% are hard to interpret.

**Recommendation:** Clamp to [0, 100] or flag unusual values with a note explaining the phenomenon.

---

### M3. LOO-CV SE Calculation Naïvely Adds Variances

**File:** `bayesian.py`, line 1219  
**Code:**
```python
se_diff = float(np.sqrt(loo_full.se**2 + loo_reduced.se**2))
```

**Problem:** This treats the two ELPD estimates as independent. However, they are computed on the **same data**, so they are correlated. The correct SE of the difference is already provided by `az.compare()` (the `dse` column in the comparison table). Using uncorrelated SE overestimates uncertainty.

**Impact:** The decision thresholds (`elpd_diff > 2 * se_diff`) become overly conservative, biasing toward "inconclusive" when group effects may actually be meaningful.

**Recommendation:** Extract the SE of the difference directly from the `az.compare()` output, which accounts for the correlation between pointwise log-likelihoods.

---

### M4. Prior Sensitivity Is Hard-Coded

**File:** `bayesian.py`, lines 361–370 (default priors)

**Problem:** The prior configuration is set with fixed defaults:
```python
"mu_pop_sigma": 2.0,
"sigma_entity_sigma": 1.0,
"sigma_group_sigma": 0.5,
"sigma_participant_sigma": 1.0,
"kappa_alpha": 2.0,
"kappa_beta": 0.1,
```

These are reasonable for 0-100 endpoint-bounded scores, but there is no built-in prior sensitivity analysis facility. The `kappa` prior `Gamma(2, 0.1)` has mean 20 and a wide tail — this may dominate the data when observation counts are small.

**Impact:** Results may be sensitive to prior choices without the user being aware.

**Recommendation:** 
1. Add a `prior_sensitivity_check()` method that re-fits with 2–3 alternative prior specs.
2. Document the implied prior ranges (e.g., prior predictive check of score distributions).

---

### M5. Bayes Factor Density Estimation Can Be Unreliable

**File:** `bayesian.py`, lines 1313–1318

**Problem:** The posterior density at zero is estimated via Gaussian KDE. For posterior distributions concentrated far from zero (i.e., when groups clearly differ), the density at zero is estimated in the tail of the distribution, where KDE is known to be unreliable. The BF can swing wildly depending on KDE bandwidth.

Additionally, the Savage-Dickey ratio is only valid when the null hypothesis (group_effect = 0) is a "nested" model with the same prior on all other parameters. Since group effects share the `sigma_group` hyperprior, and `sigma_group` is estimated from the data, the Savage-Dickey ratio is technically a Savage-Dickey **approximation**, not exact.

**Impact:** BF values in the tails (BF > 100 or BF < 0.01) may be numerically unstable.

**Recommendation:**
1. Use bridge sampling (e.g., `bridgesampling` package) for more robust BF estimation.
2. At minimum, log a warning when the posterior mean of the contrast is more than 3 SDs from zero, indicating tail-density estimation is unreliable.

---

### M6. No Handling of 0 or 100 Scores Before Squeeze

**File:** `bayesian.py`, `prepare_beta_data()` (lines ~190–250)

**Problem:** The Smithson & Verkuilen squeeze `y = (score/100 * (N-1) + 0.5) / N` correctly maps (0, 100) to (0, 1) without hitting the boundaries. However, if `score` is exactly 0 or 100 (which is possible in the integer-scored data), the squeezed values are `0.5/N` and `(N-0.5)/N` respectively. These are valid but extremely close to the boundary, which can cause the Beta likelihood to have very steep gradients. No special handling or warning is given.

**Impact:** Extreme boundary scores may cause sampling difficulties (divergences) without clear diagnostic feedback pointing to the cause.

**Recommendation:** Log a warning when boundary scores are present. Consider using a slightly wider squeeze (e.g., `(score/100 * (N-2) + 1) / N`) for boundary robustness.

---

## 3. Low-Severity / Style Findings

### L1. `summarize_posteriors()` Mismatched Sample Lengths

**File:** `bayesian.py`, lines 699–702

When computing group-level means on the probability scale, `mu_pop_samples` and `samples` (group effect) are both `.flatten()`'d. If chains have different lengths (which shouldn't happen normally but could in interrupted runs), the element-wise operation would silently produce incorrect results.

**Recommendation:** Assert that both arrays have the same length.

---

### L2. `_run_diagnostics` Does Not Check for NaN R-hat

When convergence fails dramatically, `az.rhat()` can return NaN for some parameters. The current code computes `rhat_max` which would be NaN (propagated by `np.max`), but then the comparison `rhat_max < 1.01` returns False, causing convergence to be flagged as failed. This is the right outcome but for the wrong reason — the log message would show `R-hat=nan`, which is confusing.

**Recommendation:** Explicitly check for and report NaN R-hat values.

---

### L3. Ternary Plot Dimension Ordering May Confuse Users

**File:** `bayesian.py`, lines 1928–1933 and 1952

The ternary plot maps dimensions to vertices as `(bottom-right=dim[0], top=dim[1], bottom-left=dim[2])`, but this ordering isn't explained in axis labels. The `_bary_to_cart` arguments `(a=top, b=bottom-left, c=bottom-right)` are called with `(row[1], row[2], row[0])`, making dimension 1 top, dimension 2 bottom-left, dimension 0 bottom-right.

**Recommendation:** Add a clear legend or documentation showing which dimension maps to which vertex.

---

### L4. `compute_group_contrasts` Computes Contrast on Probability Scale Only

There is no option to report contrasts on the logit scale (which is the natural scale for the model's linear predictor). For some use cases, the logit-scale contrast is more directly interpretable and avoids the nonlinear-compression issues noted in H2.

**Recommendation:** Add an option or secondary output for logit-scale contrasts (already computed internally: `eff_a - eff_b`).

---

### L5. `BayesianModelResult` Has No `model` Reference

The `BayesianModelResult` dataclass stores the trace and diagnostics but not a reference to the PyMC model object. Methods like `posterior_predictive_check()` must therefore rebuild the model from scratch. This is wasteful and introduces a risk of model/trace mismatch if `build_model` behavior changes between calls.

**Recommendation:** Store the compiled model in the result or cache it.

---

### L6. No Seed Propagation for Posterior Predictive

**File:** `bayesian.py`, lines 960–982

`posterior_predictive_check()` does not accept or propagate a random seed, making results non-reproducible across calls.

**Recommendation:** Accept an optional `random_seed` parameter.

---

## 4. Test Coverage Gaps

| Area | Current Coverage | Gap |
|------|-----------------|-----|
| `prepare_beta_data` | ✅ Thorough | — |
| `build_model` | ✅ Good (shape, variables) | No test for prior predictive distribution range |
| `fit` / `fit_all_dimensions` | ✅ Integration tests on synthetic data | Tests only use 2 chains / 500 draws; no test for `nutpie` sampler |
| `compute_group_contrasts` | ✅ Direction + ROPE tested | No test for > 2 groups; no test for ROPE boundary behavior |
| `compute_icc` | ⚠️ Only checks ICC ordering | No test against known variance decomposition |
| `compute_shrinkage` | ❌ Not tested at all | Missing |
| `compare_models` (LOO-CV) | ❌ Not tested | Missing |
| `compute_bayes_factor` | ❌ Not tested | Missing |
| `summarize_posteriors` | ❌ Not tested | Missing |
| `BayesianVisualizer` | ❌ Not tested | No tests for plot generation |
| `save_results` | ❌ Not tested | No round-trip test |
| Edge cases (1 entity, 1 run, many groups) | ❌ Not tested | Missing |
| Boundary scores (0 or 100) | ❌ Not tested | Missing |
| `posterior_predictive_check` | ❌ Not tested beyond compilation | Missing |

---

## 5. Recommendations Summary

### Priority Actions (Should Fix)

| # | Finding | Severity | Effort |
|---|---------|----------|--------|
| H1 | Fix ICC run-variance formula | High | Medium |
| H3 | Standardize HDI vs ETI labeling | High | Low |
| H4 | Fix posterior predictive inverse transform | High | Low |
| H5 | Fix undefined `chains` variable in CLI LOO-CV | High | Trivial |
| M1 | Compute posterior means from samples, not point transforms | Medium | Low |
| M3 | Use correlated SE from `az.compare()` in LOO-CV | Medium | Low |

### Recommended Improvements (Should Consider)

| # | Finding | Severity | Effort |
|---|---------|----------|--------|
| H2 | Document or fix marginal vs. conditional contrast | High | Medium |
| M2 | Clamp or flag extreme shrinkage values | Medium | Low |
| M4 | Add prior sensitivity analysis utility | Medium | Medium |
| M5 | Improve BF tail-density reliability | Medium | Medium |
| M6 | Handle or warn about boundary scores | Medium | Low |

### Nice-to-Have

| # | Finding | Severity | Effort |
|---|---------|----------|--------|
| L1–L6 | Style and minor usability items | Low | Low each |
| Tests | Expand coverage per table above | — | Medium |

---

*This audit covers the implementation as of the commit present on 2026-02-17. It is a static analysis — findings H2, M4, and M5 could benefit from empirical validation with real study data.*
