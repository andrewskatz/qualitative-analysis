# Entity Pipeline Audit Findings

**Date:** 2026-02-03
**Auditor:** Senior dev team (elevated review)
**Scope:** Full entity analysis pipeline — `qa entity` subcommands
**Codebase:** ~8,437 lines across 8 Python files in `qualitative-analysis/src/qualitative_analysis/entity/`

---

## Overview

This document captures all findings from a full-stack audit of the entity analysis pipeline, covering statistical methodology, architecture, testing gaps, and documentation. Findings are organized by severity.

---

## CRITICAL Findings

These would produce incorrect research conclusions if left unaddressed.

### C1. Confidence Interval t-value Is Wrong for n=3 Runs

**File:** `models.py:219` — `DimensionScore.from_scores()`
**Status:** TO FIX

`from_scores()` uses a hardcoded `t_value = 2.0` for all n < 30. With the typical n=3 scoring runs (df=2), the correct 95% CI t-value is **4.303**, not 2.0. Reported confidence intervals are roughly **2x too narrow**, systematically understating scoring uncertainty.

**Impact:** Any downstream analysis relying on `{dim}_ci_low` / `{dim}_ci_high` columns will underestimate uncertainty. The Bayesian model is unaffected (it uses raw run scores directly).

**Fix:** Replace the hardcoded approximation with `scipy.stats.t.ppf(0.975, n-1)`.

---

### C2. Aitchison Distance Misapplied to Non-Compositional Data

**File:** `comparison.py:50-82` — `clr_transform()`
**Status:** TO FIX

The CLR transform normalizes input vectors to sum to 1 (lines 76/80) before computing log-ratios. This treats entity scores as compositional data (parts of a whole). However, Social/Ecological/Technological scores are **independent dimensions** — an entity can legitimately score 80/80/80.

The forced normalization destroys information about absolute score levels and manufactures artificial negative correlations. Concretely: an entity scored [80, 80, 80] and another scored [20, 20, 20] both normalize to [1/3, 1/3, 1/3], producing an Aitchison distance of **zero** despite very different absolute scores.

**Impact:** Default distance metric produces misleading results when absolute score levels matter (which they do in this research context).

**Fix:** Change the default metric from `aitchison` to `euclidean`. Retain Aitchison as a non-default option with documentation that it is appropriate only when the *ratio* between dimensions matters (not absolute levels). Add a clear docstring warning.

---

### C3. Smithson-Verkuilen Squeeze Uses Global N Instead of Per-Dimension N

**File:** `bayesian.py:148-149` — `prepare_beta_data()`
**Status:** TO FIX

The squeeze formula `y = (score * (N-1) + 0.5) / (N * 100)` uses `N = len(long_df)` — the total number of rows in the long-format DataFrame across all entities, participants, runs, **and dimensions**. The original Smithson & Verkuilen (2006) paper specifies N as the sample size of the variable being modeled.

Using global N (which can be thousands) makes the squeeze effectively a no-op: `(N-1)/N ≈ 1.0`, so boundary scores are barely moved. With N=2000, a score of 0 maps to ~0.0000025, which just gets clipped to `eps=1e-6`. A score of 100 maps to ~0.99975. The `clip(eps, 1-eps)` is doing all the real work rather than the S&V transform.

**Impact:** Boundary handling is less principled than intended, though not catastrophically wrong due to the clip. The transform doesn't match the referenced methodology.

**Fix:** Compute N per dimension: `N = len(dim_records)` where `dim_records` are the rows for the specific dimension being transformed, or use the total number of observations feeding into the Beta model for that dimension.

---

## HIGH Findings

Significant impact on reliability or maintainability.

### H1. Group Effect Prior sigma_group ~ HalfNormal(0.5) Is Restrictive

**File:** `bayesian.py:348`
**Status:** DOCUMENT / CONSIDER WIDENING

On the logit scale, HalfNormal(0.5) places 95% of group effect mass within ~1.0 logit units. At p=0.5 this translates to ~19 percentage points, but at p=0.8 only ~10 points. Compared with sigma_entity ~ HalfNormal(2) (4x wider), the model strongly assumes entities vary much more than groups.

**Risk:** May pull group contrasts toward zero, masking real group differences.
**Recommendation:** Widen to HalfNormal(1.0) or make configurable.

### H2. kappa Prior Gamma(5, 0.1) Is Heavily Informative

**File:** `bayesian.py:371`
**Status:** DOCUMENT

Mean = 50, sd ~ 22. Strongly assumes moderate run consistency. No sensitivity analysis exists.

**Recommendation:** Consider Gamma(2, 0.1) or Exponential(0.1), or document rationale with sensitivity analysis.

### H3. pi^2/(3*kappa) ICC Approximation Breaks at Extremes

**File:** `bayesian.py:746`
**Status:** DOCUMENT

The logistic variance approximation is reasonable when mu ~ 0.5 but degrades for extreme scores. With many entities scoring 85+, run-level variance is overestimated, biasing ICC decomposition.

### H4. Delta Method SD Conversion Is Crude

**File:** `bayesian.py:815`
**Status:** TO FIX

`posterior_sd_prob = theta_sd * 100 / 4` uses the Jacobian at theta=0 (p=0.5). For entities at theta=2 (p~88%), the actual Jacobian is ~0.10, overestimating SD by 2.5x.

**Fix:** `posterior_sd_prob = theta_sd * sigmoid(theta_mean) * (1 - sigmoid(theta_mean)) * 100`

### H5. Hard-Coded Seed in _sliced_wasserstein Corrupts Global RNG State

**File:** `comparison.py:262`
**Status:** TO FIX

`np.random.seed(42)` is called inside the function on every invocation, resetting global random state. Affects downstream code relying on numpy RNG (including permutation tests).

**Fix:** Use `rng = np.random.default_rng(seed)` as a local generator.

### H6. No Tests for Distance Metric Correctness

**Status:** TO FIX

`clr_transform`, `aitchison_distance`, `wasserstein_distance_compositional`, `cosine_distance`, and `euclidean_distance` have zero unit tests with known expected values. This gap allowed C2 to persist.

### H7. No Tests for Permutation Test or Group Comparison

**Status:** TO FIX

`run_permutation_test()` and `compute_group_distances()` have no formal tests. The permutation test computes a one-tailed p-value (`>= observed`) which is correct for testing "groups more different than chance" but this directionality is undocumented and unvalidated.

### H8. Two Independent Data Loading Paths Create Consistency Risk

**Status:** DOCUMENT / FUTURE REFACTOR

`ParticipantComparison.load_scores()` reads `{dim}_mean` columns. `prepare_beta_data()` reads `{dim}_run{k}` columns. These paths are independent — could disagree on dimension detection, participant filtering, NaN handling, or column naming.

---

## MEDIUM Findings

### M1. comparison.py at 2,347 Lines Violates Single-Responsibility

Distance metrics, pairwise comparison, group comparison, permutation testing, and all visualizations in one file. Should be split into `distances.py`, `comparison_viz.py`, and `comparison.py`.

### M2. entity_cli.py at 2,273 Lines with Inline Report Generation

`run_entity_report()` builds markdown via ~160 lines of string concatenation. Report logic mixed with CLI argument parsing hurts testability.

### M3. Missing Dimension Values Silently Default to 0.0

`comparison.py`, `load_scores()` — If a dimension column is missing for some rows, values default to 0.0 without warning, silently producing wrong distance computations.

### M4. Shrinkage Metric Is Non-Standard and Directionally Ambiguous

`bayesian.py:826-828` — `shrinkage_pct = |post_mean - raw| / |raw - 50| * 100`. Uses absolute value (loses direction), normalizes by distance from 50 (arbitrary center), produces 0% when raw ~ 50 regardless of actual shrinkage.

### M5. Bayesian "HDI" Uses Percentiles, Not True HDI

`bayesian.py:697-698` — Uses `np.percentile(delta, 3)` and `np.percentile(delta, 97)`, which gives an equal-tailed interval (ETI), not a highest density interval. For skewed posteriors these can differ substantially. ArviZ provides `az.hdi()`.

### M6. ESS Threshold of 400 Is Below Common Recommendations

Convergence check requires ESS > 400. Vehtari et al. (2021) recommend ESS > 1000 for reliable posterior summaries.

### M7. SingleRunScore Metadata Lost During Save/Load

`EntityScorer.save()` calls `result.to_dict()` without `include_runs=True`. Individual run metadata (initial_observations, raw_response, processing_time) is permanently lost. Only numeric scores survive via flattened `{dim}_run{k}` columns.

### M8. PPC Scale Conversion Ignores Squeeze Transform

`bayesian.py`, `plot_posterior_predictive()` — Replicated y values converted back via `* 100`, but original data went through S&V squeeze. Inverse transform should be applied.

---

## LOW Findings

### L1. Bayesian Exports Missing from __all__

`__init__.py:69-100` — `BayesianEntityModel`, `BayesianVisualizer`, etc. are imported (lines 57-67) but not in `__all__`.

### L2. Stale Docstring in comparison.py

Module docstring references "Bayesian hierarchical modeling (future)" — the Bayesian module is now implemented.

### L3. Mode Calculation Defaults to First Score When Multimodal

`models.py:208-210` — `statistics.mode()` fallback uses `scores[0]`, which is arbitrary and order-dependent.

### L4. Single-Participant Group Not Flagged for Identifiability

Bayesian model fits with single-participant groups without warning, but group effect is confounded with participant effect.

### L5. get_scores_as_tuple Silently Creates Empty DimensionScore for Missing Dimensions

`models.py:323-328` — Returns 0.0 for missing dimensions without warning.

---

## Implementation Priority

| # | Severity | Finding | Effort | Status |
|---|----------|---------|--------|--------|
| C1 | Critical | CI t-value wrong for n=3 | Small | TO FIX |
| C2 | Critical | Aitchison misapplied to non-compositional data | Medium | TO FIX |
| C3 | Critical | S&V squeeze uses wrong N | Small | TO FIX |
| H4 | High | Delta method SD crude | Small | TO FIX |
| H5 | High | Hard-coded seed corrupts RNG | Small | TO FIX |
| H6 | High | No distance metric tests | Medium | TO FIX |
| H7 | High | No permutation/group tests | Medium | TO FIX |
| H1 | High | Group prior too restrictive | Small | CONSIDER |
| H2 | High | kappa prior informative | Small | DOCUMENT |
| H3 | High | ICC approximation at extremes | Small | DOCUMENT |
| H8 | High | Dual data loading paths | Medium | FUTURE |
