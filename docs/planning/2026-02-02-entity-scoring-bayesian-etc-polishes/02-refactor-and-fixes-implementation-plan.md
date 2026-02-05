# Implementation Plan: Comparison Refactor + Remaining Fixes

**Date:** 2026-02-03
**Prerequisite:** Audit findings (01-entity-pipeline-audit-findings.md), Critical fixes C1–C3 complete, H6/H7 tests complete.
**Goal:** Address M1 (comparison.py split), M2 (report extraction), H4 (delta method), H5 (RNG seed).

---

## Phase 1: Split comparison.py (M1)

`comparison.py` is 2,360 lines mixing three concerns: distance math, comparison logic, and visualization. Split into three focused modules.

### Step 1A: Extract `distances.py`

**Source lines:** comparison.py:47–417 (distance metrics + `compute_pairwise_distances`)

**New file:** `src/qualitative_analysis/entity/distances.py`

Contents (in order):
- Module docstring (metric choice guidance, compositional warning)
- `clr_transform()` (line 52)
- `ilr_transform()` (line 95)
- `aitchison_distance()` (line 142)
- `aitchison_distance_matrix()` (line 173)
- `wasserstein_distance_1d()` (line 200)
- `wasserstein_distance_compositional()` (line 216)
- `_sliced_wasserstein()` (line 259)
- `_exact_wasserstein()` (line 287)
- `cosine_similarity()` (line 317)
- `cosine_distance()` (line 340)
- `compute_pairwise_distances()` (line 359)

**Imports needed:** `numpy`, `logging`, `typing`, `scipy.stats.wasserstein_distance` (lazy)

### Step 1B: Extract `comparison_viz.py`

**Source lines:** comparison.py:1207–2322 (`ComparisonVisualizer` class)

**New file:** `src/qualitative_analysis/entity/comparison_viz.py`

Contents:
- Module docstring
- `ComparisonVisualizer` class (all visualization methods)

**Imports needed:** `numpy`, `logging`, `pathlib`, `typing`, `matplotlib` (lazy inside methods). Will import `ComparisonResult` and `GroupComparisonResult` from `comparison.py`.

### Step 1C: Trim `comparison.py`

**Remaining contents (~800 lines):**
- Module docstring (updated)
- Imports from `distances.py` (re-export for backward compat)
- `ComparisonResult` dataclass (line 426)
- `GroupComparisonResult` dataclass (line 481)
- `ParticipantComparison` class (line 571–1205): load_scores, compute_distances, set_groups, compute_group_distances, run_permutation_test
- `compare_participants()` convenience function (line 2324)

**Key:** `comparison.py` will import distance functions from `distances.py` and use them in `ParticipantComparison`. The public API remains identical.

### Step 1D: Update imports across codebase

Files that import from `comparison.py`:
- `entity/__init__.py` — update to also export from `distances.py` and `comparison_viz.py`
- `entity_cli.py` — imports `ParticipantComparison`, `ComparisonVisualizer` (update source)
- `tests/test_comparison.py` — distance imports now from `distances.py` (or keep via `comparison.py` re-exports)

**Backward compatibility:** `comparison.py` will re-export all distance functions so existing `from qualitative_analysis.entity.comparison import aitchison_distance` still works.

### Step 1E: Verify

- Run `pytest tests/test_comparison.py tests/test_bayesian.py -v` — all 80 tests must pass
- Verify no import errors: `python -c "from qualitative_analysis.entity import *"`

---

## Phase 2: Extract report generation (M2)

### Step 2A: Create `entity_report.py`

**Source lines from entity_cli.py:**
- `add_entity_report_args()` (line 2051)
- `run_entity_report()` (line 2105)

**New file:** `src/qualitative_analysis/entity_report.py`

Move the report-building logic (markdown generation) into a standalone function `generate_comparison_report()` that takes data objects as input (not CLI args). The CLI function in `entity_cli.py` becomes a thin wrapper that parses args and calls the new function.

### Step 2B: Update entity_cli.py

- Remove inline report logic (~170 lines)
- Import and delegate to `entity_report.py`
- Keep `add_entity_report_args()` in CLI file (argument definitions stay with CLI)

### Step 2C: Verify

- Run full test suite

---

## Phase 3: Quick fixes (H4, H5)

### Step 3A: Fix delta method SD (H4)

**File:** `bayesian.py:815`

Replace:
```python
posterior_sd_prob = theta_sd * 100 / 4
```

With:
```python
sigmoid_mean = 1 / (1 + np.exp(-theta_mean))
posterior_sd_prob = theta_sd * sigmoid_mean * (1 - sigmoid_mean) * 100
```

This uses the per-entity Jacobian instead of assuming θ=0.

### Step 3B: Fix hard-coded RNG seed (H5)

**File:** `comparison.py` (now `distances.py`) `_sliced_wasserstein()`, line 262

Replace:
```python
np.random.seed(42)
for _ in range(n_projections):
    direction = np.random.randn(d)
```

With:
```python
rng = np.random.default_rng(42)
for _ in range(n_projections):
    direction = rng.standard_normal(d)
```

This uses a local RNG that doesn't corrupt global state.

### Step 3C: Verify

- Run full test suite — 80+ tests must pass

---

## Execution Order

| Step | Task | Depends on |
|------|------|-----------|
| 1A | Extract `distances.py` | — |
| 1B | Extract `comparison_viz.py` | 1A (needs trimmed comparison.py) |
| 1C | Trim `comparison.py`, add re-exports | 1A, 1B |
| 1D | Update imports in __init__.py, entity_cli.py | 1C |
| 1E | Run tests, verify | 1D |
| 2A | Create `entity_report.py` | 1E |
| 2B | Thin out entity_cli.py | 2A |
| 2C | Verify | 2B |
| 3A | Fix H4 (delta method) | — |
| 3B | Fix H5 (RNG seed) | 1A (seed is in distances.py now) |
| 3C | Verify | 3A, 3B |

---

## Success Criteria

- All existing tests pass (80+)
- No module has >1,000 lines
- `comparison.py` reduced from 2,360 to ~800 lines
- `entity_cli.py` reduced from 2,274 to ~2,100 lines
- Distance functions are independently importable from `distances.py`
- Report generation is independently importable from `entity_report.py`
- All public API backward-compatible (re-exports in place)

---
---

# Addendum: Completion Status + Remaining Work

**Date:** 2026-02-03
**Context:** Phases 1–3 above are complete. This addendum records outcomes and proposes next steps for the remaining audit findings.

---

## Completion Summary

All items from the original plan have been implemented and verified.

| Step | Task | Status | Outcome |
|------|------|--------|---------|
| 1A | Extract `distances.py` | **Done** | 411 lines. H5 (local RNG) applied during extraction. |
| 1B | Extract `comparison_viz.py` | **Done** | 1,133 lines (single cohesive `ComparisonVisualizer` class). |
| 1C | Trim `comparison.py` | **Done** | 885 lines (was 2,360). Re-exports in place. |
| 1D | Update imports | **Done** | `__init__.py`, `entity_cli.py` updated. Backward-compat verified. |
| 1E | Verify | **Done** | 80/80 tests pass. All import paths validated. |
| 2A | Create `entity_report.py` | **Done** | 174 lines. Standalone `generate_comparison_report()`. |
| 2B | Thin `entity_cli.py` | **Done** | 2,170 lines (was 2,274). Thin wrapper delegates to report module. |
| 2C | Verify | **Done** | Tests pass. |
| 3A | Fix H4 (delta method SD) | **Done** | Per-entity Jacobian: `theta_sd * sigmoid(θ_mean) * (1 - sigmoid(θ_mean)) * 100` |
| 3B | Fix H5 (RNG seed) | **Done** | `np.random.default_rng(42)` local generator in `_sliced_wasserstein()` |
| 3C | Verify | **Done** | 80/80 tests pass (including all Bayesian integration tests). |

### Complete Fix Ledger (all sessions)

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| C1 | CI t-value wrong for n=3 | Critical | **Fixed** — `scipy.stats.t.ppf(0.975, df=n-1)` in `models.py` |
| C2 | Aitchison misapplied to non-compositional data | Critical | **Fixed** — Default metric changed to `euclidean`; warning docstring added to `clr_transform()` |
| C3 | S&V squeeze uses wrong N | Critical | **Fixed** — Per-dimension N via `groupby("dimension").transform("count")` in `bayesian.py` |
| H4 | Delta method SD crude | High | **Fixed** — Per-entity Jacobian in `bayesian.py` |
| H5 | Hard-coded seed corrupts RNG | High | **Fixed** — Local `default_rng(42)` in `distances.py` |
| H6 | No distance metric tests | High | **Fixed** — 34 tests added in `test_comparison.py` |
| H7 | No permutation/group tests | High | **Fixed** — 19 tests added in `test_comparison.py` |
| M1 | comparison.py too large | Medium | **Fixed** — Split into 3 modules |
| M2 | entity_cli.py inline report | Medium | **Fixed** — Extracted to `entity_report.py` |
| L2 | Stale docstring in comparison.py | Low | **Fixed** — Module docstring rewritten during split |

---

## Remaining Findings

### Tier 1: Fixable Now (small effort, clear fix)

#### M3. Silent 0.0 Default for Missing Dimension Values
**File:** `comparison.py:292–295` — `load_scores()`
**Risk:** Missing or misnamed dimension columns silently produce 0.0, corrupting all downstream distance computations with no indication to the user.
**Fix:** Log a warning on the first occurrence of a missing/unparseable value per participant-dimension pair. Optionally raise if more than N% of values are missing.
**Effort:** Small.

#### M6. ESS Threshold of 400 Is Below Recommendations
**File:** `bayesian.py:559–560`
**Risk:** ESS < 1000 can produce unreliable posterior summaries (Vehtari et al. 2021). The current threshold of 400 may mark a poorly-converged chain as "converged."
**Fix:** Raise threshold to 1000 (or make configurable with 1000 as default). Add a "warning" tier at 400–1000 that reports convergence as "marginal."
**Effort:** Small.

#### L1. Bayesian Exports Missing from `__all__`
**File:** `entity/__init__.py:69–100`
**Risk:** `BayesianEntityModel`, `BayesianVisualizer`, `BayesianModelResult`, `check_pymc_available`, `prepare_beta_data` are imported but not listed in `__all__`. Tools that use `__all__` (autocompletion, linters, `from pkg import *`) won't expose them.
**Fix:** Add Bayesian symbols to `__all__` (conditional, matching the try/except import pattern).
**Effort:** Small.

#### L3. Mode Calculation Defaults to First Score When Multimodal
**File:** `models.py:208–210`
**Risk:** `statistics.mode()` fallback returns `scores[0]` when multimodal, which is order-dependent and arbitrary.
**Fix:** Use `statistics.multimode()` and return the first (or median of modes). Or document the tie-breaking rule.
**Effort:** Small.

#### L4. Single-Participant Group Not Flagged
**File:** `bayesian.py` — model build
**Risk:** A group with one participant confounds the group effect with the participant effect. The model fits silently, producing uninterpretable group posteriors.
**Fix:** Add a warning when any group has < 2 participants. Not a hard error (the model is still identifiable), just a diagnostic.
**Effort:** Small.

#### L5. `get_scores_as_tuple` Silently Returns 0.0 for Missing Dimensions
**File:** `models.py:323–328`
**Risk:** Silently filling zeros when a dimension is absent produces misleading score tuples.
**Fix:** Log a warning.
**Effort:** Small.

### Tier 2: Methodological Improvements (moderate effort, domain judgment required)

#### H1. Group Effect Prior sigma_group ~ HalfNormal(0.5) Is Restrictive
**File:** `bayesian.py:350`
**Current:** HalfNormal(0.5) places 95% of group effect mass within ~1.0 logit unit.
**Concern:** May pull group contrasts toward zero, masking real differences.
**Options:**
1. Widen to HalfNormal(1.0) — simple, broadly reasonable.
2. Make configurable via a `prior_config` dict parameter.
3. Leave as-is but document the choice with a sensitivity analysis.
**Recommendation:** Option 2 (configurable) addresses H1 and H2 together. Sensible defaults with override capability.

#### H2. kappa Prior Gamma(5, 0.1) Is Heavily Informative
**File:** `bayesian.py:373`
**Current:** Mean=50, sd~22. Strongly assumes moderate run consistency.
**Options:**
1. Weaken to Gamma(2, 0.1) or Exponential(0.1).
2. Make configurable (see H1 recommendation).
3. Document rationale; run sensitivity analysis.
**Recommendation:** Same as H1 — make configurable with current values as documented defaults.

#### M4. Shrinkage Metric Is Non-Standard
**File:** `bayesian.py:831–834`
**Current:** `|post_mean - raw| / |raw - 50| * 100` — loses direction, normalizes by distance from 50, produces 0% when raw ~ 50.
**Options:**
1. Replace with directional shrinkage: `(raw - post_mean) / (raw - grand_mean) * 100` where `grand_mean` is the overall posterior mean. This is the standard hierarchical shrinkage formula.
2. Keep the current metric but rename to `absolute_movement_pct` and add the standard formula as a second column.
**Recommendation:** Option 1. The standard formula is well-understood and directly interpretable.

#### M5. Bayesian "HDI" Uses Percentiles (ETI), Not True HDI
**File:** `bayesian.py:697–700`
**Current:** `np.percentile(delta, 3)` and `np.percentile(delta, 97)` — equal-tailed interval, not highest density interval.
**Impact:** For skewed posteriors these can differ substantially. The labels say "hdi_3%" / "hdi_97%" but the values are ETI.
**Options:**
1. Switch to `az.hdi(delta, hdi_prob=0.94)` — correct HDI, already have ArviZ as a dependency.
2. Rename columns to `eti_3%` / `eti_97%` to be accurate.
**Recommendation:** Option 1. ArviZ is already imported elsewhere. True HDI is more appropriate for reporting group contrasts.

#### M8. PPC Scale Conversion Ignores Squeeze Transform
**File:** `bayesian.py`, `plot_posterior_predictive()`
**Current:** Replicated y values multiplied by 100, but original data went through S&V squeeze.
**Impact:** PPC plots show slightly biased back-transformed values near boundaries.
**Fix:** Apply inverse S&V: `score = (y * N_dim * 100 - 0.5) / (N_dim - 1)`, using the same per-dimension N from C3 fix.
**Effort:** Small-medium (need to pass N_dim through to the viz method).

### Tier 3: Larger Structural Work (future)

#### H3. ICC pi^2/(3*kappa) Approximation at Extremes
**File:** `bayesian.py:746`
**Nature:** The logistic variance approximation overestimates run-level variance for extreme scores (p > 0.85 or p < 0.15). Affects ICC decomposition.
**Fix:** Replace with numerical integration or posterior predictive variance. Significant refactor of the ICC computation.
**Recommendation:** Document the limitation for now. Fix when ICC results are used for publication-level claims.

#### H8. Dual Data Loading Paths
**Files:** `comparison.py:load_scores()` reads `{dim}_mean`; `bayesian.py:prepare_beta_data()` reads `{dim}_run{k}`.
**Risk:** Independent paths could disagree on dimension detection, filtering, NaN handling.
**Fix:** Unify into a shared data loading layer. Large refactor touching both pipelines.
**Recommendation:** Future work. The current paths are consistent in practice because they read the same CSV produced by `EntityScorer.save()`.

#### M7. SingleRunScore Metadata Lost During Save
**File:** `scorer.py` — `EntityScorer.save()` calls `result.to_dict()` without `include_runs=True`.
**Impact:** Run-level metadata (initial_observations, raw_response, processing_time) is permanently discarded. Only flattened `{dim}_run{k}` numeric scores survive.
**Fix:** Default `include_runs=True` or add a separate metadata save. Touches serialization contract.
**Recommendation:** Future work. Low urgency since the numeric scores (the critical data) are preserved.

---

## Option C Completion (2026-02-04)

**Selected option:** C — Comprehensive (all Tier 1 + Tier 2). Implemented 2026-02-04.

### Tier 1 Fixes Completed

| Finding | File | Change |
|---------|------|--------|
| M3 | `comparison.py` | Warning on missing/unparseable dimension values in `load_scores()` |
| M6 | `bayesian.py` | ESS threshold raised to 1000; added "marginal" tier (400–1000) with warning; `convergence_status` field |
| L1 | `__init__.py` | 5 Bayesian symbols conditionally added to `__all__` |
| L3 | `models.py` | `statistics.multimode()` + median of modes (replaces arbitrary `scores[0]`) |
| L4 | `bayesian.py` | Warning when any group has < 2 participants |
| L5 | `models.py` | Warning on missing dimension in `get_scores_as_tuple()` |

### Tier 2 Fixes Completed

| Finding | File | Change |
|---------|------|--------|
| H1+H2 | `bayesian.py` | `prior_config` dict on constructor (6 keys: `mu_pop_sigma`, `sigma_entity_sigma`, `sigma_group_sigma`, `sigma_participant_sigma`, `kappa_alpha`, `kappa_beta`). Unknown keys warned and ignored. `build_model()` reads from `self.prior_config`. |
| M4 | `bayesian.py` | Standard shrinkage: `(raw - posterior) / (raw - grand_mean) * 100`, using posterior `mu_pop` as grand mean |
| M5 | `bayesian.py` | `az.hdi(delta, hdi_prob=0.94)` in both `compute_group_contrasts()` and `compute_icc()` |
| M8 | `bayesian.py` | Inverse S&V transform in PPC: `(y * N * 100 - 0.5) / (N - 1)` |

### Final Audit Ledger

**21 of 24 findings fixed.** Remaining 3 are Tier 3 (deferred by design):
- H3: ICC pi²/(3κ) approximation at extremes
- H8: Dual data loading paths
- M7: SingleRunScore metadata lost during save

**Tests:** 80/80 pass.
**Progress note:** `docs/progress-notes/2026-02-04-audit-option-c-tier1-tier2-fixes.md`

---

## Phase 4: Statistical Methods Expansion

**Date planned:** 2026-02-04
**Source:** Remaining work tracker (`00-remaining-work.md`, §5 — Statistical Methods Not Yet Implemented)
**Goal:** Implement five additional statistical methods that complete the original comparison plan's statistical catalog.

### Step 4A: Krippendorff's Alpha (Multi-Rater Agreement)

**Priority:** Medium
**Source:** Comparison plan §3.2
**Purpose:** Quantifies agreement among multiple raters (LLM scoring runs or participants) on ordinal/interval data. Unlike Cohen's kappa (2 raters only), Krippendorff's alpha handles arbitrary numbers of raters and missing data.

**Design:**
- Add `compute_krippendorff_alpha()` to a new file `src/qualitative_analysis/entity/agreement.py`
- Input: scored DataFrame + dimension name + unit config (per-entity or per-participant)
- Output: alpha value, bootstrap 95% CI, interpretation label (poor/fair/moderate/good/excellent)
- Support both run-level agreement (are 3 LLM runs consistent?) and participant-level agreement (do participants agree on entity scores?)
- Use MASI or interval distance metric depending on score type

**Effort:** Medium. Core algorithm is ~80 lines. Bootstrap CI adds ~30 lines.

### Step 4B: LOO-CV Model Comparison

**Priority:** Low
**Source:** Model Spec §6.2
**Purpose:** Use leave-one-out cross-validation (`az.loo()`) to compare alternative Bayesian model specifications. Enables principled model selection (e.g., does adding group effects improve fit?).

**Design:**
- Add `compare_models()` method to `BayesianEntityModel`
- Fits a reduced model (no group effect) alongside the full model
- Returns `az.compare()` table with ELPD, SE, weights
- Requires computing `log_likelihood` in the PyMC model (add `pm.compute_log_likelihood()` call after sampling)

**Effort:** Medium. Needs `log_likelihood` added to model spec + wrapper around ArviZ comparison.

### Step 4C: Participant Clustering

**Priority:** Low
**Source:** Comparison plan §5.1
**Purpose:** Discover natural participant groupings from scoring patterns, without pre-specified group labels. Useful for exploratory analysis.

**Design:**
- Add `src/qualitative_analysis/entity/clustering.py`
- Implement:
  1. **Hierarchical clustering** (scipy.cluster.hierarchy) with dendrogram visualization
  2. **k-means** (sklearn) with elbow/silhouette analysis for optimal k
  3. **HDBSCAN** (hdbscan) for density-based clustering with noise detection
- Input: participant score matrix (from `ParticipantComparison.compute_distances()`)
- Output: cluster labels, silhouette scores, dendrogram/scatter plots
- GMM (soft assignments) as optional extension

**Effort:** Medium-large. Multiple algorithms, each with visualization. sklearn/hdbscan are optional deps.

### Step 4D: Bayes Factor

**Priority:** Low
**Source:** Comparison plan §4.3
**Purpose:** Evidence ratio for "groups differ" vs "no difference." Complements posterior p(direction) with a measure of evidence strength.

**Design:**
- Add `compute_bayes_factor()` method to `BayesianEntityModel`
- Savage-Dickey density ratio: evaluate posterior and prior density of group contrast at zero
- Output: BF₁₀, interpretation (anecdotal/moderate/strong/decisive per Jeffreys scale)
- Alternative: bridge sampling via `az.bridgesampling` if available

**Effort:** Medium. Savage-Dickey is straightforward with KDE on posterior samples.

### Step 4E: PCA on CLR Scores

**Priority:** Low
**Source:** Comparison plan §5.3
**Purpose:** 2D participant map showing linear relationships between scoring patterns. Reduces dimensionality for visualization when there are many dimensions.

**Design:**
- Add to `clustering.py` or a new `dimensionality.py`
- Apply CLR transform to participant mean score vectors, then PCA
- Output: explained variance ratios, biplot with participant labels colored by group, loading vectors
- Use sklearn PCA

**Effort:** Small-medium. ~50 lines for PCA + ~80 lines for biplot visualization.

### Execution Order

| Step | Task | Depends on | Effort |
|------|------|-----------|--------|
| 4A | Krippendorff's Alpha | — | Medium |
| 4B | LOO-CV model comparison | Bayesian model fitted | Medium |
| 4C | Participant clustering | Distance matrix computed | Medium-large |
| 4D | Bayes Factor | Bayesian model fitted | Medium |
| 4E | PCA on CLR scores | — | Small-medium |

### Success Criteria

- Each method has at least basic unit tests
- Methods integrate with existing CLI where appropriate
- Documentation strings follow existing conventions
- All existing tests continue to pass
