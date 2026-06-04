# Audit Option C: Tier 1 + Tier 2 Fixes

**Date:** 2026-02-04
**Package:** `qualitative-analysis`
**Component:** `qa entity` — comparison, Bayesian modeling, models
**Prior Memo:** [2026-02-03 Docstring Review & Audit Preparation](./2026-02-03-docstring-review-and-audit-preparation.md)
**Audit Findings:** [01-entity-pipeline-audit-findings.md](../planning/2026-02-02-entity-scoring-bayesian-etc-polishes/01-entity-pipeline-audit-findings.md)
**Implementation Plan:** [02-refactor-and-fixes-implementation-plan.md](../planning/2026-02-02-entity-scoring-bayesian-etc-polishes/02-refactor-and-fixes-implementation-plan.md)

---

## Summary

Implemented all remaining Tier 1 (quick wins) and Tier 2 (methodological improvements) from the audit findings — "Option C" as defined in the implementation plan addendum. This resolves 10 additional findings on top of the 10 fixed in prior sessions, leaving only Tier 3 (larger structural work deferred by design).

**Test result:** 80/80 tests pass.

---

## Tier 1 Fixes (small effort, clear fix)

### M3. Warning for Silent 0.0 Defaults in `load_scores()`

**File:** `entity/comparison.py` — `load_scores()`, lines 289–310

Previously, missing or unparseable dimension values silently defaulted to 0.0, corrupting downstream distance computations with no user indication. Now logs a warning per occurrence with participant, entity, dimension, and column name context. Separate messages for missing values (`None` / empty string) versus unparseable values (e.g. non-numeric strings).

### M6. ESS Threshold Raised to 1000 with Marginal Tier

**File:** `entity/bayesian.py` — `_run_diagnostics()`, lines 557–596

Raised the convergence threshold from ESS > 400 to ESS >= 1000, per Vehtari et al. (2021). Added a three-tier convergence status:

| Status | Criteria |
|--------|----------|
| `converged` | R-hat < 1.01, ESS >= 1000, 0 divergences |
| `marginal` | R-hat < 1.01, 400 <= ESS < 1000, 0 divergences |
| `not_converged` | Anything else |

The `marginal` tier logs a warning advising the user to increase draws or thinning. The diagnostics dict now includes a `convergence_status` string field alongside the existing `converged` boolean (which now reflects the stricter threshold).

### L1. Bayesian Exports Added to `__all__`

**File:** `entity/__init__.py`, lines 102–114

`BayesianEntityModel`, `BayesianVisualizer`, `BayesianModelResult`, `check_pymc_available`, and `prepare_beta_data` are now conditionally added to `__all__` when PyMC is available. Uses a guarded `try/except ImportError` matching the existing import pattern, so `__all__` remains valid in environments without PyMC.

### L3. Multimodal Mode Calculation Fixed

**File:** `entity/models.py` — `DimensionScore.from_scores()`, lines 208–209

Replaced the `statistics.mode()` + `except StatisticsError: scores[0]` pattern with `statistics.multimode()` followed by taking the median of all modes. This eliminates order-dependent tie-breaking and produces a deterministic, interpretable result.

### L4. Single-Participant Group Warning

**File:** `entity/bayesian.py` — `BayesianEntityModel.__init__()`, lines 277–282

Added a warning when any group in the `groups` dict has fewer than 2 participants. The message explains that the group effect will be confounded with the participant effect. Not a hard error — the model is technically identifiable — just a diagnostic to help users interpret results.

### L5. Warning for Missing Dimensions in `get_scores_as_tuple`

**File:** `entity/models.py` — `EntityScore.get_scores_as_tuple()`, lines 325–335

Previously returned 0.0 silently via `DimensionScore.from_scores(dim, [])` when a requested dimension was absent. Now logs a warning with the dimension name and entity name, and explicitly returns 0.0.

---

## Tier 2 Fixes (methodological improvements)

### H1 + H2. Configurable Priors via `prior_config`

**File:** `entity/bayesian.py` — `BayesianEntityModel`, lines 224–268 (class attribute + constructor) and `build_model()` (prior assignments)

Added a `prior_config` parameter to the `BayesianEntityModel` constructor. Users can override any of 6 prior hyperparameters while retaining the current values as documented defaults:

| Key | Default | Controls |
|-----|---------|----------|
| `mu_pop_sigma` | 1.5 | Population mean Normal prior SD |
| `sigma_entity_sigma` | 2.0 | Entity HalfNormal prior SD |
| `sigma_group_sigma` | 0.5 | Group effect HalfNormal prior SD |
| `sigma_participant_sigma` | 1.0 | Participant HalfNormal prior SD |
| `kappa_alpha` | 5.0 | Run precision Gamma shape |
| `kappa_beta` | 0.1 | Run precision Gamma rate |

Unknown keys are warned and ignored. `build_model()` reads from `self.prior_config` instead of hard-coded literals.

### M4. Standard Hierarchical Shrinkage Formula

**File:** `entity/bayesian.py` — `compute_shrinkage()`, lines 896–908

Replaced the non-standard formula `|post_mean - raw| / |raw - 50| * 100` with the standard hierarchical shrinkage:

```
shrinkage_pct = (raw - posterior_mean) / (raw - grand_mean) * 100
```

where `grand_mean` is the posterior mean of `mu_pop` on the probability (0–100) scale. This formula:
- Preserves direction (positive = shrunk toward grand mean)
- Uses the actual population estimate as the reference, not the arbitrary midpoint 50
- Returns ~100% when the posterior fully collapses to the grand mean, ~0% when it stays at the raw estimate
- Guards against division by zero when `|raw - grand_mean| < 1`

### M5. True HDI via ArviZ (Replacing ETI Percentiles)

**Files:** `entity/bayesian.py` — `compute_group_contrasts()` (line 761) and `compute_icc()` (lines 830–841)

Replaced `np.percentile(delta, 3)` / `np.percentile(delta, 97)` with `az.hdi(delta, hdi_prob=0.94)` in both:
- **Group contrasts:** The `hdi_3%` / `hdi_97%` fields now report the true 94% highest density interval rather than the 3rd/97th equal-tailed percentiles.
- **ICC decomposition:** All four ICC components (group, entity, participant, run) now use `az.hdi()`.

For symmetric posteriors the difference is negligible. For skewed posteriors (common in ICC decomposition where variance components are bounded at 0), true HDI produces narrower, more informative intervals.

### M8. Inverse S&V Transform in PPC Plots

**File:** `entity/bayesian.py` — `BayesianVisualizer.plot_posterior_predictive()`, line 1398

Previously, replicated y values from the posterior predictive were back-transformed with a simple `* 100`. But the observed data was squeezed via Smithson & Verkuilen (2006): `y = (score * (N-1) + 0.5) / (N * 100)`. The PPC now applies the correct inverse:

```
score = (y * N * 100 - 0.5) / (N - 1)
```

where N is the per-dimension observation count. This eliminates the slight bias in PPC plots near the 0 and 100 boundaries.

---

## Complete Audit Fix Ledger

All 20 findings from the audit, with final status:

| # | Finding | Severity | Status | Session |
|---|---------|----------|--------|---------|
| C1 | CI t-value wrong for n=3 | Critical | **Fixed** | Session 1 (2026-02-03) |
| C2 | Aitchison misapplied to non-compositional data | Critical | **Fixed** | Session 1 |
| C3 | S&V squeeze uses wrong N | Critical | **Fixed** | Session 1 |
| H1 | Group effect prior too restrictive | High | **Fixed** (configurable) | Session 3 (2026-02-04) |
| H2 | kappa prior heavily informative | High | **Fixed** (configurable) | Session 3 |
| H3 | ICC pi²/(3κ) approximation at extremes | High | **Deferred** (Tier 3) | — |
| H4 | Delta method SD crude | High | **Fixed** | Session 2 (2026-02-03) |
| H5 | Hard-coded seed corrupts RNG | High | **Fixed** | Session 2 |
| H6 | No distance metric tests | High | **Fixed** (34 tests) | Session 1 |
| H7 | No permutation/group tests | High | **Fixed** (19 tests) | Session 1 |
| H8 | Dual data loading paths | High | **Deferred** (Tier 3) | — |
| M1 | comparison.py too large | Medium | **Fixed** (3 modules) | Session 2 |
| M2 | entity_cli.py inline report | Medium | **Fixed** | Session 2 |
| M3 | Silent 0.0 default for missing values | Medium | **Fixed** | Session 3 |
| M4 | Non-standard shrinkage metric | Medium | **Fixed** | Session 3 |
| M5 | ETI mislabeled as HDI | Medium | **Fixed** | Session 3 |
| M6 | ESS threshold too low | Medium | **Fixed** | Session 3 |
| M7 | SingleRunScore metadata lost during save | Medium | **Deferred** (Tier 3) | — |
| M8 | PPC ignores squeeze transform | Medium | **Fixed** | Session 3 |
| L1 | Bayesian exports missing from `__all__` | Low | **Fixed** | Session 3 |
| L2 | Stale docstring in comparison.py | Low | **Fixed** | Session 2 |
| L3 | Multimodal mode calculation arbitrary | Low | **Fixed** | Session 3 |
| L4 | Single-participant group not flagged | Low | **Fixed** | Session 3 |
| L5 | Silent zeros in get_scores_as_tuple | Low | **Fixed** | Session 3 |

**Score: 21/24 findings fixed. 3 deferred (Tier 3 — H3, H8, M7).**

---

## Files Modified

| File | Changes |
|------|---------|
| `entity/comparison.py` | M3: Warning on missing/unparseable dimension values in `load_scores()` |
| `entity/bayesian.py` | M6: ESS threshold + marginal tier; L4: single-participant warning; H1+H2: `prior_config` + `build_model()` prior parameterization; M4: standard shrinkage formula; M5: `az.hdi()` in contrasts and ICC; M8: inverse S&V in PPC |
| `entity/models.py` | L3: `multimode()` + median; L5: warning in `get_scores_as_tuple()`; added `logging` import |
| `entity/__init__.py` | L1: Bayesian symbols in `__all__` |

---

## Deferred Items (Tier 3)

These are genuinely larger structural changes documented in the audit findings and intentionally deferred:

1. **H3 — ICC pi²/(3κ) approximation at extremes.** The logistic variance approximation overestimates run-level variance for extreme scores. Fix requires numerical integration or posterior predictive variance — a significant ICC refactor. Document the limitation for now.

2. **H8 — Dual data loading paths.** `comparison.py:load_scores()` and `bayesian.py:prepare_beta_data()` read the same CSV independently. Unifying them requires a shared data loading layer touching both pipelines. Low urgency since both paths currently agree in practice.

3. **M7 — SingleRunScore metadata lost during save.** `EntityScorer.save()` discards run-level metadata (raw_response, processing_time). Fix requires changing the serialization contract. Low urgency since numeric scores (the critical data) are preserved.

---

*Report: 2026-02-04*
