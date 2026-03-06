# Bayesian Audit Fixes — Implementation Plan

Implements all 17 findings from [AUDIT_REPORT.md](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/docs/planning/2026-02-17-bayesian-hierarchical-modeling-audit/AUDIT_REPORT.md). Work is organized into 5 phases, ordered so that foundational fixes land first and tests validate everything at the end.

## User Review Required

> [!IMPORTANT]
> **H2 — Group contrast computation:** The current code reports contrasts at the "typical entity, typical participant" point. The fix options are:
> 1. **Document-only** — keep current behavior, add clear docstring/output labels (`conditional_contrast`).
> 2. **Add marginal contrast** — compute a second quantity that integrates over entity/participant effects. Report both.
> 3. **Replace** — switch entirely to the marginal contrast.
>
> **Recommendation:** Option 2 (add marginal, keep both). This is non-breaking and gives users the richer picture.

> [!IMPORTANT]
> **M4 — Prior sensitivity:** Should we add a full `prior_sensitivity_check()` method that re-fits with alternative priors, or just improve documentation of the default priors?
>
> **Recommendation:** Add a lightweight method that re-fits with 4 alternative prior specs and returns a comparison table. Skip full cross-validation of priors to keep runtime manageable.

---

## Phase 1 — Critical Bug Fixes

Trivial / low-risk fixes that resolve crashes or clearly wrong outputs.

### Core module

#### [MODIFY] [bayesian.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/entity/bayesian.py)

- **H5 — Fix CLI LOO-CV crash:** In `entity_cli.py` line 1819, change `chains=chains` → `chains=args.chains`. *(Actually in CLI file, listed here for grouping.)*
- **H4 — Fix PPC inverse transform** (line 1807): Change `flat[r_idx] * n_dim * 100 - 0.5` to `(flat[r_idx] * n_dim - 0.5) / max(n_dim - 1, 1) * 100`.
- **H3 — Standardize interval labeling:**
  - In `summarize_posteriors()`: Replace `np.percentile(samples, 3/97)` with `az.hdi(samples, hdi_prob=0.94)` and rename keys to `"hdi_lower"` / `"hdi_upper"`.
  - In `compute_group_contrasts()`, `compute_icc()`: Rename `"hdi_3%"` / `"hdi_97%"` → `"hdi_lower"` / `"hdi_upper"`.
  - Add `"hdi_prob": 0.94` to every dict containing HDI bounds for self-documentation.

#### [MODIFY] [entity_cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/entity_cli.py)

- **H5:** Line 1819: `chains=chains` → `chains=args.chains`.

#### [MODIFY] [entity_report.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/entity_report.py)

- Update references to `hdi_3%` / `hdi_97%` → `hdi_lower` / `hdi_upper` in the report template.

---

## Phase 2 — Statistical Corrections

Fixes that change numerical outputs. These require careful verification.

#### [MODIFY] [bayesian.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/entity/bayesian.py)

- **H1 — Fix ICC run-variance formula** (lines 819–821): Replace `π²/(3κ)` with delta-method approximation `1/(κ+1)` (for μ≈0.5 case), and add a comment explaining the derivation.
- **M1 — Fix shrinkage posterior mean** (lines 889–898): Compute `invlogit` per sample, then take mean/SD, instead of transforming point estimates.
- **M2 — Clamp shrinkage values** (lines 915–923): Clamp `shrinkage_pct` to `[0, 100]` and add a note field when clamping occurs.
- **M3 — Use correlated SE for LOO-CV** (line 1219): Extract `dse` from `az.compare()` output instead of naïvely summing variances.
- **H2 — Add marginal group contrast** (lines 748–784): Add a `compute_marginal_group_contrasts()` method that integrates over entity and participant effects. Keep the existing method as `compute_group_contrasts()` (conditional). Update `save_results()` to output both.

---

## Phase 3 — Robustness & Diagnostics

Hardening and improved user feedback.

#### [MODIFY] [bayesian.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/entity/bayesian.py)

- **M5 — BF tail-density warning:** Add a check in `compute_bayes_factor()` — if posterior mean of the contrast is > 3 SD from zero, log a warning that the Savage-Dickey estimate may be unreliable and add `"reliability": "low"` to the output.
- **M6 — Boundary score handling:** In `prepare_beta_data()`, log a warning when any score is exactly 0 or 100, reporting the count.
- **L1 — Assert sample length match** in `summarize_posteriors()`.
- **L2 — Check for NaN R-hat** in `_run_diagnostics()` and report explicitly.
- **L5 — Cache model in result:** Store the compiled PyMC model in `BayesianModelResult`.
- **L6 — Add `random_seed` to `posterior_predictive_check()`.**

---

## Phase 4 — New Capabilities

#### [MODIFY] [bayesian.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/entity/bayesian.py)

- **L4 — Add logit-scale contrasts:** Add `scale` parameter to `compute_group_contrasts()` accepting `"probability"` (default) or `"logit"`.
- **M4 — Prior sensitivity method:** Add `prior_sensitivity_check(dimension, alternative_configs)` that re-fits with 2 alternative prior specs and returns a comparison summary.
- **L3 — Ternary plot clarification:** Add axis-vertex mapping to plot title or legend annotation.

---

## Phase 5 — Tests

#### [MODIFY] [test_bayesian.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/tests/test_bayesian.py)

Add tests covering the gaps identified in the audit:

| Test | Validates |
|------|-----------|
| `test_icc_sums_to_one` | H1 fix — ICC components sum to ~1.0 |
| `test_icc_known_variance` | H1 — known-variance synthetic data |
| `test_hdi_keys_consistent` | H3 — all output dicts use `hdi_lower`/`hdi_upper` |
| `test_shrinkage_bounds` | M2 — shrinkage values in [0, 100] |
| `test_shrinkage_direction` | M1 — posterior means closer to grand mean than raw |
| `test_summarize_posteriors_keys` | Output structure |
| `test_save_and_load_results` | Round-trip for `save_results()` |
| `test_boundary_scores_warning` | M6 — warning logged for 0/100 scores |
| `test_loo_compare_runs` | M3 + H5 — LOO-CV doesn't crash, uses correct SE |
| `test_bayes_factor_output_keys` | M5 — output structure + reliability flag |
| `test_marginal_contrast` | H2 — new marginal contrast method |
| `test_logit_scale_contrast` | L4 — logit-scale output |
| `test_posterior_predictive_seed` | L6 — reproducibility |
| `test_prior_sensitivity` | M4 — sensitivity check returns comparison |

---

## Verification Plan

### Automated Tests
```bash
# Run full test suite after each phase
cd qualitative-analysis
python -m pytest tests/test_bayesian.py -v

# Run with PyMC tests (requires pymc environment)
python -m pytest tests/test_bayesian.py -v -k "Bayesian"
```

### Manual Verification
- After Phase 2, compare ICC/shrinkage/contrast outputs on the existing ground-truth dataset before and after to confirm the statistical corrections produce more accurate values.
- After Phase 4, run the full CLI pipeline with `--method bayesian --loo-compare --bayes-factor` to verify end-to-end.
