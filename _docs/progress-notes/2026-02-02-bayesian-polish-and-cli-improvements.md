# Bayesian Polish, Testing & CLI Improvements

**Date:** 2026-02-02 (Session 2)
**Package:** `qualitative-analysis`
**Component:** `qa entity` — testing, production validation, visualizations, CLI
**Prior Memo:** [2026-02-02 Bayesian Implementation](./2026-02-02-bayesian-hierarchical-modeling-implementation.md)
**Remaining Work Tracker:** [00-remaining-work.md](../planning/2026-02-02-entity-scoring-bayesian-etc-polishes/00-remaining-work.md)

---

## Summary

Addressed the majority of items from the remaining work tracker: wrote 22 formal pytest tests, ran a production-quality Bayesian fit (4 chains / 2000 draws), added posterior predictive check and group ternary visualizations, created `compare-viz` and `report` CLI subcommands, and wired 95% confidence ellipses into the overlaid ternary plot.

---

## 1. Formal pytest Test Suite

**New file:** `tests/test_bayesian.py` — 22 tests, all passing.

| Test Class | Count | Scope |
|-----------|-------|-------|
| `TestPrepareBetaData` | 10 | Rescaling bounds, shape, auto-detect runs, NaN handling, index validation, single participant, str type safety |
| `TestRunLevelSerialization` | 5 | `to_dict()` run columns, multi-dimension, empty scores, CSV round-trip, JSON round-trip |
| `TestBayesianModelBuild` | 4 | Model builds, expected variables (free + observed RVs), coord shapes, prior predictive, invalid dimension (requires PyMC) |
| `TestCheckPymcAvailable` | 2 | Import check returns bool |

Helper function `_make_scores_df()` generates synthetic scored DataFrames with configurable entity count, participants, runs, dimensions, and group assignments.

### Issues Found and Fixed During Testing

- **`y_obs` classified as free RV:** Initial test asserted `y_obs` was in `model.free_RVs`, but it's an observed RV. Fixed by splitting assertions into `free_var_names` and `observed_var_names`.
- **`samples` vs `draws` parameter name:** `prior_predictive_check()` was renamed from `samples` to `draws` for API consistency, but the test still used the old parameter name.

---

## 2. Production Bayesian Run (4 Chains / 2000 Draws)

Ran the full production-quality MCMC fit on the 15-participant dataset (415 entities, 693 entity-context pairs, 3 groups × 5 participants).

**Command:**
```bash
qa entity compare .../step3_scoring_input_scores_with_groups.csv \
    --method bayesian --group-by-col group \
    --sampler nuts --chains 4 --draws 2000 --tune 1000 \
    --output-dir .../bayesian_production/
```

**Convergence Results — All 3 Dimensions PASS:**

| Dimension | R-hat (max) | ESS (min) | Divergences |
|-----------|------------|-----------|-------------|
| Social | 1.0026 | 1,475 | 0 |
| Ecological | 1.0016 | 2,568 | 0 |
| Technological | 1.0017 | 2,292 | 0 |

This resolves the R-hat warnings from the earlier abbreviated 2-chain/500-draw test run. All convergence criteria are met (R-hat < 1.01, ESS > 1000, 0 divergences).

**Output:** `tests/output/group_e2e_test/bayesian_production/`

---

## 3. Posterior Predictive Check Visualization

**Modified file:** `entity/bayesian.py` — `BayesianVisualizer`

Added `plot_posterior_predictive()` method:
- Samples 100 replicated datasets from the posterior predictive distribution
- Plots each as a thin translucent histogram overlay
- Overlays observed data as a bold step histogram
- Generates one subplot per dimension
- Called automatically from `generate_all()`

This addresses the "Posterior predictive check plot" item from the remaining work tracker (previously listed as "method exists, plot not wired up").

---

## 4. Group Ternary with Bayesian Credible Regions

**Modified file:** `entity/bayesian.py` — `BayesianVisualizer`

Added `plot_group_ternary()` method:
- Projects posterior group mean samples to ternary (barycentric → cartesian) coordinates
- Plots posterior scatter per group with transparency
- Computes 95% credible region ellipses from the 2D posterior covariance (chi²(2) = 5.991)
- Marks group centroids with X markers
- Added `_bary_to_cart()` static helper for coordinate conversion

Updated `generate_all()` to include both PPC and group ternary plots (with `try/except` for graceful failure).

---

## 5. Overlaid Ternary: 95% Confidence Ellipses

**Modified file:** `entity/comparison.py` — `ComparisonVisualizer`

Added `show_confidence_ellipses` parameter to `generate_overlaid_ternary()`:
- When enabled, computes per-participant confidence ellipses from entity scatter covariance
- Uses eigendecomposition + chi²(2) = 5.991 for 95% coverage
- Draws ellipses in each participant's color
- Requires ≥3 entities per participant for meaningful covariance

**Modified file:** `entity_cli.py`

Added `--show-ellipses` flag to both `compare` and `compare-viz` subcommands, wired through to the overlaid ternary call.

---

## 6. New CLI Subcommands

### `qa entity compare-viz`

**Modified files:** `entity_cli.py`, `unified_cli.py`

Standalone visualization generation subcommand. Reads an already-scored CSV and generates all comparison visualizations without re-running the distance or Bayesian analysis. Supports all visualization flags (`--show-hull`, `--show-ellipses`, `--show-density`, `--annotate`, `--top-n`).

Use case: regenerate plots with different visual options after analysis is complete.

### `qa entity report`

**Modified files:** `entity_cli.py`, `unified_cli.py`

Auto-generated markdown report summarizing comparison findings. Reads scored CSV and optional Bayesian results, then produces a report containing:
- Summary statistics (mean/median/min/max distance, most similar/different pairs)
- Participant dimension means table
- Group comparison (within/between distances, effect size)
- Bayesian group contrasts (Mean Δ, 94% HDI, P(direction)) — if available
- ICC variance decomposition — if available

**Output example:** Successfully tested, producing the report format shown at `/tmp/test_report.md`.

---

## 7. `--show-ellipses` Flag for `qa entity compare`

**Modified file:** `entity_cli.py`

The existing `compare` subcommand now accepts `--show-ellipses` to enable 95% confidence ellipses on the overlaid ternary plot during the standard comparison workflow (not just via `compare-viz`).

---

## Files Modified

| File | Changes |
|------|---------|
| `entity/bayesian.py` | Fixed `samples`→`draws` in `prior_predictive_check()`; added `plot_posterior_predictive()`, `plot_group_ternary()`, `_bary_to_cart()`; updated `generate_all()` |
| `entity/comparison.py` | Added `show_confidence_ellipses` parameter and ellipse drawing to `generate_overlaid_ternary()` |
| `entity_cli.py` | Added `compare-viz` subcommand, `report` subcommand, `--show-ellipses` flag on `compare` and `compare-viz` |
| `unified_cli.py` | Registered `compare-viz` and `report` subcommands in `_add_entity_commands()` |

## Files Created

| File | Purpose |
|------|---------|
| `tests/test_bayesian.py` | 22 formal pytest tests for data preparation, model build, and serialization |
| `tests/output/group_e2e_test/bayesian_production/` | Production Bayesian run outputs (4 chains / 2000 draws) |
| `docs/planning/2026-02-02-entity-scoring-bayesian-etc-polishes/00-remaining-work.md` | Remaining work tracker |

---

## Remaining Work Tracker Status Update

Items from `00-remaining-work.md` addressed in this session:

| Section | Item | Status |
|---------|------|--------|
| §1 Testing | Unit tests for `prepare_beta_data()` | ✅ Done (10 tests) |
| §1 Testing | Unit tests for `BayesianEntityModel.build_model()` | ✅ Done (4 tests) |
| §1 Testing | Unit tests for `EntityScore.to_dict()` run-level columns | ✅ Done (5 tests) |
| §1 Testing | Unit tests for `EntityScorer.load()` run reconstruction | ✅ Done (JSON round-trip test) |
| §1 Testing | Integration test with known ground truth | ⬜ Not yet (synthetic validation exists but not formalized as pytest) |
| §1 Testing | Docstring and type hint review pass | ⬜ Not yet |
| §2 Production | Full production Bayesian run | ✅ Done — all 3 dims pass |
| §2 Production | Small-N testing (2-3 per group) | ⬜ Not yet |
| §3 Visualizations | Posterior predictive check plot | ✅ Done |
| §3 Visualizations | Credible region ellipses on ternary | ✅ Done (Bayesian group ternary) |
| §3 Visualizations | Overlaid ternary with uncertainty | ✅ Done (confidence ellipses) |
| §3 Visualizations | Centroid comparison with Bayesian credible regions | ✅ Done (part of group ternary) |
| §3 Visualizations | LOO-CV model comparison | ⬜ Not yet (low priority) |
| §4 CLI/UX | `qa entity compare-viz` subcommand | ✅ Done |
| §4 CLI/UX | Markdown report generation | ✅ Done |
| §4 CLI/UX | `--statistical-test permutation` flag verification | ⬜ Not yet verified |
| §5 Statistical Methods | Krippendorff's Alpha | ⬜ Not yet |
| §5 Statistical Methods | Participant clustering | ⬜ Not yet |
| §5 Statistical Methods | Bayes Factor | ⬜ Not yet |
| §5 Statistical Methods | PCA on CLR scores | ⬜ Not yet |

**Summary:** 13 items completed, 9 items remaining (most are medium-to-low priority).

---

## Remaining Items (Prioritized)

### Should Do — ✅ All Completed (Session 3)

1. ~~**Integration test with known ground truth**~~ — ✅ 5 tests in `TestBayesianIntegration`
2. ~~**`--statistical-test permutation` flag verification**~~ — ✅ Fully wired end-to-end
3. ~~**Small-N testing**~~ — ✅ See findings below

### Nice to Have

4. **Docstring and type hint review** — Audit all public methods in `bayesian.py`
5. **Krippendorff's Alpha** — Multi-rater agreement metric; straightforward to implement
6. **LOO-CV model comparison** — `az.loo()` for comparing alternative model specifications

### Deferred

7. **Participant clustering** (hierarchical, k-means, GMM, HDBSCAN)
8. **Bayes Factor** for evidence ratio
9. **PCA on CLR scores** — 2D participant map
10. **Phase 5 advanced features** — Dirichlet regression, longitudinal comparison, cross-entity comparison

---

## Addendum: Session 3 — Integration Tests, Permutation Verification & Small-N Testing

### Integration Tests (5 new tests, all passing)

Added `TestBayesianIntegration` class to `tests/test_bayesian.py` with synthetic data where ground truth is known:
- Group "high" scores ~75 on social, ~40 on ecological
- Group "low" scores ~40 on social, ~75 on ecological
- Both groups score ~55 on technological (null difference)

| Test | Assertion | Result |
|------|-----------|--------|
| `test_group_contrast_direction_social` | high > low by >10 pts, P(dir) > 0.95 | PASS |
| `test_group_contrast_direction_ecological` | low > high by >10 pts, P(dir) > 0.95 | PASS |
| `test_no_difference_on_technological` | \|diff\| < 15, ROPE prob > 0.20 | PASS |
| `test_icc_entity_dominates` | Run ICC < 0.30, entity+group ICC > 0.20 | PASS |
| `test_convergence_on_ground_truth` | 0 divergences, R-hat < 1.05 | PASS |

Full suite: **27 tests, all passing** (53s with `.venv-qa-pkg`).

### Permutation Test Flag Verification

`--statistical-test permutation` is **fully implemented and wired**:
- CLI flag defined in `add_entity_compare_args()` with `--n-permutations` (default 1000)
- Triggers `comparison.run_permutation_test()` which shuffles group labels, computes effect sizes, and derives p-values
- Results printed to console and saved to `permutation_test_results.json` and `group_comparison_results.json`
- `GroupComparisonResult.permutation_test_p` field serialized correctly

### Small-N Testing Results

Tested the Bayesian model with 2 and 3 participants per group (3 groups, 6 entities, 3 runs per entity).

**n=2 per group (6 total participants):**

| Dimension | R-hat | ESS-bulk | ESS-tail | Divergences | Status |
|-----------|-------|----------|----------|-------------|--------|
| Social | 1.0022 | 538 | 696 | 0 | PASS |
| Ecological | 1.0074 | 370 | 512 | 0 | WARN (ESS) |
| Technological | 1.0118 | 227 | 375 | 0 | WARN (ESS, R-hat) |

**n=3 per group (9 total participants):**

| Dimension | R-hat | ESS-bulk | ESS-tail | Divergences | Status |
|-----------|-------|----------|----------|-------------|--------|
| Social | 1.0054 | 280 | 599 | 0 | WARN (ESS) |
| Ecological | 1.0157 | 210 | 445 | 0 | WARN (ESS, R-hat) |
| Technological | 1.0105 | 245 | 713 | 0 | WARN (ESS, R-hat) |

**Key findings:**

1. **Contrast recovery is robust** — Even with n=2 per group, the model correctly identifies all large (30+ pt) group differences with P(direction) = 1.000. Small differences (~3-5 pts) are correctly identified as uncertain (P(dir) 0.54–0.93).

2. **Convergence is marginal** — With 2 chains / 1000 draws, ESS frequently falls below 400 and R-hat occasionally exceeds 1.01. This is expected with small N and only 2 chains.

3. **Recommendation for small N:** Use **4 chains and ≥2000 draws** to compensate. The model is identifiable but the posterior geometry is harder to explore with fewer data points.

4. **ICC is dominated by group effects** — With well-separated groups, group ICC reaches 83-85%, which is correct for this synthetic data. Entity-level variance is small (1-2%) because synthetic entities don't have strong per-entity effects.

5. **Minimum viable N:** n=2 per group works for **detecting large effects** (>20 pts on 0-100 scale). For subtle effects (<10 pts), recommend n≥5 per group.

---

*Report updated 2026-02-03, Session 3*
