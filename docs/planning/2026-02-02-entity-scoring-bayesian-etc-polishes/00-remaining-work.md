# Remaining Work: Entity Scoring, Bayesian Modeling & Polish

**Date:** 2026-02-02 (updated after Session 2)
**Package:** `qualitative-analysis`
**Component:** `qa entity` — scoring, comparison, Bayesian modeling
**Progress References:**
- [2026-02-02 Bayesian Implementation](../../progress-notes/2026-02-02-bayesian-hierarchical-modeling-implementation.md)
- [2026-02-02 Polish & CLI Improvements](../../progress-notes/2026-02-02-bayesian-polish-and-cli-improvements.md)
**Parent Plan:** [Multi-Participant Comparison Plan](../2026-01-23-qa-package-entity-scoring-enhancements/00-multi-participant-comparison-plan.md)

---

## Purpose

This document tracks all unfinished tasks identified during the 2026-02-02 review of the multi-participant comparison plan and Bayesian implementation plan. It serves as a single reference for remaining work so nothing is lost across sessions.

---

## 1. Testing (Phase E of Bayesian Plan)

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Unit tests for `prepare_beta_data()` | High | ✅ Done | 10 tests in `tests/test_bayesian.py` |
| Unit tests for `BayesianEntityModel.build_model()` | High | ✅ Done | 4 tests (requires PyMC) |
| Integration test with known ground truth | High | ✅ Done | 5 tests in `TestBayesianIntegration`: contrast direction (social, ecological), no-diff detection (technological), ICC decomposition, convergence |
| Unit tests for `EntityScore.to_dict()` run-level columns | Medium | ✅ Done | 5 tests in `TestRunLevelSerialization` |
| Unit tests for `EntityScorer.load()` run reconstruction | Medium | ✅ Done | JSON round-trip test |
| Docstring and type hint review pass | Low | ✅ Done | 7 type hint additions, 5 docstring additions/fixes across `bayesian.py` |

---

## 2. Production Validation

| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Full production Bayesian run (4 chains / 2000 draws / 1000 tune) | High | ✅ Done | All 3 dims pass: R-hat<1.01, ESS>1000, 0 divergences |
| Small-N testing (2-3 participants per group) | Medium | ✅ Done | Tested n=2 and n=3 per group. Model recovers correct contrast directions with P(dir)≥0.99 for 30+ pt differences. ESS low with 2 chains — recommend 4 chains for small N. See progress note. |
| nutpie compatibility monitoring | Low | Ongoing | Track upstream fix for ArrowStringArray issue with pandas 3.0 |

---

## 3. Incomplete Visualizations

| Visualization | Source Plan Section | Priority | Status | Notes |
|---------------|-------------------|----------|--------|-------|
| Posterior predictive check plot | Bayesian Plan Phase C, Step C5 | High | ✅ Done | `plot_posterior_predictive()` added to `BayesianVisualizer`, wired into `generate_all()` |
| Credible region ellipses on ternary | Comparison Plan Viz Spec §3 | High | ✅ Done | `plot_group_ternary()` with 95% credible regions from Bayesian posteriors |
| Overlaid ternary with uncertainty | Comparison Plan Viz Spec §2 | Medium | ✅ Done | `show_confidence_ellipses` parameter on `generate_overlaid_ternary()` |
| Centroid comparison with Bayesian credible regions | Comparison Plan Viz Spec §3 | Medium | ✅ Done | Part of `plot_group_ternary()` |
| LOO-CV model comparison | Model Spec §6.2 | Low | Not implemented | `az.loo()` for model comparison if alternative specs are tested |

---

## 4. CLI / UX Gaps

| Feature | Source | Priority | Status | Notes |
|---------|--------|----------|--------|-------|
| `qa entity compare-viz` subcommand | Comparison Plan §CLI Example | Medium | ✅ Done | Standalone visualization generation |
| Markdown report generation | Comparison Plan Phase 4 | Medium | ✅ Done | `qa entity report` subcommand |
| `--statistical-test permutation` flag verification | 2026-01-29 Progress Note | Medium | ✅ Verified | Fully wired: CLI flag → `run_permutation_test()` in comparison.py → JSON output. Includes effect size, p-value, null distribution summary. |

---

## 5. Statistical Methods Not Yet Implemented

These are from the "Statistical Methods Catalog" in the comparison plan. They are lower priority but represent the full vision.

| Method | Plan Section | Priority | Notes |
|--------|-------------|----------|-------|
| Krippendorff's Alpha | §3.2 | Medium | Multi-rater agreement metric; straightforward to implement |
| Participant clustering (hierarchical, k-means, GMM, HDBSCAN) | §5.1 | Low | Discover natural participant groupings from scoring patterns |
| Gaussian Mixture Model (soft assignments) | §5.1 | Low | Unknown group discovery with uncertainty |
| Bayes Factor | §4.3 | Low | Evidence ratio for difference vs no difference |
| PCA on CLR scores | §5.3 | Low | 2D participant map showing linear relationships |

---

## 6. Open Questions (From Original Plan, Still Unanswered)

| Question | Context | Recommendation |
|----------|---------|---------------|
| How to handle missing data in distance pipeline? | Comparison Plan §Open Questions #1 | Bayesian handles naturally; document for distance-based; add CLI note |
| Minimum sample size for Bayesian? | Comparison Plan §Open Questions #3 | ✅ Answered: n=2 works for large effects (30+ pts); recommend 4 chains and more draws. See small-N findings in progress note. |
| Weighted comparisons? | Comparison Plan §Open Questions #4 | Defer until user need is clear |

---

## 7. Phase 5: Advanced Features (Deferred)

These are explicitly out of scope for now but documented for future reference:

| Feature | Notes |
|---------|-------|
| Dirichlet regression | For truly compositional data if future datasets warrant it |
| Longitudinal comparison | Same participant over time (pre/post intervention) |
| Cross-entity comparison | Same participant, different entity types |

---

## Execution Order (Recommended — Updated)

Completed items struck through; remaining items renumbered:

1. ~~Formal pytest tests~~ ✅
2. ~~Production Bayesian run~~ ✅
3. ~~Posterior predictive check visualization~~ ✅
4. ~~Credible region ellipses on ternary~~ ✅
5. ~~CLI/UX gaps — compare-viz subcommand, markdown report~~ ✅
6. ~~Overlaid ternary with uncertainty~~ ✅

7. ~~Integration test with known ground truth~~ ✅
8. ~~`--statistical-test permutation` flag verification~~ ✅
9. ~~Small-N testing~~ ✅

10. ~~Docstring and type hint review~~ ✅

**Remaining recommended order:**

1. **Krippendorff's Alpha** — multi-rater agreement metric
2. **LOO-CV model comparison** — if alternative model specs are tested
3. **Participant clustering, Bayes Factor, PCA** — lower priority exploratory methods
