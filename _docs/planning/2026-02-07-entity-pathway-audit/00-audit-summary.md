# Entity Pathway Audit Summary

**Date:** 2026-02-07
**Scope:** Full audit of the entity analysis pathway in `qualitative-analysis/src/qualitative_analysis/entity/`
**Auditor:** Claude (computational social science / statistical modeling review)

---

## Executive Summary

The entity pathway is a substantial (~8,300 lines across 12 modules + 2,900 lines of CLI) multi-stage system for LLM-driven entity extraction, scoring, comparison, and visualization in qualitative research. The architecture is well-structured and modular, but the audit identified **5 critical bugs**, **15 high-severity issues**, and numerous medium/low findings across statistical correctness, data flow, and visualization methodology.

**The most notable design observation** is that the scoring model treats dimensions as independent (each 0-100), while ternary plots normalize them to relative proportions. This normalization is valid for showing profile shapes, but loses magnitude information. The fix is to encode magnitude as point size and uncertainty as opacity. This has been implemented. The independent-vs-compositional distinction remains important for distance metric choice (prefer Euclidean over Aitchison for independent dimensions).

---

## Findings by Severity

### CRITICAL (6 findings)

| # | Module | Finding | Impact |
|---|--------|---------|--------|
| C2 | **entity_cli.py:1531** | Undefined variable `chains` in LOO-CV | Runtime crash when `--loo-compare` flag is used |
| C3 | **entity_cli.py:2516** | Wrong argument order in `cluster` command's `load_scores()` call | Dimension list passed as `participant_col` |
| C4 | **bayesian.py:1191-1203** | `pm.compute_log_likelihood()` return value not captured | LOO-CV comparison silently fails |
| C5 | **bayesian.py:1326-1331** | Bayes Factor computation fundamentally incorrect | Savage-Dickey ratio doesn't integrate out sigma from posterior |
| C6 | **clustering.py:381** | `compute_pairwise_distances()` receives numpy array instead of expected dict | Runtime crash for non-Euclidean metrics in clustering |

### HIGH (15 findings)

| # | Module | Finding |
|---|--------|---------|
| H1 | scorer.py:505 | Silent skip of missing dimensions in `_aggregate_runs()` |
| H2 | scorer.py:465-472 | No validation that LLM scores are within [0, 100] range |
| H3 | models.py:227 | CV defaults to 0.0 when mean=0 (should be undefined/NaN) |
| H4 | models.py:211-212 | Mode fabricated from median of multimode when no true mode exists |
| H5 | models.py:217-224 | CI bounds clamped to [0,100] creating asymmetric intervals |
| H6 | bayesian.py:175 | Participant ID type mismatch (str vs original type) causes key errors |
| H7 | bayesian.py:607 | Rhat convergence threshold (1.01) arguably too loose for small ESS |
| H8 | agreement.py:157 | Expected disagreement formula incorrect with structured missing data |
| H9 | clustering.py:145-150 | Ward linkage used with non-Euclidean distances without validation |
| H10 | clustering.py:366-382 | K-means ignores the `metric` parameter entirely |
| H11 | visualizer.py:158-163 | Ternary coordinate dimension-to-vertex mapping scrambled |
| H12 | comparison_viz.py:232,389 | Different coordinate scrambling than visualizer.py (incompatible) |
| H13 | comparison_viz.py:415-440 | Confidence ellipses computed with Euclidean covariance on simplex data |
| H14 | entity_cli.py:334 | Run-level data discarded in CSV export (`include_runs=False`), breaking agreement analysis |
| H15 | entity_cli.py (pipeline) | Group information lost between `detect` and `score` steps |

### MEDIUM (18 findings)

| # | Module | Finding |
|---|--------|---------|
| M1 | detector.py:144-151 | JSON extraction greedy (first `{` to last `}`) can grab wrong block |
| M2 | detector.py:209 | Case-sensitive entity deduplication |
| M3 | detector.py:199-201 | Silent failure on LLM errors (empty list, no tracking) |
| M4 | detector.py:213-215 | Context truncation at arbitrary character boundary (mid-word) |
| M5 | scorer.py:507-517 | Single justification selected for aggregated result (information loss) |
| M6 | scorer.py:484-525 | Dimension name matching is lowercase-only (fragile) |
| M7 | bayesian.py:820-821 | ICC variance approximation (`pi^2/3`) only valid at symmetric point |
| M8 | bayesian.py:919-921 | Shrinkage set to 0% when raw mean near grand mean (should be NaN) |
| M9 | agreement.py:194-197 | Bootstrap assumes exchangeability; no stratification by entity |
| M10 | distances.py:266 | Sliced Wasserstein uses hardcoded seed (42) |
| M11 | comparison.py:662-668 | Group centroid not weighted by entity count per participant |
| M12 | comparison_viz.py:748-764 | Bootstrap CIs violate compositional constraint (can sum > 1) |
| M13 | comparison_viz.py:146 | Hardcoded `np.ones(3)/3` fallback assumes 3 dimensions |
| M14 | visualizer.py:175-181 | Uncertainty opacity formula produces incorrect alpha values |
| M15 | visualizer.py:307 | Missing radar chart values default to 50 (masks data quality issues) |
| M16 | entity_cli.py:719-735 | Missing dimension columns silently default to 50 |
| M17 | entity_cli.py:1754 | `prepare-scoring` silently ignores JSON parse errors |
| M18 | Both viz files | Colorblind-unfriendly palette (red/green combination) |

---

## Detailed Reports

- [01-design-contradiction.md](01-design-contradiction.md) - The fundamental independent-vs-compositional design flaw
- [02-statistical-methods.md](02-statistical-methods.md) - Bayesian, agreement, clustering, and distance metric issues
- [03-scoring-pipeline.md](03-scoring-pipeline.md) - Entity detection, scoring models, and data flow
- [04-visualization.md](04-visualization.md) - Ternary plot, radar chart, comparison visualization issues
- [05-cli-pipeline.md](05-cli-pipeline.md) - CLI integration, argument handling, and pipeline orchestration
- [06-recommendations.md](06-recommendations.md) - Prioritized remediation plan

---

## How to Read This Audit

**Severity levels:**
- **CRITICAL** - Will cause runtime errors, produce fundamentally incorrect results, or represents a design-level flaw that undermines the validity of the analysis
- **HIGH** - Produces incorrect statistical results or silently corrupts data in ways that could mislead research conclusions
- **MEDIUM** - Creates data quality issues, loses information, or uses suboptimal methods that may affect results in edge cases
- **LOW** - Documentation, style, efficiency, or accessibility concerns
