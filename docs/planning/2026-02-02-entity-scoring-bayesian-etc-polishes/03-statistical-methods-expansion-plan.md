# Plan: 5 Statistical Methods for Entity Analysis Pipeline

**Date:** 2026-02-04
**Package:** `qualitative-analysis`
**Component:** `qa entity` — agreement, clustering, Bayesian extensions
**Prerequisite:** Audit Option C complete (21/24 findings fixed). Pipeline production-ready.
**Source:** Remaining work tracker (`00-remaining-work.md`, §5 — Statistical Methods Not Yet Implemented)

---

## Overview

Implement Krippendorff's Alpha, LOO-CV, participant clustering, Bayes Factor, and PCA. Two new files (`agreement.py`, `clustering.py`), modifications to `bayesian.py`, `entity_cli.py`, `unified_cli.py`, `__init__.py`.

---

## New Files

### 1. `entity/agreement.py` — Krippendorff's Alpha

**Public API:**
- `krippendorff_alpha(reliability_data: np.ndarray) -> float` — core algorithm (interval distance)
- `bootstrap_ci(reliability_data, n_bootstrap=1000, confidence=0.95, random_seed=42) -> (lo, hi)`
- `compute_agreement(scores_df, dimensions, mode="run_level"|"participant_level", ...) -> Dict[str, AgreementResult]`
- `AgreementResult` dataclass: `alpha, ci_lower, ci_upper, n_units, n_raters, mode, dimension, interpretation`

**Two modes:**
- **Run-level**: raters = LLM runs, units = entity-participant pairs. Uses `{dim}_run1..N` columns.
- **Participant-level**: raters = participants, units = entities. Uses `{dim}_mean` columns.

**Interpretation thresholds** (Krippendorff): ≥0.80 = good, 0.67–0.80 = fair, <0.67 = poor.

Implement from scratch (~120 lines) — the algorithm is straightforward for interval data and avoids adding a dependency.

### 2. `entity/clustering.py` — Clustering + PCA

**Clustering API:**
- `ClusteringResult` dataclass: `method, labels, participant_ids, n_clusters, quality_metrics, metadata`
- `cluster_hierarchical(distance_matrix, participant_ids, n_clusters=None, linkage_method="ward")` — scipy
- `cluster_kmeans(score_matrix, participant_ids, n_clusters=None, max_k=10)` — sklearn, with elbow/silhouette
- `cluster_hdbscan(distance_matrix, participant_ids, min_cluster_size=3)` — optional dep
- `cluster_participants(comparison, method, metric, n_clusters, ...)` — convenience wrapper

**PCA API:**
- `PCAResult` dataclass: `components, explained_variance_ratio, cumulative_variance, transformed_scores, participant_ids, dimension_names`
- `pca_on_scores(comparison, n_components=None, use_clr=False, standardize=True)` — sklearn PCA
- `plot_biplot(result, output_path, groups=None)` — scatter + loading arrows
- `plot_scree(result, output_path)` — explained variance bar chart

**Visualization:**
- `plot_dendrogram(result, output_path)` — from hierarchical linkage
- `plot_elbow(result, output_path)` — inertia + silhouette per k
- `plot_cluster_scatter(result, score_matrix, participant_ids, output_path)` — 2D via PCA/MDS

**Existing `ParticipantComparison.cluster_participants()`** at `comparison.py:419-471` — keep for backward compat since it's a simpler API that serves its purpose.

---

## Modifications to Existing Files

### 3. `entity/bayesian.py` — LOO-CV + Bayes Factor

**LOO-CV:**
- `build_reduced_model(dimension)` — same model but no `group_effect`/`sigma_group` params
- `fit_reduced(dimension, **kwargs)` — fits reduced model, stores in `self.reduced_results`
- `compare_models(dimension, **kwargs)` — lazily computes log_likelihood via `pm.compute_log_likelihood()`, calls `az.loo()` + `az.compare()`, returns dict with elpd_diff, preferred_model, interpretation
- New instance var: `self.reduced_results: Dict[str, BayesianModelResult] = {}`

**Bayes Factor (Savage-Dickey):**
- `compute_bayes_factor(dimension)` — for each group pair:
  1. Prior density at delta=0: sample `sigma_group` from prior, compute `Normal(0, sqrt(2)*sigma_group).pdf(0)`, average
  2. Posterior density at delta=0: `scipy.stats.gaussian_kde` on posterior contrast samples
  3. `BF10 = prior_at_0 / posterior_at_0`
- Returns dict per pair: `bf10, log10_bf, interpretation` (Jeffreys scale)

### 4. `entity_cli.py` — New CLI commands + flags

**New commands:**
- `add_entity_agreement_args(parser)` + `run_entity_agreement(args)` — loads CSV, calls `compute_agreement()`, prints table, saves JSON
- `add_entity_cluster_args(parser)` + `run_entity_cluster(args)` — loads CSV via `ParticipantComparison`, runs clustering + optional PCA, saves results + plots

**Extended `compare` command:**
- Add `--loo-compare` flag → calls `compare_models()` after Bayesian fit
- Add `--bayes-factor` flag → calls `compute_bayes_factor()` after Bayesian fit

### 5. `unified_cli.py` — Register new subcommands

Add `agreement` and `cluster` subparsers in `_add_entity_commands()`, import from entity_cli.

### 6. `entity/__init__.py` — Export new symbols

Add `AgreementResult`, `compute_agreement`, `ClusteringResult`, `PCAResult`, `cluster_participants` (new), `pca_on_scores` to `__all__`.

---

## Implementation Order

| Phase | Task | Files |
|-------|------|-------|
| 1 | Krippendorff's Alpha | `agreement.py` (new), `entity_cli.py`, `unified_cli.py` |
| 2 | Clustering + PCA | `clustering.py` (new), `entity_cli.py`, `unified_cli.py` |
| 3 | LOO-CV model comparison | `bayesian.py`, `entity_cli.py` |
| 4 | Bayes Factor | `bayesian.py`, `entity_cli.py` |
| 5 | Integration + exports | `__init__.py`, run full test suite |

---

## Testing

**`tests/test_agreement.py`** (~10 tests):
- Perfect agreement → alpha = 1.0
- Known hand-computed example
- Missing data handling (NaN)
- Bootstrap CI contains point estimate
- Both run-level and participant-level modes
- Interpretation thresholds

**`tests/test_clustering.py`** (~12 tests):
- Well-separated clusters recovered (hierarchical, k-means)
- Auto k-selection works
- HDBSCAN optional dep handling
- PCA variance sums to 1, correct shapes
- CLR flag applied when set
- Plot functions create output files

**`tests/test_bayesian.py` additions** (~4 tests):
- LOO-CV: full model preferred with group differences
- LOO-CV: output structure validation
- BF: BF10 > 1 with clear difference
- BF: Jeffreys interpretation labels

---

## Verification

1. `source .venv-qa-pkg/bin/activate && pytest tests/ -v` — all tests pass
2. `python -c "from qualitative_analysis.entity import *"` — no import errors
3. Spot-check CLI: `qa entity agreement --help`, `qa entity cluster --help`
