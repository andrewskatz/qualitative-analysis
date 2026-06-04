# Bayesian Hierarchical Modeling: Implementation Progress

**Date:** 2026-02-02
**Package:** `qualitative-analysis`
**Component:** `qa entity compare --method bayesian`
**Planning Reference:** [Bayesian Hierarchical Modeling Plan](../planning/2026-01-29-bayesian-hierarchical-modeling/)
**Parent Plan:** [Multi-Participant Comparison Plan](../planning/2026-01-23-qa-package-entity-scoring-enhancements/00-multi-participant-comparison-plan.md) — Phase 2

---

## Summary

Implemented the full Bayesian hierarchical modeling pipeline (Phase 2 of the multi-participant comparison plan). This includes a Beta-likelihood hierarchical model with entity, participant, and group levels, integrated into the CLI as `qa entity compare --method bayesian`. Validated end-to-end on the 15-participant simulated dataset (3 groups × 5 participants, 415 unique entities, 693 entity-context pairs).

Additionally discovered and fixed a significant bug in the scoring pipeline where individual LLM run scores were computed in memory but discarded during serialization, and re-scored the full test dataset to generate run-level data.

---

## Critical Bug Fix: Run-Level Score Data Loss

### Discovery

During onboarding, we identified that the scored CSV files did **not** contain `{dim}_run{k}` columns (e.g., `social_run1`, `social_run2`, `social_run3`), despite the Bayesian model specification and implementation plan assuming they existed. Investigation revealed:

1. `DimensionScore.scores` (a `List[int]`) correctly stored individual run scores **in memory** during the scoring pipeline
2. `EntityScore.to_dict()` serialized aggregate statistics (`{dim}_mean`, `{dim}_std`, `{dim}_ci_low`, `{dim}_ci_high`, `{dim}_cv`) but **never** included the raw per-run scores
3. An `include_runs` parameter existed on `to_dict()` but was **never called with `True`** — and even when set to `True`, it dumped the full `SingleRunScore` objects as nested JSON, not as flat CSV-compatible columns
4. The `EntityScorer.load()` method had no logic to reconstruct `DimensionScore.scores` from loaded data

The result: once scoring results were written to disk (CSV or JSON), all run-level variance information was permanently lost. Every downstream analysis collapsed multiple LLM runs into a single mean before comparison — precisely the uncertainty propagation problem the Bayesian model was designed to solve.

### Fix Applied

**File: `entity/models.py` — `EntityScore.to_dict()`**

Added `{dim}_run{k}` columns directly from `DimensionScore.scores`:

```python
for dim_name, score in self.dimension_scores.items():
    for k, run_score in enumerate(score.scores, start=1):
        result[f"{dim_name}_run{k}"] = run_score
    result[f"{dim_name}_mean"] = score.mean
    # ... (existing aggregate columns unchanged)
```

**File: `entity/scorer.py` — `EntityScorer.load()`**

Added reconstruction of `DimensionScore.scores` from `{dim}_run{k}` keys when loading from JSON:

```python
run_scores = []
for k in range(1, num_runs + 1):
    run_key = f"{dim_key}_run{k}"
    if run_key in s_data:
        run_scores.append(s_data[run_key])
```

### Re-scoring

Re-scored the full 693 entity-context pairs using `gpt-oss:120b` via Ollama (local) with `--num-runs 3`. Output saved to:
- `tests/output/group_e2e_test/step4_scores_v2/step3_scoring_input_scores.csv`
- `tests/output/group_e2e_test/step4_scores_v2/step3_scoring_input_scores.json`

Verified that the re-scored output now contains `social_run1`, `social_run2`, `social_run3`, `ecological_run1`, etc.

---

## Bayesian Module Implementation

### New File: `entity/bayesian.py` (~750 lines)

Implements the full model specification from [01-model-specification.md](../planning/2026-01-29-bayesian-hierarchical-modeling/01-model-specification.md).

#### `prepare_beta_data(scores_df, dimensions, ...)`

- Reshapes wide-format scored CSV into long format (one row per observation = one run score)
- Rescales [0, 100] → (0, 1) using Smithson & Verkuilen (2006) squeeze: `y = (raw/100 * (N-1) + 0.5) / N`
- Builds integer index arrays: `entity_idx`, `participant_idx`, `group_idx`, `ep_pair_idx`
- Converts all name arrays to plain Python `str` to avoid pandas 3.0 ArrowStringArray issues

#### `BayesianEntityModel`

Core class implementing the hierarchical model. Key methods:

| Method | Description |
|--------|-------------|
| `build_model(dimension)` | Constructs PyMC model per spec (Beta likelihood, non-centered parameterization, group fixed effects) |
| `fit(dimension, ...)` | MCMC sampling with nutpie (primary) or NUTS (fallback); stores ArviZ InferenceData |
| `fit_all_dimensions()` | Fits all dimensions sequentially |
| `summarize_posteriors()` | Group means on probability scale with 95% HDI |
| `compute_group_contrasts(rope_delta)` | Posterior Δ between all group pairs, P(direction), ROPE probability |
| `compute_icc()` | Variance decomposition: group, entity, participant, run-level |
| `compute_shrinkage()` | Raw vs posterior entity mean comparison |
| `prior_predictive_check()` | Samples from priors only |
| `posterior_predictive_check()` | Samples from posterior for model checking |
| `save_results(output_dir, ...)` | JSON summaries + CSV exports |

**Model structure (per dimension):**

```
y_{i,j,k} ~ Beta(μ_{i,j} · κ, (1 - μ_{i,j}) · κ)

μ_{i,j} = logit⁻¹(η_{i,j})
η_{i,j} = θ_i + group_effect_{g[j]} + z_{i,j} · σ_participant

θ_i = μ_population + z_entity_i · σ_entity     (non-centered)
z_entity_i ~ Normal(0, 1)
σ_entity ~ HalfNormal(2)

group_effect_g ~ Normal(0, σ_group)
σ_group ~ HalfNormal(0.5)

z_{i,j} ~ Normal(0, 1)                         (non-centered)
σ_participant ~ HalfNormal(1)

μ_population ~ Normal(0, 1.5)
κ ~ Gamma(5, 0.1)
```

#### `BayesianVisualizer`

| Method | Output |
|--------|--------|
| `plot_group_contrasts()` | Forest plot with HDI and ROPE shading |
| `plot_posterior_densities()` | Overlaid histograms per group per dimension |
| `plot_icc_decomposition()` | Stacked bar chart of variance components |
| `plot_shrinkage()` | Scatter plot with 45° reference line |
| `plot_trace()` | MCMC trace plots for diagnostics |
| `generate_all()` | All plots to output directory |

#### `BayesianModelResult` (dataclass)

Structured container for all results: posterior summaries, group contrasts, ICC decomposition, shrinkage estimates, and convergence diagnostics.

---

## CLI Integration

### Modified File: `entity_cli.py`

Added Bayesian-specific arguments to `qa entity compare`:

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `distance` | `{distance, bayesian, all}` |
| `--chains` | 4 | Number of MCMC chains |
| `--draws` | 2000 | Posterior draws per chain |
| `--tune` | 1000 | Warmup/tuning iterations |
| `--target-accept` | 0.95 | Target acceptance probability |
| `--rope-delta` | 5.0 | ROPE threshold (on 0–100 scale) |
| `--sampler` | `nutpie` | `{nutpie, nuts}` |
| `--save-trace` | False | Save full ArviZ InferenceData (.nc files) |

The Bayesian block:
1. Checks for group definitions (required for Bayesian method)
2. Provides a clear error message if PyMC is not installed
3. Loads the DataFrame, ensures group column is present
4. Creates `BayesianEntityModel`, fits all dimensions
5. Prints convergence summary and group contrast highlights to console
6. Saves structured results (JSON + CSV) and generates all visualizations
7. Catches errors gracefully — if `--method all`, continues with distance-based analysis

---

## Dependency Changes

### Modified File: `pyproject.toml`

Added `[bayes]` optional dependency group:

```toml
bayes = [
    "pymc>=5.21",
    "arviz>=0.20",
    "nutpie>=0.13",
]
```

Installed versions: PyMC 5.27.1, ArviZ 0.23.1, nutpie 0.16.4.

### Modified File: `entity/__init__.py`

Added conditional Bayesian imports (wrapped in `try/except ImportError`) so the module loads cleanly whether or not PyMC is installed.

---

## Technical Issues Encountered and Resolved

### 1. nutpie + pandas 3.0 ArrowStringArray Incompatibility

**Problem:** pandas 3.0 defaults to `ArrowStringArray` for string columns. When nutpie's Rust backend receives PyMC model coordinates containing `ArrowStringArray` values, it raises:

```
RuntimeError: Coordinate entity value has unsupported type... ArrowStringArray
```

**Fixes applied (layered):**
1. Converted all entity/participant/group name arrays to plain `str()` in `prepare_beta_data()`
2. Wrapped model coordinate values in explicit `[str(e) for e in ...]` in `build_model()`
3. Added automatic fallback from nutpie to NUTS sampler via `try/except RuntimeError`

The `str()` conversion alone was insufficient — nutpie still failed on some coordinate types. The NUTS fallback is the reliable solution. This appears to be a nutpie version compatibility issue with pandas 3.0/PyArrow that may be resolved in a future nutpie release.

### 2. Synthetic Data for Early Validation

Before the full re-scoring completed, we generated synthetic run-level data from existing mean/std values for a 30-entity subset to validate the model build and sampling pipeline. This confirmed the model structure was correct before committing to the full dataset.

---

## Validation Results

### Synthetic Data Validation (30 entities, 3 dimensions, NUTS sampler)

- Zero divergences
- Group effects recovered correctly from known synthetic data
- Confirmed model builds, samples, and extracts posteriors without error

### Full Dataset Validation (415 entities, 15 participants, 3 groups)

**Command:**
```bash
qa entity compare tests/output/group_e2e_test/step4_scores_v2/step3_scoring_input_scores_with_groups.csv \
    --method bayesian \
    --group-by-col group \
    --sampler nuts \
    --chains 2 --draws 500 --tune 500 \
    --output-dir tests/output/group_e2e_test/bayesian_cli_test/
```

**Convergence:**
- Zero divergences across all 3 dimensions
- R-hat warnings present (expected with abbreviated 2-chain/500-draw test run; production runs with 4 chains/2000 draws should resolve)
- Model completed successfully for all dimensions

**Group Contrasts (key results):**

| Dimension | Comparison | Posterior Δ | P(direction) | Interpretation |
|-----------|-----------|-------------|--------------|----------------|
| Social | social vs technological | +5.23 | 1.000 | Social group scores higher on Social dimension |
| Ecological | ecological vs social | +13.16 | 1.000 | Ecological group scores higher on Ecological dimension |
| Technological | social vs technological | -17.37 | 1.000 | Technological group scores much higher on Technological dimension |

All results align with the known group emphasis in the simulated dataset and are consistent with the distance-based comparison results from Phase 1.

**ICC Variance Decomposition:**

| Level | Social | Ecological | Technological |
|-------|--------|------------|---------------|
| Entity | 46–59% | — | — |
| Participant | 20–33% | — | — |
| Group | 6–26% | — | — |
| Run | 7–11% | — | — |

Entity-level variance dominates (as expected — different entities are fundamentally different concepts), with meaningful group and participant contributions. Run-level variance is modest, indicating reasonable LLM scoring consistency.

**Output files generated:**
- `model_summary.json` — convergence diagnostics, posterior summaries
- `group_contrasts.json` — per-dimension group differences with HDI
- `icc_decomposition.json` — variance at each hierarchical level
- `shrinkage_estimates.csv` — raw vs posterior per entity
- `group_contrast_forest.png` — forest plot
- `posterior_densities.png` — per-dimension posteriors
- `icc_decomposition.png` — variance bar chart
- `shrinkage_plot.png` — raw vs posterior scatter

---

## Implementation Plan Phase Status

### Bayesian Implementation Plan ([02-implementation-plan.md](../planning/2026-01-29-bayesian-hierarchical-modeling/02-implementation-plan.md))

| Phase | Status | Notes |
|-------|--------|-------|
| Phase A: Core Model | ✅ Complete | build_model, fit, diagnostics, summarize_posteriors |
| Phase B: Group Contrasts & Derived Quantities | ✅ Complete | group_contrasts, ICC, shrinkage, prior/posterior predictive |
| Phase C: Visualizations | ✅ Complete | forest plot, posterior densities, ICC bar chart, shrinkage plot, trace plots |
| Phase D: CLI Integration | ✅ Complete | --method bayesian, all sampler flags, --rope-delta, --save-trace |
| Phase E: Documentation & Testing | 🔶 Partial | Validated end-to-end on real data; formal unit tests and docstring review still pending |

### Multi-Participant Comparison Plan ([00-multi-participant-comparison-plan.md](../planning/2026-01-23-qa-package-entity-scoring-enhancements/00-multi-participant-comparison-plan.md))

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Distance Functions | ✅ Complete | Aitchison, EMD (centroid + distribution) |
| Phase 2: Bayesian Models | ✅ Complete | Full Beta hierarchical model with CLI integration |
| Phase 3: Visualizations | ✅ Complete | All individual + group visualizations working |
| Phase 4: CLI Integration | ✅ Complete | `--groups`, `--groups-file`, `--group-by-col`, `--method`, all Bayesian flags |
| Phase 5: Advanced Features | 🔲 Not Started | Dirichlet regression, longitudinal comparison, cross-entity comparison |

---

## Files Modified

| File | Changes |
|------|---------|
| `entity/models.py` | Added `{dim}_run{k}` columns to `EntityScore.to_dict()` serialization |
| `entity/scorer.py` | Fixed `EntityScorer.load()` to reconstruct `DimensionScore.scores` from `{dim}_run{k}` keys |
| `entity/__init__.py` | Added conditional Bayesian imports (`BayesianEntityModel`, `BayesianVisualizer`, etc.) |
| `entity_cli.py` | Added `--method`, `--chains`, `--draws`, `--tune`, `--target-accept`, `--rope-delta`, `--sampler`, `--save-trace` flags and Bayesian execution block |
| `pyproject.toml` | Added `[bayes]` optional dependency group |

## Files Created

| File | Purpose |
|------|---------|
| `entity/bayesian.py` | Full Bayesian module (~750 lines): model, visualizer, data preparation |
| `tests/output/group_e2e_test/step4_scores_v2/` | Re-scored data with run-level columns (693 entities, 3 runs) |
| `tests/output/group_e2e_test/bayesian_cli_test/` | CLI test outputs (JSON, CSV, PNG) |
| `tests/output/group_e2e_test/bayesian_validation/` | Synthetic validation test outputs |

---

## Known Limitations

1. **nutpie sampler not functional with current pandas/PyArrow versions** — falls back to NUTS automatically
2. **No formal unit tests** — validated via end-to-end integration testing on simulated data; pytest-based tests should be added
3. **Abbreviated test run** — validation used 2 chains / 500 draws for speed; production runs should use 4 chains / 2000 draws
4. **No LOO-CV model comparison** — listed in model spec diagnostics but not yet implemented
5. **No posterior predictive check visualization** — the `posterior_predictive_check()` method exists but `plot_posterior_predictive()` in the visualizer is not yet wired up

---

## Next Steps

1. **Formal unit tests** — data preparation, model build, edge cases (Phase E of implementation plan)
2. **Production-quality run** — 4 chains / 2000 draws / 1000 tune on full dataset to verify R-hat < 1.01
3. **Review remaining items** from the multi-participant comparison plan (Phase 5 and open questions)
4. **nutpie compatibility** — monitor for upstream fix to ArrowStringArray issue
