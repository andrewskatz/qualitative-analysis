# Implementation Plan: Bayesian Hierarchical Model

**Date:** 2026-01-29
**Parent:** [00-overview.md](./00-overview.md)
**Model Spec:** [01-model-specification.md](./01-model-specification.md)

---

## 1. Module Architecture

### New File: `entity/bayesian.py`

```
qualitative-analysis/src/qualitative_analysis/entity/bayesian.py
├── class BayesianEntityModel
│   ├── __init__(scores_df, groups, dimensions)
│   ├── build_model(dimension)           → pm.Model
│   ├── fit(dimension, **sampler_kwargs) → az.InferenceData
│   ├── fit_all_dimensions()             → Dict[str, az.InferenceData]
│   ├── summarize_posteriors()           → Dict
│   ├── compute_group_contrasts()        → Dict
│   ├── compute_icc()                    → Dict
│   ├── compute_shrinkage()              → Dict
│   ├── run_diagnostics()               → Dict
│   ├── prior_predictive_check()         → az.InferenceData
│   ├── posterior_predictive_check()     → az.InferenceData
│   └── save_results(output_dir)         → None
│
├── class BayesianVisualizer
│   ├── plot_group_contrasts()           → Figure   # forest plot of group diffs
│   ├── plot_posterior_densities()       → Figure   # per-dimension posteriors
│   ├── plot_icc_decomposition()         → Figure   # stacked bar of variance
│   ├── plot_shrinkage()                 → Figure   # raw vs posterior estimates
│   ├── plot_posterior_predictive()      → Figure   # observed vs predicted
│   └── plot_trace()                     → Figure   # MCMC trace plots
│
└── helper functions
    ├── prepare_beta_data(scores_df)     → arrays   # rescale, squeeze boundaries
    ├── encode_groups(groups)            → arrays   # integer-encode group labels
    └── check_pymc_available()           → bool     # graceful import check
```

### Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `entity/comparison.py` | `ParticipantComparison` gains `.run_bayesian_model()` method that delegates to `BayesianEntityModel` |
| `entity_cli.py` | New `--method bayesian` option on `qa entity compare` |
| `entity/__init__.py` | Conditional export of `BayesianEntityModel` (only if PyMC installed) |

---

## 2. Dependencies

### New Optional Dependency Group: `[bayes]`

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
bayes = [
    "pymc>=5.21",
    "arviz>=0.20",
    "nutpie>=0.13",
]
```

### Installation

```bash
# Base install (no Bayesian)
pip install -e .

# With Bayesian support
pip install -e ".[bayes]"
```

### Graceful Degradation

If PyMC is not installed:
- `qa entity compare --method bayesian` prints a clear error message with install instructions
- All other functionality (distance-based, permutation test) works normally
- No import errors at module load time — use lazy imports

```python
def check_pymc_available():
    try:
        import pymc
        return True
    except ImportError:
        return False
```

---

## 3. Data Preparation

### Input Format

Same CSV as the existing comparison pipeline:

```
text_id,entity,context,dim1_run1,dim1_run2,dim1_run3,dim2_run1,...,group
interview_001,river,flooding context,75,78,72,45,48,43,...,ecological
interview_002,river,flooding context,60,62,58,55,53,57,...,social
```

### Preparation Steps

1. **Identify dimensions and runs** from column naming pattern (`{dim}_run{k}`)
2. **Rescale** scores from [0, 100] to (0, 1): `y = (raw * (N-1) + 0.5) / N`
3. **Build index arrays:**
   - `entity_idx[obs]` → integer index for entity i
   - `participant_idx[obs]` → integer index for participant j
   - `group_idx[obs]` → integer index for group g
   - `run_idx[obs]` → integer index for run k
4. **Handle missing data** — rows where a participant didn't score an entity are simply absent (no imputation needed)

---

## 4. Implementation Phases

### Phase A: Core Model (Minimum Viable)

**Goal:** Build, fit, and extract basic posteriors for one dimension.

| Step | Task | Output |
|------|------|--------|
| A1 | Create `entity/bayesian.py` with `BayesianEntityModel` class | Module skeleton |
| A2 | Implement `prepare_beta_data()` | Rescaled arrays + index mappings |
| A3 | Implement `build_model(dimension)` | PyMC model object |
| A4 | Implement `fit(dimension)` | ArviZ InferenceData |
| A5 | Implement `run_diagnostics()` | R-hat, ESS, divergence summary |
| A6 | Implement `summarize_posteriors()` | Group means, entity means with HDI |

**Validation:** Fit on the 15-participant simulated dataset. Check convergence (R-hat < 1.01, no divergences). Verify that posterior group means align with the known simulated group differences.

### Phase B: Group Contrasts and Derived Quantities

**Goal:** Extract the inferential outputs that distinguish this from the distance-based approach.

| Step | Task | Output |
|------|------|--------|
| B1 | Implement `compute_group_contrasts()` | Posterior Δ, P(direction), HDI per pair |
| B2 | Implement `compute_icc()` | Variance decomposition at each level |
| B3 | Implement `compute_shrinkage()` | Raw vs posterior comparison per entity |
| B4 | Implement `prior_predictive_check()` | Prior predictive samples |
| B5 | Implement `posterior_predictive_check()` | Posterior predictive samples |

**Validation:** Compare group contrast posteriors against permutation test results — they should agree on which groups are most/least different. ICC should show meaningful between-group variance.

### Phase C: Visualizations

**Goal:** Produce publication-quality figures from Bayesian outputs.

| Step | Task | Description |
|------|------|-------------|
| C1 | Group contrast forest plot | Per-dimension posterior Δ with HDI for each group pair |
| C2 | Posterior density plot | Overlaid densities per group per dimension |
| C3 | ICC bar chart | Stacked bar showing variance at each level |
| C4 | Shrinkage plot | Scatter of raw vs posterior means with 45-degree reference |
| C5 | Posterior predictive check | Observed vs replicated score distributions |
| C6 | Trace plots | MCMC chain mixing (diagnostic, not for publication) |

### Phase D: CLI Integration

**Goal:** Make Bayesian modeling accessible from the command line.

| Step | Task | Description |
|------|------|-------------|
| D1 | Add `--method bayesian` to `qa entity compare` | Triggers Bayesian pipeline |
| D2 | Add Bayesian-specific flags | `--chains`, `--draws`, `--tune`, `--target-accept` |
| D3 | Add `--rope-delta` flag | Threshold for Region of Practical Equivalence |
| D4 | JSON output | Structured results including posteriors summary |
| D5 | Integration with existing output directory structure | Fits alongside distance-based outputs |

### Phase E: Documentation and Testing

| Step | Task | Description |
|------|------|-------------|
| E1 | Unit tests for data preparation | Rescaling, index encoding, edge cases |
| E2 | Integration test on simulated data | Full pipeline with known ground truth |
| E3 | Docstrings and type hints | All public methods |
| E4 | Update progress memo | Record implementation outcomes |

---

## 5. CLI Interface (Proposed)

```bash
# Run Bayesian model (default settings)
qa entity compare scored_data.csv \
    --method bayesian \
    --group-by-col group \
    --output-dir results/bayesian/

# With custom sampler settings
qa entity compare scored_data.csv \
    --method bayesian \
    --group-by-col group \
    --chains 4 \
    --draws 2000 \
    --tune 1000 \
    --target-accept 0.95 \
    --rope-delta 5.0 \
    --output-dir results/bayesian/

# Run both distance-based and Bayesian
qa entity compare scored_data.csv \
    --method all \
    --group-by-col group \
    --statistical-test permutation \
    --output-dir results/
```

### Output Directory Structure

```
results/bayesian/
├── model_summary.json           # Convergence diagnostics, posterior summaries
├── group_contrasts.json         # Per-dimension group differences with HDI
├── icc_decomposition.json       # Variance at each hierarchical level
├── shrinkage_estimates.csv      # Raw vs posterior per entity
├── group_contrast_forest.png    # Forest plot
├── posterior_densities.png      # Per-dimension posteriors
├── icc_decomposition.png        # Variance bar chart
├── shrinkage_plot.png           # Raw vs posterior scatter
├── posterior_predictive.png     # Model check
└── trace/                       # Optional: saved InferenceData (netCDF)
    ├── dim_social.nc
    ├── dim_ecological.nc
    └── dim_technological.nc
```

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Convergence failures with ~400 entities | Medium | High | Non-centered parameterization; increase tune/target_accept; reduce model if needed |
| Long sampling time (>1 hour) | Medium | Low | nutpie sampler; parallel dimensions; progress reporting |
| PyMC API breaking changes | Low | Medium | Pin minimum version; test in CI |
| Memory issues with large datasets | Low | Medium | Process dimensions sequentially; limit stored samples |
| Identifiability with small groups (n=5) | Medium | Medium | Regularizing priors; prior predictive checks; document limitations |

---

## 7. Simplification Options

If the full model proves too complex or slow, these simplifications can be applied incrementally:

| Simplification | Trade-off | When to Use |
|---------------|-----------|-------------|
| Drop entity-level random effects | Lose entity-specific shrinkage | If >500 entities cause convergence issues |
| Fix κ (run precision) | Lose run-level variance estimation | If run variance is negligible |
| Use Normal instead of Beta | Lose boundary handling | If scores rarely approach 0 or 100 |
| Fit dimensions jointly (multivariate) | Gain dimension correlations, lose speed | If dimension correlations are of interest |
| Use variational inference (ADVI) | Much faster, lose exact posteriors | For quick exploratory analysis |

---

## 8. Success Criteria

The implementation is considered successful when:

1. **Convergence:** R-hat < 1.01, ESS > 400, 0 divergences on the 15-participant simulated dataset
2. **Agreement:** Bayesian group contrasts align with permutation test conclusions (same groups identified as most/least different)
3. **Added value:** ICC decomposition shows non-trivial variance at multiple levels; shrinkage analysis shows meaningful partial pooling
4. **Usability:** Single CLI command produces all outputs; runs in under 90 minutes on the simulated dataset
5. **Robustness:** Graceful error messages when PyMC is not installed or convergence fails
