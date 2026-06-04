# Prioritized Remediation Plan

---

## Priority 1: Immediate Fixes (Runtime Crashes)

These will cause crashes when triggered. Fix before any further use.

### 1.1 Fix undefined `chains` variable (entity_cli.py:1531)
```
Change: chains=chains
To:     chains=args.chains
```
**Effort:** 1 line | **Risk:** None

### 1.2 Fix argument order in cluster command (entity_cli.py:2516)
```python
# Change:
comparison.load_scores(str(input_path), dimension_names, ...)

# To:
comparison.load_scores(str(input_path), participant_col=args.participant_col,
                       entity_col=args.entity_col, dimensions=dimension_names)
```
**Effort:** 1 line | **Risk:** None

### 1.3 Fix compute_pairwise_distances input type (clustering.py:381)
```python
# Change:
distance_matrix = compute_pairwise_distances(centroids, metric=metric, aggregate=None)

# To:
scores_dict = {pid: centroids[i] for i, pid in enumerate(participant_ids)}
distance_matrix, _ = compute_pairwise_distances(scores_dict, metric=metric)
```
**Effort:** 2-3 lines | **Risk:** Low

### 1.4 Fix log-likelihood capture in LOO-CV (bayesian.py:1191-1203)
```python
# Change:
pm.compute_log_likelihood(full_trace)

# To:
full_trace = pm.compute_log_likelihood(full_trace)
```
Same for reduced_trace at line 1203.
**Effort:** 2 lines | **Risk:** None

---

## Priority 2: Statistical Correctness (High Impact on Results)

These produce incorrect statistical results and should be fixed before publishing any analyses.

### 2.1 Rewrite Bayes Factor computation (bayesian.py:1326-1331)

The Savage-Dickey density ratio needs to properly integrate sigma_group out of the posterior. Replace the KDE-on-contrast-samples approach with:

```python
# For each posterior draw:
for draw_idx in range(n_draws):
    sigma_g = posterior_sigma_group[draw_idx]
    posterior_density_at_0[draw_idx] = norm.pdf(0, loc=0, scale=np.sqrt(2) * sigma_g)
posterior_at_0 = np.mean(posterior_density_at_0)
```

**Effort:** ~30 lines | **Risk:** Medium (need to verify posterior parameter extraction)

### 2.2 Add score range validation (scorer.py:465-472)

```python
if not (0 <= score <= 100):
    logger.warning(f"Score {score} for {dim_name} outside [0, 100], clamping")
    score = max(0, min(100, score))
```

**Effort:** 3 lines | **Risk:** None

### 2.3 Fix participant ID type mismatch (bayesian.py:175)

```python
# Change:
participant_to_group[pid] = gid

# To:
participant_to_group[str(pid)] = gid
```

**Effort:** 1 line | **Risk:** Low

### 2.4 Validate Ward linkage requires Euclidean (clustering.py:145)

```python
if linkage_method == "ward" and metric != "euclidean":
    logger.warning("Ward linkage assumes Euclidean distances. Results may be invalid "
                   "with metric='%s'. Consider using 'average' or 'complete' linkage.", metric)
```

**Effort:** 3 lines | **Risk:** None

### 2.5 Document/fix K-means metric limitation (clustering.py:366)

Either raise an error when metric != "euclidean" for K-means, or switch to K-medoids:
```python
if method == "kmeans" and metric != "euclidean":
    raise ValueError("K-means only supports Euclidean distance. "
                     "Use method='hierarchical' for other metrics.")
```

**Effort:** 3 lines | **Risk:** None

---

## Priority 3: Design Decision (Requires Team Discussion)

### 3.1 Resolve independent vs. compositional contradiction

This is the most consequential finding and requires a design decision. See [01-design-contradiction.md](01-design-contradiction.md) for full analysis. The three options are:

**Option A: Enforce compositional** - Modify LLM prompt to "distribute 100 points"; validate sum=100
**Option B: Enforce independent** - Remove ternary normalization; use radar charts; remove Aitchison distance
**Option C: Hybrid** - Keep both but clearly label ternary as "relative proportions" and add magnitude encoding

Recommendation: **Option C** is most pragmatic, but requires:
1. Adding point-size encoding for total score on ternary plots
2. Clear documentation that ternary shows relative proportions only
3. Defaulting distance metric to Euclidean (not Aitchison) everywhere
4. Warning when Aitchison distance is selected for non-compositional data

**Effort:** ~50-100 lines across multiple files | **Risk:** Medium

### 3.2 Fix ternary coordinate mapping consistency

Both visualizer.py and comparison_viz.py need to use the same dimension-to-vertex mapping. Define it once:

```python
# In a shared constants module:
TERNARY_VERTEX_ORDER = {
    0: "top",      # First dimension → top vertex
    1: "bottom_left",   # Second dimension → bottom-left
    2: "bottom_right",  # Third dimension → bottom-right
}
```

Then use consistently in both files.
**Effort:** ~20 lines | **Risk:** Low

---

## Priority 4: Pipeline Integrity

### 4.1 Preserve run-level data in CSV export

Add `--include-runs` flag to `score` command:
```python
parser.add_argument("--include-runs", action="store_true",
                    help="Include per-run scores in CSV (needed for agreement analysis)")
```

Then pass `include_runs=args.include_runs` to `to_flat_dict()`.
**Effort:** ~10 lines | **Risk:** None

### 4.2 Propagate group information through pipeline

Add `--group-col` to `score` command. Read group from input CSV and populate `EntityScore.group`.
**Effort:** ~15 lines | **Risk:** Low

### 4.3 Add input validation to CLI commands

Add a shared `_validate_csv_columns()` function:
```python
def _validate_csv_columns(path, required_cols, optional_cols=None):
    df = pd.read_csv(path, nrows=0)
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {path}: {missing}")
```

Call at the start of each command.
**Effort:** ~20 lines + 1 call per command | **Risk:** None

---

## Priority 5: Statistical Improvements (Correctness Refinements)

### 5.1 Fix CV for zero mean (models.py:227)
```python
cv = std_dev / mean_val if mean_val > 0 else float('nan')
```

### 5.2 Fix mode fabrication (models.py:211-212)
```python
modes = statistics.multimode(scores)
mode_val = float(modes[0]) if len(modes) < len(scores) else None
```

### 5.3 Fix shrinkage edge case (bayesian.py:919-921)
```python
shrinkage_pct = float('nan') if abs(raw - grand_mean) <= 1.0 else ...
```

### 5.4 Fix agreement expected disagreement for missing data (agreement.py:157)

Replace `2 * np.var(all_values, ddof=0)` with coincidence-matrix-marginal formulation per Krippendorff (2011). This is a larger change requiring computation from the coincidence matrix itself.
**Effort:** ~20-30 lines | **Risk:** Medium (need to verify against reference implementation)

---

## Priority 6: Usability Improvements (Nice to Have)

### 6.1 Add colorblind-safe palette option
### 6.2 Improve prompt specificity for entity extraction
### 6.3 Add `--dry-run` / `--validate` mode
### 6.4 Add retry logic for LLM calls in detector
### 6.5 Implement case-insensitive entity deduplication option
### 6.6 Document expected CSV formats for each pipeline step

---

## Summary

| Priority | Items | Effort | Impact |
|----------|-------|--------|--------|
| P1: Crash fixes | 4 | ~1 hour | Prevents runtime errors |
| P2: Statistical correctness | 5 | ~4 hours | Fixes incorrect results |
| P3: Design decision | 2 | ~8 hours + discussion | Resolves fundamental contradiction |
| P4: Pipeline integrity | 3 | ~4 hours | Enables full pipeline flow |
| P5: Statistical refinements | 4 | ~3 hours | Improves accuracy |
| P6: Usability | 6 | ~8 hours | Better user experience |
