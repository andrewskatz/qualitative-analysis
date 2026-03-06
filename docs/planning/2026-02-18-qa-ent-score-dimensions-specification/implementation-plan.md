# Configurable Scoring Scale for `qa entity score`

## Context

The entity scoring pipeline currently hardcodes a 0-100 scale throughout. The user needs to run scoring with a 1-10 scale instead. The LLM prompt layer already supports configurable `scale_min`/`scale_max` via `DimensionDefinition`, but there are no CLI args to set it, and all downstream components (CI clamping, Bayesian modeling, visualizations) hardcode `100` in ~45 locations. This plan makes the scale configurable end-to-end via `--scale-min` / `--scale-max` CLI args.

## Architecture

**Propagation strategy:** Add a `ScaleConfig` dataclass to `models.py` with helper methods (`fraction()`, `midpoint`, `rope_default()`). CLI args override dimension definitions at scoring time and persist to `score_metadata.json`. Downstream commands auto-detect scale from metadata, with CLI override.

## Files to Modify

| File | Changes | Complexity |
|------|---------|------------|
| `entity/models.py` | +`ScaleConfig`, fix CI clamping in `from_scores()` | Low |
| `entity/scorer.py` | Pass scale bounds to `from_scores()` (1 call site) | Low |
| `entity/prompts/entity_scoring_v2.txt` | Remove hardcoded "Scale: 0-100" from example | Low |
| `entity_cli.py` | +CLI args, metadata persistence, auto-detection wiring | Medium |
| `entity/bayesian.py` | +`_invlogit_to_scale()` helper, ~15 replacements, scale params | High |
| `entity/visualizer.py` | Scale params on size encoding, radar chart, legend | Medium |
| `entity/comparison_viz.py` | Scale params on size encoding, CI clamping, axes, midpoint | Medium |
| `tests/test_bayesian.py` | +custom-scale test cases | Medium |

---

## Step 1: Foundation — `models.py`

### 1A. Add `ScaleConfig` dataclass (after `DimensionSet`, ~line 128)

```python
@dataclass
class ScaleConfig:
    scale_min: float = 0
    scale_max: float = 100

    @property
    def scale_range(self) -> float:
        return self.scale_max - self.scale_min

    @property
    def midpoint(self) -> float:
        return (self.scale_min + self.scale_max) / 2

    def fraction(self, value: float) -> float:
        """Convert raw score to [0, 1] fraction."""
        return max(0, min((value - self.scale_min) / self.scale_range, 1.0))

    def from_fraction(self, frac: float) -> float:
        """Convert [0, 1] fraction back to raw scale."""
        return self.scale_min + frac * self.scale_range

    def rope_default(self) -> float:
        """Proportional ROPE: 5% of scale range."""
        return self.scale_range * 0.05

    def reference_levels(self) -> list:
        """Reference sizes for legends at 25%, 50%, 75% of scale."""
        r = self.scale_range
        return [
            (self.scale_min + 0.25 * r, f"{self.scale_min + 0.25 * r:.0f}"),
            (self.scale_min + 0.50 * r, f"{self.scale_min + 0.50 * r:.0f}"),
            (self.scale_min + 0.75 * r, f"{self.scale_min + 0.75 * r:.0f}"),
        ]

    @classmethod
    def from_metadata_file(cls, metadata_path) -> "ScaleConfig":
        import json
        from pathlib import Path
        with open(Path(metadata_path)) as f:
            meta = json.load(f)
        return cls(
            scale_min=meta.get("scale_min", 0),
            scale_max=meta.get("scale_max", 100),
        )
```

### 1B. Fix CI clamping in `DimensionScore.from_scores()` (line 176)

Add `scale_min=0, scale_max=100` params to the method signature. Replace lines 221-222:
```python
# Before:
ci_low = max(0, mean_val - margin)
ci_high = min(100, mean_val + margin)
# After:
ci_low = max(scale_min, mean_val - margin)
ci_high = min(scale_max, mean_val + margin)
```

---

## Step 2: Scorer — `scorer.py`

### 2A. Pass scale bounds in `_aggregate_runs()` (line 541)

The `dim` variable (a `DimensionDefinition`) is already available in the loop. Change:
```python
# Before:
aggregated[dim_name] = DimensionScore.from_scores(
    dimension=dim_name, scores=scores, justification=combined_justification,
)
# After:
aggregated[dim_name] = DimensionScore.from_scores(
    dimension=dim_name, scores=scores, justification=combined_justification,
    scale_min=dim.scale_min, scale_max=dim.scale_max,
)
```

---

## Step 3: Prompt Template — `entity_scoring_v2.txt`

### 3A. Remove hardcoded scale from example (lines 86-88)

Replace:
```
- SOCIAL (Scale: 0-100): Human aspects including community, governance, economics, culture, and equity
- ECOLOGICAL (Scale: 0-100): Natural environment and biophysical processes
- TECHNOLOGICAL (Scale: 0-100): Human-made systems and engineered infrastructure
```
With:
```
- SOCIAL: Human aspects including community, governance, economics, culture, and equity
- ECOLOGICAL: Natural environment and biophysical processes
- TECHNOLOGICAL: Human-made systems and engineered infrastructure
(Use the same scale ranges specified in the dimension definitions above.)
```

---

## Step 4: CLI — `entity_cli.py`

### 4A. Add `--scale-min` / `--scale-max` to `add_entity_score_args()` (after line 59)

```python
parser.add_argument("--scale-min", type=int, default=None,
    help="Minimum score value. Overrides scale_min on all dimensions (default: per-dimension, typically 0).")
parser.add_argument("--scale-max", type=int, default=None,
    help="Maximum score value. Overrides scale_max on all dimensions (default: per-dimension, typically 100).")
```

### 4B. Apply overrides in `run_entity_score()` (after line 172)

```python
if args.scale_min is not None or args.scale_max is not None:
    for dim in dimensions:
        if args.scale_min is not None:
            dim.scale_min = args.scale_min
        if args.scale_max is not None:
            dim.scale_max = args.scale_max
    print(f"Scale override: {dimensions[0].scale_min}-{dimensions[0].scale_max}")
```

### 4C. Persist scale in metadata (line 318)

Add to the metadata dict:
```python
"scale_min": dimensions[0].scale_min,
"scale_max": dimensions[0].scale_max,
```

### 4D. Fix checkpoint reconstruction (line 481)

Pass scale bounds from `dim_def`:
```python
dim_scores[dim_key] = DimensionScore.from_scores(
    dimension=dim_key, scores=run_scores,
    justification=data.get(f"{dim_key}_justification", ""),
    scale_min=dim_def.scale_min, scale_max=dim_def.scale_max,
)
```

### 4E. Add `--scale-min` / `--scale-max` to downstream commands

Add these optional args to `add_entity_viz_args()`, `add_entity_compare_args()`, and `add_entity_compare_viz_args()`. Default `None` (auto-detect from metadata).

### 4F. Scale auto-detection helper

Add a helper function:
```python
def _resolve_scale(args, input_csv_dir: Path) -> ScaleConfig:
    """Resolve scale from CLI args or score_metadata.json."""
    from qualitative_analysis.entity.models import ScaleConfig
    scale_min, scale_max = 0, 100
    meta_path = input_csv_dir / "score_metadata.json"
    if meta_path.exists():
        sc = ScaleConfig.from_metadata_file(meta_path)
        scale_min, scale_max = sc.scale_min, sc.scale_max
    if getattr(args, 'scale_min', None) is not None:
        scale_min = args.scale_min
    if getattr(args, 'scale_max', None) is not None:
        scale_max = args.scale_max
    return ScaleConfig(scale_min=scale_min, scale_max=scale_max)
```

Wire this into `run_entity_viz()`, `run_entity_compare()`, and Bayesian model init.

### 4G. Update ROPE default (line 1126)

Change `--rope-delta` default to `None` and help text to `"ROPE half-width for practical significance (default: 5% of scale range)."`. Resolve `None` to `scale.rope_default()` before passing to the Bayesian model.

### 4H. Update print statement (line 1792)

```python
# Before:
print("\nGroup Contrasts (population-level mean difference, 0-100 scale):")
# After:
print(f"\nGroup Contrasts (population-level mean difference, {scale.scale_min}-{scale.scale_max} scale):")
```

---

## Step 5: Bayesian Module — `bayesian.py`

### 5A. Add `_invlogit_to_scale()` module-level helper (near top, after imports)

```python
def _invlogit_to_scale(logit_values, scale_min=0, scale_max=100):
    """Convert logit values to configured score scale."""
    prob = 1 / (1 + np.exp(-logit_values))
    return prob * (scale_max - scale_min) + scale_min
```

### 5B. Add `scale_min`/`scale_max` to `BayesianEntityModel.__init__()` (line 247)

Add params, store as `self.scale_min`, `self.scale_max`, `self.scale_range`.

### 5C. Fix `prepare_beta_data()` (line 147-160)

Add `scale_min`/`scale_max` params. Replace:
```python
# Before (line 152):
long_df["y"] = (long_df["score"] * (dim_counts - 1) + 0.5) / (dim_counts * 100.0)
# After:
scale_range = scale_max - scale_min
long_df["y"] = (((long_df["score"] - scale_min) / scale_range) * (dim_counts - 1) + 0.5) / dim_counts
```

Fix boundary detection (line 156):
```python
boundary_mask = long_df["score"].isin([float(scale_min), float(scale_max)])
```

### 5D. Replace all `* 100` invlogit conversions (11 locations)

Replace each `1 / (1 + np.exp(-x)) * 100` with `_invlogit_to_scale(x, self.scale_min, self.scale_max)`:

- Line 733: `summarize_posteriors()` group effects
- Lines 796-797: `compute_group_contrasts()` probability scale
- Line 885: `compute_marginal_group_contrasts()`
- Line 1025: `compute_shrinkage()` prob_samples
- Line 1031: `compute_shrinkage()` grand_mean
- Line 1649: Prior sensitivity (LOO-CV reduced model)
- Line 1684: Prior sensitivity (LOO-CV full model)
- Line 1942: `BayesianVisualizer.plot_posterior_densities()` — needs access to model's scale
- Line 2245: `BayesianVisualizer.plot_group_ternary()` — needs access to model's scale

### 5E. Fix inverse S&V squeeze (line 2152)

```python
# Before:
rep_scores = (flat[r_idx] * n_dim - 0.5) / max(n_dim - 1, 1) * 100
# After:
prob = (flat[r_idx] * n_dim - 0.5) / max(n_dim - 1, 1)
rep_scores = prob * self.model.scale_range + self.model.scale_min
```

### 5F. Fix `BayesianVisualizer` plot labels and axes

`BayesianVisualizer.__init__` already stores `self.model`, so access scale via `self.model.scale_min` / `self.model.scale_max`.

| Line | Current | Replacement |
|------|---------|-------------|
| 1898 | `"Difference (0-100 scale)"` | `f"Difference ({self.model.scale_min}-{self.model.scale_max} scale)"` |
| 1950 | `"Score (0-100)"` | `f"Score ({self.model.scale_min}-{self.model.scale_max})"` |
| 2044 | `ax.plot([0, 100], [0, 100], ...)` | `ax.plot([self.model.scale_min, self.model.scale_max], ...)` |
| 2046 | `"Raw Mean (0-100)"` | `f"Raw Mean ({self.model.scale_min}-{self.model.scale_max})"` |
| 2048 | `"Posterior Mean (0-100)"` | `f"Posterior Mean ({self.model.scale_min}-{self.model.scale_max})"` |
| 2050-51 | `set_xlim(0, 100)` / `set_ylim(0, 100)` | Use `self.model.scale_min`/`scale_max` |
| 2154 | `range=(0, 100)` | `range=(self.model.scale_min, self.model.scale_max)` |
| 2160 | `range=(0, 100)` | same |
| 2165 | `"Score (0-100)"` | f-string with model scale |

### 5G. Note: Shrinkage clamping (line 1051-1052) — NO CHANGE

The `max(0.0, min(100.0, raw_shrinkage))` clamps a *percentage* (0-100%), not a score. This is scale-independent.

---

## Step 6: Visualizer — `visualizer.py`

### 6A. Add `scale_min`/`scale_max` params to `generate_ternary_plot()` and internal methods

### 6B. Fix size encoding (line 193)
```python
# Before:
abs_fraction = max(0, min(mean_score / 100.0, 1.0))
# After:
abs_fraction = max(0, min((mean_score - scale_min) / (scale_max - scale_min), 1.0))
```

### 6C. Fix size legend `_add_size_legend()` (lines 736-745)

Use `ScaleConfig.reference_levels()` instead of hardcoded 25/50/75. Fix the fraction computation similarly.

### 6D. Fix radar chart (lines 351-353)
```python
# Before:
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
# After:
ax.set_ylim(scale_min, scale_max)
r = scale_max - scale_min
ticks = [scale_min + i * r / 5 for i in range(1, 6)]
ax.set_yticks(ticks)
ax.set_yticklabels([f'{t:.0f}' for t in ticks])
```

---

## Step 7: Comparison Viz — `comparison_viz.py`

### 7A. Add `scale_min`/`scale_max` to `ComparisonVisualizer.__init__()` (line 37)

Store as instance attributes.

### 7B. Fix size encoding (lines 299, 482, 709)

Replace `/ 100.0` with `/ (self.scale_max - self.scale_min)` offset by `self.scale_min` (same pattern as visualizer).

### 7C. Fix forest plot (lines 1123-1136)

- CI clamping: use `self.scale_min`/`self.scale_max` instead of 0.0/100.0
- Axis limits: `x_lim = (self.scale_min, self.scale_max)`
- Midpoint: `ref_line_x = (self.scale_min + self.scale_max) / 2`
- Subtitle: f-string with actual scale values

---

## Step 8: Tests — `test_bayesian.py`

### 8A. Add custom-scale test fixture

Create `_make_ground_truth_df_custom_scale()` generating scores on a 1-10 scale.

### 8B. Add parameterized tests

Test that:
- `prepare_beta_data()` correctly normalizes 1-10 scores to (0, 1)
- Posterior group means are on the 1-10 scale
- ROPE proportional default returns 0.45 for a 1-10 scale
- Plot labels contain "1-10" not "0-100"

---

## Implementation Order

1. **Step 1** — Foundation (`models.py`): `ScaleConfig` + CI fix
2. **Step 2** — Scorer (`scorer.py`): pass scale to `from_scores()`
3. **Step 3** — Prompt template: remove hardcoded example scale
4. **Step 4** — CLI (`entity_cli.py`): args, metadata, auto-detection, ROPE
5. **Step 5** — Bayesian (`bayesian.py`): helper fn, S&V squeeze, invlogit, plots
6. **Step 6** — Visualizer (`visualizer.py`): size, legend, radar
7. **Step 7** — Comparison viz (`comparison_viz.py`): size, forest plot, axes
8. **Step 8** — Tests (`test_bayesian.py`): custom-scale tests

---

## Verification

1. **Unit tests**: Run `pytest qualitative-analysis/tests/test_bayesian.py -v` — existing tests pass (default 0-100 scale unchanged)
2. **Scoring test**: Run `qa entity score <test_csv> --scale-min 1 --scale-max 10 --limit 3` — verify prompt shows "1-10 scale", output scores are in 1-10 range, metadata JSON contains `scale_min: 1, scale_max: 10`
3. **Viz test**: Run `qa entity viz <scored_csv>` from a scored directory with metadata — verify auto-detects scale, axes/labels reflect actual scale
4. **Bayesian test**: Run `qa entity compare <scored_csv> --method bayesian` — verify S&V squeeze normalizes correctly, posterior means on correct scale, plot labels correct

---

## Documentation Location

Full plan and audit docs: `docs/planning/2026-02-18-qa-ent-score-dimensions-specification/`
