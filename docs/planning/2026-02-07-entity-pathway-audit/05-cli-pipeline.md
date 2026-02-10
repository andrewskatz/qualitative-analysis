# CLI & Pipeline Orchestration Audit

**Files:** entity_cli.py, unified_cli.py

---

## 1. Critical Runtime Bugs

### C2: Undefined Variable `chains` (Line 1531)

```python
loo_result = bayesian_model.compare_models(
    dim,
    chains=chains,  # NameError: 'chains' is not defined
    draws=args.draws,
    tune=args.tune,
    sampler=args.sampler,
)
```

The variable `chains` is never assigned in the scope of `run_entity_compare()`. Should be `chains=args.chains`.

**Impact:** Using `--loo-compare` flag will crash immediately.

### C3: Wrong Argument Order in Cluster Command (Line 2516)

```python
comparison.load_scores(
    str(input_path),
    dimension_names,           # This becomes participant_col!
    participant_col=args.participant_col,
    entity_col=args.entity_col,
)
```

`load_scores()` signature is:
```python
def load_scores(self, scores_path, participant_col="text_id", entity_col="entity",
                dimension_pattern="{dim}_mean", dimensions=None)
```

`dimension_names` (a list) is passed as the 2nd positional arg, which is `participant_col`. The actual dimensions keyword arg is never set.

**Fix:** Use keyword argument:
```python
comparison.load_scores(
    str(input_path),
    participant_col=args.participant_col,
    entity_col=args.entity_col,
    dimensions=dimension_names,
)
```

---

## 2. Pipeline Data Flow Breaks

### H14: Run-Level Data Lost in CSV Export

The entity scoring pipeline discards per-run scores when writing CSV:

```python
# In _write_scores_csv(), EntityScore.to_flat_dict() is called with include_runs=False
# This means columns like social_run1, social_run2, social_run3 are NOT written
```

But the `agreement` command expects run-level columns:

```python
# agreement._build_run_level_matrix() looks for:
# "{dim}_run1", "{dim}_run2", "{dim}_run3" columns
```

**Result:** The `score` → `agreement` pipeline is broken. Users cannot compute run-level agreement from the standard scoring CSV output. They would need to:
1. Use the JSON output (which does include runs)
2. Manually reformat data
3. Or modify `_write_scores_csv()` to include run data

**Fix:** Add `--include-runs` flag to `score` command, or always include run columns.

### H15: Group Information Lost Between detect and score

```
qa entity detect --group-col group → entities.csv (has "group" column)
qa entity score --input entities.csv → scores.csv (NO "group" column)
qa entity compare → requires manual --groups specification
```

The `score` command doesn't read or propagate the "group" column from its input CSV. The `EntityScorer` class stores `group` in `EntityScore` objects but the CLI doesn't populate it from input data.

**Fix:** Add `--group-col` parameter to `score` command and propagate through to output.

### M16: Silent Dimension Fallback to 50

```python
# entity_cli.py:719-735
mean_key = f"{dim_key}_mean"
if mean_key in row:
    entity_data["dimensions"][dim_key] = {"mean": float(row[mean_key]), ...}
else:
    entity_data["dimensions"][dim_key] = {"mean": 50}  # SILENT FALLBACK
```

If a dimension column is missing from the CSV, it's silently set to 50 (the scale midpoint). No warning, no error. Visualizations and comparisons proceed with fabricated data.

### M17: prepare-scoring Silently Ignores JSON Parse Errors

```python
# Line 1754
try:
    entities = json.loads(entities_json)
except json.JSONDecodeError:
    entities = []  # Silent empty list
```

If the `entities_json` column contains malformed JSON (e.g., from a failed LLM extraction), the entities for that row are silently dropped. No warning logged, no count of dropped rows.

---

## 3. Argument Validation Issues

### Missing Column Validation

No CLI command validates that the input CSV contains required columns before processing. The general pattern is:

1. Open CSV
2. Start iterating rows
3. Access expected columns
4. Fail silently or raise confusing errors deep in processing

**Better approach:** Validate required columns upfront:
```python
required_cols = {"entity", "context", "text_id"}
actual_cols = set(df.columns)
missing = required_cols - actual_cols
if missing:
    raise SystemExit(f"Missing required columns: {missing}")
```

### Dimension Names Not Validated Against Data

When `--dimensions social,ecological,technological` is specified, the CLI doesn't verify these dimensions exist in the input CSV. Missing dimensions silently default to 0 or 50 depending on the command.

### Metric/Aggregation Mode Mismatch

```python
if aggregate_mode == "distribution" and args.metric not in ("emd", "wasserstein"):
    print(f"\nNote: --aggregate=distribution is only meaningful with --metric=emd")
```

This prints a note but allows the invalid combination to proceed. Should be an error or at minimum a prominent warning.

---

## 4. CLI Design Observations

### Strengths

- Comprehensive help text for each command
- Consistent output directory structure with timestamped defaults
- Support for multiple output formats (JSON, CSV, PNG, HTML)
- `--verbose` flag for debugging LLM prompts/responses
- `--limit` flag for testing with subset of data

### Weaknesses

- No `--dry-run` mode to validate inputs without running analysis
- No `--validate` flag to check data format compatibility
- Error messages reference internal exceptions rather than user-friendly guidance
- No pipeline orchestration command (user must run each step manually)
- Output directory structure isn't documented; users must discover it empirically

### Pipeline Orchestration Gap

The entity pathway is conceptually a pipeline:
```
detect → [consolidate] → score → [viz | compare | agreement | cluster]
```

But each step is a separate CLI invocation with no built-in chaining. Users must:
1. Know the correct order
2. Manually pass output paths from one step to the next
3. Ensure data format compatibility between steps
4. Respecify options (dimensions, groups) at each step

A `qa entity pipeline` command that chains these steps with a single configuration file would improve usability.

---

## 5. unified_cli.py

No issues found. The subcommand registration is straightforward and correctly delegates to `entity_cli.py` functions. The entity subparser aliases (`ent`) are convenient.
