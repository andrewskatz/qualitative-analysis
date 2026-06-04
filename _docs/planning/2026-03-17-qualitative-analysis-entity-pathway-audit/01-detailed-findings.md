# Detailed Findings

Status note: these were the confirmed findings at audit time. Most have since been remediated; see `00-audit-summary.md` and `02-recommendations.md` in this folder for current closeout status.

## 1. Critical: non-Euclidean clustering is currently unusable

Location:

- `qualitative-analysis/src/qualitative_analysis/entity/clustering.py:371-382`

What happens:

- `cluster_participants()` handles Euclidean and cosine directly.
- For every other metric, it calls `compute_pairwise_distances(centroids, ...)`.
- `centroids` is a NumPy array, but `compute_pairwise_distances()` expects a `Dict[str, np.ndarray]` and immediately calls `.keys()`.

Evidence:

- Confirmed in `.venv-qa-pkg` with `metric='aitchison'`.
- Runtime error: `AttributeError: 'numpy.ndarray' object has no attribute 'keys'`.

Impact:

- `qa entity cluster --metric aitchison` is broken.
- Any future non-Euclidean extension through this path will fail the same way.

Why this matters:

- The CLI exposes these metrics as valid options, so this is a real broken user path, not a theoretical refactor bug.

## 2. Critical: Bayesian prep fails for numeric participant IDs

Location:

- `qualitative-analysis/src/qualitative_analysis/entity/bayesian.py:183-201`

What happens:

- `prepare_beta_data()` converts `entity_names`, `participant_names`, and `group_names` to `str`.
- It then builds `participant_map` with string keys.
- But `long_df["participant"]` is not normalized to string before `.map(participant_map)`.

Impact:

- If participant IDs are numeric in the source DataFrame, `participant_idx` becomes `NaN` and `.astype(int)` raises.

Evidence:

- Confirmed in `.venv-qa-pkg` with a tiny DataFrame using numeric `text_id` values.
- Runtime error: `pandas.errors.IntCastingNaNError`.

Why this matters:

- Numeric IDs are common in CSV workflows.
- This breaks Bayesian analysis before model fitting starts.

## 3. High: `--participants` is ignored for the actual comparison result

Locations:

- `qualitative-analysis/src/qualitative_analysis/entity_cli.py:1526-1531`
- `qualitative-analysis/src/qualitative_analysis/entity_cli.py:1624-1626`
- `qualitative-analysis/src/qualitative_analysis/entity_cli.py:2517-2521`
- `qualitative-analysis/src/qualitative_analysis/entity_cli.py:2537-2539`

What happens:

- Both `run_entity_compare()` and `run_entity_compare_viz()` parse the participant filter into `participants`.
- That filtered list is passed to some plotting helpers.
- But the actual distance matrix is still computed from the full `comparison` object with `comparison.compute_distances(...)`.

Impact:

- `comparison_results.json` still includes participants the user explicitly filtered out.
- Heatmaps and similarity maps that depend on `result.participant_ids` still operate on the full participant set.
- The printed summary statistics are also computed on the full set.

Why this matters:

- This is a silent correctness issue: the command appears to honor the filter, but much of the analysis does not.

## 4. High: comparison loading fabricates missing data as hard zeroes

Location:

- `qualitative-analysis/src/qualitative_analysis/entity/comparison.py:290-312`

What happens:

- Missing values and unparseable values are logged and then replaced with `0.0`.

Impact:

- The pipeline silently converts schema problems, export bugs, and partial data into substantively meaningful “lowest possible” scores.
- Distance metrics, centroids, group comparisons, clustering, and visualizations then operate on invented data.

Why this matters:

- For analytical code, fail-fast is much safer than silently manufacturing extreme values.

Recommended direction:

- Raise by default on missing/malformed score columns.
- If permissive behavior is needed, track explicit missingness and require an opt-in imputation mode.

## 5. High: entity alignment is never established, but some downstream code assumes it

Locations:

- `qualitative-analysis/src/qualitative_analysis/entity/comparison.py:314-325`
- `qualitative-analysis/src/qualitative_analysis/entity/comparison_viz.py:424-427`
- `qualitative-analysis/src/qualitative_analysis/entity/comparison_viz.py:704-708`

What happens:

- `ParticipantComparison.load_scores()` preserves per-participant row order and stores entity names separately per participant.
- It also builds `self.entity_names = sorted(entity_set)`, which is a global set-based ordering unrelated to each participant’s row order.
- `generate_overlaid_ternary(... connect_same_entities=True)` then uses global `self.comparison.entity_names[i]` to connect row `i` across participants.

Impact:

- If participants do not have identical entity inventories in identical order, those connection lines are wrong.
- More broadly, the comparison layer never creates a shared participant × entity matrix, so any future analysis that assumes row alignment will be fragile or incorrect.

Why this matters:

- This is exactly the kind of silent structural bug that produces persuasive but false visuals.

## 6. Medium: visualization loaders fabricate missing scores as `50`

Locations:

- `qualitative-analysis/src/qualitative_analysis/entity_cli.py:1044-1060`
- `qualitative-analysis/src/qualitative_analysis/entity/visualizer.py:334-341`
- `qualitative-analysis/src/qualitative_analysis/entity/visualizer.py:426-442`

What happens:

- `_read_scored_entities_csv()` substitutes invalid or missing scores with `{"mean": 50}`.
- The radar chart code also falls back to `50`.
- Ternary prep falls back to `50` when a score is absent.

Impact:

- Missing data becomes visually “moderate” data.
- Charts can look smooth and plausible even when the source file is incomplete or malformed.

Why this matters:

- This is worse than a loud error because it masks upstream failures.

## 7. Medium: `entity prepare-scoring` destroys original entity casing

Location:

- `qualitative-analysis/src/qualitative_analysis/entity_cli.py:2212-2224`

What happens:

- Entities are deduplicated by lowercase key.
- The output entity string is then set to that lowercase form rather than the first-seen original form.

Impact:

- Proper nouns, acronyms, branded terms, and domain-specific capitalization are lost before scoring.
- That can degrade prompt quality and reduce traceability back to the source material.

Why this matters:

- The code comment already acknowledges the problem: “Could be improved to preserve original case.”

## 8. Medium: package import has unnecessary heavy side effects

Locations:

- `qualitative-analysis/src/qualitative_analysis/entity/__init__.py:31-78`
- `qualitative-analysis/src/qualitative_analysis/entity/consolidator.py:33`
- `qualitative-analysis/src/qualitative_analysis/entity/visualizer.py:33-36`

What happens:

- Importing `qualitative_analysis.entity` eagerly imports scoring, visualization, consolidator, comparison, clustering, detector, and optional Bayesian symbols.
- That pulls in matplotlib unconditionally and embeddings/`sentence_transformers` transitively through the consolidator.

Observed behavior:

- In the package venv, a simple import of `qualitative_analysis.entity.models` succeeded, but only after triggering matplotlib/font-cache initialization.
- In the base environment earlier, this same import path cascaded into torch-related failure before test collection.

Impact:

- Lightweight code paths pay unnecessary import-time cost.
- Environment sensitivity increases because unrelated optional dependencies become import-time requirements.

Why this matters:

- A module import should not behave like full application startup.

## 9. Medium: entity detection deduplicates case-sensitively

Location:

- `qualitative-analysis/src/qualitative_analysis/entity/detector.py:191-193`

What happens:

- `all_entities` treats `"Climate Change"` and `"climate change"` as different unique entities.

Impact:

- `EntityDetectionResult.entities` can over-count unique entities.
- Downstream scoring inputs can contain near-duplicate entries that differ only by case.

Why this matters:

- The prepare-scoring pathway later lowercases entities, so the pipeline is inconsistent about case normalization.

## Test Coverage Gaps

Current tests do not appear to cover:

- Non-Euclidean clustering through `cluster_participants()`
- Numeric participant IDs in Bayesian prep
- CLI-level `--participants` filtering semantics
- Visualization behavior when score columns are missing or malformed
- Cross-participant entity alignment assumptions in comparison visualizations
