# Recommendations

Status: the main remediation items in this plan have been implemented. This document now functions as a closeout checklist plus a record of any residual follow-up.

## Priority Order

1. Fix hard runtime failures first. Completed.
2. Remove silent data fabrication next. Completed.
3. Normalize comparison semantics around entity alignment. Completed for the current pathway.
4. Reduce import-time side effects and add focused regression tests. Completed.

## P0

### 1. Repair non-Euclidean clustering

- In `entity/clustering.py`, pass a participant-id keyed dict into `compute_pairwise_distances()` instead of a raw centroid matrix.
- Add tests for `metric='aitchison'` and `metric='cosine'` through the public `cluster_participants()` wrapper.

Status: completed.

### 2. Normalize ID types in Bayesian prep

- In `prepare_beta_data()`, cast `entity`, `participant`, and `group` columns to `str` before building index maps and before `.map(...)`.
- Add a regression test using numeric participant IDs.

Status: completed.

## P1

### 3. Make `--participants` a real data filter

- Subset the `ParticipantComparison` object before calling `compute_distances()`, or add a subset-aware comparison method.
- Ensure the same filtered participant set drives:
  - summary statistics
  - saved JSON
  - heatmaps
  - similarity maps
  - group logic

Status: completed for `entity compare` and `entity compare-viz`.

### 4. Stop substituting missing scores with `0.0` or `50`

- Comparison loaders should raise on missing/malformed score columns by default.
- Visualization loaders should either:
  - fail with a clear schema error, or
  - mark missing data explicitly and skip affected entities.

Status: completed via fail-fast validation and clearer CLI surfacing.

## P2

### 5. Build an aligned participant × entity representation

- Add a canonical alignment step in `ParticipantComparison`.
- Make downstream consumers opt into either:
  - centroid-only comparison, or
  - entity-aligned comparison on a shared entity index.
- Do not let visualization code infer entity identity from row index.

Status: completed for the current comparison visualization path that previously relied on row-index identity.

### 6. Preserve original entity surface forms in prepare-scoring

- Deduplicate with a normalized key, but store the first-seen original surface form separately.
- Carry both fields if needed:
  - `entity`
  - `entity_normalized`

Status: completed by preserving the first-seen surface form while deduplicating on a normalized key.

## P3

### 7. Slim down package imports

- Avoid eager imports in `qualitative_analysis.entity.__init__`.
- Import optional/heavy modules lazily:
  - consolidator
  - visualizer
  - Bayesian components

Status: completed in `qualitative_analysis.entity.__init__`, with import regression tests and README documentation.

### 8. Normalize case behavior across the pipeline

- Decide once whether entity uniqueness is case-sensitive.
- Apply that rule consistently in:
  - detection
  - prepare-scoring
  - comparison loading

Status: partially addressed. `prepare-scoring` now preserves original surface form while deduplicating by normalized key, but detector/comparison case policy is still a design choice rather than a fully unified package-wide rule.

## Suggested Regression Tests

- `test_cluster_participants_aitchison_does_not_crash`
- `test_prepare_beta_data_accepts_numeric_participant_ids`
- `test_entity_compare_participants_filter_applies_to_distance_matrix`
- `test_compare_viz_participants_filter_applies_to_heatmap_and_similarity_map`
- `test_comparison_load_scores_raises_on_missing_dimension_values`
- `test_visualizer_rejects_missing_scores_instead_of_imputing_50`
- `test_prepare_scoring_preserves_original_entity_case`
- `test_overlaid_ternary_does_not_connect_entities_by_row_index_without_alignment`

Implemented in this remediation pass:

- `test_cluster_participants_aitchison_does_not_crash`
- `test_prepare_beta_data_accepts_numeric_participant_ids`
- `test_entity_compare_participants_filter_applies_to_distance_matrix`
- `test_comparison_load_scores_raises_on_missing_dimension_values`
- `test_visualizer_rejects_missing_scores_instead_of_imputing_50`
- `test_prepare_scoring_preserves_original_entity_case`
- `test_overlaid_ternary_does_not_connect_entities_by_row_index_without_alignment`
- package import/lazy-load regression coverage

Residual follow-up:

- decide whether entity uniqueness should be case-sensitive across the entire pipeline
- keep optional Bayesian/runtime checks and tests aligned with dependency evolution
- optionally reduce third-party warnings in the broader test suite
- decide whether full Bayesian integration tests should run in a dedicated runtime-ready environment as part of required CI, rather than being opportunistically skipped in constrained environments
