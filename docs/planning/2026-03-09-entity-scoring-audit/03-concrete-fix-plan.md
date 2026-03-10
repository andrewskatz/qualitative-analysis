# Concrete Fix Plan

**Goal:** harden the package-side entity identification and SETS scoring pathway without redesigning the overall architecture.

**Important clarification:** prompt versions `v2`, `v3`, and `v4` should be treated as **intentional output contracts**. The fix plan below assumes that version differences are deliberate and focuses on documenting and testing those contracts rather than removing them.

---

## Scope of implementation work

In scope:
- `entity/scorer.py`
- `entity/models.py`
- `entity/detector.py`
- `relationships/components/entity_extractor.py` only where shared parsing/schema extraction is involved
- `entity_cli.py`
- new package tests under `qualitative-analysis/tests/`
- package-side docs/comments where needed

Out of scope:
- web app/backend scoring code
- changing the basic SETS framework design
- removing intentional prompt versions

---

## Phase 1: Fix scorer persistence fidelity

### Changes
1. **Align save/load contracts**
   - In `entity/models.py`, update `EntityScore.to_dict()` so flattened outputs include `{dim}_mode`.
   - In `entity/scorer.py`, update `load()` to restore:
     - `window_index`
     - `group`
     - `runs` if present in saved JSON
2. **Make load behavior explicit**
   - If `runs` are absent, continue reconstructing aggregated `DimensionScore` objects from flat columns.
   - Add a docstring note that load is “full fidelity when runs are present, aggregate fidelity otherwise.”

### Tests
- Add/extend a scorer serialization test covering:
  - run scores
  - `mode`
  - `window_index`
  - `group`
  - non-default scale bounds

### Acceptance criteria
- A saved `EntityScoreResult` reloads without losing `mode`, `window_index`, or `group`.
- Existing JSON without `runs` still loads successfully.

---

## Phase 2: Add direct scorer tests

### New test file
- `qualitative-analysis/tests/test_entity_scorer.py`

### Required cases
1. `_parse_response()` accepts valid fenced JSON.
2. `_parse_response()` accepts embedded JSON surrounding a `dimension_scores` payload.
3. `_parse_response()` rejects missing expected dimensions.
4. `_parse_response()` handles `v4`-style outputs with no `justification` / `initial_observations`.
5. `_aggregate_runs()` computes stable summary stats and selects a deterministic justification.
6. `score_entity()` preserves partial success when one run fails.
7. `score_entities()` increments `errors` when an entity fully fails.
8. `score_entities()` passes through `text_id`, `window_index`, and `group`.

### Acceptance criteria
- Core scorer behavior is directly covered by unit tests instead of only indirectly through Bayesian tests.

---

## Phase 3: Add checkpoint/resume coverage

### New test file
- `qualitative-analysis/tests/test_entity_score_checkpointing.py`

### Required cases
1. `_append_to_checkpoint()` writes a score that `_load_checkpoint()` can reconstruct.
2. Resume skips previously scored entities using `entity|text_id|window_index` keys.
3. Final merge preserves checkpointed + newly scored entities.
4. Optional `group` and `window_index` fields survive checkpoint round-trip.

### Acceptance criteria
- Resume behavior is locked down by tests and matches the live CLI pathway.

---

## Phase 4: Harden parser behavior

### Changes
1. In `entity/scorer.py`, decide and document the float-score policy:
   - preferred: reject non-integer numeric scores with a clear error, or
   - acceptable: round explicitly instead of truncating with `int(score)`
2. Validate scores against each dimension’s `scale_min` / `scale_max`.
3. Improve parse errors to report:
   - expected dimensions
   - parsed dimensions
   - first 200-300 chars of the raw response when helpful
4. Keep support for intentional prompt-version differences.

### Acceptance criteria
- Parser failures become diagnosable from logs/test failures.
- Out-of-range scores do not silently propagate.

---

## Phase 5: Document and test prompt-version contracts

### Changes
1. Add scorer tests that explicitly cover `v2`, `v3`, and `v4` output shapes.
2. Add short inline comments or docstrings in `entity/scorer.py` and/or prompt docs stating:
   - `v2`: reasoning + justifications
   - `v3`: reasoning-first ordering
   - `v4`: score-only mode
3. Ensure recommendations/docs describe version choice as a workflow decision, not an implementation bug.

### Acceptance criteria
- Maintainers can tell which fields are intentionally optional under each prompt version.
- Future prompt edits that accidentally break a version contract will fail tests.

---

## Phase 6: Reduce detector coupling and parsing duplication

### Changes
1. Move the shared entity-extraction response schema and JSON extraction helper into a neutral shared module, e.g.:
   - `qualitative_analysis/entity/shared_parsing.py`, or
   - `qualitative_analysis/core/json_extraction.py`
2. Update both:
   - `entity/detector.py`
   - `relationships/components/entity_extractor.py`
   to import from the shared location.
3. Decide whether detector deduplication should remain case-sensitive.
   - If not, normalize for dedupe while preserving original surface form in outputs.

### Acceptance criteria
- Detector no longer depends on a relationship-side schema definition.
- JSON extraction changes happen in one place.

---

## Phase 7: Clarify CLI scale semantics

### Changes
1. Either wire `_resolve_scale()` into `run_entity_score()` or remove/refactor it.
2. Add a small test covering scoring scale precedence:
   - dimensions file / SETS defaults
   - explicit `--scale-min` / `--scale-max`
   - no hidden metadata auto-detection unless intentionally added

### Acceptance criteria
- The scoring command has one clear scale-resolution contract.

---

## Recommended execution order

1. Phase 1: persistence fidelity
2. Phase 2: direct scorer tests
3. Phase 3: checkpoint/resume tests
4. Phase 4: parser hardening
5. Phase 5: prompt-version contract tests/docs
6. Phase 6: detector coupling cleanup
7. Phase 7: CLI scale semantics cleanup

This order fixes correctness first, then adds regression protection, then cleans up architecture.

