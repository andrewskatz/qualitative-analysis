# Prioritized Recommendations

---

## Priority 1: Fix persistence-contract bugs

1. **Align `EntityScore.to_dict()` and `EntityScorer.load()`**
   - Emit `{dim}_mode` in flattened outputs, or stop expecting it during load.
   - Restore `window_index` and `group` in `EntityScorer.load()`.
   - Decide explicitly whether JSON load should also reconstruct `runs`; document the choice either way.

2. **Add a focused JSON round-trip test for scorer outputs**
   - Assert preservation of run scores, `mode`, `window_index`, and `group`.
   - Include one entity scored on a non-default scale.

**Why first:** this is the clearest correctness issue in the current package implementation.

---

## Priority 2: Add direct scorer tests

Create package tests for:
- valid `_parse_response()` with `v2`-style output
- valid `_parse_response()` with `v4`-style output (no justifications)
- rejection of missing expected dimensions
- behavior on unexpected dimensions and invalid score types
- `_aggregate_runs()` justification selection and statistics
- `score_entity()` partial-success behavior when one run fails
- `score_entities()` error counting and callback invocation
- CLI checkpoint/resume using a tiny synthetic scoring input

**Why second:** the pathway’s highest-risk logic currently lacks the most direct tests.

---

## Priority 3: Reduce architectural drift in detection

1. Move `EntityResponse` into a package-neutral shared module, or define a package-local entity schema in `entity/`.
2. Centralize JSON extraction into one helper used by both entity and relationship extraction code.
3. Decide whether detector deduplication should stay case-sensitive; if not, normalize while preserving original display text.

**Why third:** these are design-quality issues rather than immediate runtime bugs, but they affect maintainability and package independence.

---

## Priority 4: Harden scoring parser behavior

Recommended changes:
- validate scores against each dimension’s `scale_min` / `scale_max`
- choose an explicit policy for float scores: reject, round, or preserve
- improve error messages so omitted dimensions are reported alongside the raw parsed payload
- optionally surface parse warnings in returned metadata for later auditability

**Why fourth:** current behavior is usable, but failure analysis will be harder than it needs to be during real scoring runs.

---

## Priority 5: Document and test intentional prompt-version contracts

1. Document that `v2` and `v3` preserve interpretability through justifications.
2. Document that `v4` is a deliberate score-only mode that intentionally omits reasoning fields.
3. Document when each version should be used, rather than treating the versions as interchangeable.
4. Add a test asserting that the scorer tolerates the missing `initial_observations` / `justification` fields under `v4`.

**Why fifth:** this is mainly a product-contract issue inside the package, but it directly affects research interpretability.

---

## Priority 6: Clean up CLI scale semantics

- Either use `_resolve_scale()` inside `run_entity_score()` or remove/refactor it so scale behavior is not split between code paths.
- Document clearly that the scoring command uses:
  1. dimension definitions from `--dimensions`
  2. optional explicit CLI overrides
  3. not score-metadata auto-detection, unless that is intentionally added later

**Why sixth:** this is a smaller issue, but it is the kind of subtle drift that becomes confusing during reuse.

---

## Suggested Minimal Test Plan

A good near-term package-only regression suite would include:
- `tests/test_entity_scorer.py`
- `tests/test_entity_score_checkpointing.py`

Suggested cases:
- parse valid fenced JSON
- parse valid embedded JSON
- fail on missing dimensions
- aggregate three runs with stable summary statistics
- save/load round-trip preserves flattened semantics
- checkpoint resume skips already-scored entities and merges outputs correctly

---

## Overall Assessment

The package-side entity identification and SETS scoring pathway is **implemented and operational**, not experimental. The main need is **hardening**, not redesign.

If I were sequencing follow-up work, I would do it in this order:
1. fix scorer persistence fidelity
2. add direct scorer tests
3. reduce detector/schema coupling and duplicated parsing logic
4. clarify prompt-version and scale-resolution contracts

