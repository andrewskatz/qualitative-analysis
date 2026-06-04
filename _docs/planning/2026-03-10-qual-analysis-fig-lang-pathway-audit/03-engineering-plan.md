# Figurative Pathway Engineering Plan

## Objective

Harden the package-side figurative language pathway in `qualitative-analysis` without redesigning the public workflow. The implementation order below prioritizes reliability first, then result fidelity, then contract clarity.

## Phase 1: Detection hardening

### Goal

Reduce false positives/false negatives caused by malformed LLM output and make failures visible in package metadata.

### Files to change

- `qualitative-analysis/src/qualitative_analysis/figurative/components/scanner.py`
- `qualitative-analysis/src/qualitative_analysis/figurative/strategies/two_step.py`
- `qualitative-analysis/tests/test_mock_pipeline.py`
- `qualitative-analysis/tests/test_figurative_scanner.py`

### Planned changes

1. Use provider-level `generate_json(...)` for binary detection.
2. Use provider-level structured output for extraction with a top-level `instances` envelope.
3. Keep a conservative raw-response fallback path for JSON salvage.
4. Remove the permissive `"yes"/"true"` heuristic detection fallback.
5. Surface scanner parse/fallback counts in `DetectionResult.metadata`.
6. Fix context ordering so detection sees only prior summaries, not the current window summary.

### Acceptance criteria

- Malformed detector output does **not** become a false positive.
- Parse failures are visible in aggregated metadata.
- Existing happy-path detector behavior still passes.
- Window 0 detection sees `No prior context.` rather than its own summary.

## Phase 2: Result fidelity and provenance

### Goal

Improve the semantic quality of emitted instances and make downstream counting safer.

### Current status

Implemented in the package detector and CLI export path:

- canonical deduplication across overlapping windows
- deterministic window-local alignment with conservative normalized-exact recovery
- detect→map compatibility without requiring offsets
- instance CSV provenance export for `alignment_status`, `support_count`, and `supporting_window_indices`
- stable representative `window_text` export when writing instances output

### Files to change

- `qualitative-analysis/src/qualitative_analysis/figurative/strategies/two_step.py`
- `qualitative-analysis/src/qualitative_analysis/figurative/models.py`
- `qualitative-analysis/src/qualitative_analysis/core/text.py`
- `qualitative-analysis/src/qualitative_analysis/cli.py`
- `qualitative-analysis/tests/test_figurative_detect_to_map.py`

### Planned changes

1. Decide and implement an overlap policy.
2. Populate `start_char` / `end_char` only via deterministic window-local alignment, allowing conservative normalized-exact matching but never LLM-generated or approximate offsets.
3. Make `window_text` output behavior explicit and stable for instance CSV export.
4. Export instance provenance/support columns: `alignment_status`, `support_count`, and `supporting_window_indices`.
5. Add a detect→map contract test using real CSV columns.

### Acceptance criteria

- Overlap behavior is deterministic and tested.
- The emitted instance schema no longer promises unsupported provenance.
- Minor text drift does not suppress instance rows when offsets cannot be recovered.
- CLI instances CSV exports provenance/support fields with stable representative `window_text` behavior.
- `detect` output can feed `map` without undocumented assumptions.

## Phase 3: Provider and CLI contract clarity

### Goal

Align package interfaces with what is actually implemented today.

### Files to change

- `qualitative-analysis/src/qualitative_analysis/figurative/detector.py`
- `qualitative-analysis/src/qualitative_analysis/figurative/domains/extractor.py`
- `qualitative-analysis/src/qualitative_analysis/unified_cli.py`
- package docs/help text

### Planned changes

1. Either expose non-Ollama providers directly or document Ollama as the first-class path.
2. Clarify `qa figurative pipeline` as post-detection, or expand it to include detection.
3. Tighten help text around detect→map handoff expectations.

### Acceptance criteria

- CLI help text matches real behavior.
- Provider support is explicit rather than implied.

## Phase 4: Normalization async/sync boundary

### Goal

Remove the event-loop hazard in LLM-based canonicalization.

### Files to change

- `qualitative-analysis/src/qualitative_analysis/figurative/domains/normalizer.py`
- `qualitative-analysis/tests/test_domain_normalizer_llm.py`

### Planned changes

1. Split normalization into async and sync-safe entry points.
2. Avoid `run_until_complete(...)` inside the core synchronous path.
3. Add regression coverage for active-event-loop behavior.

### Acceptance criteria

- LLM normalization works predictably from both CLI and test contexts.
- No nested-event-loop runtime failure in supported use.

## Test strategy

### Near-term tests

1. `test_mock_pipeline.py`
   - keep the happy-path detector integration signal green
2. `test_figurative_scanner.py`
   - valid structured detection
   - malformed detection output stays conservative
   - extraction fallback salvages valid items and counts skipped invalid ones
   - type filtering still works
   - strategy metadata aggregates parse/fallback counts
   - strategy context excludes current-window summary

### Validation command

- `pytest tests/test_mock_pipeline.py tests/test_figurative_scanner.py`

## Immediate implementation slice

Implement **Phase 1** now.

This gives the highest reliability return with limited blast radius because it changes figurative detector internals, result metadata, and focused tests only.