# Phase 2 Spec: Provenance and Output Fidelity

## Purpose

This spec defines how the `qualitative-analysis` package should represent figurative-language detection outputs so they are traceable, semantically honest, and stable for downstream package consumers such as domain mapping.

This phase applies only to the package-side figurative pathway.

## Core principles

1. **Do not ask the LLM for offsets.**
2. **Do not emit provenance the package cannot justify deterministically.**
3. **Primary outputs should represent canonical instances, not accidental window duplicates.**
4. **Downstream mapping must depend on documented fields only.**

## Explicit non-goal

The package will **not** request `start_char`, `end_char`, or global span offsets from the LLM. LLM-generated offsets are too unreliable for a provenance contract.

## Current package contract summary

### Detector output today

- Summary CSV includes `contains_figurative`, `confidence`, `instance_count`, `window_count`, `strategy`
- Instances CSV includes:
  - `text_id`
  - `window_index`
  - `window_text`
  - `instance_text`
  - `type`
  - `confidence`
  - `explanation`
  - `context_dependent`
  - `start_char`
  - `end_char`
  - `alignment_status`
  - `support_count`
  - `supporting_window_indices`
- Windows CSV includes:
  - `window_index`
  - `window_text`
  - `has_figurative`
  - `confidence`
  - `instances_count`
  - `summary`

In the current CLI export, `supporting_window_indices` is serialized as a JSON array string inside the CSV field.

### Domain-mapping handoff today

The domain extractor actually relies on a narrower contract:

- required: figurative text (`text` / CSV `instance_text`), `type`
- optional but useful: `window_text`, `window_index`, `text_id`, `confidence`, `explanation`

This means `start_char` / `end_char` are not currently required for detect→map interoperability.

The current CLI also ensures representative `window_text` is available in instances output by requesting window metadata whenever instances are written, not only when a separate windows CSV is requested.

## Phase 2 target contract

### Canonical instance semantics

Each row in the primary detector instance output should mean:

> one canonical figurative instance supported by one or more window-level detections

It should **not** mean:

> one raw detector hit from one arbitrary overlapping window

### Required canonical fields

- `text_id`
- `instance_text`
- `type`
- `confidence`
- `explanation`
- `context_dependent`
- `window_index`
- `window_text`

### Optional provenance fields

- `supporting_window_indices`
- `support_count`
- `alignment_status`
- `start_char`
- `end_char`

Optional fields may be empty when the package cannot populate them deterministically.

## Provenance policy

### Provenance guarantees for every canonical instance

For every emitted canonical instance, the package must be able to identify:

1. the source text (`text_id` when available)
2. at least one supporting window (`window_index`)
3. the supporting window text (`window_text`) when window output is enabled or available
4. the extracted figurative phrase (`instance_text`)
5. the detector-emitted type/confidence/explanation that survived post-processing

### Provenance representation

- `window_index` and `window_text` represent the **representative** supporting window for the canonical row
- if multiple windows support the same canonical instance, retain the full support set in metadata and/or explicit support columns
- detector metadata should preserve raw window-level evidence separately from canonical row emission

## Overlap and deduplication policy

### Problem

Overlapping windows can produce duplicate detections for the same figurative expression.

### Policy

Primary outputs should be **deduplicated canonically** within a text.

Two detections are eligible for canonical merging when all of the following match:

1. same `text_id`
2. same normalized `instance_text`
3. same normalized `type`
4. windows overlap or are adjacent enough to plausibly refer to the same source occurrence

### Canonical row selection

When duplicates merge, choose the representative row by:

1. highest confidence
2. then the row with deterministic span alignment
3. then the earliest `window_index`

### Evidence preservation

Deduplication must not discard evidence. Preserve:

- all supporting `window_index` values
- support count
- optionally the list of raw window hits in metadata

## Span policy

### Rule

Offsets may only be emitted when derived by deterministic package-side alignment.

### Allowed alignment sources

1. exact match of `instance_text` within `window_text`
2. exact match after conservative normalization within `window_text`
3. exact match of `instance_text` within the original source text, if available in scope

### Disallowed alignment sources

- LLM-provided offsets
- fuzzy semantic guesses from the LLM
- package heuristics that choose among multiple matches without marking ambiguity

### Alignment algorithm requirements

1. try raw exact match first
2. if raw exact match fails, try conservative normalized-exact matching
3. if there is exactly one deterministic match in the selected text scope, populate offsets
4. if there are multiple deterministic matches, mark alignment as ambiguous and leave offsets empty
5. if there is no deterministic match, mark alignment as missing and leave offsets empty
6. approximate or fuzzy candidate detection may be recorded in metadata, but must not populate offsets

### Conservative normalization policy

Allowed normalization for alignment recovery is intentionally narrow and mechanical:

- case folding
- Unicode quote/apostrophe normalization
- dash/hyphen normalization
- collapsing repeated internal whitespace
- trimming surrounding punctuation that is clearly outside the phrase

This normalization exists to recover from small textual drift such as quote variants, apostrophes, spacing, or punctuation.

The package should not use broad fuzzy matching, semantic similarity, edit-distance ranking, or "best approximate substring" search to produce public offsets.

### Alignment status tiers

Every attempted alignment should fall into one of these statuses:

- `exact`
- `normalized_exact`
- `ambiguous`
- `missing`
- `approximate`

Status semantics:

- `exact`: raw substring match; offsets allowed
- `normalized_exact`: deterministic match after conservative normalization; offsets allowed only if mapping back to original `window_text` is unambiguous
- `ambiguous`: multiple deterministic candidates; no offsets
- `missing`: no deterministic candidate; no offsets
- `approximate`: a fuzzy/near candidate exists only for debugging/metadata; no offsets

### Row inclusion rule

Failure to align must **not** suppress the canonical instance row itself.

The detector should preserve the instance when figurative detection/extraction is otherwise valid, and treat alignment as a separate provenance concern.

### Preferred scope

Phase 2 should implement **window-local alignment first**.

That means:
- `start_char` / `end_char` should be interpreted as offsets within `window_text`
- global text offsets are out of scope unless a separate deterministic mapping layer is added later

## Output fidelity policy

### Instance CSV semantics

- `instance_text` = extracted figurative phrase text
- `window_text` = representative supporting window text
- `window_index` = representative supporting window index
- `support_count` = number of supporting windows merged into the canonical row
- `supporting_window_indices` = supporting window indices for the canonical row; in CLI CSV export this is a JSON array string
- `start_char` / `end_char` = offsets within `window_text`, only when deterministically aligned
- `alignment_status` may indicate `exact`, `normalized_exact`, `ambiguous`, `missing`, or `approximate`
- blank offset fields mean unavailable or ambiguous, not zero

### Windows CSV semantics

Window rows remain raw processing evidence and are **not** deduplicated.

### Summary CSV semantics

`instance_count` should reflect canonical deduplicated instances, not raw duplicate hits across overlapping windows.

## Proposed model and metadata changes

### `Instance` model

Retain current fields, but update semantics/documentation so:

- `start_char` / `end_char` are explicitly window-local and optional
- empty values mean unresolved
- future support metadata can be attached without breaking the public core fields

### `DetectionResult.metadata`

Add canonicalization metadata such as:

- `raw_instance_count`
- `canonical_instance_count`
- `deduplicated_instance_count`
- optional per-instance support summaries

## Detect → map compatibility requirement

The domain extractor must continue to work when given detector output containing only:

- `instance_text`
- `type`
- `window_text`
- `window_index`
- `text_id`
- `confidence`
- `explanation`

The domain-mapping path must not require offsets.

## File-level implementation targets

- `qualitative-analysis/src/qualitative_analysis/figurative/strategies/two_step.py`
  - canonicalize merged instances before final aggregation
- `qualitative-analysis/src/qualitative_analysis/figurative/models.py`
  - tighten span/provenance semantics in model docs
- `qualitative-analysis/src/qualitative_analysis/cli.py`
  - emit canonical semantics consistently, including provenance/support columns and blank offset behavior
- `qualitative-analysis/tests/test_figurative_detect_to_map.py`
  - add explicit detect→map contract coverage

## Acceptance criteria

1. Duplicate detections from overlapping windows are merged deterministically in primary instance output.
2. Summary `instance_count` reflects canonical rows.
3. Offsets are emitted only when exact or normalized-exact deterministic alignment succeeds.
4. Ambiguous, missing, or approximate-only alignments leave offsets blank.
5. No code path requests offsets from the LLM.
6. Domain mapping works with detector outputs that omit offsets.
7. Failure to align does not suppress an otherwise valid canonical instance row.

## Required tests

1. **Dedup across overlapping windows**
   - same figurative phrase in two overlapping windows produces one canonical instance row
2. **Evidence preservation**
   - canonical instance metadata records multiple supporting windows
3. **Unique exact match alignment**
   - exact one-time match in `window_text` yields window-local offsets
4. **Normalized-exact alignment**
   - small text drift recoverable via conservative normalization yields offsets
5. **Ambiguous exact match handling**
   - repeated phrase in a window leaves offsets blank
6. **Approximate-only handling**
   - a near match may be classified in metadata but must not yield offsets
7. **No-match handling**
   - extracted phrase absent from `window_text` leaves offsets blank
8. **Row preservation on alignment failure**
   - an otherwise valid instance remains in output even when offsets are blank
9. **Detect→map contract**
   - detector-style instance rows feed domain extraction successfully without offsets

## Implementation order

1. define canonical row semantics in tests
2. implement deterministic deduplication
3. implement window-local deterministic alignment with conservative normalization
4. lock detect→map compatibility with an integration-style test
5. update docs/help text to match the new semantics