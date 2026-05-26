# Prioritized Recommendations

---

## Priority 1: Harden structured-output handling in detection

1. **Use provider-level structured output where possible**
   - Prefer `generate_json(...)` in the scanner and possibly the summarizer.
   - Reuse Pydantic models or explicit schemas for detector and extractor outputs.

2. **Make parse failure observable**
   - Distinguish `no figurative language found` from `LLM output could not be parsed`.
   - Surface parse-failure counts in detection metadata and/or CSV outputs.

3. **Replace the current heuristic fallback for binary detection**
   - The current `"yes" in response.lower() or "true" in response.lower()` rule is too permissive.
   - If a fallback must exist, it should be deliberately conservative and explicitly marked in metadata.

**Why first:** this is the single biggest reliability issue in the package-side detection path.

---

## Priority 2: Improve detection-result fidelity

1. **Decide on an overlap policy**
   - Either deduplicate extracted instances across overlapping windows or document that downstream consumers must do so.

2. **Align the instance schema with actual outputs**
   - Either populate `start_char` / `end_char` or remove/de-emphasize them from the public output contract.

3. **Clarify context semantics**
   - If “prior context” should mean previous windows only, do not add the current summary before detection/extraction.

4. **Make `window_text` behavior explicit**
   - Either always include it in instances output or document that it depends on windows-mode output.

**Why second:** these are correctness and provenance issues that directly affect downstream research use.

---

## Priority 3: Clarify package contracts around providers and CLI semantics

1. **Make provider support explicit**
   - Either expose non-Ollama providers directly in the detector/domain extractor constructors and CLI, or document that Ollama is the only first-class provider path today.

2. **Fix or clarify `qa figurative pipeline` semantics**
   - Either rename/reword it as a post-detection pipeline or wire in raw-text detection as the first stage.

3. **Document the detect→map bridge clearly**
   - Call out that detection emits `instance_text`, which matches mapper defaults.
   - Call out the conditional nature of `window_text` population.

**Why third:** the implementation mostly works, but some of its public-facing contracts are currently more implied than explicit.

---

## Priority 4: Remove the event-loop hazard in normalization

Recommended changes:
- avoid `run_until_complete(...)` inside synchronous `normalize()`
- provide either:
  - a fully async normalization path for LLM canonicalization, or
  - a separate synchronous wrapper that is clearly documented for CLI-only use
- add a regression test that exercises the LLM canonical-label path from a context where an event loop is already running

**Why fourth:** this is the clearest runtime integration risk in the downstream analysis layer.

---

## Priority 5: Add focused package regression tests

Suggested additions:
- `tests/test_figurative_scanner.py`
- `tests/test_figurative_detect_to_map.py`
- `tests/test_domain_normalizer_llm.py`

Suggested cases:
- binary detection parses valid JSON
- binary detection parse failure does not silently become an unsafe false positive
- extraction parse failure is surfaced in metadata
- type filtering accepts normalized supported labels and rejects unsupported ones
- overlapping-window output behavior is explicitly asserted
- detect instances CSV feeds `DomainExtractor.extract_from_csv(...)` without column mismatch
- normalization LLM path works or fails predictably under an active event loop

**Why fifth:** the current tests prove the pathway exists, but not that its fragile edges are safe.

---

## Priority 6: Consolidate response-cleanup logic

Recommended changes:
- share one JSON-cleanup / fence-stripping / embedded-object extraction helper across figurative detection and domain mapping
- use common logging conventions for parse warnings
- standardize whether malformed items are skipped, downgraded, or surfaced explicitly

**Why sixth:** parsing behavior is implemented multiple ways across the package, which encourages drift.

---

## Suggested Minimal Test Plan

A good near-term package-only hardening pass would include:

1. **Scanner tests**
   - valid detection JSON
   - malformed detection JSON
   - valid extraction JSON with one invalid item
   - type-filtered extraction

2. **Interoperability tests**
   - `qa figurative detect` instances output schema matches mapper expectations
   - `window_text` presence/absence is intentional and documented

3. **Normalization tests**
   - representative canonical labeling on a tiny synthetic cluster set
   - LLM canonical labeling path with checkpoint save/load
   - active-event-loop safety behavior

---

## Overall Assessment

The package-side figurative identification and analysis pathway is **implemented and operational**. The main need is **hardening and contract clarification**, not wholesale redesign.

If follow-up work is sequenced, the most sensible order is:
1. harden detector/scanner structured-output handling
2. improve result fidelity and overlap provenance
3. clarify provider and CLI contracts
4. fix normalization async/sync boundary risks
5. add focused regression coverage around the fragile edges