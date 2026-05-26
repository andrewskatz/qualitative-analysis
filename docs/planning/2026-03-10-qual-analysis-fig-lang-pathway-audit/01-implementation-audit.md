# Detailed Implementation Audit

**Audited package surface:**
- `qualitative-analysis/src/qualitative_analysis/figurative/detector.py`
- `qualitative-analysis/src/qualitative_analysis/figurative/strategies/two_step.py`
- `qualitative-analysis/src/qualitative_analysis/figurative/components/*`
- `qualitative-analysis/src/qualitative_analysis/figurative/prompts/*`
- `qualitative-analysis/src/qualitative_analysis/cli.py`
- `qualitative-analysis/src/qualitative_analysis/figurative/domains/*`
- `qualitative-analysis/src/qualitative_analysis/figurative/domains_cli.py`
- `qualitative-analysis/src/qualitative_analysis/unified_cli.py`
- package tests under `qualitative-analysis/tests/`

---

## 1. High-Level Pathway Shape

### Observed package design

The figurative-language pathway inside the package is split into two layers:

1. **Detection layer**
   - `FigurativeDetector`
   - `TwoStepWithSummariesStrategy`
   - summarizer / scanner / summary buffer
   - sliding-window text processing

2. **Post-detection analysis layer**
   - `DomainExtractor`
   - `DomainNormalizer`
   - `DomainGraph`
   - CLI commands `map`, `normalize`, `graph`, `pipeline`

### Audit judgment

This is a coherent package architecture. The main issue is not missing structure, but that some command naming and contracts make the boundary between “detection” and “analysis after detection” slightly ambiguous.

---

## 2. Figurative Detection Entry Point (`figurative/detector.py`)

### What works well
- Exposes a clean public detector entry point.
- Wires text chunking and strategy selection in one place.
- Allows injected `BaseLLMProvider` instances for testing.
- Supports window size, stride, chunk unit, tokenizer, threshold, and type filtering.

### Findings
- Direct provider selection only supports `provider="ollama"` unless a provider object is injected.
- The package core defines `BaseLLMProvider`, `OllamaProvider`, and `MLXProvider`, but the figurative detector constructor does not expose that breadth directly.
- `detect()` is intentionally thin and delegates all substantive work to the strategy.

### Audit judgment

The public entry point is appropriately minimal. The main design gap is provider exposure, not control flow.

---

## 3. Two-Step Detection Strategy (`strategies/two_step.py`)

### What works well
- Windowing, summarization, binary detection, and extraction are clearly separated.
- Empty/whitespace-safe behavior exists through the text processor and empty-window handling.
- Optional `return_windows` metadata is useful for debugging and export.

### Findings
- The algorithm proceeds window-by-window as:
  1. summarize current window using prior summaries
  2. add summary to buffer
  3. run binary detection on the current window using formatted context
  4. if above threshold, extract instances
- Because the summary is added **before** detection/extraction, the formatted “prior context” includes a model-generated summary of the current window itself.
- Aggregate detection confidence is based on extracted-instance confidences, not the binary detector confidence.
- Instances from overlapping windows are appended directly with no deduplication pass.

### Audit judgment

The orchestration is understandable and operational, but the context semantics are slightly muddied and overlap effects are left unresolved.

---

## 4. Summarizer / Scanner / Buffer Components

### Summarizer (`components/summarizer.py`)

#### What works well
- Uses package-local prompt loading.
- Accepts prior summaries and formats them consistently.
- Parses either `summary_points` or `summary` from JSON.

#### Findings
- Always uses `system_prompt` version 1 even when the summarization prompt version changes.
- Calls `llm.generate(...)` rather than a structured-output helper.
- On JSON parse failure, it logs a warning and falls back to raw model text.

### Scanner (`components/scanner.py`)

#### What works well
- Uses a small Pydantic schema for extraction item validation.
- Supports optional figurative-type filtering.
- Switches to prompt version 2 automatically when type filtering is enabled.

#### Findings
- Binary detection parses raw JSON with `json.loads(...)`, then falls back to a heuristic: response contains `yes` or `true`.
- Confidence is clamped into `[0, 1]`, which is helpful, but only after successful parse.
- Extraction expects `{"instances": [...]}` and silently drops invalid items.
- Extraction returns `[]` on JSON parse failures or other exceptions.
- The scanner never uses `generate_json(...)` even though the provider layer supports it.
- Extracted `Instance` objects do not populate `start_char` or `end_char`.

### Buffer (`components/buffer.py`)

#### Findings
- The FIFO summary buffer is simple and predictable.
- `get_formatted_context()` returns either `PRIOR CONTEXT:\n...` or `No prior context.`

### Audit judgment

These components are modular and easy to reason about, but they rely on permissive best-effort parsing where stronger structured-output enforcement already exists elsewhere in the package.

---

## 5. Text Processing and Output Models

### Text chunking (`core/text.py`)

### What works well
- Supports sentence-based and token-based chunking.
- Handles empty input safely.
- Ensures a trailing window is produced when the final span would otherwise be missed.

### Findings
- Sentence splitting is naive regex-based splitting.
- If the text is shorter than the configured window, the processor returns the original text as a single window.
- Overlap behavior is intentional and useful for recall, but there is no downstream consolidation of overlap duplicates.

### Output/data models (`figurative/models.py`)

### Findings
- `Instance` includes `text`, `type`, `confidence`, `explanation`, `context_dependent`, `window_index`, `start_char`, and `end_char`.
- Current extraction flow only populates the semantic fields plus `window_index`.
- `DetectionResult` and `DetectionCheckpoint` give the CLI a stable package-side contract.

### Audit judgment

The model layer is reasonably shaped, but the exported instance schema promises finer provenance than the current implementation actually supplies.

---

## 6. Prompt Loading, Prompt Versions, and Type Filtering

### What works well
- Prompt files are versioned and loaded by name.
- Type filtering is implemented through normalized type validation utilities.
- Prompt version 2 is used automatically for binary detection and extraction when filtering is active.

### Findings
- Supported normalized types are: `metaphor`, `simile`, `personification`, `hyperbole`, `idiom`, `irony`, `extended_metaphor`, `analogy`, `other`.
- Unknown figurative types are warned on and normalized out.
- The prompt system strongly instructs JSON-only responses, but enforcement remains caller-side.

### Audit judgment

Prompt versioning and type filtering are genuine features, not stubs. The weakness is not prompt design but insufficiently strict response handling around those prompts.

---

## 7. Detection CLI Data Flow (`cli.py`)

### What works well
- `run_figurative_detect(...)` handles CSV input validation, output selection, checkpoint/resume, and run metadata.
- Output types are clearly separated into summary, instances, and windows CSVs.
- Errors per text are caught and surfaced in summary output rather than aborting the whole run.

### Findings
- The detector is always constructed with `provider="ollama"` in the CLI.
- Detection output uses `instance_text` in the instances CSV, which correctly matches the downstream mapper’s default `text_col`.
- `window_text` in the instances CSV is only available if windows metadata was collected.
- This means the detect→map bridge is implemented, but not every run produces equally rich context columns.

### Audit judgment

The CLI detection flow is operationally strong. The main issues are contract clarity and the amount of provenance retained in outputs.

---

## 8. Downstream Domain Mapping (`domains/extractor.py`)

### What works well
- `DomainExtractor.extract_from_csv(...)` is the concrete bridge from detection output into domain analysis.
- The default `text_col="instance_text"` aligns with the detection instances CSV.
- Response parsing handles markdown fences, embedded JSON, and multilevel domain outputs.

### Findings
- Provider wiring mirrors the detector: Ollama directly, otherwise dependency injection.
- Default prompt mode is multilevel extraction.
- If parsing succeeds and domains are dicts, the extractor stores full levels and uses `moderate` as the primary backwards-compatible domain.
- Error handling during batch extraction produces partial results with embedded error info rather than halting the run.

### Audit judgment

The mapping layer is more robust than the detector scanner in some ways, especially around response cleanup. It still relies on permissive parsing rather than true schema-enforced generation.

---

## 9. Domain Normalization (`domains/normalizer.py`)

### What works well
- Uses sentence embeddings plus agglomerative clustering in a straightforward way.
- Supports different conservativeness levels and a custom threshold.
- Supports `cluster_mode="separate"` or `"together"` and representative vs. LLM canonical labels.
- Lazy-loads the embedding model and includes checkpointing for LLM canonicalization.

### Findings
- Representative canonical labels are chosen by the most central member and uppercased.
- LLM canonical labels are generated with another JSON-only prompt, parsed permissively.
- `normalize()` uses `asyncio.get_event_loop().run_until_complete(...)` when `canonical_method="llm"`.
- That synchronous wrapper is convenient for CLI use but risky when called from already-running event-loop environments.

### Audit judgment

Normalization is featureful and practical, but the async/sync boundary is the most important implementation risk in this layer.

---

## 10. Graph Generation (`domains/graph.py`)

### What works well
- Builds source→target domain graphs from raw or normalized instances.
- Supports JSON, CSV, and PNG exports.
- Supports abstraction-level selection and optional clustered PNG rendering.

### Findings
- Graph generation itself appears structurally sound from the reviewed code and tests.
- Visualization paths introduce optional dependencies (`networkx`, `matplotlib`) and additional complexity, but those are appropriately kept downstream of the main data model.

### Audit judgment

Graphing is a mature downstream feature and not a primary risk area compared with detection and normalization boundaries.

---

## 11. Unified CLI and Pipeline Semantics

### What works well
- `unified_cli.py` clearly exposes `figurative detect|map|normalize|graph|pipeline`.
- The command family makes the package-side figurative feature set discoverable.

### Findings
- The `pipeline` command is described as `detect → map → normalize → graph`, but the implementation in `domains_cli.py` starts from an input CSV of figurative instances and then runs map/normalize/graph.
- In practice, this is a **post-detection pipeline**, not a raw-text full pipeline.

### Audit judgment

This is mostly a naming/documentation mismatch, but it is important because it affects how users interpret package capabilities.

---

## 12. Test Coverage Assessment

### Confirmed coverage
- `tests/test_mock_pipeline.py` exercises a narrow happy path with a mock async LLM.
- `tests/test_live_ollama_pipeline.py` provides an optional live integration check.
- `tests/test_unified_cli.py` verifies CLI help and command registration.
- `tests/test_domains.py` covers data models, domain-response parsing, and graph generation.

### Confirmed gaps
- No focused tests were found for scanner heuristic fallback behavior.
- No focused tests were found for extraction parse failures or silent item dropping.
- No focused tests were found for overlap deduplication behavior because the implementation currently does not deduplicate.
- No end-to-end detect→map compatibility test was found.
- No focused test was found for `canonical_method="llm"` normalization behavior in an event-loop-sensitive environment.

### Audit judgment

The figurative package is implemented and partially tested, but the highest-risk package behaviors are under-tested.