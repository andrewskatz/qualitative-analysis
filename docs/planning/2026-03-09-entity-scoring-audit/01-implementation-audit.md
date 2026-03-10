# Detailed Implementation Audit

**Audited package surface:**
- `src/qualitative_analysis/entity/detector.py`
- `src/qualitative_analysis/entity/scorer.py`
- `src/qualitative_analysis/entity/models.py`
- `src/qualitative_analysis/entity/prompts/*`
- `src/qualitative_analysis/entity_cli.py`
- package tests under `qualitative-analysis/tests/`

---

## 1. Entity Identification (`entity/detector.py`)

### What works well
- Uses package-local prompts via `load_entity_prompt(...)` rather than relationship prompts.
- Uses `SlidingWindowProcessor`, so extraction can operate over sentence or token windows.
- Returns both `entities` and `entity_contexts`, which is the right shape for downstream scoring preparation.
- Handles empty input safely and records simple detection metadata.

### Findings
- **Coupling to relationships schema:** the detector imports `EntityResponse` from `qualitative_analysis.relationships.components.entity_extractor`. That means the package-side detector is not actually schema-independent even though its docstring describes it as standalone.
- **Duplicated JSON extraction logic:** `_extract_json_payload()` in `entity/detector.py` is effectively duplicated in `relationships/components/entity_extractor.py`. Shared behavior is implemented twice, not centralized once.
- **Case-sensitive deduplication:** `if entity not in all_entities` treats `Climate Change` and `climate change` as different entities.
- **Failure handling favors continuity over observability:** parse/extraction errors become empty results for that window. This is operationally resilient, but it also makes true “no entities found” indistinguishable from “LLM/parsing failed.”
- **Context capture is crude but predictable:** context is just the window prefix truncated to `max_context_length`, with `...` appended. This preserves provenance cheaply, but can cut text mid-word or mid-sentence.

### Audit judgment
The detector is serviceable and well tested, but there is architectural drift between “standalone entity pathway” and the actual implementation dependency on relationship-side schema code.

---

## 2. SETS Scoring Orchestration (`entity/scorer.py`)

### What works well
- `score_entity()` performs repeated scoring passes and preserves partial success when some runs fail.
- `score_entities()` carries `text_id`, `window_index`, and `group` through the scoring path.
- `_build_prompt()` correctly injects dimensions, scale descriptions, research context, and scale-aware examples.
- `_aggregate_runs()` uses `DimensionScore.from_scores(...)`, so the main scorer delegates statistics rather than re-implementing them.

### Findings
- **Parser strict/loose mismatch:** `_parse_response()` skips unexpected dimensions and nonnumeric scores with warnings, but then raises only if an expected dimension is missing. This creates a debugging gap between raw LLM output and the final exception.
- **Float truncation:** valid numeric scores are coerced with `int(score)`. A model output of `74.9` becomes `74` without an explicit design note that truncation is intended.
- **Intentional prompt-contract variation:** prompt `v4` deliberately requests only dimension scores, while `v2`/`v3` request richer reasoning fields. The implementation supports all three, but the chosen prompt version materially changes what interpretive information survives downstream.
- **Justification aggregation is lossy:** the final justification per dimension is selected from the run whose score is closest to the mean. That is simple and defensible, but it discards cross-run reasoning diversity.
- **Load/save fidelity gap:** `load()` reconstructs aggregated `DimensionScore` objects from flattened score columns, but does not restore full `SingleRunScore` objects.
- **Metadata loss on load:** `score_entity()` returns `window_index` and `group`, but `EntityScorer.load()` does not restore them.
- **Mode reconstruction bug:** `load()` expects `{dim}_mode`, but `EntityScore.to_dict()` never writes it. Reloaded objects therefore default `mode` to `0`.

### Audit judgment
The scorer is the package’s most important implementation surface and is conceptually solid, but persistence fidelity and parser behavior need hardening before it can be considered robust.

---

## 3. Score Models and Statistics (`entity/models.py`)

### What works well
- `DimensionSet.sets_framework()` defines a clear package-native SETS default.
- `ScaleConfig` supports reusable scale propagation and metadata loading.
- `DimensionScore.from_scores(...)` computes mean, median, mode, standard deviation, CI, and CV in one place.
- `EntityScore.to_dict()` flattens run-level outputs for CSV compatibility, which matches downstream package expectations.

### Findings
- **Serialization inconsistency:** `DimensionScore.mode` exists in memory and in `DimensionScore.to_dict()`, but not in flattened `EntityScore.to_dict()` output.
- **Dataset statistics assume consistency from the first score:** `EntityScoreResult.compute_statistics()` chooses dimensions from the first scored entity and summarizes those. If a malformed or partially reconstructed score is first, statistics can become incomplete.
- **Round-trip design is aggregation-first, not full-fidelity:** this appears intentional for flat outputs, but it means persisted JSON is not equivalent to an in-memory scorer result that still has per-run objects.

### Audit judgment
The model layer contains good abstractions, but the flattened output contract and the loader are not fully aligned.

---

## 4. CLI Data Flow and Checkpointing (`entity_cli.py`)

### What works well
- `run_entity_score()` cleanly handles dimensions, research context, provider construction, scoring, outputs, metadata, and checkpoint cleanup.
- `_read_entities_csv()` preserves optional `text_id`, `window_index`, and `group` columns when present.
- `_load_checkpoint()` reconstructs dimension scores from run columns and restores `window_index` and `group` for checkpoint-resume behavior.
- `run_entity_prepare_scoring()` provides a practical package-side bridge from relationship windows into entity scoring input.

### Findings
- **Scale helper drift:** `_resolve_scale()` exists near the top of the file and can infer scale from `score_metadata.json`, but `run_entity_score()` does not appear to use it. In practice, scoring uses the dimensions source plus explicit CLI overrides.
- **Checkpoint path is more faithful than JSON load path:** `_load_checkpoint()` restores `window_index` and `group`, while `EntityScorer.load()` does not. The package has two reconstruction paths with different fidelity.
- **CSV fieldnames come from the first score:** `_write_scores_csv()` derives columns from `result.scores[0].to_flat_dict()`. This is safe when dimensions are uniform, but fragile if score shapes diverge.
- **Input rows are not deduplicated in scoring:** repeated entity/context rows are all scored, which may be desired, but it means deduplication is intentionally upstream, not enforced here.

### Audit judgment
CLI orchestration is practical and easier to follow than the lower-level scorer internals, but it exposes minor contract mismatches between scoring outputs, checkpoint state, and persisted JSON results.

---

## 5. Prompt Loading and Prompt Versions (`entity/prompts/*`)

### Findings
- `loader.py` defaults entity scoring to version `2`, while the CLI also defaults `--prompt-version` to `v2`; those defaults are aligned.
- `v2` asks for `initial_observations`, scores, and justifications.
- `v3` keeps the same information shape but explicitly asks the model to justify before scoring.
- `v4` intentionally strips the output down to dimension names and numeric scores only.
- `scoring_system_prompt_v1.txt` is intentionally minimal; that simplicity is workable, but it places nearly all reliability pressure on the main prompt template.

### Audit judgment
Prompt versioning is a real package feature, not a placeholder, and the differences between versions appear intentional. The main implementation risk is therefore not accidental mismatch, but insufficient documentation/tests around those intentionally different downstream contracts.

---

## 6. Test Coverage Assessment

### Confirmed coverage
- `tests/test_entity_detector.py` directly covers detector initialization, windowing, empty input, deduplication, truncation, metadata, parse failures, token-based chunking, and prompt usage.
- `tests/test_bayesian.py` indirectly covers score serialization assumptions, run-column persistence, JSON save/load, and custom-scale behavior.

### Confirmed gaps
No direct package tests were found for:
- `EntityScorer._parse_response(...)`
- `EntityScorer._aggregate_runs(...)`
- `EntityScorer.score_entity(...)`
- `EntityScorer.score_entities(...)`
- scoring prompt-version differences (`v2` vs `v3` vs `v4`)
- CLI checkpoint/resume behavior for `qa entity score`

### Audit judgment
The scoring pathway depends heavily on conventions that are only partially locked down by tests. That is the biggest package-only risk surfaced by this audit.

