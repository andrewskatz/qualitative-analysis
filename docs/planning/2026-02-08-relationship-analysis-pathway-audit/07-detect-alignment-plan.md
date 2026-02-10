# Plan: Align Relationship Detect with Entity Detect Pathway

## Context

The entity detect pathway (`qa entity detect`) has been more battle-tested and follows cleaner patterns than the relationship detect pathway. This plan aligns the relationship detect pathway with the entity pathway's architecture, CLI UX, and prompt usage. The user also wants entity extraction in the relationship pipeline to use the same prompt as the entity detect pathway.

---

## Changes Overview

### A. Prompt Alignment — Use Entity Pathway's Entity Extraction Prompt

**Problem**: Relationship detector uses `relationships/prompts/entity_extraction_v3.txt` (simple, flat text) while entity detector uses `entity/prompts/entity_extraction_v1.txt` (XML-structured, example-driven, granularity guidelines). Both expect the same JSON schema (`{"entities_and_concepts": [...]}`), so they're compatible.

**Plan**:
1. Copy `entity/prompts/entity_extraction_v1.txt` → `relationships/prompts/entity_extraction_v1.txt`
   - Avoids circular imports (entity pathway already imports from relationships pathway)
   - Prompts can evolve independently if needed
2. Change default `entity_prompt_version` from `3` to `1`:
   - In `relationships/detector.py` constructor: `entity_prompt_version: int = 1`
   - In `relationships/components/entity_extractor.py` constructor: `prompt_version: int = 1`
   - In `relationships_cli.py` `add_relationships_detect_args`: `--entity-prompt-version` default → `1`
   - Keep v3 available for backward compat (just change the default)

**Files**:
- NEW: `qualitative-analysis/src/qualitative_analysis/relationships/prompts/entity_extraction_v1.txt`
- EDIT: `qualitative-analysis/src/qualitative_analysis/relationships/detector.py:35` (default param)
- EDIT: `qualitative-analysis/src/qualitative_analysis/relationships/components/entity_extractor.py:24` (default param)
- EDIT: `qualitative-analysis/src/qualitative_analysis/relationships_cli.py:166` (CLI default)
- EDIT: `qualitative-analysis/src/qualitative_analysis/relationships/prompts/loader.py` (DEFAULT_VERSIONS dict: entity_extraction → 1)

---

### B. Detector Class: DI-Only Constructor

**Problem**: `RelationshipDetector.__init__` has a hybrid DI pattern — it can build its own `OllamaProvider` from `model_name`/`provider`/`provider_config`, coupling it to a specific provider. Entity detector requires `llm_provider: BaseLLMProvider` (clean DI).

**Plan**: Make `llm_provider` required, remove internal provider construction:
1. Remove `model_name`, `provider`, `provider_config` params from constructor
2. Make `llm_provider: BaseLLMProvider` a required positional parameter
3. Remove the `if llm_provider:` / `else: OllamaProvider(...)` block
4. Update tests (already use `llm_provider=mock_llm`, just remove the extra params)

**Files**:
- EDIT: `qualitative-analysis/src/qualitative_analysis/relationships/detector.py:29-65`
- EDIT: `qualitative-analysis/tests/test_relationship_detector.py` (remove `model_name`/`provider` from constructors)

---

### C. Detector Class: Per-Window Error Handling

**Problem**: If any window extraction fails, the entire `detect()` call propagates the exception and all results for that text are lost. Entity detector wraps each window in try/except.

**Plan**: Wrap the extraction block inside the window loop (lines ~138-155 in detector.py) in try/except:
```python
for i, window in enumerate(windows):
    try:
        # ... existing extraction logic ...
    except Exception as e:
        logger.error(f"Extraction failed for window {i} (text_id={text_id}): {e}")
        continue
```

**File**: `qualitative-analysis/src/qualitative_analysis/relationships/detector.py:119-175`

---

### D. Detector Class: Empty Text Guard

**Problem**: No explicit early return for empty/whitespace text.

**Plan**: Add guard at top of `detect()`:
```python
if not text or not text.strip():
    logger.warning(f"Empty text for text_id={text_id}, returning empty result")
    return RelationshipResult(relationships=[], entities=[], metadata={"window_count": 0, ...})
```

**File**: `qualitative-analysis/src/qualitative_analysis/relationships/detector.py` (top of `detect()`)

---

### E. CLI: Use Shared Utilities from `core/cli_utils.py`

**Problem**: `add_relationships_detect_args` duplicates windowing args, LLM args, and CSV args that exist as shared utilities in `core/cli_utils.py`.

**Plan**: Replace inline arg definitions with shared utilities:
1. Use `add_common_windowing_args(parser)` — replaces `--window-size`, `--stride`, `--chunk-unit`, `--tokenizer`, `--no-windowing`
2. Use `add_common_llm_args(parser)` — replaces `--model`, `--base-url`, `--timeout`, `--log-llm`
3. Use `add_common_csv_args(parser)` — replaces `input_csv`, `--text-col`, `--id-col`
4. Keep relationship-specific args inline: `--entities-col`, `--output`, `--no-windows`, `--strategy`, summary args, prompt version args, `--coref`, `--context-buffer-size`

**Note**: `add_common_llm_args` uses `--model` default `"qwen3:30b-a3b-instruct-2507-q4_K_M"` which matches the current relationship CLI default. The entity CLI uses a different default (`"gpt-oss:120b"`) — this is fine, each pathway has its own preferred model.

**File**: `qualitative-analysis/src/qualitative_analysis/relationships_cli.py:66-188`

---

### F. CLI: Add Missing Features

#### F1. `--group-col` support
- Add `--group-col` argument
- Build `text_id_to_group` mapping in `run_relationships_detect`
- Propagate group to edges CSV and summary CSV as additional column

#### F2. `--limit N` flag
- Add `--limit` argument (type=int, default=None)
- Slice `rows = rows[:args.limit]` after reading CSV (requires reading all rows first — see F5)

#### F3. `--temperature` flag
- Add `--temperature` argument (type=float, default=0.1)
- Pass to provider construction in CLI

#### F4. `--provider` choices
- Add `--provider` argument with choices `["ollama"]` (keep honest about what's supported, unlike entity pathway which advertises unsupported providers)
- Build provider in CLI: construct `OllamaProvider(model_name=args.model, base_url=args.base_url, temperature=args.temperature, ...)` instead of passing strings to detector

#### F5. Progress format: `(N/M)` with total
- Read all CSV rows into a list first (like entity CLI does) so `len(rows)` is known
- Change progress output to: `Processing {text_id} ({i+1}/{len(rows)})... found X relationships`

#### F6. `detect_metadata.json`
- Use `write_run_metadata()` from `core/cli_utils` OR write manually (entity CLI writes its own format)
- Include: input_file, model, provider, prompt_version, entity_prompt_version, strategy, windowing params, total_texts, total_relationships, unique_entities, timestamp, package_version

#### F7. Final summary banner
```
==================================================
Relationship Detection Complete
==================================================
Texts processed:           X
Total relationships:       Y
Unique entities:           Z

Output files:
  <path1>
  <path2>
  ...
```

**File**: `qualitative-analysis/src/qualitative_analysis/relationships_cli.py` (detect args + run function)

---

### G. Update Tests

- Update `test_relationship_detector.py`: remove `model_name="test", provider="ollama"` from detector constructors (now DI-only, just pass `llm_provider=mock_llm`)
- Verify all 14 existing tests still pass
- Run full test suite (129 tests across 4 files)

**Files**:
- EDIT: `qualitative-analysis/tests/test_relationship_detector.py`

---

## Files Summary

| File | Action |
|------|--------|
| `relationships/prompts/entity_extraction_v1.txt` | CREATE (copy from entity pathway) |
| `relationships/prompts/loader.py` | EDIT (default version 3→1) |
| `relationships/detector.py` | EDIT (DI-only, per-window error handling, empty text guard) |
| `relationships/components/entity_extractor.py` | EDIT (default prompt version 3→1) |
| `relationships_cli.py` | EDIT (shared utils, new flags, provider in CLI, metadata, progress, summary) |
| `tests/test_relationship_detector.py` | EDIT (remove old constructor params) |

---

## Verification

1. Run all relationship tests: `cd qualitative-analysis && python -m pytest tests/test_relationship_detector.py tests/test_relationship_normalizer.py tests/test_relationship_models.py tests/test_relationship_pipeline.py -v`
2. Verify `qa relationships detect --help` shows new args (`--group-col`, `--limit`, `--temperature`, `--provider`)
3. Verify no circular imports: `python -c "from qualitative_analysis.relationships.detector import RelationshipDetector"`
