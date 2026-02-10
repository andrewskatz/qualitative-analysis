# Entity Detect Command Implementation

**Date:** 2026-02-06
**Package:** `qualitative-analysis`
**Component:** `qa entity detect` — Standalone Entity Extraction
**Planning Docs:** [docs/planning/2026-02-05-entity-detect-pathway/00-entity-detect-plan.md](../planning/2026-02-05-entity-detect-pathway/00-entity-detect-plan.md)

---

## Summary

Implemented a new `qa entity detect` command that extracts entities from raw text, filling a significant UX gap in the entity analysis pipeline. Previously, users wanting to analyze entities had to either:

1. Run the full relationships pipeline (`qa relationships detect`) which extracts entities AND relationships — overkill for entity-only analysis
2. Manually extract entities before scoring with `qa entity score`

The new command provides a direct **text → entities** pathway, outputting CSV ready for `qa entity score`.

**Test result:** 160/160 tests pass (14 new tests for entity detection).

---

## Problem Statement

The entity analysis pipeline had a missing link:

```
BEFORE:
Interview Transcripts (.txt/.csv)
        │
        ├──── [qa relationships detect]  ← extracts entities + relationships
        │     Overkill if you only want entities
        │
        └──── [Manual extraction]        ← tedious, error-prone
                │
                ▼
        [qa entity score]                ← expects entities already extracted
```

Users repeatedly hit this friction point when trying to run the entity scoring pathway without needing relationship analysis.

---

## Solution: `qa entity detect`

```
AFTER:
Interview Transcripts (.txt/.csv)
        │
        ├──── [qa relationships detect]  ← entities + relationships
        │
        └──── [qa entity detect]         ← NEW: entities only
                │
                ▼
        [qa entity score]                ← seamless handoff
```

### CLI Interface

```bash
qa entity detect input.csv \
  --text-col response_text \
  --id-col interview_id \
  --output-dir output/entities \
  --model gpt-oss:120b \
  --log-llm
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `input_csv` | (required) | Path to input CSV with text to analyze |
| `--text-col` | `text` | Column name for text content |
| `--id-col` | `None` | Column for participant/document IDs (auto-generates if missing) |
| `--output-dir` | creates timestamped dir | Output directory |
| `--model` | `gpt-oss:120b` | LLM model name |
| `--provider` | `ollama` | LLM provider (ollama, openai, anthropic) |
| `--base-url` | `http://localhost:11434` | Ollama base URL |
| `--temperature` | `0.1` | LLM temperature (low for consistency) |
| `--window-size` | `3` | Sentences per window |
| `--stride` | `2` | Window stride |
| `--no-windowing` | `False` | Process entire text as single unit |
| `--prompt-version` | `3` | Entity extraction prompt version |
| `--log-llm` | `False` | Print prompts and responses |
| `--limit` | `None` | Limit rows processed (for testing) |

### Output Files

**Primary output: `entities.csv`**
```csv
text_id,entity,context,window_index
interview_001,renewable energy,"...exploring renewable heating options...",0
interview_001,community support,"...need community support programs...",1
```

**Summary output: `entities_summary.csv`**
```csv
text_id,entity_count,entities_json
interview_001,15,"[""renewable energy"",""community support"",...]"
```

**Metadata: `detect_metadata.json`**
```json
{
  "input_file": "abc-sim.csv",
  "model": "gpt-oss:120b",
  "total_texts": 15,
  "total_entity_occurrences": 127,
  "unique_entities": 84,
  "timestamp": "2026-02-05T12:00:00"
}
```

---

## Implementation Details

### New File: `entity/detector.py` (~214 lines)

Core detection module providing:

- **`EntityDetectionResult`** — Dataclass holding:
  - `text_id`: Document/participant identifier
  - `entities`: List of unique entities (deduplicated within text)
  - `entity_contexts`: List of `{entity, context, window_index}` dicts
  - `window_count`: Number of windows processed
  - `metadata`: Processing configuration
  - `to_dict()`: JSON-serializable conversion

- **`EntityDetector`** — Main detection class:
  - Reuses `EntityExtractor` from relationships module (avoids duplication)
  - Uses `SlidingWindowProcessor` for text chunking
  - Supports windowed or full-text modes
  - `detect()`: Single text extraction
  - `detect_batch()`: Multiple texts with auto-ID generation

**Key design decision:** Reuse existing `EntityExtractor` from `relationships/components/entity_extractor.py` rather than duplicating the LLM prompting logic. This ensures both pathways use identical entity extraction.

### Modified: `entity_cli.py` (~200 lines added)

Added at end of file:

- **`add_entity_detect_args()`** — Argument definitions matching existing CLI patterns
- **`run_entity_detect()`** — Async execution function:
  - Reads input CSV
  - Initializes LLM provider (supports ollama/openai/anthropic)
  - Creates `EntityDetector` with configured windowing
  - Iterates over rows with progress output
  - Writes `entities.csv`, `entities_summary.csv`, `detect_metadata.json`
  - Prints completion summary

### Modified: `unified_cli.py`

Added subparser registration in `_add_entity_commands()`:

```python
# detect subcommand
detect_parser = entity_subs.add_parser(
    "detect",
    help="Detect entities in text",
    description="Extract entities from text using LLM analysis. Outputs CSV ready for 'qa entity score'.",
)
add_entity_detect_args(detect_parser)
detect_parser.set_defaults(func=lambda args: asyncio.run(run_entity_detect(args)))
```

### Modified: `entity/__init__.py`

Added exports:

```python
# Entity detection
from qualitative_analysis.entity.detector import (
    EntityDetector,
    EntityDetectionResult,
)

__all__ = [
    # ... existing exports ...
    # Detection
    "EntityDetector",
    "EntityDetectionResult",
]
```

---

## Test Coverage

**New file: `tests/test_entity_detector.py`** — 14 tests covering:

| Test | Coverage |
|------|----------|
| `test_basic_creation` | EntityDetectionResult stores fields correctly |
| `test_to_dict_serialization` | JSON serialization works |
| `test_detect_single_text` | Basic extraction from text |
| `test_detect_with_windowing` | Multiple windows processed |
| `test_detect_no_windowing` | Full-text mode (single window) |
| `test_detect_empty_text` | Empty input handled gracefully |
| `test_detect_whitespace_only` | Whitespace-only returns empty |
| `test_entity_deduplication` | Duplicate entities tracked uniquely |
| `test_entity_contexts_include_window_info` | Context includes window_index |
| `test_context_truncation` | Long contexts truncated |
| `test_detect_batch` | Multiple texts processed |
| `test_detect_batch_auto_id` | IDs auto-generated when not provided |
| `test_metadata_stored` | Processing metadata preserved |
| `test_llm_parse_error_handled` | Invalid JSON from LLM handled |

**Test architecture:** Uses synchronous wrapper with `asyncio.new_event_loop()` to avoid pytest-asyncio dependency and event loop conflicts with other tests.

---

## Reused Components

| Component | Path | Purpose |
|-----------|------|---------|
| `EntityExtractor` | `relationships/components/entity_extractor.py` | Core LLM entity extraction prompting |
| `SlidingWindowProcessor` | `core/text.py` | Text windowing/chunking |
| `BaseLLMProvider` | `core/llm.py` | LLM provider interface |
| `OllamaProvider` | `core/providers.py` | Ollama implementation |
| Prompt v3 | `relationships/prompts/entity_extraction_v3.txt` | Entity extraction prompt template |

---

## Files Modified

| File | Changes |
|------|---------|
| `entity/detector.py` | **NEW** — Core detection module (214 lines) |
| `entity_cli.py` | Added `add_entity_detect_args()` + `run_entity_detect()` (~200 lines) |
| `unified_cli.py` | Registered `detect` subparser + imports |
| `entity/__init__.py` | Added `EntityDetector`, `EntityDetectionResult` exports |
| `tests/test_entity_detector.py` | **NEW** — 14 tests |

---

## Verification

1. **Unit tests:** All 160 tests pass including 14 new entity detector tests
2. **CLI help:** `qa entity detect --help` works correctly
3. **Module exports:** `from qualitative_analysis.entity import EntityDetector` works

---

## Next Steps

The entity detect pathway is implemented and tested. Recommended next steps:

1. **Stress test on real data** — Run on `abc-sim_combined_with_groups.csv` with full LLM
2. **Integration test** — Verify `qa entity detect` → `qa entity score` handoff
3. **Documentation** — Update user-facing docs with new command

---

*Report: 2026-02-06*
