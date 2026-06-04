# Plan: `qa entity detect` — Standalone Entity Extraction Pathway

## Overview

Add a `qa entity detect` command that extracts entities from text, filling the UX gap between raw text input and entity scoring. This creates a self-contained entity analysis pathway parallel to the existing relationships pathway.

**Current Gap:**
- `qa relationships detect` → extracts entities + relationships (overkill for entity-only analysis)
- `qa entity score` → expects entities already extracted
- No direct: **text → entities** command

**Solution:** `qa entity detect` command that outputs CSV ready for `qa entity score`.

---

## Design

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
| `--log-llm` | `False` | Print prompts and responses |
| `--window-size` | `3` | Sentences per window |
| `--stride` | `2` | Window stride |
| `--no-windowing` | `False` | Process entire text as single unit |
| `--prompt-version` | `3` | Entity extraction prompt version |
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
  "total_entities": 127,
  "unique_entities": 84,
  "timestamp": "2026-02-05T12:00:00"
}
```

---

## Implementation

### Files to Modify

| File | Changes |
|------|---------|
| `entity/detector.py` | **NEW** — Core detection logic |
| `entity_cli.py` | Add `add_entity_detect_args()` + `run_entity_detect()` |
| `unified_cli.py` | Register `detect` subparser in `_add_entity_commands()` |
| `entity/__init__.py` | Export `EntityDetector`, `EntityDetectionResult` |

### New File: `entity/detector.py` (~200 lines)

```python
"""Entity detection from raw text using LLM analysis."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging

from qualitative_analysis.relationships.components.entity_extractor import EntityExtractor
from qualitative_analysis.core.text import SlidingWindowProcessor
from qualitative_analysis.core.llm import get_llm_provider

logger = logging.getLogger(__name__)

@dataclass
class EntityDetectionResult:
    """Result of entity detection for a single text."""
    text_id: str
    entities: List[str]
    entity_contexts: List[Dict[str, Any]]  # {entity, context, window_index}
    window_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_id": self.text_id,
            "entities": self.entities,
            "entity_count": len(self.entities),
            "window_count": self.window_count,
            "metadata": self.metadata,
        }

class EntityDetector:
    """Detect entities in text using LLM analysis."""

    def __init__(
        self,
        llm_provider,
        window_size: int = 3,
        stride: int = 2,
        prompt_version: int = 3,
        log_llm: bool = False,
    ):
        self.extractor = EntityExtractor(llm_provider, prompt_version=prompt_version)
        self.text_processor = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride,
            chunk_unit="sentences",
        )
        self.log_llm = log_llm

    async def detect(
        self,
        text: str,
        text_id: str,
        use_windowing: bool = True,
    ) -> EntityDetectionResult:
        """Extract entities from text."""
        if use_windowing:
            windows = self.text_processor.process(text)
        else:
            windows = [text]

        all_entities = []
        entity_contexts = []

        for idx, window in enumerate(windows):
            entities = await self.extractor.extract(window)

            for entity in entities:
                if entity not in all_entities:
                    all_entities.append(entity)
                entity_contexts.append({
                    "entity": entity,
                    "context": window[:500],  # Truncate for storage
                    "window_index": idx,
                })

        return EntityDetectionResult(
            text_id=text_id,
            entities=all_entities,
            entity_contexts=entity_contexts,
            window_count=len(windows),
        )
```

### Modifications to `entity_cli.py`

Add after line ~1850 (after `run_entity_cluster`):

```python
def add_entity_detect_args(parser: argparse.ArgumentParser) -> None:
    """Add entity detection arguments."""
    parser.add_argument("input_csv", help="Path to input CSV file with text to analyze.")
    parser.add_argument("--text-col", default="text", help="Column name for text (default: text).")
    parser.add_argument("--id-col", default=None, help="Column for text IDs (default: auto-generate).")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--model", default="gpt-oss:120b", help="LLM model name.")
    parser.add_argument("--provider", choices=["ollama", "openai", "anthropic"], default="ollama")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature.")
    parser.add_argument("--window-size", type=int, default=3, help="Sentences per window.")
    parser.add_argument("--stride", type=int, default=2, help="Window stride.")
    parser.add_argument("--no-windowing", action="store_true", help="Process entire text as single unit.")
    parser.add_argument("--prompt-version", type=int, default=3, help="Entity extraction prompt version.")
    parser.add_argument("--log-llm", action="store_true", help="Print LLM prompts and responses.")
    parser.add_argument("--limit", type=int, default=None, help="Limit rows processed (for testing).")


async def run_entity_detect(args: argparse.Namespace) -> int:
    """Run entity detection on input CSV."""
    import csv
    import json
    from datetime import datetime
    from qualitative_analysis.entity.detector import EntityDetector
    from qualitative_analysis.core.llm import get_llm_provider

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = input_path.parent / f"entities_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM and detector
    llm = get_llm_provider(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        log_llm=args.log_llm,
    )

    detector = EntityDetector(
        llm_provider=llm,
        window_size=args.window_size,
        stride=args.stride,
        prompt_version=args.prompt_version,
        log_llm=args.log_llm,
    )

    # Read input CSV
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit:
        rows = rows[:args.limit]

    # Process each row
    all_results = []
    all_entity_rows = []

    for i, row in enumerate(rows):
        text = row.get(args.text_col, "")
        text_id = row.get(args.id_col, f"text_{i}") if args.id_col else f"text_{i}"

        if not text.strip():
            logger.warning(f"Empty text for {text_id}, skipping")
            continue

        print(f"Processing {text_id} ({i+1}/{len(rows)})...")

        result = await detector.detect(
            text=text,
            text_id=text_id,
            use_windowing=not args.no_windowing,
        )
        all_results.append(result)

        for ec in result.entity_contexts:
            all_entity_rows.append({
                "text_id": text_id,
                "entity": ec["entity"],
                "context": ec["context"],
                "window_index": ec["window_index"],
            })

    # Write entities.csv (one row per entity occurrence)
    entities_path = output_dir / "entities.csv"
    with open(entities_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_id", "entity", "context", "window_index"])
        writer.writeheader()
        writer.writerows(all_entity_rows)

    # Write entities_summary.csv (one row per text)
    summary_path = output_dir / "entities_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text_id", "entity_count", "entities_json"])
        writer.writeheader()
        for result in all_results:
            writer.writerow({
                "text_id": result.text_id,
                "entity_count": len(result.entities),
                "entities_json": json.dumps(result.entities),
            })

    # Write metadata
    metadata = {
        "input_file": str(input_path),
        "model": args.model,
        "total_texts": len(all_results),
        "total_entity_occurrences": len(all_entity_rows),
        "unique_entities": len(set(r["entity"] for r in all_entity_rows)),
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "detect_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n=== Entity Detection Complete ===")
    print(f"Texts processed: {len(all_results)}")
    print(f"Total entity occurrences: {len(all_entity_rows)}")
    print(f"Unique entities: {metadata['unique_entities']}")
    print(f"Output: {output_dir}")

    return 0
```

### Modifications to `unified_cli.py`

In `_add_entity_commands()` (around line 260), add after imports:

```python
from qualitative_analysis.entity_cli import (
    # ... existing imports ...
    add_entity_detect_args,
    run_entity_detect,
)
```

Add subparser registration (before `score` subcommand):

```python
# detect subcommand
detect_parser = entity_subs.add_parser(
    "detect",
    help="Detect entities in text",
    description="Extract entities from text using LLM analysis.",
)
add_entity_detect_args(detect_parser)
detect_parser.set_defaults(func=lambda args: asyncio.run(run_entity_detect(args)))
```

### Modifications to `entity/__init__.py`

Add imports and exports:

```python
from qualitative_analysis.entity.detector import EntityDetector, EntityDetectionResult

__all__ = [
    # ... existing exports ...
    "EntityDetector",
    "EntityDetectionResult",
]
```

---

## Reused Components

| Component | Path | Usage |
|-----------|------|-------|
| `EntityExtractor` | `relationships/components/entity_extractor.py` | Core LLM entity extraction |
| `SlidingWindowProcessor` | `core/text.py` | Text windowing/chunking |
| `get_llm_provider` | `core/llm.py` | LLM provider factory |
| Entity extraction prompt v3 | `relationships/prompts/entity_extraction_v3.txt` | Prompt template |

---

## Testing

**New file: `tests/test_entity_detector.py`** (~8 tests)

```python
# Test cases:
# 1. Single text detection produces entities
# 2. Windowing produces multiple windows
# 3. No-windowing mode processes full text
# 4. Empty text handled gracefully
# 5. Entity deduplication within text
# 6. Output CSV has correct columns
# 7. Metadata JSON is valid
# 8. CLI help works
```

---

## Verification

1. **Unit tests:** `pytest tests/test_entity_detector.py -v`
2. **CLI help:** `qa entity detect --help`
3. **End-to-end test:**
   ```bash
   qa entity detect tests/data/ent-rel-datasets/abc-sim_combined_with_groups.csv \
     --text-col response_text \
     --id-col interview_id \
     --output-dir output/test-detect \
     --limit 2 \
     --log-llm
   ```
4. **Integration with scoring:**
   ```bash
   qa entity score output/test-detect/entities.csv \
     --entity-col entity \
     --context-col context \
     --text-id-col text_id
   ```

---

## Implementation Order

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Create `entity/detector.py` | ~30 min |
| 2 | Add CLI functions in `entity_cli.py` | ~30 min |
| 3 | Register in `unified_cli.py` and `__init__.py` | ~10 min |
| 4 | Write tests | ~20 min |
| 5 | End-to-end verification | ~15 min |

**Total estimated effort:** ~2 hours
