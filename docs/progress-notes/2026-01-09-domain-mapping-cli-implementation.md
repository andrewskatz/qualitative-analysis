# Progress Update: Domain Mapping CLI Implementation

**Date:** 2026-01-09  
**Project:** qualitative-analysis package  
**Status:** Implementation Complete

---

## Executive Summary

Implemented a comprehensive domain mapping and normalization CLI for the `qualitative-analysis` Python package. This enables researchers to extract source/target conceptual domains from figurative language, normalize domain labels using semantic clustering, and generate relationship graphs—all via command-line tools.

---

## Features Implemented

### 1. Type Filtering for Figurative Detection (`--types`)

Added the ability to filter figurative language detection to specific types.

**Usage:**
```bash
qualitative-analysis input.csv --types "metaphor,analogy,extended_metaphor"
```

**Files Modified:**
- `cli.py` — Added `--types` argument
- `detector.py` — Added `figurative_types` parameter
- `strategies/two_step.py` — Pass types to Scanner
- `components/scanner.py` — Type-aware prompt selection

**New Files:**
- `prompts/types.py` — Type definitions and prompt section generator
- `prompts/two_step_binary_detection_v2.txt` — Dynamic type prompt
- `prompts/two_step_instance_extraction_v2.txt` — Dynamic type prompt

---

### 2. Domain Mapping Module (`qualitative_analysis.figurative.domains`)

Created a new submodule with three core classes:

| Class | Purpose |
|-------|---------|
| `DomainExtractor` | Extract source/target domains from figurative instances |
| `DomainNormalizer` | Cluster semantically similar domain labels |
| `DomainGraph` | Generate source→target relationship graphs |

**Key Features:**
- Multi-level domain extraction (specific, moderate, abstract)
- Embedding-based clustering with configurable conservativeness
- Checkpoint support for resumable extraction
- JSON and CSV export formats
- GPU/MPS/CPU auto-detection for embeddings

**New Files:**
- `domains/__init__.py`
- `domains/models.py` — Data classes for all domain entities
- `domains/extractor.py` — Domain extraction with LLM
- `domains/normalizer.py` — Semantic clustering with SentenceTransformer
- `domains/graph.py` — Graph generation and export
- `domains/prompts/domain_extraction_v1.txt`
- `domains/prompts/domain_extraction_v3_system.txt`
- `domains/prompts/domain_extraction_v3_user.txt`

---

### 3. Domain CLI (`qualitative-domains`)

New CLI tool with four subcommands:

```bash
qualitative-domains extract    # Extract domains from instances
qualitative-domains normalize  # Cluster domain labels
qualitative-domains graph      # Generate relationship graph
qualitative-domains pipeline   # Run full workflow
```

**Options include:**
- `--multi-level` for three abstraction levels
- `--conservativeness` (conservative/moderate/aggressive)
- `--log-llm` to print prompts and responses
- `--checkpoint` for resumable processing

---

### 4. Run Metadata JSON

Each run now outputs `run_metadata.json` containing:

```json
{
  "run_id": "run_20260109-1305",
  "timestamp_start": "2026-01-09T13:05:00",
  "timestamp_end": "2026-01-09T13:12:30",
  "duration_seconds": 450.0,
  "cli_args": { ... },
  "stats": {
    "texts_processed": 62,
    "windows_analyzed": 62,
    "instances_found": 45,
    "errors": 0
  },
  "outputs": { ... },
  "package_version": "0.1.0"
}
```

---

## Bug Fixes

### 1. System Prompt Separation

**Issue:** Domain extraction prompts had `<system>` tags embedded in the user prompt, rather than being passed as a proper system message to the LLM.

**Fix:** Split `domain_extraction_v3_multilevel.txt` into:
- `domain_extraction_v3_system.txt` (system message)
- `domain_extraction_v3_user.txt` (user message template)

Updated `DomainExtractor` to pass these separately via the `system_prompt` parameter.

### 2. Missing Window Context in Instances CSV

**Issue:** The instances CSV did not include `window_text`, making it impossible to provide context during domain extraction.

**Fix:** 
- Added `window_text` column to instances CSV output
- Built lookup from `result.metadata["windows"]` to populate the column

### 3. `--log-llm` Flag

**Issue:** Domain CLI lacked the `--log-llm` flag available in the main CLI.

**Fix:** Added `--log-llm` to both `extract` and `pipeline` subcommands.

---

## Design Decisions

### Multi-Level Domain Abstraction

Implemented three abstraction levels for domain extraction:
1. **Specific** — Concrete terms tied to the text (e.g., "highway driving")
2. **Moderate** — Conceptual categories (e.g., "vehicle motion")
3. **Abstract** — High-level cognitive frames (e.g., "physical movement")

**Rationale:** Different research questions require different granularity. The moderate level is used as the primary domain for backward compatibility.

### Embedding Model Selection

Default: `Qwen/Qwen3-Embedding-0.6B`

**Rationale:** Lightweight, high-quality embeddings suitable for domain label clustering. Auto-detects CUDA, MPS (Apple Silicon), or CPU.

### Conservativeness Levels for Clustering

| Level | Threshold | Use Case |
|-------|-----------|----------|
| Conservative | 0.9 | Only merge near-identical labels |
| Moderate | 0.75 | Balance between merging and preserving |
| Aggressive | 0.6 | Maximize consolidation |

### Separate System/User Prompts

Ollama's API properly handles separate system and user messages. Separating prompts:
- Improves LLM behavior
- Makes debugging easier (clear `[LLM SYSTEM PROMPT]` section)
- Aligns with best practices

---

## Dependencies Added

```toml
sentence-transformers >= 3.0.0  # Embedding models
scikit-learn >= 1.5.0           # Agglomerative clustering
numpy >= 1.24.0                 # Array operations
```

---

## Testing

All tests pass:

```
tests/test_domains.py — 10 passed
  - Model serialization
  - Extractor parsing (v1, v3, markdown-wrapped)
  - Graph generation (basic, normalized)
  
tests/test_mock_pipeline.py — 1 passed
```

---

## Remaining Work / Future Enhancements

1. **Unit tests for DomainNormalizer** — Currently only graph/extractor parsing tested
2. **LLM-based canonical label generation** — Implemented but not tested end-to-end
3. **Integration with web app backend** — Port this CLI functionality to the FastAPI backend
4. **README documentation** — Update package README with new CLI commands

---

## File Summary

| Path | Description |
|------|-------------|
| `src/qualitative_analysis/cli.py` | Updated with `--types`, `window_text`, metadata JSON |
| `src/qualitative_analysis/figurative/domains/` | New module (extractor, normalizer, graph) |
| `src/qualitative_analysis/figurative/domains_cli.py` | New CLI tool |
| `pyproject.toml` | New entry point + dependencies |
| `tests/test_domains.py` | Unit tests for domain module |
| `tests/sample_instances.csv` | Sample test data |

---

## Usage Examples

### Full Workflow

```bash
cd qualitative-analysis

# Step 1: Detect figurative language
PYTHONPATH=src python -m qualitative_analysis tests/paired-figurative-statements-60.csv \
  --text-col text --id-col id \
  --types "metaphor,analogy,extended_metaphor" \
  --output instances+windows --output-dir output

# Step 2: Extract domains
PYTHONPATH=src python -m qualitative_analysis.figurative.domains_cli extract \
  output/run_*/paired-figurative-statements-60_figurative_instances*.csv \
  --text-col instance_text --type-col type --id-col text_id \
  --window-col window_text --multi-level \
  -o output/domains.csv

# Step 3: Normalize domains
PYTHONPATH=src python -m qualitative_analysis.figurative.domains_cli normalize \
  output/domains.csv --conservativeness moderate \
  --save-config output/normalization.json

# Step 4: Generate graph
PYTHONPATH=src python -m qualitative_analysis.figurative.domains_cli graph \
  output/domains.csv --format both \
  --load-normalization output/normalization.json
```
