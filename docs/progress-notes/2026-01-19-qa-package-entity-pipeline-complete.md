# Qualitative Analysis Package Progress Update

**Date:** 2026-01-19
**Status:** Active Development
**Focus:** Entity Pipeline Completion & Relationship Verification

---

## Summary

Completed the full entity analysis pipeline and relationship verification for the `qualitative-analysis` Python package. This session added:
- Entity Visualizer (`qa entity viz`)
- Entity Consolidator (`qa entity consolidate`)
- Relationship Verifier (`qa rel verify`)

**Current pipeline status:**
- **Relationship pipeline:** `detect → normalize → verify → causal → graph` ✅
- **Entity pipeline:** `consolidate → score → viz` ✅

---

## Session Progress

### 1. Entity Visualizer (`qa entity viz`)

Created visualization capabilities for scored entity data, ported from the backend's `VisualizationService`.

#### Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/entity/visualizer.py`

| Component | Description |
|-----------|-------------|
| `EntityVisualizer` class | Main visualization service |
| `generate_ternary_plot()` | Ternary plot for 3-dimension frameworks (SETS) |
| `generate_radar_chart()` | Radar chart for multi-dimension comparisons |
| `_barycentric_to_cartesian()` | Convert ternary to Cartesian coordinates |
| `_draw_triangle()` | Triangle with grid lines and dimension labels |
| `_add_numbered_labels()` | Smart label placement with collision avoidance |

**Key Features:**
- **Ternary plots** for 3-dimension SETS framework
  - Equilateral triangle with 10% grid lines
  - Dimension labels at vertices (100% markers)
  - Points colored by primary dimension (Social=blue, Ecological=green, Technological=red)
  - Uncertainty visualization via opacity (lower CV = higher opacity)
  - Numbered labels with legend (handles many entities cleanly)
  - Direct labels option for fewer entities

- **Radar charts** for any number of dimensions
  - Polygonal score profiles for each entity
  - Configurable max entities
  - Color-coded by entity

#### CLI Integration
```bash
# Ternary plot (default for SETS)
qa entity viz scored_entities.csv \
    --type ternary \
    --dimensions social,ecological,technological \
    --title "Entity Classifications (SETS)"

# Radar chart
qa entity viz scored_entities.csv \
    --type radar \
    --max-entities 10
```

**CLI Options:**
| Option | Description |
|--------|-------------|
| `--type` | Visualization type: `ternary` or `radar` |
| `--dimensions` | Comma-separated dimension names |
| `--title` | Custom plot title |
| `--no-labels` | Hide entity labels |
| `--direct-labels` | Use direct labels instead of numbered legend |
| `--max-entities` | Max entities for radar chart |
| `--figsize` | Figure size as `width,height` |

#### Test Results
```
Input: 1 scored entity from Entity Scorer test
Output:
  Ternary plot: 184KB PNG (2186x1508)
    - Correct triangle with grid lines
    - Dimension labels at vertices
    - Entity point colored by primary dimension (Ecological)
    - Numbered label with legend

  Radar chart: 202KB PNG (1751x1498)
    - 3-axis radar for SETS dimensions
    - Entity polygon showing score profile
```

---

### 2. Entity Consolidator (`qa entity consolidate`)

Created semantic deduplication for entities using embedding-based clustering, ported from the backend's `EntityConsolidationService`.

#### Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/entity/consolidator.py`

| Component | Description |
|-----------|-------------|
| `EntityConsolidator` class | Main consolidation service |
| `consolidate()` | Main consolidation pipeline |
| `consolidate_from_csv()` | Direct CSV processing |
| `_select_canonical()` | Canonical form selection with multiple methods |
| `save()` / `load()` | JSON serialization |

**Key Features:**
- **Semantic clustering** using `EmbeddingService`
  - Agglomerative clustering with cosine distance
  - Configurable similarity threshold (0.0-1.0)
  - Complete linkage for tight clusters

- **Multiple canonical selection methods:**
  | Method | Description |
  |--------|-------------|
  | `shortest` | Shortest string (default, often root form) |
  | `frequent` | Most frequent in dataset (requires frequency column) |
  | `representative` | Entity closest to cluster centroid |
  | `first` | First occurrence in input |

- **Output formats:**
  - `_mapping.csv` - Original entity → canonical entity
  - `_clusters.csv` - Cluster details (canonical, variants, similarity)
  - `_consolidation.json` - Full result with statistics

#### CLI Integration
```bash
qa entity consolidate entities.csv \
    --threshold 0.85 \
    --canonical-method shortest \
    --embedding-model all-MiniLM-L6-v2 \
    --output-format csv,json
```

**CLI Options:**
| Option | Description |
|--------|-------------|
| `--entity-col` | Column name for entities |
| `--frequency-col` | Optional frequency column |
| `--threshold` | Similarity threshold (0.0-1.0) |
| `--canonical-method` | Selection method for canonical form |
| `--embedding-model` | Sentence transformer model |
| `--output-format` | Output formats (csv, json) |

#### Test Results
```
Input: 13 entities with variations
  - "climate change", "Climate Change", "global climate change", "climate crisis"
  - "flooding", "Flooding", "flood events"
  - "renewable energy", "Renewable Energy", "clean energy"
  - "smart grid", "urban planning", "community garden"

Output (threshold=0.75):
  Consolidation Summary:
    Original: 13 entities
    Consolidated: 8 entities (38.5% reduction)

  Merged clusters:
    - "climate change" ← 3 variants (0.92 avg similarity)
    - "flooding" ← 3 variants (0.84 avg similarity)
    - "renewable energy" ← 2 variants (1.0 avg similarity)
```

---

### 3. Relationship Verifier (`qa rel verify`)

Created LLM-based verification to validate extracted relationships against source text, ported from the backend's `RelationshipVerifier`.

#### Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/relationships/verifier.py`

| Component | Description |
|-----------|-------------|
| `RelationshipVerifier` class | Main verification service |
| `verify()` | Verify relationships against source texts |
| `_verify_batch()` | Batch verification with single LLM call |
| `_parse_response()` | Parse LLM verification output |
| `VerificationResult` | Result dataclass with statistics |

**Key Features:**
- **Window-level verification** (preferred) - Verifies against specific text windows where relationships were extracted
- **Document-level fallback** - Falls back to full document text when windows unavailable
- **Batch verification** - Groups relationships by (text_id, window_index), verifies in batches
- **Confidence scoring** - Each relationship gets a confidence score (0.0-1.0)
- **Threshold filtering** - Configurable minimum confidence to accept
- **Evidence tracking** - Captures supporting quotes from LLM
- **Correction notes** - Records suggested corrections for borderline cases

**Verification Modes:**
| Mode | Input | Use Case |
|------|-------|----------|
| Window-level | `--windows windows.csv` | Preferred - verifies against the specific text chunk |
| Document-level | `source_texts.csv` | Fallback - verifies against full document |

**Source Text Tracking:**
| Field | Purpose |
|-------|---------|
| `window_index` | Index of text window where relationship was found |
| `text_id` | Identifier for the source document |
| `description` | Evidence/snippet text supporting the relationship |

#### CLI Integration
```bash
# Window-level verification (recommended)
qa rel verify relationships.csv --windows windows.csv \
    --confidence-threshold 0.7 \
    --model qwen3:8b \
    --verified-only

# Document-level verification (fallback)
qa rel verify relationships.csv source_texts.csv \
    --confidence-threshold 0.7
```

**CLI Options:**
| Option | Description |
|--------|-------------|
| `input_csv` | CSV with relationships (source, target, type, text_id, window_index) |
| `source_csv` | Optional: CSV with full source texts (text_id, text) |
| `--windows` | Optional: CSV with window texts (text_id, window_index, window_text) |
| `--confidence-threshold` | Minimum confidence to accept (default: 0.7) |
| `--batch-size` | Max relationships per LLM call (default: 20) |
| `--verified-only` | Only output verified relationships |
| `--include-rejected` | Also output rejected relationships |

#### Verification Prompt
The verifier uses a prompt that asks the LLM to:
1. Check if each relationship is explicitly supported by the text
2. Assess confidence in the support (0.0-1.0)
3. Provide corrections if partially correct
4. Quote supporting evidence if verified

---

## File Changes Summary

| File | Change |
|------|--------|
| `entity/visualizer.py` | **NEW** - EntityVisualizer class with ternary/radar charts |
| `entity/consolidator.py` | **NEW** - EntityConsolidator class for semantic deduplication |
| `entity/__init__.py` | **UPDATED** - Export EntityVisualizer, EntityConsolidator |
| `entity_cli.py` | **EXTENDED** - Implemented viz and consolidate commands |
| `relationships/verifier.py` | **NEW** - RelationshipVerifier class for LLM verification |
| `relationships/prompts/relationship_verification_v1.txt` | **NEW** - Verification prompt template |
| `relationships/__init__.py` | **UPDATED** - Export RelationshipVerifier, VerificationResult |
| `relationships_cli.py` | **EXTENDED** - Implemented verify command |
| `unified_cli.py` | **UPDATED** - Added verify subcommand to relationships |

---

## Complete CLI Command Reference

### Relationship Pipeline
| Command | Status | Description |
|---------|--------|-------------|
| `qa rel detect` | ✅ Existing | Extract entities and relationships from text |
| `qa rel normalize` | ✅ Complete | Cluster entities/types, merge relationships |
| `qa rel verify` | ✅ Complete | LLM verification pass to filter hallucinations |
| `qa rel causal` | ✅ Complete | Classify causal attributes (polarity, certainty) |
| `qa rel graph` | ✅ Complete | Generate network graph with metrics |

### Entity Pipeline
| Command | Status | Description |
|---------|--------|-------------|
| `qa entity consolidate` | ✅ Complete | Semantic entity deduplication |
| `qa entity score` | ✅ Complete | Multi-dimensional scoring (SETS) |
| `qa entity viz` | ✅ Complete | Ternary plots and radar charts |

---

## Architecture Overview

```
qualitative-analysis/src/qualitative_analysis/
├── core/
│   ├── embeddings.py          # Shared embedding service (clustering, similarity)
│   ├── llm.py                 # LLM interface
│   └── providers.py           # Ollama, OpenAI providers
├── entity/
│   ├── __init__.py            # Module exports
│   ├── models.py              # DimensionDefinition, EntityScore, ConsolidationResult
│   ├── scorer.py              # EntityScorer - multi-dimensional LLM scoring
│   ├── visualizer.py          # EntityVisualizer - ternary plots, radar charts
│   ├── consolidator.py        # EntityConsolidator - semantic deduplication
│   └── prompts/
│       └── entity_scoring_v2.txt
├── relationships/
│   ├── models.py              # Relationship, CausalRelationship, GraphData
│   ├── detector.py            # RelationshipDetector
│   ├── normalizer.py          # RelationshipNormalizer
│   ├── verifier.py            # RelationshipVerifier - LLM verification pass
│   ├── causal.py              # CausalAnalyzer
│   ├── graph.py               # RelationshipGraph
│   └── prompts/
│       ├── causal_enrichment.txt
│       └── relationship_verification_v1.txt
├── entity_cli.py              # Entity CLI commands
├── relationships_cli.py       # Relationship CLI commands
└── unified_cli.py             # Main CLI entry point (qa command)
```

---

## Workflow Examples

### Complete Entity Analysis Workflow
```bash
# 1. Extract entities from text (using relationship detection)
qa rel detect texts.csv --output-dir ./analysis

# 2. Consolidate similar entities
qa entity consolidate ./analysis/entities.csv \
    --threshold 0.85 \
    --output-dir ./analysis/consolidated

# 3. Score entities on SETS dimensions
qa entity score ./analysis/consolidated/entities_mapping.csv \
    --dimensions sets \
    --num-runs 3 \
    --output-dir ./analysis/scored

# 4. Visualize scores
qa entity viz ./analysis/scored/entities_mapping_scores.csv \
    --type ternary \
    --output ./analysis/ternary_plot.png
```

### Complete Relationship Analysis Workflow
```bash
# 1. Detect relationships (now outputs windows.csv by default)
qa rel detect texts.csv --output-dir ./analysis
# Output: relationships.csv, windows.csv, summary.txt

# 2. Normalize entities and relationship types
qa rel normalize ./analysis/relationships.csv \
    --entity-threshold 0.85 \
    --type-threshold 0.75 \
    --output-dir ./analysis/normalized

# 3. Verify relationships using window-level text (recommended)
qa rel verify ./analysis/normalized/relationships_normalized.csv \
    --windows ./analysis/windows.csv \
    --confidence-threshold 0.7 \
    --verified-only \
    --output-dir ./analysis/verified

# 4. Analyze causal attributes
qa rel causal ./analysis/verified/relationships_verified.csv \
    --model qwen3:8b \
    --output-dir ./analysis/causal

# 5. Generate relationship graph
qa rel graph ./analysis/causal/relationships_causal.csv \
    --layout semantic \
    --output-format csv,json,png,gexf
```

---

## Next Steps

1. **Pipeline Orchestration** - Combined commands for full workflows (`qa rel pipeline`, `qa entity pipeline`)
2. **Comprehensive Testing** - End-to-end testing with research datasets
3. **Performance Optimization** - Caching, parallel processing for large datasets

---

## Technical Notes

### Ternary Plot Coordinate System
The ternary plot uses barycentric coordinates converted to Cartesian:
- Bottom-left vertex: Technological (100%)
- Bottom-right vertex: Social (100%)
- Top vertex: Ecological (100%)
- Entity position determined by normalized score proportions

### Consolidation Threshold Guidelines
| Threshold | Use Case |
|-----------|----------|
| 0.90+ | Very strict - only near-identical entities merged |
| 0.85 | Default - good balance for typos and case variations |
| 0.75 | Moderate - catches more semantic variations |
| 0.60 | Aggressive - may over-merge distinct concepts |

### Embedding Model Options
| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| `all-MiniLM-L6-v2` | Fast | Good | Default, 384 dimensions |
| `all-mpnet-base-v2` | Slower | Better | Higher quality embeddings |
| `paraphrase-multilingual-MiniLM-L12-v2` | Fast | Good | Multi-language support |

### Windowing and Verification
The relationship detection uses sliding windows to chunk long texts:
- Default: 3 sentences per window, stride of 2
- Each relationship records its `window_index` for traceability

**Important change**: `qa rel detect` now outputs `windows.csv` by default (previously opt-in). This enables accurate window-level verification:
- Window-level verification matches each relationship to its exact source chunk
- Document-level verification (full text) may miss context or exceed token limits
- Use `--no-windows` to disable window output if not needed
