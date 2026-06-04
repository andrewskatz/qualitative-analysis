# Entity Scoring Context Fix & Pipeline Tracing

**Date:** 2026-01-26
**Package:** `qualitative-analysis`
**Component:** `qa entity` - scoring input preparation

---

## Summary

Identified and fixed a critical issue where entity scoring was receiving placeholder context text instead of actual raw text from source documents. Traced the full pipeline and implemented a new CLI command to properly link entities to their original context.

---

## Problem Identified

When running entity scoring with `--verbose` to inspect LLM prompts, discovered that the `<context>` tag contained placeholder text:

```
<context>
This entity "residents" was mentioned in the context of community energy issues.
</context>
```

**Expected:** Actual raw text from the source document where the entity was extracted.

---

## Root Cause Analysis

Traced the full entity extraction and scoring pipeline:

```
Raw Text → SlidingWindowProcessor → EntityExtractor → Graph/Nodes → Scoring Input CSV → EntityScorer
```

### Where Context EXISTS (Preserved)

1. **Windows CSV** (`*_windows.csv`) from `qa relationships detect --save-windows`:
   - `window_text` column: Actual raw text from sliding windows
   - `entities_json` column: List of entities extracted from that window
   - Context IS properly captured here

2. **Relationship edges** (`*_edges.csv`):
   - `description` column: Evidence text for each relationship
   - Window tracking via `window_index` and `text_id`

### Where Context Was LOST

When preparing scoring input for multi-participant comparison, I created a CSV file with fabricated placeholder context instead of looking up the actual `window_text` from the windows file.

---

## Solution Implemented

### New CLI Command: `qa entity prepare-scoring`

Extracts entity-context pairs from windows files, ensuring real source text is used.

```bash
qa entity prepare-scoring windows.csv \
  --nodes-csv graph_nodes.csv \
  --output scoring_input.csv \
  --context-mode window \
  --min-frequency 2
```

**Options:**

| Flag | Description |
|------|-------------|
| `--nodes-csv` | Optional: Filter to entities in graph nodes file |
| `--output` | Output path (default: `<windows>_scoring_input.csv`) |
| `--context-mode` | `window` (full), `sentence` (just sentence), `combined` (all windows) |
| `--max-context-length` | Truncate context at N characters (default: 2000) |
| `--min-frequency` | Minimum entity frequency to include (default: 1) |
| `--participants` | Comma-separated list of participants to include |

### Implementation Details

**Files modified:**

| File | Changes |
|------|---------|
| [entity_cli.py](../src/qualitative_analysis/entity_cli.py) | Added `add_entity_prepare_scoring_args()`, `run_entity_prepare_scoring()`, helper functions |
| [unified_cli.py](../src/qualitative_analysis/unified_cli.py) | Registered `prepare-scoring` subcommand with alias `prep` |

**Key functions added:**

- `_extract_entity_contexts_from_windows()` - Reads windows CSV and extracts entity-context pairs
- `_filter_by_nodes()` - Optionally filters by graph nodes file
- `_extract_sentence_with_entity()` - Extracts just the sentence containing an entity
- `_write_scoring_input_csv()` - Writes properly formatted output

---

## Test Results

Generated proper scoring input from real data:

```bash
qa entity prepare-scoring \
  tests/output/pipeline_test/run_20260119-1353/combined_rows_relationships_windows_20260119-1353.csv \
  --nodes-csv tests/output/pipeline_test/run_20260119-1353/graph/combined_rows_relationships_edges_20260119-1353_causal_graph_nodes.csv \
  --output tests/output/entity_comparison_input_v2.csv \
  --context-mode window \
  --min-frequency 2
```

**Output:**
- 214 entity-context pairs
- 153 unique entities
- 5 unique participants

**Before (placeholder context):**
```csv
entity,context,text_id
residents,"This entity ""residents"" was mentioned in the context of community energy issues.",response_Alex
```

**After (real context):**
```csv
entity,context,text_id
abeesee,"Abeesee's biggest issue is definitely the high cost of heating due to its harsh winters and remote location. The rising price of fossil fuels means many residents can't afford heat for the entire winter, leading to serious health risks and even deaths. It's a critical problem that needs innovative solutions, maybe something like smart grid technologies or renewable energy sources to make heating more affordable and accessible.",response_Alex
```

---

## Pipeline Documentation

### Complete Entity Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: RAW TEXT ENTRY                                             │
│ Input: CSV with "text" column                                       │
│ Command: Input to `qa relationships detect`                         │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: RELATIONSHIP DETECTION                                     │
│ Command: `qa relationships detect input.csv --save-windows`         │
│ Output: *_windows.csv, *_edges.csv                                  │
│ Context: window_text preserved with entities                        │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: GRAPH CONSTRUCTION                                         │
│ Command: `qa relationships graph edges.csv`                         │
│ Output: *_graph_nodes.csv, *_graph_edges.csv                        │
│ Context: Node IDs and source_text_ids (participants)                │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: SCORING INPUT PREPARATION  ⭐ NEW                          │
│ Command: `qa entity prepare-scoring windows.csv --nodes-csv ...`    │
│ Output: scoring_input.csv with real context                         │
│ Context: Actual window_text linked to each entity                   │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: ENTITY SCORING                                             │
│ Command: `qa entity score scoring_input.csv --num-runs 3`           │
│ Output: *_scores.csv with dimension scores                          │
│ Context: Used in LLM prompt for accurate scoring                    │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 6: COMPARISON & VISUALIZATION                                 │
│ Command: `qa entity compare scores.csv`                             │
│ Output: Distance matrices, ternary plots, heatmaps                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Relation to Implementation Plan

This work supports **Phase 4 (CLI Integration)** of the multi-participant comparison plan by ensuring the scoring input preparation step properly preserves context.

### Plan Status Update

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Distance Functions | ✅ Complete | Aitchison, EMD, pairwise matrix |
| Phase 2: Bayesian Models | 🔲 Not Started | PyMC/NumPyro hierarchical model |
| Phase 3: Visualizations | ✅ Mostly Complete | Faceted ternary, overlaid, heatmap done; forest plots, MDS pending |
| Phase 4: CLI Integration | ✅ In Progress | `compare` command done; `prepare-scoring` added today |
| Phase 5: Advanced Features | 🔲 Not Started | Dirichlet regression, longitudinal |

---

## Next Steps

1. **Run full scoring** with the corrected input file (214 entities × 3 runs)
2. **Generate comparison visualizations** with properly scored data
3. **Implement remaining Phase 3 items:**
   - Forest plots (per-dimension comparisons)
   - MDS/UMAP similarity map
4. **Phase 2: Bayesian modeling** for proper uncertainty quantification

---

## Files Created/Modified

| File | Action |
|------|--------|
| `entity_cli.py` | Modified - Added prepare-scoring command |
| `unified_cli.py` | Modified - Registered prepare-scoring subcommand |
| `tests/output/entity_comparison_input_v2.csv` | Created - Proper scoring input with real context |
