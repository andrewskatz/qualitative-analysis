# CLI Design: Entity/Relationship Commands

**Date:** 2026-01-15
**Reference:** Follows patterns established by `qa fig` commands

---

## Command Hierarchy

```
qa
├── fig                     # Figurative language (existing)
│   ├── detect
│   ├── map
│   ├── normalize
│   ├── graph
│   └── pipeline
│
├── rel                     # Relationships (extend existing)
│   ├── detect              # EXISTS
│   ├── normalize           # NEW
│   ├── verify              # NEW
│   ├── causal              # NEW
│   ├── graph               # NEW
│   └── pipeline            # NEW
│
└── entity                  # Entity analysis (NEW)
    ├── consolidate         # NEW
    ├── score               # NEW
    └── viz                 # NEW
```

---

## Relationship Commands

### `qa rel detect` (Existing - No Changes)

```bash
qa rel detect <input.csv> [options]
```

**Current functionality:** Extract entities and relationships from text.

---

### `qa rel normalize` (NEW)

Consolidate entities and normalize relationship types using semantic clustering.

```bash
qa rel normalize <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | CSV with relationship data (from `qa rel detect`) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--entity-col` | auto | Column containing entity names (auto-detects `source`, `target`) |
| `--type-col` | `type` | Column containing relationship types |
| `--entity-threshold` | `0.85` | Similarity threshold for entity merging (0.0-1.0) |
| `--type-threshold` | `0.75` | Similarity threshold for type merging (0.0-1.0) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence transformer model for embeddings |
| `--canonical-method` | `shortest` | How to select canonical form: `shortest`, `frequent`, `llm` |
| `--model` | `qwen3:30b-a3b-...` | LLM model (only used if `--canonical-method llm`) |
| `--output` | auto | Output file path (defaults to `normalized.csv` in run dir) |
| `--output-format` | `csv` | Output format: `csv`, `json`, `both` |
| `--save-maps` | False | Save entity and type mapping files separately |

**Input CSV Expected Columns:**
- `source` - Source entity
- `target` - Target entity
- `type` - Relationship type
- `description` - Relationship description/evidence
- `window_index` - Window number (optional)
- `text_id` - Source text identifier (optional)

**Output CSV Columns:**
- All input columns, plus:
- `original_source` - Source before normalization
- `original_target` - Target before normalization
- `original_type` - Type before normalization
- `merge_confidence` - Confidence score for the merge
- `instance_count` - Number of instances merged into this row

**Example:**
```bash
qa rel normalize output/run_001/relationships.csv \
    --entity-threshold 0.85 \
    --type-threshold 0.75 \
    --canonical-method llm \
    --model qwen3:30b \
    --save-maps
```

---

### `qa rel verify` (NEW)

Verify extracted relationships against source text using LLM second-pass.

```bash
qa rel verify <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | CSV with relationships to verify |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--text-col` | `window_text` | Column containing source text for verification |
| `--source-col` | `source` | Column containing source entity |
| `--target-col` | `target` | Column containing target entity |
| `--type-col` | `type` | Column containing relationship type |
| `--threshold` | `0.5` | Minimum confidence to keep relationship (0.0-1.0) |
| `--model` | `qwen3:30b-a3b-...` | LLM model for verification |
| `--prompt-version` | `v1` | Prompt template version |
| `--output` | auto | Output file path |
| `--include-rejected` | False | Include rejected relationships in output (marked) |

**Output CSV Columns:**
- All input columns, plus:
- `verification_confidence` - LLM confidence score (0.0-1.0)
- `verification_note` - LLM comments/corrections
- `verified` - Boolean: passed threshold

**Example:**
```bash
qa rel verify output/run_001/normalized.csv \
    --text-col window_text \
    --threshold 0.6 \
    --model qwen3:30b \
    --include-rejected
```

---

### `qa rel causal` (NEW)

Classify relationships for causal attributes.

```bash
qa rel causal <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | CSV with relationships to analyze |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--source-col` | `source` | Column containing source entity |
| `--target-col` | `target` | Column containing target entity |
| `--type-col` | `type` | Column containing relationship type |
| `--evidence-col` | `description` | Column containing evidence/description |
| `--model` | `qwen3:30b-a3b-...` | LLM model for analysis |
| `--prompt-version` | `v3` | Prompt template version (v1, v2, v3) |
| `--output` | auto | Output file path |
| `--stats-output` | auto | Separate file for statistics summary |

**Causal Attributes Added:**

| Column | Type | Values |
|--------|------|--------|
| `is_causal` | bool | True/False |
| `causal_polarity` | str | positive, negative, neutral |
| `causal_certainty` | str | certain, possible, uncertain |
| `causal_explicit` | str | explicit, implicit |
| `causal_explanation` | str | LLM reasoning |

**Statistics Output (JSON):**
```json
{
  "total_relationships": 150,
  "causal_count": 87,
  "causal_percentage": 58.0,
  "polarity": {
    "positive": 45,
    "negative": 32,
    "neutral": 10
  },
  "certainty": {
    "certain": 40,
    "possible": 35,
    "uncertain": 12
  },
  "explicit_vs_implicit": {
    "explicit": 52,
    "implicit": 35
  }
}
```

**Example:**
```bash
qa rel causal output/run_001/verified.csv \
    --evidence-col description \
    --model qwen3:30b \
    --prompt-version v3 \
    --stats-output causal_stats.json
```

---

### `qa rel graph` (NEW)

Generate relationship network graph with visualization.

```bash
qa rel graph <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | CSV with relationships |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--source-col` | `source` | Column for source entities |
| `--target-col` | `target` | Column for target entities |
| `--type-col` | `type` | Column for relationship types |
| `--weight-col` | None | Column for edge weights (uses count if not specified) |
| `--format` | `json` | Output formats: `json`, `csv`, `gexf`, `png`, `all` |
| `--output` | auto | Output directory |

**Visualization Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--visualize` | False | Generate PNG visualization |
| `--layout` | `spring` | Layout algorithm: `spring`, `circular`, `kamada_kawai`, `semantic` |
| `--cluster-labels` | False | Enable HDBSCAN clustering with labels |
| `--min-cluster-size` | `3` | Minimum nodes per cluster |
| `--noise-handling` | `label` | Handle outliers: `label`, `hide`, `other` |
| `--embedding-model` | `all-MiniLM-L6-v2` | Model for semantic layout |
| `--model` | `qwen3:30b-...` | LLM for cluster label generation |

**Filtering Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--min-edge-weight` | `1` | Minimum edge weight to include |
| `--relationship-types` | None | Filter to specific types (comma-separated) |
| `--include-isolated` | True | Include nodes with no connections |
| `--top-n-nodes` | None | Limit to top N nodes by degree |

**Graph Metrics Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--compute-metrics` | True | Compute NetworkX metrics |
| `--metrics-output` | auto | Separate file for metrics |

**Output Files:**

For `--format all`:
```
output/
├── graph.json           # Full graph data with metrics
├── nodes.csv            # Node list with attributes
├── edges.csv            # Edge list with attributes
├── graph.gexf           # Gephi-compatible format
├── graph.png            # Network visualization
└── metrics.json         # Graph metrics summary
```

**Example:**
```bash
qa rel graph output/run_001/causal.csv \
    --visualize \
    --layout semantic \
    --cluster-labels \
    --format all \
    --min-edge-weight 2 \
    --model qwen3:30b
```

---

### `qa rel pipeline` (NEW)

Run complete relationship analysis pipeline.

```bash
qa rel pipeline <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | Input CSV with text data |

**Pipeline Steps (all optional except detect):**

| Option | Default | Description |
|--------|---------|-------------|
| `--normalize` | False | Run normalization step |
| `--verify` | False | Run verification step |
| `--causal` | False | Run causal analysis step |
| `--graph` | True | Generate final graph |
| `--skip-detect` | False | Skip detection (use pre-detected input) |

**Inherited Options:**
All options from individual commands are available with prefixes:
- `--detect-*` options from `qa rel detect`
- `--normalize-*` options from `qa rel normalize`
- `--verify-*` options from `qa rel verify`
- `--causal-*` options from `qa rel causal`
- `--graph-*` options from `qa rel graph`

**Output Structure:**
```
output/run_001/
├── detect/
│   ├── relationships.csv
│   ├── entities.csv
│   └── windows.csv
├── normalize/
│   ├── normalized.csv
│   ├── entity_map.json
│   └── type_map.json
├── verify/
│   └── verified.csv
├── causal/
│   ├── causal.csv
│   └── stats.json
└── graph/
    ├── graph.json
    ├── nodes.csv
    ├── edges.csv
    └── graph.png
```

**Example:**
```bash
qa rel pipeline input.csv \
    --text-col transcript \
    --normalize \
    --normalize-entity-threshold 0.85 \
    --verify \
    --verify-threshold 0.6 \
    --causal \
    --graph \
    --graph-layout semantic \
    --graph-cluster-labels \
    --model qwen3:30b
```

---

## Entity Commands

### `qa entity consolidate` (NEW)

Deduplicate and merge similar entities using semantic clustering.

```bash
qa entity consolidate <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | CSV with entity data |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--entity-col` | auto | Column(s) containing entities (auto-detects) |
| `--threshold` | `0.85` | Similarity threshold for merging (0.0-1.0) |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `--canonical-method` | `shortest` | Selection method: `shortest`, `frequent`, `llm` |
| `--model` | `qwen3:30b-...` | LLM for canonical label generation |
| `--output` | auto | Output file path |
| `--output-format` | `json` | Format: `json`, `csv`, `both` |

**Output JSON Structure:**
```json
{
  "consolidations": [
    {
      "canonical": "climate change",
      "variants": ["Climate Change", "climate changes", "global climate change"],
      "confidence": 0.92,
      "frequency": 15
    }
  ],
  "mapping": {
    "Climate Change": "climate change",
    "climate changes": "climate change",
    "global climate change": "climate change"
  },
  "stats": {
    "original_count": 150,
    "consolidated_count": 87,
    "reduction_percentage": 42.0
  }
}
```

**Example:**
```bash
qa entity consolidate output/run_001/entities.csv \
    --entity-col entity \
    --threshold 0.85 \
    --canonical-method llm \
    --model qwen3:30b
```

---

### `qa entity score` (NEW)

Score entities on multiple dimensions with uncertainty quantification.

```bash
qa entity score <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | CSV with entities and context |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--entity-col` | `entity` | Column containing entity names |
| `--context-col` | auto | Column(s) for context (auto-detects window_text, text) |
| `--text-id-col` | `text_id` | Column for text identifier |
| `--dimensions` | required | Path to dimensions JSON file |
| `--num-runs` | `3` | Number of scoring runs for uncertainty |
| `--model` | `qwen3:30b-...` | LLM model for scoring |
| `--prompt-version` | `v2` | Prompt template version |
| `--output` | auto | Output file path |
| `--output-format` | `csv` | Format: `csv`, `json`, `both` |
| `--include-raw-scores` | False | Include individual run scores in output |

**Dimensions JSON Format:**
```json
{
  "dimensions": [
    {
      "name": "Social",
      "description": "Degree to which the entity represents social/human factors",
      "min_anchor": "Purely technical or natural, no social dimension",
      "max_anchor": "Fundamentally about human behavior, institutions, or culture",
      "scale_min": 0,
      "scale_max": 100
    },
    {
      "name": "Ecological",
      "description": "Degree to which the entity relates to environmental/ecological systems",
      "min_anchor": "No environmental relevance",
      "max_anchor": "Core environmental or ecological concept",
      "scale_min": 0,
      "scale_max": 100
    },
    {
      "name": "Technological",
      "description": "Degree to which the entity represents technological factors",
      "min_anchor": "No technological component",
      "max_anchor": "Fundamentally technological or engineering-based",
      "scale_min": 0,
      "scale_max": 100
    }
  ]
}
```

**Output CSV Columns:**
- `entity` - Entity name
- `text_id` - Source text identifier
- `context` - Context used for scoring
- `{dimension}_mean` - Mean score for dimension
- `{dimension}_median` - Median score
- `{dimension}_std` - Standard deviation
- `{dimension}_ci_low` - 95% CI lower bound
- `{dimension}_ci_high` - 95% CI upper bound
- `{dimension}_cv` - Coefficient of variation
- `{dimension}_justification` - LLM justification

**Example:**
```bash
qa entity score output/run_001/entities.csv \
    --entity-col entity \
    --context-col window_text \
    --dimensions sets_dimensions.json \
    --num-runs 5 \
    --model qwen3:30b \
    --include-raw-scores
```

---

### `qa entity viz` (NEW)

Generate visualizations for scored entities.

```bash
qa entity viz <input.csv> [options]
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `input.csv` | Yes | CSV with entity scores (from `qa entity score`) |

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | `ternary` | Visualization type (see below) |
| `--dimensions` | auto | Dimensions to visualize (comma-separated) |
| `--output` | auto | Output file path |
| `--format` | `png` | Output format: `png`, `svg`, `pdf` |
| `--width` | `12` | Figure width in inches |
| `--height` | `10` | Figure height in inches |
| `--dpi` | `150` | Resolution for raster formats |

**Visualization Types:**

| Type | Description | Dimensions |
|------|-------------|------------|
| `ternary` | Triangular plot for 3D scores | Exactly 3 |
| `radar` | Spider/radar chart | 3+ |
| `heatmap` | Entity × Dimension matrix | Any |
| `boxplot` | Score distributions | Any |
| `scatter` | 2D scatter with optional size/color | 2-4 |
| `confidence` | Scores with error bars | Any |

**Ternary-Specific Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--color-by` | `primary` | Color scheme: `primary`, `cluster`, `text_id` |
| `--show-uncertainty` | False | Vary opacity by coefficient of variation |
| `--label-top-n` | `20` | Number of entities to label |
| `--legend-position` | `right` | Legend position: `right`, `bottom`, `none` |

**Example:**
```bash
# Ternary plot for SETS framework
qa entity viz output/run_001/scores.csv \
    --type ternary \
    --dimensions "Social,Ecological,Technological" \
    --show-uncertainty \
    --label-top-n 30 \
    --output sets_ternary.png

# Radar chart
qa entity viz output/run_001/scores.csv \
    --type radar \
    --output radar.png

# Heatmap
qa entity viz output/run_001/scores.csv \
    --type heatmap \
    --output heatmap.png
```

---

## Common Options

All commands support these common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `qwen3:30b-a3b-instruct-2507-q4_K_M` | LLM model name |
| `--provider` | `ollama` | LLM provider: `ollama`, `openai`, `mlx` |
| `--temperature` | `0.1` | LLM temperature |
| `--verbose` | False | Enable verbose output |
| `--quiet` | False | Suppress progress output |
| `--checkpoint` | False | Enable checkpoint/resume |
| `--checkpoint-interval` | `10` | Items between checkpoints |

---

## Output Directory Structure

Following the pattern established by figurative language:

```
output/
└── run_001/                    # Auto-generated run ID
    ├── detect/
    │   └── detect_001/
    │       ├── relationships.csv
    │       ├── entities.csv
    │       ├── windows.csv
    │       └── summary.json
    ├── normalize/
    │   └── normalize_001/
    │       ├── normalized.csv
    │       ├── entity_map.json
    │       └── type_map.json
    ├── verify/
    │   └── verify_001/
    │       └── verified.csv
    ├── causal/
    │   └── causal_001/
    │       ├── causal.csv
    │       └── stats.json
    ├── graph/
    │   └── graph_001/
    │       ├── graph.json
    │       ├── nodes.csv
    │       ├── edges.csv
    │       ├── graph.gexf
    │       └── graph.png
    └── entity/
        ├── consolidate_001/
        │   └── consolidation.json
        ├── score_001/
        │   └── scores.csv
        └── viz_001/
            └── ternary.png
```

---

## Backward Compatibility

The existing `qa rel detect` command remains unchanged. New commands extend functionality without breaking existing workflows.

Legacy command aliases continue to work:
- `qualitative-relationships` → `qa rel detect` (with deprecation warning)
