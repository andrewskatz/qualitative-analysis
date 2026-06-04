# Qualitative Analysis Package Progress Update

**Date:** 2026-01-16
**Status:** Active Development
**Focus:** Entity/Relationship Pipeline Foundation, Normalization, Causal Analysis, Graph Generation, Entity Scoring, Visualization & Consolidation

---

## Summary

Ported entity/relationship analysis functionality from the fullstack web app backend to the `qualitative-analysis` Python package. Completed the foundation layer (data models, embedding service), the Relationship Normalizer, the Causal Analyzer, the Relationship Graph Generator, the Entity Scorer, the Entity Visualizer, and the Entity Consolidator with full CLI integration.

**Completed pipelines:**
- **Relationship pipeline:** `detect → normalize → causal → graph` ✅
- **Entity pipeline:** `consolidate → score → viz` ✅

---

## Completed Work

### 1. Foundation Layer

#### Shared Embedding Service
**File:** `qualitative-analysis/src/qualitative_analysis/core/embeddings.py`

Created a unified embedding service for semantic operations across all modules:

| Feature | Description |
|---------|-------------|
| `embed()` | Generate embeddings for text lists |
| `compute_similarity()` | Pairwise semantic similarity |
| `compute_batch_similarity()` | Query vs. multiple candidates |
| `cluster()` | Agglomerative clustering with threshold |
| `cluster_with_info()` | Clustering with canonical selection |
| `find_similar()` | Search for similar items |

**Key design decisions:**
- Lazy model loading for efficiency
- Auto-device detection (CPU/CUDA/MPS)
- Default model: `all-MiniLM-L6-v2` (fast, good quality)
- Threshold presets: conservative (0.85), moderate (0.75), aggressive (0.60)

#### Extended Relationship Models
**File:** `qualitative-analysis/src/qualitative_analysis/relationships/models.py`

Added 12 new dataclasses to support the complete pipeline:

| Category | Models |
|----------|--------|
| **Normalization** | `EntityCluster`, `NormalizedEntity`, `NormalizedRelationship`, `NormalizationResult` |
| **Verification** | `VerifiedRelationship` |
| **Causal Analysis** | `CausalAttributes`, `CausalRelationship`, `CausalAnalysisResult` |
| **Graph** | `RelationshipGraphNode`, `RelationshipGraphEdge`, `RelationshipGraphData` |
| **Utilities** | `Checkpoint` |

#### New Entity Module
**Directory:** `qualitative-analysis/src/qualitative_analysis/entity/`

Created new module for entity-specific analysis:

| File | Contents |
|------|----------|
| `__init__.py` | Module exports |
| `models.py` | `DimensionDefinition`, `DimensionSet`, `DimensionScore`, `EntityScore`, `EntityScoreResult`, `ConsolidationResult` |
| `prompts/` | Directory for scoring prompts (to be populated) |

**Notable feature:** `DimensionSet.sets_framework()` factory method for the SETS (Social-Ecological-Technological Systems) scoring framework.

### 2. Relationship Normalizer

#### Core Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/relationships/normalizer.py`

Implemented `RelationshipNormalizer` class with:

| Method | Purpose |
|--------|---------|
| `normalize()` | Full normalization pipeline |
| `_cluster_items()` | Semantic clustering using embedding service |
| `_assign_canonical_labels()` | Select canonical form (shortest/frequent/representative/llm) |
| `_apply_normalization_and_merge()` | Apply mappings and merge duplicate relationships |
| `save()` / `load()` | JSON serialization |

**Normalization pipeline:**
1. Extract unique entities (sources + targets) and relationship types
2. Cluster entities semantically using embeddings
3. Cluster relationship types semantically
4. Generate canonical labels for each cluster
5. Apply mappings to relationships
6. Merge relationships with identical (normalized source, type, target) tuples
7. Aggregate descriptions, window_indices, text_ids

#### CLI Integration
**Files:** `relationships_cli.py`, `unified_cli.py`

Added `qa rel normalize` command with full argument support:

```bash
qa rel normalize relationships.csv \
    --entity-threshold 0.85 \
    --type-threshold 0.75 \
    --canonical-method shortest \
    --embedding-model all-MiniLM-L6-v2 \
    --output-format both \
    --save-maps
```

**CLI options:**
- Column mapping (source-col, target-col, type-col, etc.)
- Threshold presets or custom floats
- Canonical method selection (shortest, frequent, representative, llm)
- Output format (csv, json, both)
- Optional entity/type map files

### 3. Causal Analyzer

#### Core Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/relationships/causal.py`

Implemented `CausalAnalyzer` class for classifying relationships as causal or non-causal:

| Method | Purpose |
|--------|---------|
| `analyze()` | Analyze relationships for causal attributes |
| `_analyze_single()` | LLM-based analysis of single relationship |
| `_parse_response()` | Parse LLM JSON response with error handling |
| `save()` / `load()` | JSON serialization |

**Causal attributes extracted:**
- `is_causal` (bool): Whether relationship represents cause-effect
- `polarity` (string): "positive", "negative", or "neutral"
- `certainty` (string): "certain", "likely", or "possible"
- `explicit_vs_implicit` (string): Whether causality is stated or inferred
- `reasoning` (string): LLM's step-by-step analysis

#### Prompt Template
**File:** `qualitative-analysis/src/qualitative_analysis/relationships/prompts/causal_enrichment.txt`

Ported v3 prompt from backend with detailed definitions for:
- Causal vs non-causal relationships (with explicit NOT causal examples)
- Polarity classification
- Certainty levels
- Explicit vs implicit causation

#### CLI Integration
**Files:** `relationships_cli.py`, `unified_cli.py`

Added `qa rel causal` command:

```bash
qa rel causal relationships.csv \
    --model qwen3:8b \
    --temperature 0.3 \
    --max-evidence 3 \
    --output-format both \
    --causal-only
```

**CLI options:**
- Input CSV column mapping (source, target, type, description)
- LLM model and provider configuration
- Temperature control
- Maximum evidence snippets per relationship
- Output format (csv, json, both)
- Filter to causal-only relationships

### 4. Relationship Graph Generator

#### Core Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/relationships/graph.py`

Implemented `RelationshipGraph` class for building and visualizing entity relationship networks:

| Method | Purpose |
|--------|---------|
| `build()` | Build graph from relationships (raw, normalized, or causal) |
| `filter()` | Filter by edge weight, relationship types |
| `to_png()` | Export as PNG with multiple layouts |
| `to_json()` | Export as JSON |
| `to_csv()` | Export as nodes.csv + edges.csv |
| `to_gexf()` | Export as GEXF for Gephi |
| `_compute_semantic_positions()` | UMAP projection of node embeddings |

**Graph features:**
- NetworkX-based graph construction
- Node metrics: degree, in/out degree, betweenness centrality, PageRank
- Edge weights and evidence aggregation
- Optional causal attribute integration
- Multiple layout algorithms: spring, circular, kamada_kawai, semantic (UMAP)

#### CLI Integration
**Files:** `relationships_cli.py`, `unified_cli.py`

Added `qa rel graph` command:

```bash
qa rel graph relationships.csv \
    --layout semantic \
    --output-format csv,json,png,gexf \
    --min-edge-weight 1 \
    --embedding-model all-MiniLM-L6-v2 \
    --show-edge-labels
```

**CLI options:**
- Input CSV column mapping (source, target, type, etc.)
- Layout selection (spring, circular, kamada_kawai, semantic)
- Multiple output formats (csv, json, png, gexf)
- Minimum edge weight filtering
- PNG customization (DPI, figure size, title, edge labels)
- Causal attribute inclusion toggle

### 5. Entity Scorer

#### Core Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/entity/scorer.py`

Implemented `EntityScorer` class for multi-dimensional entity classification:

| Method | Purpose |
|--------|---------|
| `score_entity()` | Score single entity with multiple runs |
| `score_entities()` | Score multiple entities with progress callback |
| `_score_single_run()` | Single LLM scoring run |
| `_aggregate_runs()` | Aggregate runs into statistics |
| `_build_prompt()` | Build scoring prompt from template |
| `_parse_response()` | Parse LLM JSON response |
| `save()` / `load()` | JSON serialization |

**Scoring features:**
- Multi-dimensional scoring (supports any dimension set)
- Uncertainty quantification via multiple runs
- Statistics: mean, median, mode, std_dev, 95% CI, coefficient of variation
- Built-in SETS framework (Social-Ecological-Technological)
- Research context integration

#### Prompt Template
**File:** `qualitative-analysis/src/qualitative_analysis/entity/prompts/entity_scoring_v2.txt`

Ported v2 prompt from backend with:
- Step-by-step reasoning structure
- Dimension definitions with anchors
- JSON output format
- Example for SETS framework

#### CLI Integration
**Files:** `entity_cli.py`, `unified_cli.py`

Added `qa entity score` command:

```bash
qa entity score entities.csv \
    --dimensions sets \
    --num-runs 3 \
    --model gpt-oss:120b \
    --temperature 0.3 \
    --output-format csv,json
```

**CLI options:**
- Input CSV column mapping (entity, context, text_id)
- Dimension set ('sets' or custom JSON file)
- Number of runs for uncertainty estimation
- LLM model and provider configuration
- Research context file (optional)
- Output format (csv, json)

### 6. Entity Visualizer

#### Core Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/entity/visualizer.py`

Implemented `EntityVisualizer` class for visualizing entity scores:

| Method | Purpose |
|--------|---------|
| `generate_ternary_plot()` | Ternary plot for 3-dimension frameworks (SETS) |
| `generate_radar_chart()` | Radar chart for multi-dimension comparisons |
| `_prepare_ternary_data()` | Normalize scores for barycentric coordinates |
| `_barycentric_to_cartesian()` | Convert ternary to x,y coordinates |
| `_draw_triangle()` | Triangle outline with grid lines |
| `_add_numbered_labels()` | Numbered labels with legend (for many entities) |
| `_add_direct_labels()` | Direct text labels next to points |

**Visualization features:**
- Ternary plots for 3-dimension frameworks (SETS)
- Radar charts for any number of dimensions
- Dimension color coding (based on primary dimension)
- Uncertainty visualization via opacity
- Smart label placement with collision avoidance
- Numbered labels with legend for many entities
- Configurable figure size, title, and output path

#### CLI Integration
**Files:** `entity_cli.py`, `unified_cli.py`

Added `qa entity viz` command:

```bash
# Ternary plot (default for SETS)
qa entity viz scored_entities.csv \
    --type ternary \
    --dimensions social,ecological,technological \
    --title "Entity Classifications (SETS)"

# Radar chart
qa entity viz scored_entities.csv \
    --type radar \
    --dimensions social,ecological,technological \
    --max-entities 10
```

**CLI options:**
- Input CSV from `qa entity score` output
- Visualization type (ternary, radar)
- Dimension selection (comma-separated)
- Custom title
- Label control (--no-labels, --direct-labels)
- Figure size configuration
- Maximum entities for radar chart

### 7. Entity Consolidator

#### Core Implementation
**File:** `qualitative-analysis/src/qualitative_analysis/entity/consolidator.py`

Implemented `EntityConsolidator` class for semantic entity deduplication:

| Method | Purpose |
|--------|---------|
| `consolidate()` | Main consolidation pipeline |
| `consolidate_from_csv()` | Consolidate entities from CSV file |
| `_select_canonical()` | Select canonical form for a cluster |
| `save()` / `load()` | JSON serialization |

**Consolidation features:**
- Semantic clustering using embeddings (via `EmbeddingService`)
- Multiple canonical selection methods: shortest, frequent, representative, first
- Configurable similarity threshold (0.0-1.0)
- Frequency-aware consolidation support
- Output: mapping CSV, clusters CSV, full JSON

#### CLI Integration
**Files:** `entity_cli.py`, `unified_cli.py`

Added `qa entity consolidate` command:

```bash
qa entity consolidate entities.csv \
    --threshold 0.85 \
    --canonical-method shortest \
    --embedding-model all-MiniLM-L6-v2 \
    --output-format csv,json
```

**CLI options:**
- Input CSV with entity column
- Similarity threshold (0.0-1.0)
- Canonical method (shortest, frequent, representative, first)
- Optional frequency column for frequency-weighted consolidation
- Embedding model selection
- Output format (csv, json)

### 8. Planning Documentation

**Directory:** `docs/planning/2026-01-15-qa-package-entity-relationship-functionality/`

Created comprehensive planning documents:

| Document | Contents |
|----------|----------|
| `00-overview.md` | Executive summary, feature gap analysis, success criteria |
| `01-implementation-roadmap.md` | Phased implementation plan with workstream architecture |
| `02-cli-design.md` | Detailed CLI specifications for all planned commands |

---

## Test Results

### Normalizer Test
Verified normalizer functionality with sample data:

```
Input: 6 relationships with entity variations
  - "climate change", "Climate Change", "global climate change"
  - "flooding", "Flooding", "flood events"

Output: Correct clustering
  - Entity mapping: variants → canonical forms
  - Type clustering: preserved semantic distinctions
  - Relationship merging: working correctly
```

### Causal Analyzer Test
Verified causal analysis with 15 test relationships:

```
Input: 15 relationships (mix of causal and non-causal)
Output:
  - 13 correctly identified as causal
  - 2 correctly identified as non-causal:
    - "cybersecurity requires secure systems" → prerequisite, not causal
    - "John sibling of Mary" → attribute, not causal
  - Polarity, certainty, explicit/implicit attributes correctly classified
```

### Graph Generator Test
Verified graph generation with test data:

```
Input: 15 relationships from test dataset
Output:
  Graph Summary:
    Nodes: 30
    Edges: 15
    Density: 0.0172
    Average degree: 1.00
    Connected components: 15
    Top nodes (PageRank): flooding, pollution levels, ice cream sales...

  Generated files:
    - test_relationships_causal_graph.json (18.7 KB)
    - test_relationships_causal_graph.png (267 KB)
    - test_relationships_causal_graph_nodes.csv
    - test_relationships_causal_graph_edges.csv
    - test_relationships_causal_graph.gexf
```

### Entity Scorer Test
Verified entity scoring with SETS framework:

```
Input: "renewable energy" with context about community investment
Output (2 runs):
  SETS Scores:
    Social: 75 (community investment, job creation)
    Ecological: 90 (carbon emissions reduction)
    Technological: 85 (infrastructure, engineered systems)

  Statistics: mean, median, std_dev, 95% CI, CV computed
  Output formats: CSV with flattened dimensions, JSON with full detail
```

### Entity Visualizer Test
Verified visualization generation with scored entities:

```
Input: 1 scored entity from Entity Scorer test
Output:
  Ternary plot: 184KB PNG (2186x1508)
    - Correct triangle with grid lines
    - Dimension labels at vertices (Social, Ecological, Technological)
    - Entity point colored by primary dimension
    - Numbered label with legend

  Radar chart: 202KB PNG (1751x1498)
    - 3-axis radar for SETS dimensions
    - Entity polygon showing score profile
    - Y-axis from 0-100
```

### Entity Consolidator Test
Verified semantic consolidation with entity variations:

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
    Clusters: 8

  Merged clusters:
    - "climate change" <- ["climate change", "Climate Change", "global climate change"] (0.92 similarity)
    - "flooding" <- ["flooding", "Flooding", "flood events"] (0.84 similarity)
    - "renewable energy" <- ["renewable energy", "Renewable Energy"] (1.0 similarity)

  Generated files:
    - test_entities_consolidation.json (full result)
    - test_entities_mapping.csv (entity -> canonical)
    - test_entities_clusters.csv (cluster details)
```

---

## Dependencies Updated

**File:** `pyproject.toml`

| Addition | Purpose |
|----------|---------|
| `python-ternary>=1.0.8` | Ternary plots for entity scoring visualization |
| `entity/prompts/*.txt` in build includes | Support for entity scoring prompts |

Core dependencies already present: sentence-transformers, scikit-learn, networkx, matplotlib, umap-learn, hdbscan

---

## New CLI Commands

| Command | Status | Description |
|---------|--------|-------------|
| `qa rel detect` | Existing | Extract entities and relationships |
| `qa rel normalize` | **NEW** | Cluster entities/types, merge relationships |
| `qa rel causal` | **NEW** | Causal attribute classification |
| `qa rel graph` | **NEW** | Network graph with metrics and visualization |
| `qa rel verify` | Planned | LLM verification pass |
| `qa entity score` | **NEW** | Multi-dimensional scoring with uncertainty |
| `qa entity consolidate` | **NEW** | Semantic entity deduplication |
| `qa entity viz` | **NEW** | Ternary plots, radar charts |

---

## File Changes Summary

| File | Change |
|------|--------|
| `core/embeddings.py` | **NEW** - Shared embedding service |
| `relationships/models.py` | **EXTENDED** - 12 new dataclasses (updated with Optional types for causal attrs) |
| `relationships/normalizer.py` | **NEW** - RelationshipNormalizer class |
| `relationships/causal.py` | **NEW** - CausalAnalyzer class |
| `relationships/graph.py` | **NEW** - RelationshipGraph class for network generation |
| `relationships/prompts/causal_enrichment.txt` | **NEW** - Causal analysis prompt template |
| `relationships/__init__.py` | **UPDATED** - Export CausalAnalyzer, RelationshipGraph |
| `relationships_cli.py` | **EXTENDED** - normalize, causal, and graph commands |
| `entity/__init__.py` | **UPDATED** - Export EntityScorer, EntityVisualizer, EntityConsolidator |
| `entity/models.py` | **EXTENDED** - Scoring and consolidation models |
| `entity/scorer.py` | **NEW** - EntityScorer class |
| `entity/visualizer.py` | **NEW** - EntityVisualizer class |
| `entity/consolidator.py` | **NEW** - EntityConsolidator class |
| `entity/prompts/entity_scoring_v2.txt` | **NEW** - Scoring prompt template |
| `entity_cli.py` | **EXTENDED** - Entity CLI commands (score, consolidate, viz) |
| `unified_cli.py` | **UPDATED** - qa rel, qa entity subcommands |
| `pyproject.toml` | **UPDATED** - python-ternary, build includes |

---

## Next Steps

- [x] ~~Implement Causal Analyzer (`qa rel causal`)~~ ✅ Completed
- [x] ~~Implement Relationship Graph generator (`qa rel graph`)~~ ✅ Completed
- [x] ~~Implement Entity Scorer (`qa entity score`)~~ ✅ Completed
- [x] ~~Implement Entity Visualizer (`qa entity viz`)~~ ✅ Completed
- [x] ~~Implement Entity Consolidator (`qa entity consolidate`)~~ ✅ Completed
- [ ] Implement Relationship Verifier (`qa rel verify`)
- [ ] Create pipeline orchestration (`qa rel pipeline`, `qa entity pipeline`)
- [ ] Comprehensive testing with provided datasets

---

## Architecture Diagram

```
qualitative-analysis/src/qualitative_analysis/
├── core/
│   ├── embeddings.py          ✅ NEW - Shared embedding service
│   ├── llm.py                 Existing
│   └── providers.py           Existing
├── entity/                    ✅ NEW MODULE
│   ├── __init__.py            ✅ UPDATED - Export EntityScorer, EntityVisualizer, EntityConsolidator
│   ├── models.py              ✅ EXTENDED - Scoring & consolidation models
│   ├── scorer.py              ✅ NEW - EntityScorer class
│   ├── visualizer.py          ✅ NEW - EntityVisualizer class
│   ├── consolidator.py        ✅ NEW - EntityConsolidator class
│   └── prompts/
│       └── entity_scoring_v2.txt  ✅ NEW - Scoring prompt
├── relationships/
│   ├── models.py              ✅ EXTENDED - Full pipeline models
│   ├── normalizer.py          ✅ NEW - Normalization logic
│   ├── causal.py              ✅ NEW - Causal analysis logic
│   ├── graph.py               ✅ NEW - Graph generation & visualization
│   ├── detector.py            Existing
│   └── prompts/
│       └── causal_enrichment.txt  ✅ NEW - Causal analysis prompt
├── entity_cli.py              ✅ EXTENDED - Entity CLI commands (score, viz)
├── relationships_cli.py       ✅ EXTENDED - normalize, causal, graph commands
└── unified_cli.py             ✅ UPDATED - qa rel + qa entity subcommands
```
