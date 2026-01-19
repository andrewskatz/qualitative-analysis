# Entity/Relationship Functionality: Package Porting Overview

**Date:** 2026-01-15
**Status:** Planning Phase
**Goal:** Port entity/relationship analysis functionality from the fullstack web app backend to the `qualitative-analysis` Python package

---

## Executive Summary

The `qualitative-analysis` package currently has a mature figurative language analysis pipeline (`qa fig detect → map → normalize → graph`) but the entity/relationship pipeline (`qa rel detect`) is limited to basic detection only. The web app backend contains extensive additional functionality that needs to be ported to achieve feature parity and expand capabilities.

This document outlines the comprehensive porting effort to bring the following capabilities to the CLI package:
- Entity normalization/consolidation
- Entity scoring with uncertainty quantification
- Relationship normalization
- Relationship verification
- Causal relationship classification
- Graph generation and visualization
- Multi-format export

---

## Current State Assessment

### Figurative Language Pipeline (Reference Model) ✅ Complete

```
qa fig detect    →    qa fig map    →    qa fig normalize    →    qa fig graph
     ↓                     ↓                    ↓                      ↓
Detect metaphors,    Extract source/     Cluster similar        Generate network
similes, etc.        target domains      domains semantically   visualization
```

**Features:**
- Multi-level domain extraction (specific/moderate/abstract)
- Semantic clustering with embeddings
- LLM-generated canonical labels
- UMAP semantic layout
- HDBSCAN clustering for visualization
- Checkpoint/resume support
- Multiple export formats (JSON, CSV, PNG)

### Entity/Relationship Pipeline 🔄 Partially Implemented

**In Package (Current):**
```
qa rel detect    →    ???    →    ???    →    ???
     ↓
Extract entities
& relationships
```

**In Backend (To Port):**
- Entity consolidation service
- Entity scoring service (multi-run, multi-dimensional)
- Relationship normalizer
- Relationship verifier
- Causal analysis service
- Graph aggregation with NetworkX metrics
- Visualization service (ternary plots, etc.)
- Export services (CSV, GEXF)

---

## Feature Gap Analysis

| Feature Category | Figurative Language | Relationships (Package) | Relationships (Backend) |
|-----------------|---------------------|------------------------|------------------------|
| **Detection** | ✅ Full | ✅ Basic | ✅ Full |
| **Multi-level extraction** | ✅ 3 levels | ❌ | ❌ |
| **Entity consolidation** | N/A | ❌ | ✅ Semantic clustering |
| **Entity scoring** | N/A | ❌ | ✅ Multi-run + uncertainty |
| **Type normalization** | ✅ Domains | ❌ | ✅ Relationship types |
| **Verification** | N/A | ❌ | ✅ LLM second-pass |
| **Causal analysis** | N/A | ❌ | ✅ Full pipeline |
| **Graph generation** | ✅ Full | ❌ | ✅ Full |
| **Graph metrics** | ✅ Basic | ❌ | ✅ NetworkX (degree, betweenness, PageRank) |
| **Semantic layout** | ✅ UMAP+PCA | ❌ | Partial |
| **Cluster visualization** | ✅ HDBSCAN | ❌ | ❌ |
| **Search (string/semantic)** | ❌ | ❌ | ✅ Full |
| **Export (CSV)** | ✅ | ❌ | ✅ |
| **Export (GEXF)** | ❌ | ❌ | ✅ |
| **Checkpoints** | ✅ | ❌ | ❌ |
| **Progress callbacks** | ✅ | ❌ | ✅ (via jobs) |

---

## Target CLI Structure

### Current Commands
```bash
qa fig detect <input.csv>           # Figurative language detection
qa fig map <instances.csv>          # Domain extraction
qa fig normalize <mappings.csv>     # Domain normalization
qa fig graph <normalized.csv>       # Domain graph visualization

qa rel detect <input.csv>           # Relationship extraction (BASIC)
```

### Proposed New Commands
```bash
# Relationship Analysis Pipeline
qa rel detect <input.csv>           # Extract entities & relationships (EXISTS)
qa rel normalize <results.csv>      # NEW: Consolidate entities & normalize types
qa rel verify <results.csv>         # NEW: LLM verification pass
qa rel causal <results.csv>         # NEW: Classify causal attributes
qa rel graph <results.csv>          # NEW: Generate relationship network
qa rel pipeline <input.csv>         # NEW: Run full pipeline

# Entity Analysis (New Subcommand Group)
qa entity score <results.csv>       # NEW: Multi-dimensional entity scoring
qa entity consolidate <results.csv> # NEW: Deduplicate/merge entities
qa entity viz <scores.csv>          # NEW: Visualization (ternary, radar, etc.)
```

---

## Backend Services to Port

### 1. EntityConsolidationService
**Source:** `backend/services/entity_consolidation_service.py`
**Purpose:** Semantic deduplication of entities using embeddings

**Key Functions:**
- `propose_merges(entities, threshold)` - Cluster similar entities
- `compute_similarity(text1, text2)` - Pairwise semantic similarity

**Technical Details:**
- Embedding model: `Qwen/Qwen3-Embedding-0.6B` (configurable)
- Clustering: Agglomerative with cosine distance
- Canonical selection: Shortest string heuristic

### 2. EntityScoringService
**Source:** `backend/services/entity_scoring_service.py`
**Purpose:** Multi-dimensional scoring with uncertainty quantification

**Key Functions:**
- `score_entity(entity, context, dimensions, num_runs)` - Multi-run scoring
- `_aggregate_runs(runs, dimensions)` - Statistical aggregation

**Output Structure:**
```python
{
    "entity": str,
    "aggregated": {
        "dimension_name": {
            "mean": float,
            "median": float,
            "std_dev": float,
            "confidence_interval_95": [float, float],
            "coefficient_of_variation": float,
            "justification": str
        }
    }
}
```

### 3. RelationshipNormalizer
**Source:** `backend/services/relationship_normalizer.py`
**Purpose:** Normalize relationship types via semantic clustering

**Pipeline:**
1. Fuzzy entity grouping
2. Relationship type embedding
3. Agglomerative clustering
4. Canonical type selection

### 4. RelationshipVerifier
**Source:** `backend/services/relationship_verifier.py`
**Purpose:** Second-pass LLM verification of extracted relationships

**Output:** Adds `verification_confidence` and `verification_note` to relationships

### 5. CausalAnalysisService
**Source:** `backend/services/causal_analysis_service.py`
**Purpose:** Classify relationships for causal attributes

**Causal Attributes:**
- `is_causal`: bool
- `polarity`: "positive" | "negative" | "neutral"
- `certainty`: "certain" | "possible" | "uncertain"
- `explicit_vs_implicit`: "explicit" | "implicit"
- `explanation`: str

### 6. RelationshipAnalysisService (Graph Aggregation)
**Source:** `backend/services/relationship_analysis_service.py`
**Purpose:** Build and analyze relationship networks

**Graph Features:**
- Node metrics: degree, betweenness centrality, PageRank
- Edge metrics: weight, evidence aggregation
- Global metrics: density, components, connectivity

### 7. RelationshipGraphExportService
**Source:** `backend/services/relationship_graph_export_service.py`
**Purpose:** Export to multiple formats

**Formats:**
- JSON (full graph data)
- CSV (nodes.csv, edges.csv)
- GEXF (Gephi compatible)

### 8. VisualizationService
**Source:** `backend/services/visualization_service.py`
**Purpose:** Generate visualizations for scored entities

**Visualization Types:**
- Ternary plot (3-dimensional scores)
- Radar chart (multi-dimensional)
- Heatmap (entity × dimension)
- Box plot (score distributions)
- Confidence intervals

---

## Data Models to Create

### Relationship Pipeline Models

```python
@dataclass
class NormalizedRelationship:
    source: str                    # Entity A (normalized)
    target: str                    # Entity B (normalized)
    type: str                      # Relationship type (normalized)
    original_type: str             # Original type before normalization
    description: str               # Merged descriptions
    confidence: float              # Merge confidence
    count: int                     # Number of merged instances
    window_indices: List[int]      # Source windows
    text_ids: List[str]            # Source text IDs

@dataclass
class VerifiedRelationship(NormalizedRelationship):
    verification_confidence: float
    verification_note: Optional[str]

@dataclass
class CausalRelationship(VerifiedRelationship):
    is_causal: bool
    polarity: str                  # positive/negative/neutral
    certainty: str                 # certain/possible/uncertain
    explicit_vs_implicit: str      # explicit/implicit
    causal_explanation: str

@dataclass
class RelationshipGraphNode:
    id: str
    label: str
    frequency: int
    text_count: int
    degree: int
    in_degree: int
    out_degree: int
    betweenness: float
    pagerank: float
    snippets: List[dict]

@dataclass
class RelationshipGraphEdge:
    source: str
    target: str
    type: str
    weight: int
    text_count: int
    evidence: List[str]
    causal_attributes: Optional[dict]

@dataclass
class RelationshipGraphData:
    nodes: List[RelationshipGraphNode]
    edges: List[RelationshipGraphEdge]
    metrics_summary: dict
```

### Entity Scoring Models

```python
@dataclass
class DimensionDefinition:
    name: str
    description: str
    min_anchor: str                # Description of minimum score
    max_anchor: str                # Description of maximum score
    scale_min: int = 0
    scale_max: int = 100

@dataclass
class DimensionScore:
    dimension: str
    mean: float
    median: float
    mode: int
    std_dev: float
    confidence_interval_95: Tuple[float, float]
    coefficient_of_variation: float
    scores: List[int]              # Raw scores from each run
    justification: str

@dataclass
class EntityScore:
    entity: str
    text_id: str
    context: str
    dimension_scores: Dict[str, DimensionScore]
    processing_time_ms: float
```

---

## Dependencies to Add

```toml
[project.optional-dependencies]
relationships = [
    "sentence-transformers>=2.2.0",  # Embeddings
    "scikit-learn>=1.0.0",           # Clustering
    "networkx>=3.0",                 # Graph analytics
    "adjustText>=1.0.0",             # Label placement (already in viz)
]

entity = [
    "sentence-transformers>=2.2.0",  # Embeddings
]

viz = [
    # Existing dependencies...
    "python-ternary>=1.0.0",         # Ternary plots
]
```

---

## Related Documentation

- [01-implementation-roadmap.md](./01-implementation-roadmap.md) - Phased implementation plan
- [02-cli-design.md](./02-cli-design.md) - Detailed CLI command specifications
- [03-data-models.md](./03-data-models.md) - Complete data model definitions
- [04-prompts.md](./04-prompts.md) - LLM prompt templates
- [05-testing-strategy.md](./05-testing-strategy.md) - Test plan

---

## Success Criteria

1. **Feature Parity:** All backend entity/relationship features available via CLI
2. **Pipeline Integration:** Commands chain seamlessly with nested output structure
3. **Consistent UX:** CLI mirrors figurative language pattern (`detect → normalize → graph`)
4. **Performance:** Efficient processing with checkpoint/resume support
5. **Documentation:** Comprehensive CLI help and examples
