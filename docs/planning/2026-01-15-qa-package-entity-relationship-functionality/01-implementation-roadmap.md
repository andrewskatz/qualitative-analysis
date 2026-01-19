# Implementation Roadmap: Entity/Relationship Package Porting

**Date:** 2026-01-15
**Approach:** Parallel development of all major feature areas

---

## Overview

This roadmap covers porting entity/relationship functionality from the backend to the `qualitative-analysis` package. Features are organized into workstreams that can be developed in parallel, with dependencies noted where sequential work is required.

---

## Workstream Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                   FOUNDATION LAYER                       │
                    │  - Enhanced data models                                  │
                    │  - Embedding service abstraction                         │
                    │  - Prompt templates & loaders                            │
                    └─────────────────────────────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│   ENTITY WORKSTREAM     │   │ RELATIONSHIP WORKSTREAM │   │   CAUSAL WORKSTREAM     │
│                         │   │                         │   │                         │
│ • Entity consolidation  │   │ • Rel normalization     │   │ • Causal classification │
│ • Entity scoring        │   │ • Rel verification      │   │ • Causal attributes     │
│ • Scoring aggregation   │   │ • Type clustering       │   │ • Statistics            │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                 GRAPH & VISUALIZATION                    │
                    │  - Graph aggregation with NetworkX                       │
                    │  - Network visualization                                 │
                    │  - Entity scoring visualization                          │
                    │  - Multi-format export                                   │
                    └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                    CLI INTEGRATION                       │
                    │  - qa rel normalize/verify/causal/graph commands         │
                    │  - qa entity score/consolidate/viz commands              │
                    │  - Pipeline orchestration                                │
                    └─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation Layer

**Duration:** Initial setup
**Dependency:** None (start here)

### 1.1 Enhanced Data Models

**File:** `qualitative-analysis/src/qualitative_analysis/relationships/models.py`

Extend existing models to support full pipeline:

```python
# Existing (keep)
@dataclass
class Relationship:
    source: str
    target: str
    type: str
    description: str
    window_index: int

# New additions
@dataclass
class NormalizedEntity:
    canonical: str
    variants: List[str]
    confidence: float
    frequency: int

@dataclass
class NormalizedRelationship:
    source: str
    target: str
    type: str
    original_source: str
    original_target: str
    original_type: str
    description: str
    confidence: float
    count: int
    window_indices: List[int]
    text_ids: List[str]

@dataclass
class CausalAttributes:
    is_causal: bool
    polarity: str  # positive/negative/neutral
    certainty: str  # certain/possible/uncertain
    explicit_vs_implicit: str
    explanation: str

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
    snippets: List[Dict]

@dataclass
class RelationshipGraphEdge:
    source: str
    target: str
    type: str
    weight: int
    text_count: int
    evidence: List[str]
    causal_attributes: Optional[CausalAttributes]

@dataclass
class RelationshipGraphData:
    nodes: List[RelationshipGraphNode]
    edges: List[RelationshipGraphEdge]
    metrics_summary: Dict
    is_normalized: bool = False
```

**File:** `qualitative-analysis/src/qualitative_analysis/entity/models.py` (NEW)

```python
@dataclass
class DimensionDefinition:
    name: str
    description: str
    min_anchor: str
    max_anchor: str
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
    scores: List[int]
    justification: str

@dataclass
class EntityScore:
    entity: str
    text_id: str
    context: str
    dimension_scores: Dict[str, DimensionScore]
    num_runs: int
    processing_time_ms: float
```

### 1.2 Embedding Service Abstraction

**File:** `qualitative-analysis/src/qualitative_analysis/core/embeddings.py` (NEW)

Create a shared embedding service for use across modules:

```python
class EmbeddingService:
    """Shared embedding service for semantic operations."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        self.model = SentenceTransformer(model_name)
        self.device = device or self._detect_device()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""

    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""

    def compute_batch_similarity(
        self,
        query: str,
        candidates: List[str]
    ) -> List[float]:
        """Compute similarity between query and multiple candidates."""

    def cluster(
        self,
        texts: List[str],
        threshold: float = 0.75,
        method: str = "agglomerative"
    ) -> List[List[int]]:
        """Cluster texts by semantic similarity."""
```

### 1.3 Prompt Templates

**Directory:** `qualitative-analysis/src/qualitative_analysis/relationships/prompts/`

Add new prompts (port from backend):
- `relationship_verification_v1.txt`
- `causal_enrichment_v1.txt`, `v2.txt`, `v3.txt`

**Directory:** `qualitative-analysis/src/qualitative_analysis/entity/prompts/` (NEW)
- `entity_scoring_v1.txt`, `v2.txt`
- `cluster_labeling_v1.txt`

---

## Phase 2: Entity Workstream

**Dependency:** Phase 1 (Foundation)

### 2.1 Entity Consolidation

**File:** `qualitative-analysis/src/qualitative_analysis/entity/consolidator.py` (NEW)

Port from `backend/services/entity_consolidation_service.py`:

```python
class EntityConsolidator:
    """Semantic deduplication of entities using embeddings."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        threshold: float = 0.85
    ):
        self.embeddings = embedding_service
        self.threshold = threshold

    def propose_merges(
        self,
        entities: List[str]
    ) -> List[NormalizedEntity]:
        """Propose entity consolidations based on semantic similarity."""

    def apply_merges(
        self,
        entities: List[str],
        merge_map: Dict[str, str]
    ) -> List[str]:
        """Apply a merge map to a list of entities."""
```

**Key Algorithm:**
1. Generate embeddings for all entities
2. Compute pairwise cosine similarity
3. Apply agglomerative clustering with threshold
4. Select canonical form (shortest string heuristic)
5. Return merge proposals with confidence scores

### 2.2 Entity Scoring

**File:** `qualitative-analysis/src/qualitative_analysis/entity/scorer.py` (NEW)

Port from `backend/services/entity_scoring_service.py`:

```python
class EntityScorer:
    """Multi-dimensional entity scoring with uncertainty quantification."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        dimensions: List[DimensionDefinition],
        num_runs: int = 3
    ):
        self.llm = llm_provider
        self.dimensions = dimensions
        self.num_runs = num_runs

    async def score_entity(
        self,
        entity: str,
        context: str
    ) -> EntityScore:
        """Score an entity on all dimensions with multiple runs."""

    def aggregate_runs(
        self,
        runs: List[Dict]
    ) -> Dict[str, DimensionScore]:
        """Aggregate multiple scoring runs into statistics."""
```

**Scoring Pipeline:**
1. Build prompt with entity, context, dimension definitions
2. Run LLM `num_runs` times for uncertainty estimation
3. Parse JSON responses (score 0-100 + justification per dimension)
4. Aggregate: mean, median, mode, std_dev, CI, CV
5. Return EntityScore with full statistics

### 2.3 CLI Commands

**File:** `qualitative-analysis/src/qualitative_analysis/entity_cli.py` (NEW)

```bash
# Entity consolidation
qa entity consolidate results.csv \
    --threshold 0.85 \
    --embedding-model all-MiniLM-L6-v2 \
    --output consolidation_map.json

# Entity scoring
qa entity score results.csv \
    --dimensions dimensions.json \
    --num-runs 3 \
    --context-mode window \
    --model qwen3:30b \
    --output scores.csv
```

---

## Phase 3: Relationship Workstream

**Dependency:** Phase 1 (Foundation)

### 3.1 Relationship Normalizer

**File:** `qualitative-analysis/src/qualitative_analysis/relationships/normalizer.py` (NEW)

Port from `backend/services/relationship_normalizer.py`:

```python
class RelationshipNormalizer:
    """Normalize entities and relationship types via semantic clustering."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        entity_threshold: float = 0.85,
        type_threshold: float = 0.75
    ):
        self.embeddings = embedding_service
        self.entity_threshold = entity_threshold
        self.type_threshold = type_threshold

    def normalize(
        self,
        relationships: List[Relationship]
    ) -> Tuple[List[NormalizedRelationship], Dict]:
        """Full normalization pipeline."""

    def _group_entities(
        self,
        relationships: List[Relationship]
    ) -> Dict[str, str]:
        """Fuzzy match entities to canonical forms."""

    def _normalize_types(
        self,
        relationships: List[Relationship]
    ) -> Dict[str, str]:
        """Cluster relationship types semantically."""
```

**Normalization Pipeline:**
1. Extract all unique entities (sources + targets)
2. Cluster entities semantically → entity_map
3. Extract all unique relationship types
4. Cluster types semantically → type_map
5. Apply maps to relationships
6. Merge duplicate (source, target, type) tuples
7. Aggregate counts, descriptions, text_ids

### 3.2 Relationship Verifier

**File:** `qualitative-analysis/src/qualitative_analysis/relationships/verifier.py` (NEW)

Port from `backend/services/relationship_verifier.py`:

```python
class RelationshipVerifier:
    """Second-pass LLM verification of extracted relationships."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        support_threshold: float = 0.5
    ):
        self.llm = llm_provider
        self.threshold = support_threshold

    async def verify(
        self,
        text: str,
        relationships: List[Relationship]
    ) -> List[VerifiedRelationship]:
        """Verify relationships against source text."""
```

**Verification Process:**
1. Format relationships as (source, type, target) tuples
2. Prompt LLM with evidence text
3. LLM returns confidence (0-1) + optional corrections
4. Filter by support threshold
5. Return VerifiedRelationship with confidence and notes

### 3.3 CLI Commands

**File:** Update `qualitative-analysis/src/qualitative_analysis/relationships_cli.py`

```bash
# Normalize relationships
qa rel normalize results.csv \
    --entity-threshold 0.85 \
    --type-threshold 0.75 \
    --embedding-model all-MiniLM-L6-v2 \
    --output normalized.csv

# Verify relationships
qa rel verify results.csv \
    --source-text-col text \
    --threshold 0.5 \
    --model qwen3:30b \
    --output verified.csv
```

---

## Phase 4: Causal Workstream

**Dependency:** Phase 3 (Relationship normalization recommended first)

### 4.1 Causal Analyzer

**File:** `qualitative-analysis/src/qualitative_analysis/relationships/causal.py` (NEW)

Port from `backend/services/causal_analysis_service.py`:

```python
class CausalAnalyzer:
    """Analyze relationships for causal attributes."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_version: str = "v3"
    ):
        self.llm = llm_provider
        self.prompt_version = prompt_version

    async def analyze_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        evidence: List[str]
    ) -> CausalAttributes:
        """Analyze a single relationship for causal attributes."""

    async def analyze_all(
        self,
        relationships: List[Relationship],
        progress_callback: Optional[Callable] = None
    ) -> List[CausalRelationship]:
        """Analyze all relationships with progress reporting."""

    def compute_statistics(
        self,
        results: List[CausalRelationship]
    ) -> Dict:
        """Compute summary statistics for causal analysis."""
```

**Causal Attributes Schema:**
- `is_causal`: Whether the relationship expresses causation
- `polarity`: Direction of effect (positive/negative/neutral)
- `certainty`: Confidence level (certain/possible/uncertain)
- `explicit_vs_implicit`: Whether causation is stated or implied
- `explanation`: LLM reasoning for classification

### 4.2 CLI Command

```bash
# Causal analysis
qa rel causal normalized.csv \
    --evidence-col description \
    --model qwen3:30b \
    --prompt-version v3 \
    --output causal.csv
```

---

## Phase 5: Graph & Visualization

**Dependency:** Phases 2, 3, 4 (can start visualization while others complete)

### 5.1 Relationship Graph Generator

**File:** `qualitative-analysis/src/qualitative_analysis/relationships/graph.py` (NEW)

Port from `backend/services/relationship_analysis_service.py` (graph aggregation):

```python
class RelationshipGraph:
    """Build and analyze relationship networks."""

    def __init__(self):
        self.nodes: Dict[str, RelationshipGraphNode] = {}
        self.edges: Dict[str, RelationshipGraphEdge] = {}
        self.nx_graph: Optional[nx.DiGraph] = None

    def build_from_relationships(
        self,
        relationships: List[Relationship],
        include_causal: bool = False
    ) -> RelationshipGraphData:
        """Build graph from relationship list."""

    def compute_metrics(self) -> Dict:
        """Compute NetworkX graph metrics."""

    def filter(
        self,
        min_edge_weight: int = 1,
        relationship_types: Optional[List[str]] = None,
        include_isolated: bool = True
    ) -> 'RelationshipGraph':
        """Filter graph by criteria."""

    def search(
        self,
        query: str,
        mode: str = "string",  # or "semantic"
        scope: str = "both"    # nodes, edges, or both
    ) -> List[Dict]:
        """Search graph nodes and edges."""
```

**Graph Metrics (via NetworkX):**
- Node: degree, in_degree, out_degree, betweenness centrality, PageRank
- Global: density, average degree, connected components, is_connected

### 5.2 Graph Visualization

**File:** Extend `qualitative-analysis/src/qualitative_analysis/relationships/graph.py`

```python
class RelationshipGraphVisualizer:
    """Visualize relationship networks."""

    def render_png(
        self,
        graph_data: RelationshipGraphData,
        layout: str = "spring",  # spring, circular, kamada_kawai, semantic
        output_path: str = "graph.png",
        **kwargs
    ) -> str:
        """Render graph as PNG image."""

    def render_clustered(
        self,
        graph_data: RelationshipGraphData,
        min_cluster_size: int = 3,
        label_clusters: bool = True,
        llm_provider: Optional[LLMProvider] = None,
        **kwargs
    ) -> str:
        """Render with cluster visualization (HDBSCAN + hulls)."""
```

**Layouts:**
- `spring`: Force-directed (NetworkX spring_layout)
- `circular`: Circular arrangement
- `kamada_kawai`: Energy minimization
- `semantic`: UMAP projection of node embeddings (like figurative)

### 5.3 Graph Export

**File:** Extend `qualitative-analysis/src/qualitative_analysis/relationships/graph.py`

```python
class RelationshipGraphExporter:
    """Export relationship graphs to multiple formats."""

    def export_json(
        self,
        graph_data: RelationshipGraphData,
        output_path: str
    ) -> str:
        """Export full graph as JSON."""

    def export_csv(
        self,
        graph_data: RelationshipGraphData,
        output_dir: str
    ) -> Tuple[str, str]:
        """Export as nodes.csv and edges.csv."""

    def export_gexf(
        self,
        graph_data: RelationshipGraphData,
        output_path: str
    ) -> str:
        """Export as GEXF for Gephi."""
```

### 5.4 Entity Scoring Visualization

**File:** `qualitative-analysis/src/qualitative_analysis/entity/visualizer.py` (NEW)

Port from `backend/services/visualization_service.py`:

```python
class EntityScoreVisualizer:
    """Generate visualizations for scored entities."""

    def ternary_plot(
        self,
        scores: List[EntityScore],
        dimensions: List[str],  # Exactly 3 dimensions
        output_path: str,
        **kwargs
    ) -> str:
        """Generate ternary plot for 3-dimensional scores."""

    def radar_chart(
        self,
        scores: List[EntityScore],
        output_path: str,
        **kwargs
    ) -> str:
        """Generate radar chart for multi-dimensional scores."""

    def heatmap(
        self,
        scores: List[EntityScore],
        output_path: str,
        **kwargs
    ) -> str:
        """Generate entity × dimension heatmap."""
```

### 5.5 CLI Commands

```bash
# Generate relationship graph
qa rel graph results.csv \
    --layout semantic \
    --cluster-labels \
    --min-cluster-size 3 \
    --format png,json,gexf \
    --output output/

# Entity visualization
qa entity viz scores.csv \
    --type ternary \
    --dimensions "Social,Ecological,Technological" \
    --output ternary.png
```

---

## Phase 6: CLI Integration & Pipeline

**Dependency:** All previous phases

### 6.1 Update Unified CLI

**File:** Update `qualitative-analysis/src/qualitative_analysis/unified_cli.py`

Add new subparsers:
- `qa rel normalize`
- `qa rel verify`
- `qa rel causal`
- `qa rel graph`
- `qa rel pipeline`
- `qa entity consolidate`
- `qa entity score`
- `qa entity viz`

### 6.2 Pipeline Orchestration

**File:** `qualitative-analysis/src/qualitative_analysis/relationships/pipeline.py` (NEW)

```python
async def run_relationship_pipeline(
    input_csv: str,
    output_dir: str,
    config: PipelineConfig
) -> Dict:
    """Run full relationship analysis pipeline."""

    # Step 1: Detect
    detect_results = await detect(...)

    # Step 2: Normalize (optional)
    if config.normalize:
        normalized = await normalize(detect_results, ...)

    # Step 3: Verify (optional)
    if config.verify:
        verified = await verify(normalized, ...)

    # Step 4: Causal (optional)
    if config.causal:
        causal = await analyze_causal(verified, ...)

    # Step 5: Graph
    graph = build_graph(causal or verified or normalized, ...)

    # Step 6: Export
    export_all(graph, config.formats, output_dir)

    return {"status": "complete", "outputs": {...}}
```

```bash
# Full pipeline
qa rel pipeline input.csv \
    --normalize \
    --verify \
    --causal \
    --graph \
    --layout semantic \
    --output output/
```

---

## Phase 7: Testing & Documentation

**Dependency:** All implementation complete

### 7.1 Unit Tests
- Test each component in isolation
- Mock LLM responses for deterministic tests
- Test edge cases (empty inputs, malformed data)

### 7.2 Integration Tests
- Test full pipeline end-to-end
- Test CSV input/output roundtrip
- Test checkpoint/resume functionality

### 7.3 Documentation
- Update README with new commands
- Add usage examples
- Document configuration options
- Create tutorial notebooks

---

## File Structure Summary

```
qualitative-analysis/src/qualitative_analysis/
├── core/
│   ├── embeddings.py          # NEW: Shared embedding service
│   ├── llm.py                 # Existing
│   └── ...
├── entity/                    # NEW: Entity analysis module
│   ├── __init__.py
│   ├── models.py              # Entity scoring data models
│   ├── consolidator.py        # Entity deduplication
│   ├── scorer.py              # Multi-dimensional scoring
│   ├── visualizer.py          # Ternary plots, etc.
│   └── prompts/
│       └── entity_scoring_v1.txt, v2.txt
├── relationships/
│   ├── models.py              # EXTEND: Add graph models
│   ├── normalizer.py          # NEW: Relationship normalization
│   ├── verifier.py            # NEW: LLM verification
│   ├── causal.py              # NEW: Causal analysis
│   ├── graph.py               # NEW: Graph generation & viz
│   ├── pipeline.py            # NEW: Pipeline orchestration
│   └── prompts/
│       ├── relationship_verification_v1.txt  # NEW
│       └── causal_enrichment_v1.txt, ...     # NEW
├── entity_cli.py              # NEW: Entity CLI commands
├── relationships_cli.py       # EXTEND: Add new commands
└── unified_cli.py             # EXTEND: Add entity subcommand
```

---

## Parallel Development Strategy

Since you want to tackle all areas together, here's how work can be parallelized:

| Workstream | Can Start | Depends On |
|------------|-----------|------------|
| Foundation (models, embeddings) | Immediately | Nothing |
| Entity consolidation | After foundation | Foundation |
| Entity scoring | After foundation | Foundation |
| Relationship normalization | After foundation | Foundation |
| Relationship verification | After foundation | Foundation |
| Causal analysis | After foundation | Foundation |
| Graph generation | After normalization | Normalization (soft) |
| Graph visualization | After graph gen | Graph generation |
| Entity visualization | After scoring | Scoring |
| CLI integration | After all features | All features |
| Pipeline orchestration | After CLI | CLI integration |

**Recommended parallel pairs:**
1. Entity consolidation + Relationship normalization (both use embeddings)
2. Entity scoring + Causal analysis (both use LLM)
3. Graph generation + Entity visualization (both use matplotlib/networkx)
