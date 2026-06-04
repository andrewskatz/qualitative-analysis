# Implementation Plan: Domain Mapping & Normalization for qualitative-analysis CLI

**Date:** 2026-01-06  
**Estimated Effort:** 2-3 weeks  
**Priority:** High

---

## Goal Description

Extend the `qualitative-analysis` Python package to include **post-detection domain mapping features** that currently exist only in the web application. This enables users to:

1. **Extract conceptual domains** from detected figurative language (source/target domains with multi-level tagging)
2. **Normalize domains** by clustering semantically similar labels using embeddings
3. **Generate domain graphs** showing source→target conceptual mappings
4. **Filter figurative types** during detection (enhancement to existing CLI)

The target architecture places domain functionality under `qualitative_analysis/figurative/domains/` as it's semantically coupled to figurative language analysis.

---

## User Review Required

> [!IMPORTANT]
> **Checkpoint/Resume Scope**: The handoff requests progress monitoring and pause/resume with checkpoint files. Please confirm:
> - Should checkpoints be JSON files saved periodically during processing?
> - Should we support resuming from partial results if interrupted mid-batch?

> [!NOTE]
> **Normalization State Files**: The plan includes saving/loading normalization runs as JSON files. This enables users to reuse normalization configurations across different datasets.

---

## Proposed Changes

### Phase 1: Figurative Type Filtering (Enhancement)

Enhance existing figurative detection CLI to support filtering by type.

---

#### [cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/cli.py)

- Add `--types` argument accepting comma-separated figurative types (e.g., `metaphor,analogy,extended_metaphor`)
- Pass types to `FigurativeDetector` constructor
- Default: all types (current behavior)

```python
parser.add_argument(
    "--types",
    default=None,
    help="Comma-separated figurative types to detect (e.g., 'metaphor,analogy'). Default: all types."
)
```

---

#### [detector.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/detector.py)

- Add `figurative_types: Optional[List[str]]` parameter
- Pass to strategy for prompt customization

---

#### [two_step.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/strategies/two_step.py)

- Accept `figurative_types` parameter
- Pass to prompt loader to generate type-specific prompt sections

---

#### [NEW] Updated prompt templates

- Modify `two_step_binary_detection_v1.txt` and `two_step_instance_extraction_v1.txt` to use `{figurative_types_section}` template variable
- Generate type-specific background information dynamically

---

### Phase 2: Domain Module Structure

Create the new `domains/` submodule within `figurative/`.

---

#### [NEW] [__init__.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/domains/__init__.py)

```python
from .extractor import DomainExtractor
from .normalizer import DomainNormalizer
from .graph import DomainGraph
from .models import DomainMappedInstance, NormalizationResult, DomainGraphData

__all__ = [
    "DomainExtractor",
    "DomainNormalizer", 
    "DomainGraph",
    "DomainMappedInstance",
    "NormalizationResult",
    "DomainGraphData",
]
```

---

#### [NEW] [models.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/domains/models.py)

Define dataclasses for domain mapping:

```python
@dataclass
class DomainMappedInstance:
    """A figurative instance with extracted domains."""
    # Original instance fields
    text: str
    type: str
    confidence: float
    explanation: str
    window_index: int
    # Domain fields
    source_domain: str = ""
    target_domain: str = ""
    mapping_explanation: str = ""
    # Multi-level domains
    source_domain_levels: Dict[str, str] = field(default_factory=dict)
    target_domain_levels: Dict[str, str] = field(default_factory=dict)
    # Metadata
    raw_response: Optional[dict] = None

@dataclass
class NormalizationResult:
    """Result from domain normalization."""
    source_mapping: Dict[str, str]  # original -> canonical
    target_mapping: Dict[str, str]
    source_clusters: List[DomainCluster]
    target_clusters: List[DomainCluster]
    config: dict

@dataclass 
class DomainCluster:
    """A cluster of semantically similar domains."""
    canonical: str
    members: List[str]
    count: int
    avg_similarity: float

@dataclass
class DomainGraphData:
    """Domain relationship graph."""
    nodes: List[DomainGraphNode]
    edges: List[DomainGraphEdge]
    stats: dict
```

---

### Phase 3: Domain Extraction

Implement the `DomainExtractor` class.

---

#### [NEW] [extractor.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/domains/extractor.py)

Core extraction class ported from backend service:

```python
class DomainExtractor:
    def __init__(
        self,
        model_name: str,
        provider: str = "ollama",
        provider_config: Optional[Dict] = None,
        prompt_version: str = "v3-multilevel",
        multi_level: bool = True,
    ):
        ...

    async def extract(
        self,
        instances: List[Instance],
        window_texts: Optional[Dict[int, str]] = None,
    ) -> List[DomainMappedInstance]:
        """Extract domains for each instance."""
        ...

    async def extract_from_csv(
        self,
        input_path: Path,
        text_col: str = "instance_text",
        type_col: str = "type",
        window_col: Optional[str] = "window_text",
    ) -> List[DomainMappedInstance]:
        """Extract domains from CSV of figurative instances."""
        ...
```

Key methods:
- `extract()`: Process list of instances
- `extract_from_csv()`: CSV input workflow  
- `_analyze_single()`: LLM call for one instance
- `_parse_response()`: Parse LLM JSON output (port from backend)

---

#### [NEW] Prompt templates in `domains/prompts/`

Port prompts from backend `_get_prompt()`:
- `domain_extraction_v1.txt` — Basic source/target
- `domain_extraction_v2.txt` — Enhanced with reasoning
- `domain_extraction_v3_multilevel.txt` — Three abstraction levels (specific/moderate/abstract)

---

### Phase 4: Domain Normalization

Implement clustering and canonical label generation.

---

#### [NEW] [normalizer.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/domains/normalizer.py)

```python
class DomainNormalizer:
    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
    ):
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

    def normalize(
        self,
        instances: List[DomainMappedInstance],
        conservativeness: str = "moderate",  # conservative/moderate/aggressive/custom
        similarity_threshold: Optional[float] = None,
        cluster_mode: str = "separate",  # separate/together
        canonical_method: str = "representative",  # llm/representative
        abstraction_level: Optional[str] = None,  # specific/moderate/abstract
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> NormalizationResult:
        ...

    def save(self, result: NormalizationResult, path: Path) -> None:
        """Save normalization state to JSON file."""
        ...

    def load(self, path: Path) -> NormalizationResult:
        """Load normalization state from JSON file."""
        ...

    def _cluster_domains(
        self, domains: List[str], threshold: float
    ) -> List[DomainCluster]:
        """Cluster using AgglomerativeClustering with cosine distance."""
        ...

    def _generate_canonical_labels_representative(
        self, clusters: List[DomainCluster]
    ) -> List[DomainCluster]:
        """Pick most central member as canonical label."""
        ...

    def _generate_canonical_labels_llm(
        self, clusters: List[DomainCluster], model: str, provider: str
    ) -> List[DomainCluster]:
        """Use LLM to generate abstract canonical labels."""
        ...
```

Threshold mapping (from backend):
| Conservativeness | Threshold |
|------------------|-----------|
| conservative | 0.85 |
| moderate | 0.75 |
| aggressive | 0.60 |

---

### Phase 5: Graph Generation

Implement domain relationship graphs.

---

#### [NEW] [graph.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/domains/graph.py)

```python
class DomainGraph:
    def __init__(
        self,
        instances: List[DomainMappedInstance],
        normalization: Optional[NormalizationResult] = None,
    ):
        self.instances = instances
        self.normalization = normalization

    def generate(self) -> DomainGraphData:
        """Generate graph from instances (uses normalized if available)."""
        ...

    def to_json(self) -> dict:
        """Export as JSON dict."""
        ...

    def to_csv_nodes(self, path: Path) -> None:
        """Export nodes to CSV."""
        ...

    def to_csv_edges(self, path: Path) -> None:
        """Export edges to CSV."""
        ...

    def save(
        self,
        output_dir: Path,
        format: str = "json",  # json/csv/both
        prefix: str = "domain_graph",
    ) -> List[Path]:
        """Save graph to file(s)."""
        ...
```

Output formats match backend schema:
- **Nodes**: id, label, domain_type, frequency, as_source, as_target, raw_domains (if normalized)
- **Edges**: id, source, target, weight, figurative_types, examples, text_count

---

### Phase 6: CLI Implementation

Create unified CLI entry point.

---

#### [NEW] [domains_cli.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/src/qualitative_analysis/figurative/domains_cli.py)

Subcommand-based CLI:

```bash
# Domain extraction
qualitative-domains extract instances.csv --output domains.csv --multi-level

# Normalization
qualitative-domains normalize domains.csv --conservativeness moderate --output normalized.csv

# Graph generation  
qualitative-domains graph domains.csv --format json --output graph.json

# Full pipeline
qualitative-domains pipeline instances.csv --normalize --graph --output-dir ./results

# Load/apply saved normalization
qualitative-domains normalize domains.csv --load-config saved_normalization.json
```

CLI structure using `argparse` subparsers:

```python
def main():
    parser = argparse.ArgumentParser(prog="qualitative-domains")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Extract subcommand
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("input_csv")
    extract_parser.add_argument("--output", "-o")
    extract_parser.add_argument("--multi-level", action="store_true")
    extract_parser.add_argument("--prompt-version", default="v3-multilevel")
    extract_parser.add_argument("--model", default="qwen3:30b-a3b-instruct-2507-q4_K_M")
    extract_parser.add_argument("--checkpoint-interval", type=int, default=50)
    
    # Normalize subcommand
    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("input_csv")
    normalize_parser.add_argument("--output", "-o")
    normalize_parser.add_argument("--conservativeness", choices=["conservative", "moderate", "aggressive", "custom"])
    normalize_parser.add_argument("--threshold", type=float)
    normalize_parser.add_argument("--cluster-mode", choices=["separate", "together"])
    normalize_parser.add_argument("--canonical-method", choices=["llm", "representative"])
    normalize_parser.add_argument("--abstraction-level", choices=["specific", "moderate", "abstract"])
    normalize_parser.add_argument("--save-config")
    normalize_parser.add_argument("--load-config")
    
    # Graph subcommand
    graph_parser = subparsers.add_parser("graph")
    graph_parser.add_argument("input_csv")
    graph_parser.add_argument("--format", choices=["json", "csv", "both"])
    graph_parser.add_argument("--output", "-o")
    normalize_parser.add_argument("--load-normalization")
    
    # Pipeline subcommand (unified)
    pipeline_parser = subparsers.add_parser("pipeline")
    # ... combines all options
```

---

#### [pyproject.toml](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/qualitative-analysis/pyproject.toml)

Add new entry point and dependencies:

```toml
[project.scripts]
qualitative-analysis = "qualitative_analysis.cli:main"
qualitative-relationships = "qualitative_analysis.relationships_cli:main"
qualitative-domains = "qualitative_analysis.figurative.domains_cli:main"  # NEW

[project.dependencies]
# Existing...
sentence-transformers = ">=3.0.0"  # NEW
scikit-learn = ">=1.5.0"  # NEW
```

---

### Phase 7: Progress & Checkpointing

Support for long-running jobs.

---

#### Progress Logging

- Use Python `logging` module with configurable verbosity
- Log progress every N items (configurable via `--log-interval`)
- Support `--quiet` mode for minimal output

---

#### Checkpointing

- Save partial results every N items (configurable via `--checkpoint-interval`)
- Checkpoint file format: JSON with processed indices and partial results
- Resume with `--resume checkpoint.json`

```python
@dataclass
class Checkpoint:
    processed_indices: List[int]
    partial_results: List[dict]
    config: dict
    timestamp: str
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `cli.py` | MODIFY | Add `--types` argument |
| `detector.py` | MODIFY | Add `figurative_types` parameter |
| `strategies/two_step.py` | MODIFY | Pass types to prompts |
| `prompts/*.txt` | MODIFY | Add type template variable |
| `figurative/domains/__init__.py` | NEW | Module exports |
| `figurative/domains/models.py` | NEW | Data classes |
| `figurative/domains/extractor.py` | NEW | Domain extraction |
| `figurative/domains/normalizer.py` | NEW | Clustering & normalization |
| `figurative/domains/graph.py` | NEW | Graph generation |
| `figurative/domains/prompts/*.txt` | NEW | Domain extraction prompts |
| `figurative/domains_cli.py` | NEW | CLI entry point |
| `pyproject.toml` | MODIFY | Add entry point and deps |

---

## Verification Plan

### Unit Tests (New)

Create `tests/test_domains.py` with mock LLM tests:

```bash
# Run from qualitative-analysis directory
cd qualitative-analysis
python -m pytest tests/test_domains.py -v
```

**Test cases:**
1. `test_domain_extractor_parses_v1_response` — Parse basic source/target
2. `test_domain_extractor_parses_multilevel_response` — Parse three-level domains
3. `test_normalizer_clustering_threshold` — Verify threshold behavior
4. `test_normalizer_representative_canonical` — Representative label selection
5. `test_graph_generation_from_instances` — Node/edge generation
6. `test_graph_normalized_includes_raw_domains` — Normalized graph metadata
7. `test_checkpoint_save_load_roundtrip` — Checkpoint serialization
8. `test_normalization_save_load_roundtrip` — Normalization state persistence

### Integration Tests

Extend existing mock pipeline test pattern:

```python
# tests/test_domains_pipeline.py
class TestDomainsPipeline(unittest.TestCase):
    def test_full_pipeline_mock(self):
        """Test extract -> normalize -> graph with mock LLM."""
        ...
```

Run with:
```bash
cd qualitative-analysis
python -m pytest tests/test_domains_pipeline.py -v
```

### Live Ollama Tests (New)

Following the existing pattern in `test_live_ollama_pipeline.py`, create live LLM tests:

```python
# tests/test_live_domains.py
RUN_LIVE = os.getenv("RUN_OLLAMA_LIVE_TESTS") == "1"

class TestLiveDomainExtraction(unittest.TestCase):
    @unittest.skipUnless(RUN_LIVE, "Set RUN_OLLAMA_LIVE_TESTS=1")
    def test_live_domain_extraction_quality(self):
        """Test domain extraction with real LLM responses."""
        extractor = DomainExtractor(
            model_name="qwen3:30b-a3b-instruct-2507-q4_K_M",
            provider="ollama",
            multi_level=True,
        )
        
        samples = [
            {"text": "Time is a thief", "type": "metaphor"},
            {"text": "Life is a journey", "type": "metaphor"},
        ]
        
        results = asyncio.run(extractor.extract(samples))
        
        for result in results:
            self.assertIsNotNone(result.source_domain)
            self.assertIsNotNone(result.target_domain)
            if result.source_domain_levels:
                self.assertIn("specific", result.source_domain_levels)
                self.assertIn("moderate", result.source_domain_levels)
                self.assertIn("abstract", result.source_domain_levels)
```

Run with:
```bash
cd qualitative-analysis
RUN_OLLAMA_LIVE_TESTS=1 python -m pytest tests/test_live_domains.py -v
```

### Manual CLI Testing

**Test data location:** `qualitative-analysis/tests/data/`

Create sample input file `sample_instances.csv`:
```csv
text_id,instance_text,type,confidence,window_text
1,"Time is a thief",metaphor,0.9,"We often feel that time steals our moments."
2,"Life is a journey",metaphor,0.85,"The path we walk through life..."
3,"Arguments are war",metaphor,0.8,"They attacked my position in the debate."
```

**Manual verification commands:**

```bash
# 1. Test type filtering in main CLI
cd qualitative-analysis
python -m qualitative_analysis tests/data/sample_input.csv \
  --types "metaphor,analogy" \
  --output instances \
  --output-dir /tmp/test_types

# Verify: Check output CSV contains only metaphor/analogy types

# 2. Test domain extraction
python -m qualitative_analysis.figurative.domains extract \
  tests/data/sample_instances.csv \
  --multi-level \
  --output /tmp/domains.csv

# Verify: CSV has source_domain, target_domain, and *_levels columns

# 3. Test normalization
python -m qualitative_analysis.figurative.domains normalize \
  /tmp/domains.csv \
  --conservativeness moderate \
  --canonical-method representative \
  --output /tmp/normalized.csv \
  --save-config /tmp/normalization_config.json

# Verify: CSV has canonical columns, config JSON is valid

# 4. Test graph generation
python -m qualitative_analysis.figurative.domains graph \
  /tmp/normalized.csv \
  --format both \
  --output /tmp/graph

# Verify: graph.json and nodes.csv/edges.csv are created

# 5. Test full pipeline
python -m qualitative_analysis.figurative.domains pipeline \
  tests/data/sample_instances.csv \
  --normalize \
  --graph \
  --output-dir /tmp/pipeline_output

# Verify: All output files present in /tmp/pipeline_output/
```

### Existing Tests to Run

Ensure no regressions in existing functionality:

```bash
cd qualitative-analysis
python -m pytest tests/test_mock_pipeline.py -v
python -m pytest tests/test_live_ollama_pipeline.py -v  # Requires Ollama running
```

---

## Dependencies

### Required Additions to `pyproject.toml`

```toml
[project.dependencies]
# Existing deps...
sentence-transformers = ">=3.0.0"
scikit-learn = ">=1.5.0"
numpy = ">=1.24.0"
```

### Model Download

First run will download `Qwen/Qwen3-Embedding-0.6B` (~1.2GB). Consider documenting this in README.

---

## Rollout Phases

| Phase | Scope | Est. Time |
|-------|-------|-----------|
| 1 | Type filtering enhancement | 1-2 days |
| 2 | Module structure + models | 0.5 days |
| 3 | Domain extraction | 2-3 days |
| 4 | Normalization | 2-3 days |
| 5 | Graph generation | 1-2 days |
| 6 | CLI implementation | 2-3 days |
| 7 | Progress & checkpointing | 1-2 days |
| — | Testing & documentation | 2-3 days |

**Total:** ~12-18 days

---

## Open Questions

1. **Embedding model choice**: The backend uses `Qwen/Qwen3-Embedding-0.6B`. Should we offer alternatives or make this configurable via CLI?
ANSWER: Sure, make it configurable via CLI but default to `Qwen/Qwen3-Embedding-0.6B`.

2. **GPU support**: The backend forces `device='cpu'`. Should we auto-detect GPU availability for faster embedding generation?
ANSWER: Sure, but remember that the default LLM inference will be local (Ollama) through HTTP requests. We should also abstract the LLM inference to allow for different providers at a future date.

3. **Batch processing for extraction**: Should we implement batch LLM calls (if provider supports) for faster extraction?
ANSWER: No, we should not implement batch processing for extraction for now.

4. **Checkpoint format**: Should checkpoints be JSON files saved periodically?
ANSWER: Yes, we should implement checkpointing to allow for resuming interrupted runs and I don't have a strong preference for the format.
