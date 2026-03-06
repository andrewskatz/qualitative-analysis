# Decisions/Factors Module Architecture

## Module Structure

```
qualitative_analysis/
├── decisions/
│   ├── __init__.py                     # Public API exports
│   ├── models.py                       # All dataclasses
│   ├── extractor.py                    # DecisionExtractor (3 strategies)
│   ├── validator.py                    # DecisionValidator (rule-based)
│   ├── normalizer.py                   # Semantic dedup/consolidation
│   ├── aggregator.py                   # Batch statistics
│   ├── visualizer.py                   # Matplotlib charts
│   └── prompts/
│       ├── __init__.py
│       ├── loader.py                   # Enhanced prompt loading
│       ├── decisions_only_v[2-5].txt   # Decision extraction prompts
│       ├── factors_for_decision_v[1-5].txt  # Factor extraction prompts
│       ├── decision_extraction_v[1-2].txt   # Combined single-pass
│       └── window_summary_v1.txt       # Summarization
├── decisions_cli.py                    # CLI commands
└── unified_cli.py                      # Modified: register decisions commands
```

## Data Models (`models.py`)

### Core Extraction Models

```
Factor
├── text: str                   # Factor text (verbatim from source)
├── decision_text: str          # Linked decision
├── polarity: str               # supporting | opposing | neutral
├── window_index: int
├── text_id: str
└── confidence: float

Decision
├── text: str                   # Decision action (causal clauses stripped)
├── original_text: str          # Before normalization
├── factors: List[Factor]       # Linked factors
├── window_indices: List[int]
├── text_ids: List[str]
└── validation_passed: bool

WindowSummary
├── window_index: int
├── text: str                   # Original window text
├── summary: str                # 1-2 sentence summary
└── bullet_points: List[str]    # 3-5 key points

DecisionExtractionResult
├── text_id: str
├── decisions: List[Decision]
├── factors: List[Factor]       # All factors (flat)
├── window_count: int
└── metadata: Dict
```

### Configuration

```
DecisionConfig
├── strategy: str               # one_pass | semantic_two_pass | context_aware
├── temperature: float          # LLM temperature
├── validate_decisions: bool    # Enable rule-based validation
├── normalize_decisions: bool   # Strip causal clauses
├── dedup_method: str           # exact | semantic
├── decision_dedup_threshold: float  # Default 0.9
├── factor_dedup_threshold: float    # Default 0.8
├── factor_min_words: int       # Default 3
├── factor_max_decision_overlap: float  # Default 0.85
├── top_k_windows: int          # Semantic two-pass: windows per decision
└── min_similarity: float       # Semantic two-pass: min relevance threshold
```

### Aggregation

```
AggregationResult
├── total_decisions: int
├── total_factors: int
├── unique_decisions: int
├── unique_factors: int
├── decision_frequency: Dict[str, int]
├── factor_frequency: Dict[str, int]
├── polarity_distribution: Dict[str, int]  # supporting/opposing/neutral
├── texts_per_decision: Dict[str, int]
├── texts_per_factor: Dict[str, int]
├── avg_decisions_per_text: float
└── avg_factors_per_decision: float
```

## Extraction Strategies

### 1. One-Pass (`one_pass`)
Simplest approach. Uses `decision_extraction` prompt for combined extraction.

```
Text → Window → LLM (combined prompt) → decisions + factors
```

### 2. Semantic Two-Pass (`semantic_two_pass`)
Port of `SemanticTwoPassExtractor`. RAG-based for speed + accuracy.

```
Text → Chunk into windows
  → Generate window summaries (LLM)
  → Extract decisions from ALL windows (Pass 1, LLM)
  → Embed decisions + summaries (EmbeddingService)
  → Find top-K relevant windows per decision (cosine similarity)
  → Extract factors from relevant windows only (Pass 2, LLM)
  → Deduplicate decisions + factors
```

### 3. Context-Aware (`context_aware`)
Port of context-aware extraction path. Most thorough.

```
Text → Chunk into windows
  → For each window:
    → Stage 1: Summarize (LLM)
    → Stage 2: Extract with summary context (LLM)
  → Collect all decisions across windows
  → Validate decisions (DecisionValidator)
  → Normalize decisions (strip causal clauses)
  → Per-decision factor probing: decision × window matrix (LLM)
  → Validate factors (empty, identical, too short, high overlap)
  → Deduplicate factors
```

## Decision Validator (`validator.py`)

Rule-based validation ported from web app:

- **40+ decision verbs**: decide, choose, adopt, implement, approve, reject, select, commit, agree, standardize, deploy, etc.
- **10 decision patterns**: "to ", "will ", "should ", "must ", "going to", "plan to", etc.
- **13 non-decision indicators**: discuss, consider, might, maybe, suggest, recommend, think, etc.
- **20 imperative verbs**: adopt, implement, launch, standardize, reduce, etc.

Validation logic:
1. Reject empty/whitespace
2. Reject < min_words (default 3)
3. Reject if non-decision indicator present
4. Accept if has: decision verb OR decision pattern OR imperative form

Also includes `strip_causal_clauses()` to remove "because X", "due to X", "given X", "since X" from decision text.

## Prompt System (`prompts/loader.py`)

Enhanced version of existing prompt loader pattern with custom file support:

```python
def load_prompt(
    prompt_type: str,
    version: Optional[int] = None,
    custom_path: Optional[str] = None,  # Key extensibility feature
) -> str:
```

- If `custom_path` provided: reads from that file, ignores type/version
- Otherwise: loads `{prompt_type}_v{version}.txt` from built-in prompts dir
- CLI exposes as `--decisions-prompt my_prompt.txt`, `--factors-prompt ...`, etc.

This enables researchers to iterate on prompts without modifying package code.

## Core Package Reuse

| Core Component | Used For |
|----------------|----------|
| `SlidingWindowProcessor` (core/text.py) | Text chunking with overlap |
| `EmbeddingService` (core/embeddings.py) | Semantic search, dedup, clustering |
| `BaseLLMProvider` (core/llm.py) | LLM abstraction |
| `OllamaProvider` / `MLXProvider` (core/providers.py) | Concrete LLM implementations |
| `add_common_*_args()` (core/cli_utils.py) | Shared CLI arguments |
| `write_run_metadata()` (core/cli_utils.py) | Run metadata logging |

## CLI Commands

```
qa decisions detect <csv>     # Extract decisions + factors
qa decisions normalize <json> # Semantic dedup across texts
qa decisions aggregate <json> # Compute batch statistics
qa decisions viz <json>       # Generate visualizations
```

Alias: `qa dec detect ...`

## Visualizations

1. **Decision frequency** — horizontal bar chart (top-N)
2. **Factor frequency** — horizontal bar chart (top-N)
3. **Polarity distribution** — stacked bar or pie chart
4. **Decision-factor network** — bipartite graph with polarity-colored edges
5. **Factor polarity by decision** — grouped bar chart per decision
