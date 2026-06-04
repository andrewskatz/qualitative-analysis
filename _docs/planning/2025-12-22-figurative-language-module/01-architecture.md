# Architectural Design: Qualitative Analysis Modularization

**Date:** 2025-12-22
**Status:** Planning

## 1. Vision & Scope

The objective is to refactor the existing figurative language detection workflow into a standalone, portable Python package. This package is intended to be the foundation for a broader suite of qualitative analysis tools.

### Strategic Approach: "Extensible Scaffold, Focused Implementation"

To address the question of whether to include other analyses (entity extraction, relationships, etc.) immediately:

**Recommendation:** Adopt a **"One Workflow at a Time"** but **"Platform First"** approach.
*   **Design the package as a suite (`qualitative-analysis`)**: The directory structure and base classes should assume multiple modules will exist.
*   **Implement only `figurative` first**: This serves as the "tracer bullet" to validate the shared infrastructure (LLM clients, configuration, logging) before migrating the more complex Entity/Relationship logic.
*   **Shared Core**: Invest heavily in the `core/` module (LLM adapters, Pydantic types, text chunking) so that future migrations are just about moving prompt logic, not inventing new plumbing.

## 2. Package Structure

We will use a namespace-ready structure.

```text
qualitative-analysis/
├── pyproject.toml              # Dependencies: pydantic, tenacity, litellm (optional)
├── src/
│   └── qualitative_analysis/
│       ├── __init__.py
│       ├── core/               # SHARED INFRASTRUCTURE
│       │   ├── __init__.py
│       │   ├── llm/            # Abstract LLM Interface
│       │   │   ├── base.py     # BaseLLMProvider
│       │   │   ├── ollama.py   # OllamaProvider
│       │   │   ├── openai.py   # OpenAIProvider (generic for compatible APIs)
│       │   │   └── factory.py  # ProviderFactory
│       │   ├── text/           # Text Processing
│       │   │   ├── chunker.py  # Sliding window, sentence splitting
│       │   │   └── clean.py
│       │   └── types.py        # Shared Pydantic models (BaseResult, Span, etc.)
│       │
│       ├── figurative/         # DOMAIN: FIGURATIVE LANGUAGE
│       │   ├── __init__.py
│       │   ├── detector.py     # Main Entrypoint: FigurativeDetector
│       │   ├── models.py       # Domain-specific models (Instance, Summary)
│       │   ├── strategies/
│       │   │   ├── base.py
│       │   │   └── two_step.py # The specific workflow logic
│       │   └── prompts/        # Prompt templates
│       │
│       └── (future_modules)/   # Placeholders for future expansion
│           ├── entities/
│           ├── relationships/
│           └── factors/
```

## 3. Core Components

### 3.1 Abstract LLM Provider
To ensure portability across computers (some with GPUs, some without), we abstract the LLM calls.

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """Generate text completion."""
        pass

    @abstractmethod
    async def generate_json(self, prompt: str, schema: Type[BaseModel], **kwargs) -> BaseModel:
        """Generate structured data."""
        pass
```

### 3.2 Configuration Injection
We will move away from global environment variables.

```python
@dataclass
class FigurativeConfig:
    model_name: str = "mistral-small"
    provider: str = "ollama"
    provider_config: Dict[str, Any] = field(default_factory=dict)
    summary_buffer_size: int = 5
    threshold: float = 0.5
```

## 4. Migration Strategy

1.  **Extract Shared Utils**: Identify utility functions in the current project (text splitting, LLM wrappers) that are generic and move them to `core/`.
2.  **Port Figurative Logic**: Copy the `TwoStepWithSummariesStrategy` logic, but replace specific `backend` calls with the new `BaseLLMProvider`.
3.  **Establish Testing Patterns**: Create a mock LLM provider that returns deterministic responses to test the *logic* of the pipeline without needing a running model.

## 5. Future Roadmap

Once `figurative` is stable:
1.  **Entity Extraction**: Port the `EntityExtractor` to `qualitative_analysis.entities`.
2.  **Relationship Extraction**: Port to `qualitative_analysis.relationships`.
3.  **Orchestration**: Create a high-level `Analyzer` that can run multiple modules in parallel on the same text stream.

## 6. Two-Stage Robust Variant (Sketch)

This keeps the conceptual separation (detect then extract) but closes the gaps by enforcing a single schema contract, tolerant gating, and deterministic fallbacks.

### 6.1 Shared Schema Contract (Single Source of Truth)

Both stages use the same core schema to avoid drift. Detection returns only the top-level fields; extraction returns the full payload.

```python
class FigurativeDetection(BaseModel):
    contains_figurative: bool
    confidence: float  # 0.0 - 1.0
    reason: str

class FigurativeInstance(BaseModel):
    text: str
    type: str
    confidence: float
    explanation: str
    context_dependent: bool

class FigurativeExtraction(BaseModel):
    contains_figurative: bool
    confidence: float
    reason: str
    instances: list[FigurativeInstance]
```

### 6.2 Tolerant Gating (Avoid False Negatives)

Use a three-zone gate to avoid missing true positives:

- `confidence >= 0.65` -> extract
- `0.35 <= confidence < 0.65` -> extract anyway (uncertainty zone)
- `confidence < 0.35` -> skip extract *unless* detection output is invalid or ambiguous

If detection output is invalid JSON or fails schema validation, fall back to extraction directly with a compact prompt.

### 6.3 Context Hygiene

Summaries should provide context but must not include the current window summary prior to detection:

- Build `prior_context` from summaries of *previous* windows only.
- Summarize current window *after* detection/extraction to avoid leakage.

### 6.4 Prompt Strategy

Detection prompt requests the shared schema and a numeric confidence. Extraction prompt requests the full schema with instances.
Both prompts include the same instruction for JSON output, and both are validated with the same Pydantic models.

### 6.5 Pseudocode Flow

```python
for i, window in enumerate(windows):
    prior_context = summary_buffer.get_formatted_context()

    detection = llm.generate_json(
        prompt=detect_prompt(window, prior_context),
        schema=FigurativeDetection,
    )

    if detection_invalid or detection.confidence >= 0.35:
        extraction = llm.generate_json(
            prompt=extract_prompt(window, prior_context),
            schema=FigurativeExtraction,
        )
        instances.extend(extraction.instances)

    summary = llm.generate(prompt=summarize_prompt(window, prior_context))
    summary_buffer.add(summary)
```

### 6.6 Tests (Minimal, High-Value)

1. Detection schema validation (valid/invalid JSON).
2. Uncertainty gate triggers extraction.
3. Extraction schema validation with instances.
4. Context leakage test (current window summary is not included in detection/extraction prompts).
