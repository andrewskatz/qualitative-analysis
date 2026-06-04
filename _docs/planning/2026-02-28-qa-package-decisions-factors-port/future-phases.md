# Future Phases: Bayesian Modeling & Causal Influence Diagrams

## Phase 8: Bayesian Modeling for Decisions/Factors

### Potential Models

**Decision Frequency Model**
- Hierarchical Poisson/negative-binomial model for decision counts across groups
- Group-level fixed effects, text-level random effects
- Posterior contrasts: which decisions differ between groups?
- Analogous to `entity/bayesian.py` BayesianEntityModel

**Factor Polarity Model**
- Beta-binomial for proportion of supporting vs. opposing factors per decision
- Hierarchical: decision-level + group-level
- Test hypotheses: "Is decision X more contested (more opposing factors) in group A vs B?"

**Cross-Text Consistency**
- How consistently does a decision appear across texts?
- Shrinkage estimation for decision frequency
- ICC (intra-class correlation) for decision agreement

### CLI Integration
```
qa decisions bayes <input_json> --group-col <col> [--model frequency|polarity]
```

### Dependencies
- `pymc`, `arviz`, `nutpie` (already in `[project.optional-dependencies].bayes`)

---

## Phase 9: Causal Influence Diagrams

### Vision

The ultimate goal: automatically construct **causal influence diagrams** from text. These are directed acyclic graphs (DAGs) where:

- **Nodes** = decisions + factors (and potentially entities from the entity module)
- **Edges** = causal/influence relationships between them
- **Edge attributes** = polarity (supporting/opposing), strength (frequency), certainty

### Construction Pipeline

```
Extraction Results
  → Normalize decisions + factors (semantic clustering)
  → Identify decision-factor links (from extraction)
  → Identify factor-factor relationships (new LLM pass or relationship module)
  → Identify decision-decision dependencies (new LLM pass)
  → Construct DAG with networkx
  → Apply layout algorithm (hierarchical/layered for causal DAGs)
  → Render with matplotlib / export to DOT / interactive HTML
```

### Potential Implementation

```python
# decisions/causal_diagram.py

class CausalDiagramBuilder:
    """Build causal influence diagrams from decision-factor data."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        include_inter_factor_links: bool = True,
        include_inter_decision_links: bool = True,
    ): ...

    def build_from_extraction(
        self,
        results: List[DecisionExtractionResult],
    ) -> CausalDiagram:
        """Build diagram from extraction results (decision-factor links only)."""

    async def build_enriched(
        self,
        results: List[DecisionExtractionResult],
        source_texts: List[str],
    ) -> CausalDiagram:
        """Build enriched diagram with inter-factor and inter-decision links."""

    def export_dot(self, diagram: CausalDiagram, path: str): ...
    def export_png(self, diagram: CausalDiagram, path: str): ...
    def export_html(self, diagram: CausalDiagram, path: str): ...
```

### Integration with Relationships Module

The `relationships/` module already has:
- `RelationshipGraph` for network graph construction
- `CausalAnalyzer` for causal attribute analysis
- Graph visualization with networkx

A causal influence diagram could integrate with these:
- Use `RelationshipDetector` to find inter-factor relationships
- Use `CausalAnalyzer` to classify polarity/certainty
- Use `RelationshipGraph` patterns for graph construction

### CLI Integration
```
qa decisions diagram <input_json> [--enriched] [--format png|dot|html]
```

### Research Questions This Enables

1. What decisions are most influenced by factors? (high in-degree)
2. What factors have the broadest influence? (high out-degree)
3. Are there feedback loops? (cyclic paths)
4. What's the "critical path" of influence? (longest causal chain)
5. How do causal structures differ between groups?
