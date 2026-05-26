# Qualitative Analysis

Portable tooling for qualitative data analysis using LLMs.

## CLI (CSV)

Run from the repo root:

```bash
PYTHONPATH=qualitative-analysis/src python -m qualitative_analysis --help
```

If installed editable, you can use:

```bash
qualitative-analysis --help
```

## Figurative detect output notes

The `qa figurative detect` / `qualitative-analysis figurative detect` command can write summary, instances, and windows CSV outputs.

For the instances CSV, the current package contract includes:

- core detector fields such as `text_id`, `window_index`, `window_text`, `instance_text`, `type`, `confidence`, and `explanation`
- window-local optional offsets `start_char` / `end_char` when deterministic alignment succeeds
- provenance/support fields `alignment_status`, `support_count`, and `supporting_window_indices`

Notes:

- `start_char` / `end_char` are offsets within `window_text`, not global source-text offsets
- `supporting_window_indices` is serialized as a JSON array string in the CSV field
- blank offsets mean alignment was unavailable, ambiguous, or approximate-only; the instance row may still be valid and usable for downstream domain mapping

## Relationships CLI (CSV)

```bash
PYTHONPATH=qualitative-analysis/src python -m qualitative_analysis.relationships --help
```

If installed editable:

```bash
qualitative-relationships --help
```

Notes:
- Default strategy is `two_pass` (entity extraction then relationship-only).
- Use `--strategy one_pass` to run a single-pass relationship extractor.
- Optional context buffer: `--context-buffer-size N`
- Optional coreference resolution: `--coref`
- Optional window summaries: `--include-summaries`, `--summary-buffer-size N`, or disable with `--no-summaries`
- Summary trigger threshold: `--summary-min-windows N` (default: 2)

## Entity Python API

The `qualitative_analysis.entity` package exposes the main entity-analysis API for:

- scoring
- participant comparison
- agreement and clustering
- standalone entity detection

Lightweight imports are safe at the package level:

```python
from qualitative_analysis.entity import EntityScorer, ParticipantComparison
```

Heavier or optional functionality is loaded only when first accessed:

- `EntityVisualizer`
- `EntityConsolidator` / `consolidate_entities`
- `ComparisonVisualizer`
- `cluster_hdbscan` (requires `hdbscan`)
- Bayesian helpers such as `BayesianEntityModel` (require `pymc` and `arviz`)

That means `import qualitative_analysis.entity` does not eagerly initialize
matplotlib, embedding/consolidation dependencies, or Bayesian libraries.

If you know you need a specific optional component, direct submodule imports are
also supported:

```python
from qualitative_analysis.entity.visualizer import EntityVisualizer
from qualitative_analysis.entity.bayesian import BayesianEntityModel
```
