# qualitative-analysis

A modular Python toolkit for qualitative data analysis using LLMs. Four analysis pathways are exposed through a single `qa` CLI and as importable Python modules:

- **figurative** — detect metaphors, analogies, and other figurative language; extract source/target conceptual domains
- **relationships** — extract entities and their relationships from text; classify relationships as causal with polarity/certainty/explicitness attributes; build graphs
- **entity** — score entities along configurable dimensions (e.g., SETS: social/ecological/technological); compare participants; cluster; Bayesian hierarchical modeling
- **decisions** — extract decisions and their supporting/opposing factors

The package was extracted from [`entity-id-app-v2`](https://github.com/andrewskatz/entity-id-app-v2) into its own repository on 2026-05-25. See [`docs/planning/2026-05-25-qualitative-analysis-package-fork/`](docs/planning/2026-05-25-qualitative-analysis-package-fork/) for the fork plan, path inventory, and execution log.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/andrewskatz/qualitative-analysis.git
cd qualitative-analysis
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz]"
```

Optional dependency groups:

| Group | Contents | Use when |
|---|---|---|
| `dev` | `pytest`, `pytest-asyncio`, `ruff`, `mypy` | developing the package |
| `viz` | `matplotlib`, `networkx`, `umap-learn`, `hdbscan`, `python-ternary`, `adjustText` | generating plots/graphs |
| `bayes` | `pymc`, `arviz`, `nutpie` | Bayesian hierarchical modeling for entity scoring |
| `mlx` | `mlx-vlm`, `torchvision` | Apple MLX inference provider |

## CLI

After install, the unified `qa` command exposes all four pathways:

```bash
qa --help
qa figurative detect transcripts.csv --model gpt-oss:120b
qa relationships detect transcripts.csv --entities "Company,Person"
qa entity score scored.csv --dimensions "social,ecological,technological"
qa decisions detect transcripts.csv
```

Each pathway also has dedicated entry points (`qualitative-analysis`, `qualitative-relationships`, `qualitative-domains`) preserved for backward compatibility — see `[project.scripts]` in `pyproject.toml`.

## Pathway notes

### Figurative detect output

The `qa figurative detect` command writes summary, instances, and windows CSV outputs.

The instances CSV contains:

- core detector fields: `text_id`, `window_index`, `window_text`, `instance_text`, `type`, `confidence`, `explanation`
- window-local optional offsets `start_char` / `end_char` when deterministic alignment succeeds
- provenance/support fields `alignment_status`, `support_count`, `supporting_window_indices`

Notes:

- `start_char` / `end_char` are offsets within `window_text`, not global source-text offsets
- `supporting_window_indices` is serialized as a JSON array string in the CSV field
- blank offsets mean alignment was unavailable, ambiguous, or approximate-only; the instance row may still be valid and usable for downstream domain mapping

### Relationships CLI

```bash
qa relationships --help
# equivalently:
qualitative-relationships --help
```

Notes:

- Default strategy is `two_pass` (entity extraction then relationship-only).
- Use `--strategy one_pass` to run a single-pass relationship extractor.
- Optional context buffer: `--context-buffer-size N`
- Optional coreference resolution: `--coref`
- Optional window summaries: `--include-summaries`, `--summary-buffer-size N`, or disable with `--no-summaries`
- Summary trigger threshold: `--summary-min-windows N` (default: 2)

### Entity Python API

The `qualitative_analysis.entity` package exposes the main entity-analysis API for scoring, participant comparison, agreement, clustering, and standalone entity detection.

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

So `import qualitative_analysis.entity` does not eagerly initialize matplotlib, embedding/consolidation dependencies, or Bayesian libraries.

If you know you need a specific optional component, direct submodule imports are also supported:

```python
from qualitative_analysis.entity.visualizer import EntityVisualizer
from qualitative_analysis.entity.bayesian import BayesianEntityModel
```

## Development

Run the test suite:

```bash
pytest
```

Run a single pathway's tests:

```bash
pytest tests/test_figurative_scanner.py
pytest tests/test_entity_scorer.py tests/test_entity_imports.py
```

## License

MIT — see [`LICENSE`](LICENSE).
