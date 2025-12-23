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
- Optional window summaries: `--include-summaries` and `--summary-buffer-size N`
