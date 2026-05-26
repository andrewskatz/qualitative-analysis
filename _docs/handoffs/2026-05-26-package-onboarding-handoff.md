# Onboarding Handoff Memo: qualitative-analysis Package

**Date:** 2026-05-26
**From:** Initial extraction phase (Andrew Katz)
**To:** New collaborator or future-self picking up the package
**Project:** `qualitative-analysis` — modular Python toolkit for LLM-powered qualitative data analysis

---

## Executive Summary

This repository is a freshly-extracted standalone Python package (just released as `v0.2.0` on 2026-05-26) for running LLM-driven qualitative data analysis across four pathways:

- **figurative** — detect metaphors, analogies, and other figurative language; map source/target conceptual domains
- **relationships** — extract entities and their relationships; classify as causal with polarity/certainty/explicitness attributes
- **entity** — score entities along configurable dimensions (e.g., SETS: social/ecological/technological); compare participants; cluster; Bayesian hierarchical modeling
- **decisions** — extract decisions and their supporting/opposing factors

**The package is research-grade, actively evolving, and built to support a methods-paper publication.** Four pathway audits (figurative, entity ×2, relationships) document known issues and recommended fixes — these are the most valuable starting reading.

**Key context to absorb before contributing:**
1. The package was extracted on 2026-05-25 from a larger parent repo (`entity-id-app-v2`) where it had lived as a subdirectory. Git history is preserved via `git filter-repo`; commits go back to the original entity-id-app-v2 init.
2. The four pathways are **decoupled** at the code level (no cross-pathway imports). They share only `core/` (LLM providers, embeddings, sliding-window text utils, CLI helpers).
3. Heavier optional dependencies are split into install groups (`viz`, `bayes`, `mlx`) — the core install stays slim.
4. The unified `qa` CLI is the canonical entry point; legacy per-pathway entry points (`qualitative-analysis`, `qualitative-relationships`, `qualitative-domains`) are preserved but secondary.

---

## Repository Structure

```
qualitative-analysis/
├── src/qualitative_analysis/    # the package
│   ├── core/                    # shared LLM/text/embedding utilities
│   ├── figurative/              # figurative-language pathway
│   ├── relationships/           # entity-relationship pathway
│   ├── entity/                  # entity-scoring pathway (largest, ~9.6k LOC)
│   ├── decisions/               # decisions/factors pathway
│   ├── unified_cli.py           # the `qa` command's entry point
│   ├── cli.py                   # legacy `qualitative-analysis` entry point
│   ├── relationships_cli.py     # legacy `qualitative-relationships` entry point
│   ├── decisions_cli.py         # used by `qa decisions ...`
│   └── entity_cli.py            # used by `qa entity ...` (~3.5k LOC)
├── tests/                       # pytest suite (29 test files, ~470 tests)
├── scripts/                     # standalone experiment scripts (not entry points)
├── docs/
│   ├── planning/                # design docs, audits, implementation plans (21 dirs)
│   ├── progress-notes/          # dated progress notes (11 files)
│   ├── explanations/            # conceptual explainers
│   └── guides/                  # how-to guides
├── _docs/                       # operational/team docs (this convention)
│   ├── handoffs/                # ← this memo
│   ├── feature-dev/             # dated dirs for in-progress feature work
│   └── progress-updates/        # rolling progress updates
├── pyproject.toml               # package config + dep groups
├── README.md                    # user-facing intro
├── CHANGELOG.md
└── LICENSE                      # MIT
```

**Two doc directories, different purposes:**
- `docs/` — design and audit history (carried with the package from the parent repo's `docs/planning/...` during extraction)
- `_docs/` — operational team docs (handoffs, feature-dev plans, progress updates) following the convention used in your other research repos

---

## Setup

Requires Python 3.10+ (verified working on 3.13).

```bash
git clone git@github.com:andrewskatz/qualitative-analysis.git
cd qualitative-analysis
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,viz]"
```

For full functionality also install `[bayes]` and `[mlx]`:

```bash
pip install -e ".[dev,viz,bayes,mlx]"
```

| Group | Purpose |
|---|---|
| `dev` | pytest, pytest-asyncio, ruff, mypy |
| `viz` | matplotlib, networkx, umap-learn, hdbscan, python-ternary |
| `bayes` | pymc, arviz, nutpie (entity Bayesian hierarchical modeling) |
| `mlx` | mlx-vlm, torchvision (Apple MLX inference provider) |

Verify install:

```bash
qa --version           # → qa 0.2.0
qa --help              # shows the four pathways
pytest                 # full suite; ~450 pass + skips for missing optional deps
```

---

## CLI Usage (the four pathways)

```bash
# Figurative
qa figurative detect transcripts.csv --model gpt-oss:120b
qa fig map instances.csv --multi-level
qa fig normalize ...
qa fig graph ...
qa fig pipeline ...              # full detect → map → normalize → graph

# Relationships
qa relationships detect transcripts.csv --entities "Company,Person"
qa rel verify ...                # LLM second-pass to filter hallucinated relationships
qa rel causal ...                # classify polarity / certainty / explicitness
qa rel normalize ...
qa rel graph ...

# Entity (scoring along configurable dimensions)
qa entity detect ...             # standalone entity extraction
qa entity prepare-scoring ...    # extracts entity-context pairs from rel detection windows
qa entity score scored.csv --dimensions "social,ecological,technological"
qa entity consolidate ...        # semantic dedup
qa entity viz ...                # ternary plots, radars
qa entity compare ...            # cross-participant distances
qa entity compare-viz ...
qa entity agreement ...          # Krippendorff's Alpha
qa entity cluster ...            # hierarchical / kmeans / hdbscan + PCA
qa entity report ...             # auto-markdown report

# Decisions
qa decisions detect transcripts.csv
qa dec normalize ...
qa dec aggregate ...
qa dec viz ...
```

Each subcommand has its own `--help` with detailed flags. The defaults assume a local Ollama server at `http://localhost:11434`.

---

## LLM Providers

`core/providers.py` exposes:

- **`OllamaProvider`** — the default. Uses Ollama's HTTP API. Handles `<think>...</think>` reasoning-block stripping for models like Qwen3.5 in non-thinking mode. Pass `enable_thinking=True` to keep them.
- **`MLXProvider`** — Apple MLX backend (requires `[mlx]` extras, M-series Macs). Used by `--provider mlx` on supported subcommands.

Add new providers by subclassing `core/llm.py:BaseLLMProvider` (`generate` and `generate_json` methods).

---

## Known Issues / Active Audits

The most important reading. All under `docs/planning/`:

| Audit | What it covers |
|---|---|
| [`2026-02-07-entity-pathway-audit/`](../../docs/planning/2026-02-07-entity-pathway-audit/) | First full audit of the entity pathway — 5 critical, 15 high, 18 medium findings |
| [`2026-03-09-entity-scoring-audit/`](../../docs/planning/2026-03-09-entity-scoring-audit/) | Focused audit of entity ID + SETS scoring pipeline |
| [`2026-03-17-qualitative-analysis-entity-pathway-audit/`](../../docs/planning/2026-03-17-qualitative-analysis-entity-pathway-audit/) | Final/most recent entity pathway audit, detailed findings + recommendations |
| [`2026-03-10-qual-analysis-fig-lang-pathway-audit/`](../../docs/planning/2026-03-10-qual-analysis-fig-lang-pathway-audit/) | Figurative pathway audit, with engineering plan + provenance/output fidelity spec |
| [`2026-02-08-relationship-analysis-pathway-audit/`](../../docs/planning/2026-02-08-relationship-analysis-pathway-audit/) | Relationship pathway audit |
| [`2026-02-17-bayesian-hierarchical-modeling-audit/`](../../docs/planning/2026-02-17-bayesian-hierarchical-modeling-audit/) | Bayesian model correctness audit |

**Specific items worth knowing:**

- **Ternary plot magnitude encoding** (entity viz): position carries relative proportions; size now carries absolute magnitude (0–100 scale, 0.15× to 2.5× base marker). See [`2026-02-07-entity-pathway-audit/04-visualization.md`](../../docs/planning/2026-02-07-entity-pathway-audit/04-visualization.md). The earlier "ternary normalization is invalid" concern was downgraded MEDIUM → not CRITICAL once magnitude encoding was added.
- **Entity scoring scale specification** ([`2026-02-18-qa-ent-score-dimensions-specification/`](../../docs/planning/2026-02-18-qa-ent-score-dimensions-specification/)): the scoring scale is now configurable (0-100, 1-10, 1-5, custom). 0-100 has consistently shown best reliability across models — see the experiment results in [`docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md`](../../docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md).
- **Two-step figurative strategy** ([`2025-11-04-two-step-with-summaries-figurative-language/`](https://github.com/andrewskatz/entity-id-app-v2/tree/main/_deprecated_qa_fork_20260526/docs/planning/2025-11-04-two-step-with-summaries-figurative-language) — note: lives in parent repo, not extracted): the `figurative/strategies/two_step.py` strategy uses summarization-then-detection.
- **CoT inflation effect** (entity scoring): chain-of-thought prompts deflate scores vs no-CoT (v4 ~2× faster, modest quality loss). Documented in [`docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md`](../../docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md).

---

## Testing

```bash
# Full suite
pytest

# Single pathway
pytest tests/test_figurative_scanner.py tests/test_figurative_cli.py tests/test_figurative_detect_to_map.py
pytest tests/test_entity_*.py tests/test_bayesian.py tests/test_comparison.py tests/test_clustering.py tests/test_agreement.py
pytest tests/test_relationship_*.py
pytest tests/test_decision_*.py

# Imports-only smoke (fast)
pytest tests/test_entity_imports.py

# Mock pipeline (no real LLM)
pytest tests/test_mock_pipeline.py
```

The fresh-clone smoke test from extraction Phase 6 ran `pip install -e ".[dev]"` + `pytest` and saw **448 passed / 21 skipped** (skips are Bayesian tests requiring the `[bayes]` group).

There is **one live-Ollama integration test** (`test_live_ollama_pipeline.py`) that requires a running Ollama server — skipped by default if unreachable.

---

## Key Architectural Conventions

1. **Entity pathway uses lazy imports** at the package level. `import qualitative_analysis.entity` does **not** eagerly initialize matplotlib, pymc/arviz, hdbscan, sentence-transformers consolidation, or comparison-viz code. See [`src/qualitative_analysis/entity/__init__.py`](../../src/qualitative_analysis/entity/__init__.py) for the `_LAZY_IMPORTS` table. **Don't move imports up to eager unless you mean to.**

2. **Package version is single-source-of-truth in `pyproject.toml`**, derived at runtime via `importlib.metadata.version("qualitative-analysis")` in `src/qualitative_analysis/__init__.py`. `core/cli_utils.py:PACKAGE_VERSION` re-exports it. Bumping the version means editing **only** `pyproject.toml` (and `CHANGELOG.md`). The previous hardcoded version in 5 places was a bug fixed during extraction.

3. **Prompt templates live in pathway-local `prompts/` dirs** (e.g., `entity/prompts/`, `figurative/prompts/`). Loaded via Jinja2. Wheel build includes them via the `[tool.hatch.build.targets.wheel] include = [...]` list in `pyproject.toml` — **add new prompts there** or they will be missing from installed wheels.

4. **CLI argument conventions are shared via `core/cli_utils.py`**: `add_common_llm_args`, `add_provider_args`, `add_common_output_args`, etc. Use these rather than hand-rolling per-command.

5. **Output CSV conventions for figurative** are documented in the package README and the [`2026-03-10-qual-analysis-fig-lang-pathway-audit/04-phase-2-provenance-output-fidelity-spec.md`](../../docs/planning/2026-03-10-qual-analysis-fig-lang-pathway-audit/04-phase-2-provenance-output-fidelity-spec.md). Window-local offsets (`start_char`/`end_char`), `alignment_status`, `support_count`, `supporting_window_indices` — preserve these contracts.

6. **Pathway decoupling**: there are no cross-pathway imports between `figurative/`, `relationships/`, `entity/`, `decisions/`. The closest coupling is at the data-layer: `qa entity prepare-scoring` consumes relationship-detection windows ([`entity_cli.py`](../../src/qualitative_analysis/entity_cli.py)). Maintain this — it makes future splits and individual deprecation tractable.

---

## Context: What Came Before This Repo

This package was extracted from `entity-id-app-v2` on 2026-05-25 via `git filter-repo`. The full extraction plan, path inventory, and execution log are in [`docs/planning/2026-05-25-qualitative-analysis-package-fork/`](../../docs/planning/2026-05-25-qualitative-analysis-package-fork/).

Key inherited context:

- **Parent repo still exists at [`andrewskatz/entity-id-app-v2`](https://github.com/andrewskatz/entity-id-app-v2)**. It contains a web app (frontend/backend), pilot experiments, and now installs this package via pinned pip URL in `requirements.txt`.
- **The parent's `_deprecated_qa_fork_20260526/` folder** (gitignored, local-only) holds the pre-extraction working tree as a safety net. Will be deleted once the user is confident nothing was missed.
- **The parent's `scoring-experiment-output/`** holds 3.8 GB of historical scoring experiment data the publications scripts read from. **Not** in this repo — it never was (the package's `output/` was always gitignored).
- **`publications-and-presentations/llm-qualitative-scoring-methodology/`** in the parent contains the methods-paper materials. Two scripts there (`select_sample.py`, `generate_tables.py`) read the scoring experiment output but **do not import this package**.

---

## Suggested Reading Order

If you're new and have an hour:

1. This handoff memo (you're here)
2. The package [`README.md`](../../README.md) — quick orientation
3. [`docs/planning/2026-05-25-qualitative-analysis-package-fork/README.md`](../../docs/planning/2026-05-25-qualitative-analysis-package-fork/README.md) — how the repo got here
4. [`docs/planning/2026-03-17-qualitative-analysis-entity-pathway-audit/00-audit-summary.md`](../../docs/planning/2026-03-17-qualitative-analysis-entity-pathway-audit/00-audit-summary.md) — most-recent entity audit (the highest-traffic pathway)
5. [`docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md`](../../docs/progress-notes/2026-04-08-scoring-experiment-multi-model-analysis.md) — what the package has been used for + what experiment results say about model/prompt choices
6. Run `qa --help`, `qa entity --help`, `qa figurative --help`, etc.

If you're picking up a specific pathway, the per-pathway audit is mandatory reading first.

---

## Key Files

| File | Purpose |
|---|---|
| [`pyproject.toml`](../../pyproject.toml) | Package config, deps, version, entry points |
| [`src/qualitative_analysis/__init__.py`](../../src/qualitative_analysis/__init__.py) | `__version__` (sourced from package metadata) |
| [`src/qualitative_analysis/unified_cli.py`](../../src/qualitative_analysis/unified_cli.py) | `qa` command structure; subcommands registered here |
| [`src/qualitative_analysis/core/cli_utils.py`](../../src/qualitative_analysis/core/cli_utils.py) | Shared argument-parsing helpers |
| [`src/qualitative_analysis/core/providers.py`](../../src/qualitative_analysis/core/providers.py) | OllamaProvider, MLXProvider |
| [`src/qualitative_analysis/entity/__init__.py`](../../src/qualitative_analysis/entity/__init__.py) | Lazy-import contract for the entity API |
| [`src/qualitative_analysis/entity_cli.py`](../../src/qualitative_analysis/entity_cli.py) | The biggest CLI surface (~3.5k LOC) — all `qa entity *` subcommands |
| [`tests/test_unified_cli.py`](../../tests/test_unified_cli.py) | CLI integration tests (subprocess-based) |
| [`CHANGELOG.md`](../../CHANGELOG.md) | Release history (starts at v0.2.0) |

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: pandas` after install | Old install missed `pandas` (was a bug fixed in v0.2.0). Reinstall: `pip install -e ".[dev,viz]"`. |
| `qa --version` shows `0.1.0` | Stale install. Reinstall: `pip install -e ".[dev,viz]"`. |
| Bayes tests skip | `[bayes]` deps not installed. `pip install -e ".[bayes]"`. |
| `from qualitative_analysis.entity.agreement import …` fails with pandas error | Same as the first row — reinstall. |
| Ollama HTTP timeouts | Default `--timeout 60` is short for large models. Bump with `--timeout 600` on the affected subcommand. Also verify Ollama is running: `curl http://localhost:11434/api/version`. |
| Bayesian sampling hangs on macOS | Known multiprocessing/PyMC interaction on macOS. PyMC uses `fork()`; some packages emit warnings or hang. Set `MULTIPROCESSING_FORK_SAFE=1` or use `--cores 1` if available. |
| `qa figurative detect` produces no instances | Try `--verbose` to see LLM output. Often the model is over-conservative; try `--model gpt-oss:120b` or another larger model. |
| Tests collect but several error | Usually an optional dep missing. Look at the first traceback's `ImportError` — install the matching dep group. |

---

## Open Questions / Next Steps

A few open items worth raising in any successor's first session:

1. **Are the v0/v1 sibling repos** at `~/Documents/ak fac/research/projects/qualitative-analysis-v{0,1}/` permanently archived, or should they be deleted? They had abandoned extraction attempts. (See the `project_qa_package_repos.md` memory note.)
2. **Should the package go public on GitHub** for citation purposes? Currently private. A methods-paper draft is in the parent's `publications-and-presentations/`.
3. **Tagging cadence**: this is `v0.2.0`. The next bump should follow conventional rules — bug fixes → 0.2.1; new pathway features → 0.3.0; major API rework → 1.0.0.
4. **The figurative pathway's `output/` directory** never made it across extraction (gitignored, not in history). Pathway-specific output now lives wherever you point `--output-dir` flags. If you want a canonical place, set up an `output/` (still gitignored) in the new repo's working tree.
5. **The `_docs/feature-dev/` and `_docs/progress-updates/` dirs are empty** (only `.gitkeep`). Start populating them on the next feature/progress entry.

---

**End of Handoff Memo**
