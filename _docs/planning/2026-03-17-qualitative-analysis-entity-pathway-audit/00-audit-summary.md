# Qualitative-Analysis Entity Pathway Audit

Date: 2026-03-17
Scope: `qualitative-analysis/src/qualitative_analysis/entity*` and the entity portions of `entity_cli.py`

## Scope

This audit covers the entity pathway inside the `qualitative-analysis` package only:

- Entity detection
- Entity scoring
- Visualization
- Participant/group comparison
- Bayesian/clustering utilities that are part of the entity workflow

The web app was intentionally excluded.

## Executive Summary

The entity pathway is functional in the common happy path, and the highest-risk issues identified in this audit have now been remediated. The original failures were concentrated in downstream comparison and analysis steps where the code silently invented data, assumed entity alignment that was never established, or crashed for valid inputs that were not covered by tests.

Status after remediation:

| Severity | Area | Original finding | Status |
|---|---|---|
| Critical | Clustering | Non-Euclidean clustering was broken: `cluster_participants(... metric='aitchison')` passed a NumPy array into `compute_pairwise_distances()` and crashed. | Fixed |
| Critical | Bayesian prep | `prepare_beta_data()` failed on numeric participant IDs because the lookup maps were string-keyed but the DataFrame values were not normalized before mapping. | Fixed |
| High | Comparison CLI | `--participants` was parsed but ignored for the actual distance computation in both `entity compare` and `entity compare-viz`. | Fixed |
| High | Comparison loader | Missing or malformed dimension values were silently converted to `0.0`, fabricating extreme low scores instead of failing fast. | Fixed |
| High | Comparison/viz semantics | Participant/entity rows were never aligned across participants, but some downstream code still treated row index `i` as “the same entity” across participants. | Fixed for current comparison/viz consumers via aligned entity representation |
| Medium | Visualization | Visualization loaders substituted missing scores with `50`, producing plausible-looking but false charts. | Fixed |
| Medium | Prepare-scoring | `entity prepare-scoring` lowercased entities and permanently lost original surface form. | Fixed |
| Medium | Package surface | Importing `qualitative_analysis.entity` eagerly imported heavy optional modules (`consolidator`, `visualizer`), causing unnecessary torch/matplotlib side effects for lightweight use cases. | Fixed |

Remaining lower-priority follow-up:

- Keep broad regression coverage healthy as optional Bayesian and visualization dependencies evolve.
- Reduce non-critical third-party warnings in test output where practical.
- Continue clarifying package contracts in docs when new entity-analysis surfaces are added.
- Decide whether skipped Bayesian integration tests should become part of required CI in an environment with fully writable/runtime-ready PyMC and ArviZ support.

## Verification Notes

Runtime checks were performed inside `qualitative-analysis/.venv-qa-pkg`.

Validated during and after remediation:

- `qualitative-analysis/.venv-qa-pkg/bin/python -m pytest qualitative-analysis/tests/test_comparison.py -q`
- `qualitative-analysis/.venv-qa-pkg/bin/python -m pytest qualitative-analysis/tests/test_entity_score_checkpointing.py -q`
- `MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp qualitative-analysis/.venv-qa-pkg/bin/python -m pytest qualitative-analysis/tests/test_entity_imports.py qualitative-analysis/tests/test_entity_prepare_scoring.py qualitative-analysis/tests/test_entity_score_checkpointing.py qualitative-analysis/tests/test_comparison.py::TestLoadScores qualitative-analysis/tests/test_bayesian.py::TestCheckPymcAvailable qualitative-analysis/tests/test_bayesian.py::TestPrepareBetaData::test_numeric_participant_ids_are_supported -q`
- `MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp ARVIZ_DATA=/tmp/arviz_data qualitative-analysis/.venv-qa-pkg/bin/python -m pytest qualitative-analysis/tests/test_agreement.py qualitative-analysis/tests/test_bayesian.py qualitative-analysis/tests/test_clustering.py qualitative-analysis/tests/test_comparison.py qualitative-analysis/tests/test_entity_detector.py qualitative-analysis/tests/test_entity_imports.py qualitative-analysis/tests/test_entity_prepare_scoring.py qualitative-analysis/tests/test_entity_prompt_contracts.py qualitative-analysis/tests/test_entity_score_checkpointing.py qualitative-analysis/tests/test_entity_scorer.py -q`

Targeted validation runs passed. The broader entity-package suite completed with:

- `163 passed`
- `20 skipped`
- `0 failed`

Skipped tests were environment-sensitive Bayesian integration cases that now cleanly defer when the current runtime cannot initialize the full PyMC/ArviZ stack:

- Bayesian integration tests now use runtime-aware skips rather than failing when ArviZ initialization tries to write under a restricted home directory.
- Token-based entity detector tests no longer depend on live `tiktoken` asset downloads; they use an isolated test tokenizer and the runtime now raises a clearer error if token assets are unavailable offline.

Additional direct reproductions confirmed the original failures before remediation:

- Non-Euclidean clustering crash in `entity/clustering.py`
- Numeric participant ID failure in `entity/bayesian.py`

## Deliverables

- `01-detailed-findings.md`: confirmed findings with file/line evidence
- `02-recommendations.md`: prioritized remediation plan
