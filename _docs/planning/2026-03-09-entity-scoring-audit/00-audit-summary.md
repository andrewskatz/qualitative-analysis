# Entity Identification and SETS Scoring Audit Summary

**Date:** 2026-03-10  
**Scope:** Package-only audit of `qualitative-analysis` entity identification and SETS scoring. This audit intentionally excludes the web app/backend pathway except where package boundaries needed to be confirmed.

---

## Executive Summary

The package-side pathway is coherent and usable end-to-end: `qa entity detect` can produce scoring-ready rows, `qa entity score` can run repeated LLM scoring passes, and the statistical model layer supports multi-run uncertainty summaries on customizable scales.

The strongest parts of the implementation are:
- clear package-public entry points in `entity/__init__.py` and `unified_cli.py`
- well-scoped entity detection tests in `tests/test_entity_detector.py`
- a practical multi-run scoring design in `entity/scorer.py`
- scale-aware statistical aggregation in `entity/models.py`
- resumable scoring through JSONL checkpoints in `entity_cli.py`

The main concerns are not that the pathway is missing, but that several details are fragile or internally inconsistent:
- the detector claims independence but imports the relationships package’s `EntityResponse` schema
- JSON extraction logic is duplicated across package areas instead of being shared
- scoring response parsing is permissive in some ways and brittle in others
- the prompt versions intentionally expose different output contracts, so version choice changes how much reasoning survives into aggregated outputs
- JSON round-trip fidelity is incomplete in `EntityScorer.load()`
- direct unit coverage of scorer internals is very limited compared with detector coverage

---

## Package Pathway in Scope

1. **Detection**: `entity/detector.py`
2. **Scoring models/statistics**: `entity/models.py`
3. **SETS scoring orchestration**: `entity/scorer.py`
4. **CLI wiring / checkpointing / CSV flow**: `entity_cli.py`
5. **Prompt loading and prompt versions**: `entity/prompts/*`
6. **Package tests**: `qualitative-analysis/tests/*`

Primary CLI flow audited:
- `qa entity detect`
- `qa entity prepare-scoring`
- `qa entity score`

---

## Key Findings by Severity

### High

| ID | Area | Finding | Why it matters |
|---|---|---|---|
| H1 | `entity/scorer.py` + `entity/models.py` | `EntityScore.to_dict()` does not emit `{dim}_mode`, but `EntityScorer.load()` expects it | Loaded scores reconstruct with `mode=0`, so JSON persistence is not statistically faithful |
| H2 | `entity/scorer.py` | `load()` drops `window_index`, `group`, and full `runs` data | JSON round-trips lose provenance and raw-run detail |
| H3 | `entity/scorer.py` | `_parse_response()` silently skips unexpected dimensions / nonnumeric scores, then fails only when expected dimensions are missing | Failure modes are hard to diagnose and partial parse behavior is opaque |
| H4 | `entity/prompts/entity_scoring_v4.txt` | Prompt `v4` intentionally removes `initial_observations` and `justification` fields | This is a legitimate score-only mode, but aggregated outputs become much less interpretable unless the workflow explicitly accepts that tradeoff |
| H5 | Test coverage | No direct package tests found for `_parse_response`, `_aggregate_runs`, `score_entity`, `score_entities`, or CLI checkpoint/resume | Core scoring behavior can regress without a focused package test failing |

### Medium

| ID | Area | Finding | Why it matters |
|---|---|---|---|
| M1 | `entity/detector.py` | Detector imports `EntityResponse` from `relationships.components.entity_extractor` | Package boundary is looser than the detector docstring suggests |
| M2 | `entity/detector.py` + `relationships/components/entity_extractor.py` | JSON extraction logic is duplicated in two modules | Bug fixes or parser improvements can drift between implementations |
| M3 | `entity/detector.py` | Deduplication is case-sensitive only | Same entity can be overcounted before consolidation |
| M4 | `entity/scorer.py` | Numeric scores are coerced with `int(score)` | Float outputs are truncated rather than validated or rounded intentionally |
| M5 | `entity_cli.py` | `_resolve_scale()` exists, but `run_entity_score()` directly uses loaded dimensions plus explicit overrides | Metadata-based scale auto-detection does not appear to drive the scoring command itself |
| M6 | `entity/models.py` | Dataset statistics are derived from dimensions present on the first scored entity | If dimensions ever differ across scores, summaries can be incomplete |

---

## Strengths Worth Preserving

- **Detection pathway is cleanly packaged** and returns both unique entities and entity-context occurrences.
- **Scoring supports partial success across runs**: `score_entity()` only fails when every run fails, and preserves `num_runs=len(runs)` after partial failure.
- **Scale customization is thoughtfully propagated** through prompt examples and confidence-interval clamping logic.
- **Checkpoint/resume is operationally useful** and keyed by `entity|text_id|window_index`, which fits the prepared scoring input shape.
- **The CLI data flow is understandable** and the package explicitly documents `detect -> score` compatibility.

---

## Coverage Assessment

Coverage is asymmetric:
- **Strong direct coverage:** detector behavior in `tests/test_entity_detector.py`
- **Indirect scoring-related coverage:** serialization/custom-scale assumptions in `tests/test_bayesian.py`
- **Weak direct coverage:** scorer parsing, aggregation, prompt-version behavior, and CLI checkpoint/resume

This is the single biggest maintainability gap in the package-side SETS scoring pathway.

---

## Deliverables in This Directory

- `00-audit-summary.md` — concise package-only summary
- `01-implementation-audit.md` — detailed component-by-component audit
- `02-recommendations.md` — prioritized remediation and test plan
- `03-concrete-fix-plan.md` — phased implementation plan with tests and acceptance criteria

