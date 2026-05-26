# Qualitative-Analysis Figurative Language Pathway Audit Summary

**Date:** 2026-03-10  
**Scope:** Package-only audit of the `qualitative-analysis` figurative language pathway. This audit intentionally excludes the web app/backend pathway except where package boundaries needed to be confirmed.

---

## Executive Summary

The package-side figurative pathway is implemented end-to-end, but it is best understood as **two connected stages** rather than one monolithic detector:

1. **Figurative identification** via `qa figurative detect`
2. **Post-detection analysis** via `qa figurative map|normalize|graph|pipeline`

The strongest parts of the current implementation are:
- a coherent package namespace under `qualitative_analysis/figurative`
- a practical two-step detection strategy with rolling summary context
- prompt versioning plus optional figurative-type filtering
- operational CLI checkpointing and CSV/JSON outputs
- a real downstream domain-analysis stack rather than a placeholder

The main concerns are not that the pathway is missing, but that several implementation details are fragile:
- detector/scanner structured-output handling is weaker than the provider layer already supports
- detection/extraction failure modes often degrade silently into empty or heuristic results
- overlapping window outputs are not deduplicated and character spans are not populated
- one normalization path uses `run_until_complete(...)` inside a synchronous method
- CLI contracts and help text are slightly misleading in a few places
- test coverage is much thinner for failure modes and cross-stage interoperability than for the basic happy path

---

## Package Pathway in Scope

1. **Detection entry point:** `figurative/detector.py`
2. **Two-step orchestration:** `figurative/strategies/two_step.py`
3. **Detection components:** `figurative/components/*`
4. **Prompts and type filtering:** `figurative/prompts/*`
5. **CLI detection wiring / CSV outputs:** `cli.py`
6. **Domain mapping / normalization / graphing:** `figurative/domains/*`, `figurative/domains_cli.py`
7. **Unified CLI wiring:** `unified_cli.py`
8. **Package tests:** `qualitative-analysis/tests/*`

Primary CLI flow audited:
- `qa figurative detect`
- `qa figurative map`
- `qa figurative normalize`
- `qa figurative graph`
- `qa figurative pipeline`

---

## Key Findings by Severity

### High

| ID | Area | Finding | Why it matters |
|---|---|---|---|
| H1 | `figurative/components/scanner.py` | Detection and extraction use `llm.generate(...)` plus hand-rolled JSON parsing instead of provider-level `generate_json(...)` | The package already has stronger structured-output support than the figurative path currently uses |
| H2 | `scanner.py` + `summarizer.py` | Parse failures degrade silently: detection falls back heuristically, extraction returns `[]`, summary parsing falls back to raw text | Runtime failures can look like genuine “no figurative language found” or low-quality summaries |
| H3 | `strategies/two_step.py` + `models.py` | Overlapping windows can emit duplicate instances, and `Instance.start_char` / `end_char` are never populated | Downstream consumers receive weaker provenance and may overcount repeated instances |
| H4 | `figurative/domains/normalizer.py` | LLM canonical labeling calls `asyncio.get_event_loop().run_until_complete(...)` inside synchronous `normalize()` | This is risky in environments where an event loop is already running |
| H5 | Test coverage | There is no strong package regression coverage for parse failures, overlap deduplication, detect→map interoperability, or normalization/CLI edge cases | The most fragile parts of the pathway are not well locked down |

### Medium

| ID | Area | Finding | Why it matters |
|---|---|---|---|
| M1 | `strategies/two_step.py` | The current window’s summary is added to the buffer before binary detection/extraction for that same window | “Prior context” is not purely prior; it includes a model-generated view of the current text |
| M2 | `detector.py` + `domains/extractor.py` | Provider selection is effectively Ollama-only unless a provider is dependency-injected | The core package has broader provider abstractions than this pathway exposes |
| M3 | `domains_cli.py` | `qa figurative pipeline` is described as a “full pipeline,” but its input is an instances CSV, not raw text detection input | The command name/help text can imply a broader end-to-end flow than is actually implemented |
| M4 | `core/text.py` | Sentence chunking is intentionally naive regex-based splitting | This is predictable and lightweight, but brittle around abbreviations and punctuation edge cases |
| M5 | `cli.py` | `window_text` in the instances CSV is only populated when windows output was also requested | The detect→map bridge works, but context completeness depends on output mode |

---

## Strengths Worth Preserving

- **The package pathway is real and usable end-to-end** rather than a partial prototype.
- **Two-step detection is thoughtfully separated** into summarization, binary detection, and extraction.
- **Prompt versioning is meaningful** and supports type-constrained detection.
- **Checkpoint/resume support exists** in both detection and downstream domain mapping.
- **Domain analysis goes beyond extraction** into clustering, canonicalization, and graph generation.

---

## Coverage Assessment

Coverage is asymmetric:
- **Direct happy-path coverage:** `tests/test_mock_pipeline.py`
- **Optional live integration signal:** `tests/test_live_ollama_pipeline.py`
- **CLI help coverage:** `tests/test_unified_cli.py`
- **Domain parser/graph coverage:** `tests/test_domains.py`

The weakest areas are:
- detector/scanner parse-failure behavior
- overlap/deduplication behavior
- CSV interoperability between `detect` and `map`
- normalization behavior under `canonical_method="llm"`
- full package-side figurative integration tests

---

## Deliverables in This Directory

- `00-audit-summary.md` — concise package-only summary
- `01-implementation-audit.md` — detailed component-by-component audit
- `02-recommendations.md` — prioritized remediation and test plan