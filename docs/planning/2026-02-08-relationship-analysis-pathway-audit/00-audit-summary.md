# Relationship Analysis Pathway — Audit Summary

**Date**: 2026-02-08
**Auditor**: Claude (Opus 4.6)
**Scope**: Full `qualitative_analysis/relationships/` module, `relationships_cli.py`, and unified CLI integration

---

## Executive Summary

The relationships analysis pipeline is a well-architected, modular system with 5 stages (detect → normalize → verify → causal → graph). The code is clean, well-documented, and the modular design allows flexible stage composition. However, the audit identified **11 HIGH-severity issues**, **1 DESIGN issue**, **16 MEDIUM issues**, and **6 LOW issues**, plus a critical finding that the module has **zero test coverage**.

Unlike the entity pathway audit which found runtime crashes (undefined variables, wrong argument order), this audit found **no obvious crash bugs**. The issues here are primarily about **data integrity** (information loss between stages), **incorrect pipeline semantics** (early normalization, cross-text contamination), and **missing validation**.

---

## Findings Overview

| Severity | Count | Description |
|----------|-------|-------------|
| **DESIGN** | 1 | Early normalization loses per-participant context (D1) |
| **HIGH** | 11 | Data corruption, incorrect results, significant information loss |
| **MEDIUM** | 16 | Data quality degradation, suboptimal methods, DRY violations |
| **LOW** | 9 | Test coverage, minor issues, documentation |

### No CRITICAL (Runtime Crash) Findings

No undefined variables, wrong argument orders, or other crash-inducing bugs were found. The codebase is functionally stable.

---

## Findings by Severity

### DESIGN ISSUE (1)

| ID | Finding | Location | Doc |
|----|---------|----------|-----|
| D1 | Early normalization destroys per-participant context | `normalizer.py` | [02-normalization-design.md](02-normalization-design.md) |

### HIGH (11)

| ID | Finding | Location | Doc |
|----|---------|----------|-----|
| H1 | Entity context buffer persists between detect() calls | `detector.py:88,143` | [01-detection-pipeline.md](01-detection-pipeline.md) |
| H2 | text_id not propagated to Relationship objects | `components/*.py` | [01-detection-pipeline.md](01-detection-pipeline.md) |
| H3 | Confidence always hardcoded to 1.0 | `normalizer.py:460` | [02-normalization-design.md](02-normalization-design.md) |
| H4 | Normalized relationship verification only uses first text_id/window | `verifier.py:346` | [03-verification-causal.md](03-verification-causal.md) |
| H5 | Batch verification silently loses unmatched relationships | `verifier.py:410` | [03-verification-causal.md](03-verification-causal.md) |
| H6 | Failed causal analysis counted as non-causal | `causal.py:203` | [03-verification-causal.md](03-verification-causal.md) |
| H7 | Causal analysis has no batching — O(N) LLM calls | `causal.py:137` | [03-verification-causal.md](03-verification-causal.md) |
| H8 | Causal attributes overwritten for duplicate edges | `graph.py:203` | [04-graph-visualization.md](04-graph-visualization.md) |
| H9 | GEXF export drops most causal attributes | `graph.py:1481` | [04-graph-visualization.md](04-graph-visualization.md) |
| H10 | Verify command reads normalized CSV as raw Relationships | `relationships_cli.py:1777` | [05-cli-pipeline.md](05-cli-pipeline.md) |
| H11 | Pipeline stage ordering not enforced | `relationships_cli.py` | [05-cli-pipeline.md](05-cli-pipeline.md) |

### MEDIUM (16)

| ID | Finding | Location | Doc |
|----|---------|----------|-----|
| M1 | Entity deduplication is case-sensitive | `detector.py:168` | [01-detection-pipeline.md](01-detection-pipeline.md) |
| M2 | JSON parsing inconsistency across extractors | `components/*.py` | [01-detection-pipeline.md](01-detection-pipeline.md) |
| M3 | No validation of extracted relationships | `components/*.py` | [01-detection-pipeline.md](01-detection-pipeline.md) |
| M4 | Coreference resolution loses original text | `detector.py:98` | [01-detection-pipeline.md](01-detection-pipeline.md) |
| M5 | LLM canonical label generation async anti-pattern | `normalizer.py:310` | [02-normalization-design.md](02-normalization-design.md) |
| M6 | Entity frequency counts relationships, not text occurrences | `normalizer.py:231` | [02-normalization-design.md](02-normalization-design.md) |
| M7 | Case-insensitive grouping with case-dependent display | `normalizer.py:417` | [02-normalization-design.md](02-normalization-design.md) |
| M8 | Redundant confidence check semantics in verifier | `verifier.py:247` | [03-verification-causal.md](03-verification-causal.md) |
| M9 | VerifiedRelationship loses original reference for normalized input | `verifier.py:518` | [03-verification-causal.md](03-verification-causal.md) |
| M10 | Causal validation defaults to least-informative values | `causal.py:352` | [03-verification-causal.md](03-verification-causal.md) |
| M11 | Config comparison re-reads prompt file from disk | `causal.py:236` | [03-verification-causal.md](03-verification-causal.md) |
| M12 | Type detection based on first element only | `graph.py:99` | [04-graph-visualization.md](04-graph-visualization.md) |
| M13 | Self-loops silently dropped | `graph.py:123` | [04-graph-visualization.md](04-graph-visualization.md) |
| M14 | Isolated nodes silently dropped by build() | `graph.py:239` | [04-graph-visualization.md](04-graph-visualization.md) |
| M15 | Node frequency counts relationships, not text mentions | `graph.py:154` | [04-graph-visualization.md](04-graph-visualization.md) |
| M16 | Semantic layout hardcoded embedding model may not be installed | `graph.py:767` | [04-graph-visualization.md](04-graph-visualization.md) |

### LOW (9)

| ID | Finding | Location | Doc |
|----|---------|----------|-----|
| L1 | Summary context check uses magic string | `detector.py:127` | [01-detection-pipeline.md](01-detection-pipeline.md) |
| L2 | Single-item clusters get empty canonical initially | `normalizer.py:257` | [02-normalization-design.md](02-normalization-design.md) |
| L3 | JSON parsing regex may match nested objects | `verifier.py:455` | [03-verification-causal.md](03-verification-causal.md) |
| L4 | Progress callback fires irregularly | `causal.py:225` | [03-verification-causal.md](03-verification-causal.md) |
| L5 | Hardcoded UMAP random_state=42 | `graph.py:833` | [04-graph-visualization.md](04-graph-visualization.md) |
| L6 | adjustText not installed warning buried | `graph.py:1323` | [04-graph-visualization.md](04-graph-visualization.md) |
| L7 | Convex hull handles degenerate cases (positive) | `graph.py:1051` | [04-graph-visualization.md](04-graph-visualization.md) |
| L8 | Detector instance reuse undocumented | `relationships_cli.py:260` | [05-cli-pipeline.md](05-cli-pipeline.md) |
| L9 | Timestamp-based output directories may collide | `relationships_cli.py:249` | [05-cli-pipeline.md](05-cli-pipeline.md) |

### CLI-SPECIFIC (5 additional findings in [05-cli-pipeline.md](05-cli-pipeline.md))

| ID | Severity | Finding |
|----|----------|---------|
| M17 | MEDIUM | Three duplicate CSV reading functions |
| M18 | MEDIUM | No progress reporting during detection |
| M19 | MEDIUM | Output directory defaults to package directory |
| M20 | MEDIUM | --include-causal flag is a no-op |
| M21 | MEDIUM | No --timeout for causal/verify/graph commands |

---

## Cross-Cutting Concerns

### Zero Test Coverage

The relationships module has **zero dedicated test files**. Compare:

| Module | Test Files | Test Count |
|--------|-----------|------------|
| Entity | `test_entity_detector.py`, `test_bayesian.py`, `test_comparison.py`, `test_clustering.py`, `test_agreement.py` | ~80+ |
| Figurative | `test_mock_pipeline.py`, `test_domains.py`, `test_domain_normalizer_json.py` | ~30+ |
| **Relationships** | **None** | **0** |

This is the single most impactful finding. Without tests, any fix or refactor risks introducing regressions.

### Checkpoint Model is Dead Code

The `Checkpoint` dataclass in `models.py:607-642` is never used anywhere in the codebase. It was presumably designed for resuming interrupted processing but was never implemented.

### Only Ollama Provider in CLI

All CLI commands hardcode `OllamaProvider`. The `BaseLLMProvider` abstraction supports OpenAI and Anthropic, but the CLI doesn't expose `--provider` with those options (except for a non-functional `choices=["ollama"]` in causal and verify).

---

## Audit Document Index

| Document | Content |
|----------|---------|
| [00-audit-summary.md](00-audit-summary.md) | This file — executive summary |
| [01-detection-pipeline.md](01-detection-pipeline.md) | Detection, windowing, extractors, prompts |
| [02-normalization-design.md](02-normalization-design.md) | Normalization design concerns, clustering |
| [03-verification-causal.md](03-verification-causal.md) | Verification semantics, causal analysis |
| [04-graph-visualization.md](04-graph-visualization.md) | Graph construction, metrics, visualization |
| [05-cli-pipeline.md](05-cli-pipeline.md) | CLI orchestration, data flow, UX |
| [06-recommendations.md](06-recommendations.md) | Prioritized remediation plan |
