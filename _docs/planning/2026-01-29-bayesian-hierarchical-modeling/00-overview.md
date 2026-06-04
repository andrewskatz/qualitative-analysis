# Bayesian Hierarchical Modeling for Entity Score Comparison

**Date:** 2026-01-29
**Package:** `qualitative-analysis`
**Component:** `qa entity` — Phase 2 of Multi-Participant Comparison Plan
**Parent Plan:** [00-multi-participant-comparison-plan.md](../2026-01-23-qa-package-entity-scoring-enhancements/00-multi-participant-comparison-plan.md)

---

## Motivation

The existing group comparison pipeline (Phases 1, 3, 4 — completed 2026-01-29) provides:
- **Distance-based comparison** using Aitchison and EMD metrics
- **Effect size ratios** (between-group / within-group distance)
- **Permutation testing** for statistical significance (p-values)

These methods answer *whether* groups differ, but have limitations:
1. **No uncertainty propagation** — multiple LLM scoring runs produce variance that is collapsed to means before comparison
2. **No partial pooling** — each participant's scores are treated independently, with no borrowing of information across participants or entities
3. **No dimension-level inference** — effect sizes are computed over the full multivariate profile, not per-dimension
4. **No probability statements** — we get p-values, but not posterior probabilities like P(ecological > social on dimension X)

A Bayesian hierarchical model addresses all four limitations.

---

## What This Adds to the Pipeline

| Capability | Current (Distance-based) | Proposed (Bayesian) |
|-----------|--------------------------|---------------------|
| Group separation | Effect size ratio | Posterior group contrasts |
| Significance | Permutation p-value | Full posterior distributions |
| Uncertainty | Ignored (mean of runs) | Propagated through hierarchy |
| Per-dimension | Not supported | Native (separate or joint model) |
| Variance decomposition | Not supported | ICC at each level |
| Entity-level detail | Not supported | Shrinkage estimates per entity |
| Small sample behavior | Limited | Regularized via priors |

---

## Key Design Decision: Beta, Not Dirichlet

The planning document (Phase 2, Section 2.2) originally proposed Dirichlet-based models under the assumption that entity scores are compositional (sum to 100%). After investigation:

**Scores are NOT compositional.** Each dimension (e.g., Social, Ecological, Technological) is scored independently on a 0–100 scale. A participant can score an entity as 80/80/80 or 20/20/20 — the dimensions do not trade off against each other.

Therefore:
- **Dirichlet likelihood is inappropriate** — it would impose a false sum-to-1 constraint
- **Beta likelihood is correct** — each dimension is modeled independently as a proportion on [0, 1] (after rescaling from 0–100)
- This is a meaningful departure from the original plan and should be documented as such

---

## Scope

This planning document covers:
1. **Model specification** — the full hierarchical structure ([01-model-specification.md](./01-model-specification.md))
2. **Implementation plan** — module design, dependencies, CLI integration ([02-implementation-plan.md](./02-implementation-plan.md))

### Out of Scope (for now)
- Dirichlet regression for truly compositional data (Phase 5 — if future data warrants it)
- Longitudinal comparison (Phase 5)
- Cross-entity comparison (Phase 5)

---

## Prerequisites

Before implementing, the following must be in place (all are currently satisfied):
- [x] Multi-participant scoring pipeline (`qa entity score`)
- [x] Group assignment infrastructure (`--groups`, `--groups-file`, `--group-by-col`)
- [x] Distance-based comparison baseline (`qa entity compare`)
- [x] Permutation testing for significance validation
- [x] End-to-end validation on 15-participant simulated dataset (3 groups x 5 participants)

---

## Document Index

| Document | Description |
|----------|-------------|
| [00-overview.md](./00-overview.md) | This document — motivation, scope, key decisions |
| [01-model-specification.md](./01-model-specification.md) | Statistical model structure, priors, likelihood, outputs |
| [02-implementation-plan.md](./02-implementation-plan.md) | Module design, dependencies, phased implementation, CLI |
