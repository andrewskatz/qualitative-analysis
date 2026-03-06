# Decisions/Factors Port: Web App to qualitative-analysis Package

**Date:** 2026-02-28
**Status:** Planning

## Overview

Port the decisions/factors extraction pipeline from the `entity-id-app-v2` web app backend to the standalone `qualitative-analysis` Python package. This follows the same pattern used for porting entity scoring, figurative language detection, and relationship extraction.

## What Gets Ported

### From Web App
| File | Purpose |
|------|---------|
| `backend/services/extractors/semantic_two_pass_extractor.py` | RAG-based two-pass extraction (decisions then factors) |
| `backend/services/extractors/context_aware_extractor.py` | Context-aware extraction with summarization |
| `backend/services/extractors/decision_validator.py` | Rule-based decision validation |
| `backend/services/batch_aggregator.py` | Batch aggregation statistics |
| `src/prompts/decision_extraction/*.txt` | 11 versioned prompt files |
| `src/prompts/summarization/window_summary.txt` | Window summarization prompt |

### New in Python Package
| Component | Purpose |
|-----------|---------|
| `decisions/extractor.py` | Unified extractor with 3 strategies |
| `decisions/visualizer.py` | Matplotlib visualizations |
| `decisions/normalizer.py` | Semantic dedup across texts |
| Custom prompt support | Pass custom `.txt` files via CLI flags |

## Architecture

See [architecture.md](architecture.md) for detailed module design.

## Future Work

See [future-phases.md](future-phases.md) for Bayesian modeling and causal influence diagrams.

## Related Docs
- Entity pathway audit: `docs/planning/2026-02-07-entity-pathway-audit/`
- Entity scoring dimensions: `docs/planning/2026-02-18-qa-ent-score-dimensions-specification/`
