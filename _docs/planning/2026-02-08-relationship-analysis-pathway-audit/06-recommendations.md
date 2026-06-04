# Recommendations — Prioritized Remediation Plan

**Date**: 2026-02-08

---

## Priority 1: Test Coverage Foundation (~8-12 hours)

**Rationale**: Every other fix risks regressions without tests. This must come first.

### 1a. Core Unit Tests (~4 hours)

Create `tests/test_relationship_detector.py`:
- Mock LLM provider (follow `test_entity_detector.py` pattern)
- Test two-pass strategy: entity extraction → relationship extraction
- Test one-pass strategy: combined extraction
- Test windowing behavior (multi-window text)
- Test entity deduplication
- Test context buffer isolation between detect() calls
- Test coreference resolution path
- Test empty/single-window inputs

Create `tests/test_relationship_models.py`:
- Serialization round-trips (to_dict → from_dict) for all model types
- Relationship.to_tuple() case handling
- CausalAnalysisResult.compute_statistics() edge cases
- NormalizationResult serialization with clusters

### 1b. Integration Tests (~4 hours)

Create `tests/test_relationship_normalizer.py`:
- Clustering with known similarity thresholds
- Canonical label selection methods (shortest, frequent, representative)
- Merge behavior for duplicate tuples
- Threshold presets resolution

Create `tests/test_relationship_pipeline.py`:
- End-to-end mock pipeline: detect → normalize → verify → causal → graph
- CSV input/output round-tripping
- Format detection in CLI readers

### 1c. CLI Tests (~2 hours)

Extend `tests/test_unified_cli.py`:
- `qa relationships detect --help` argument coverage
- `qa relationships normalize --help` argument coverage
- `qa relationships causal --help`
- `qa relationships verify --help`
- `qa relationships graph --help`

---

## Priority 2: Data Integrity Fixes (~4 hours)

These fixes prevent incorrect results in the current pipeline.

### 2a. Fix H1: Context Buffer Bleeding (~15 min)

Reset context buffer at start of each `detect()` call:
```python
# detector.py, top of detect()
self.context_buffer = EntityBuffer(self.context_buffer.max_size)
```

### 2b. Fix H2: Propagate text_id (~30 min)

Add `text_id` parameter to `detect()` and pass to all created Relationship objects. Update extractors to accept and propagate text_id.

### 2c. Fix H6: Separate Error Count from Non-Causal (~15 min)

In `causal.py`, don't increment `non_causal_count` for errors:
```python
except Exception as e:
    stats["errors"] += 1
    # Remove: stats["non_causal_count"] += 1
    causal_rel.causal = CausalAttributes(is_causal=False, reasoning=f"Error: {str(e)}")
```

Add `analysis_status` field: "analyzed", "error", "skipped".

### 2d. Fix H5: Log Missing Verifications (~15 min)

In `verifier.py`, warn when LLM response is incomplete:
```python
if i not in verification_map:
    logger.warning(f"Relationship {i} missing from LLM response, marking unverified")
```

### 2e. Fix H8: Causal Attribute Merging (~30 min)

In `graph.py`, accumulate causal attributes and resolve via majority vote instead of overwriting.

### 2f. Fix H9: Complete GEXF Export (~15 min)

Export all 4 causal attributes in `to_gexf()`.

### 2g. Fix H10: Verify Command Reads Normalized Format (~30 min)

Replace `_read_relationships_csv` in verify command with `_read_causal_input_csv` pattern that detects normalized format.

---

## Priority 3: Design Decision — Normalization Placement (~2 hours + discussion)

**Issue D1**: The current pipeline order (detect → normalize → verify → causal → graph) normalizes too early, losing per-participant context.

**Options**:

### Option A: Move Normalization to Just Before Graph (Recommended)

**Pipeline**: detect → verify → causal → normalize (optional) → graph

- Normalization only serves graph construction (combining similar entities/types)
- Verification operates on raw relationships with full context
- Causal analysis sees un-merged relationships with original evidence
- Per-participant analysis is possible at every stage before normalization

**Effort**: ~2 hours (reorder CLI documentation, update pipeline docs)
**Risk**: Low — normalization is already independent

### Option B: Two Normalization Modes

- **Pre-analysis normalization**: Current behavior, for quick aggregate views
- **Post-analysis normalization**: Normalize only at graph construction time

Users choose based on their needs. More flexible but more complex.

### Option C: Graph-Time Normalization (Implicit)

Move normalization logic into `graph.build()` as an optional parameter. Remove the separate normalize command entirely. The graph builder would cluster entities and types on-the-fly.

**Effort**: ~4 hours
**Risk**: Medium — changes CLI interface

---

## Priority 4: Quality Improvements (~3 hours)

### 4a. Fix M1: Case-Insensitive Entity Dedup (~15 min)

Update `_merge_entities()` to use lowercase comparison while preserving original case.

### 4b. Fix M2: Shared JSON Parsing Utility (~1 hour)

Create `core/json_utils.py` with a shared `parse_llm_json(response, expected_keys)` function. Update all extractors to use it.

### 4c. Fix M3: Post-Extraction Validation (~30 min)

Add validation for extracted relationships: non-empty source/target, reasonable length, known entity check for two-pass mode.

### 4d. Fix H3: Confidence from Cluster Similarity (~30 min)

Compute normalization confidence from source/target/type cluster similarities instead of hardcoding 1.0.

### 4e. Fix M10: Log Causal Validation Defaults (~15 min)

Add warnings when causal attribute values are invalid and defaulted.

### 4f. Fix M17: Unified CSV Reader (~1 hour)

Replace three CSV reading functions with one that auto-detects format.

---

## Priority 5: Performance & UX (~2 hours)

### 5a. Fix H7: Causal Analysis Batching (~2 hours)

Implement batch processing for causal analysis (group 5-10 relationships per LLM call). This could reduce analysis time by 5-10x.

### 5b. Fix M18: Detection Progress Reporting (~15 min)

Add per-row progress output in the detect command.

### 5c. Fix M19: Fix Default Output Directory (~15 min)

Change detect command default output to input file's parent directory.

### 5d. Fix M20: Remove --include-causal No-Op (~5 min)

Remove the ineffective flag.

### 5e. Fix M21: Add --timeout to All Commands (~15 min)

Add `--timeout` argument to causal, verify, and graph commands.

---

## Priority 6: Deferred / Lower Priority

| Finding | Effort | Notes |
|---------|--------|-------|
| M4: Coreference tracking | 30 min | Store original + resolved text |
| M5: Async anti-pattern in normalizer | 1 hour | Make normalize() async |
| M7: Case-dependent display | 15 min | Use canonical from cluster directly |
| M12: Mixed-type list detection | 30 min | Check all elements or use duck typing |
| M13: Self-loop option | 15 min | Add `allow_self_loops` parameter |
| M14: Isolated node tracking | 15 min | Add to metrics summary |
| L5: Configurable UMAP seed | 5 min | Add parameter |
| L9: Second-precision timestamps | 5 min | Add `%S` to format |
| Dead code: Checkpoint model | 5 min | Remove or implement |

---

## Summary: Effort Estimates

| Priority | Focus | Effort |
|----------|-------|--------|
| **P1** | Test coverage | 8-12 hours |
| **P2** | Data integrity fixes | 4 hours |
| **P3** | Normalization design decision | 2 hours + discussion |
| **P4** | Quality improvements | 3 hours |
| **P5** | Performance & UX | 2 hours |
| **P6** | Deferred | 3 hours |
| **Total** | | **22-26 hours** |

---

## Comparison with Entity Pathway Audit

| Dimension | Entity Pathway | Relationships Pathway |
|-----------|---------------|----------------------|
| **CRITICAL (crashes)** | 5 | 0 |
| **HIGH** | 15 | 11 |
| **MEDIUM** | 18 | 16+5 CLI |
| **Test coverage** | 80+ tests | 0 tests |
| **Architecture** | Monolithic scorer | Clean modular pipeline |
| **Design issues** | Independent vs compositional | Early normalization |
| **Maturity** | More battle-tested | Less exercised |

The relationships pathway is architecturally cleaner (better separation of concerns, modular components) but less battle-tested (zero tests, several data flow issues). The entity pathway had more severe bugs (crashes) but also had more test coverage to catch regressions.
