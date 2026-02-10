# Verification & Causal Analysis Audit

**Scope**: `relationships/verifier.py`, `relationships/causal.py`

---

## VERIFICATION FINDINGS

### H4: NormalizedRelationship Verification Only Uses First text_id/window_index

**Severity**: HIGH
**File**: `relationships/verifier.py:346-347, 502-503`
**Impact**: Merged relationships verified against wrong text window; evidence loss

When grouping normalized relationships for verification, only the first text_id and first window_index are used:

```python
# _group_by_window (line 346-347)
if isinstance(rel, NormalizedRelationship):
    text_id = rel.text_ids[0] if rel.text_ids else ""
    window_index = rel.window_indices[0] if rel.window_indices else 0

# _to_verified_relationship (line 502-503)
if isinstance(rel, NormalizedRelationship):
    window_index = rel.window_indices[0] if rel.window_indices else 0
    text_id = rel.text_ids[0] if rel.text_ids else ""
```

A normalized relationship merged from 3 texts (text_ids=["t1", "t2", "t3"]) will only be verified against `t1`'s window. If the relationship was strongest in `t3`, verification may incorrectly reject it.

**Fix**: Verify against ALL source windows and accept if any pass threshold, or verify against the window with the most evidence.

---

### H5: Batch Verification Silently Loses Relationships

**Severity**: HIGH
**File**: `relationships/verifier.py:410-415`
**Impact**: Unverified relationships default to rejected without warning

```python
verification_map = {v["original_index"]: v for v in verifications}

for i, rel in enumerate(relationships):
    ver = verification_map.get(i, {})  # missing → empty dict
    is_supported = ver.get("is_supported", False)  # defaults to False
    confidence = ver.get("confidence", 0.0)         # defaults to 0.0
```

If the LLM's response is malformed and only returns verification for 3 of 10 relationships, the other 7 silently get `is_supported=False, confidence=0.0` with no warning logged. These appear as "confidently rejected" when they were actually "not evaluated."

**Fix**: Log a warning when relationships are missing from verification response:
```python
if i not in verification_map:
    logger.warning(f"Relationship {i} ({rel.source} → {rel.target}) missing from verification response")
```

---

### M8: Redundant Confidence Check Semantics

**Severity**: MEDIUM
**File**: `relationships/verifier.py:247-249`
**Impact**: Confusing accept/reject logic

```python
if vrel.verified and vrel.verification_confidence >= threshold:
    verified_rels.append(vrel)
else:
    rejected_rels.append(vrel)
```

The `verified` field is set from `is_supported` (boolean: "is this relationship in the text?") and `verification_confidence` is a separate score (0.0-1.0). The accept condition requires BOTH to pass.

Edge cases that may confuse users:
- `verified=True, confidence=0.3` → **rejected** (supported but low confidence)
- `verified=False, confidence=0.9` → **rejected** (high confidence it's NOT supported)

The semantics of "confidence" here is ambiguous: is it confidence that the relationship IS supported, or confidence in the LLM's assessment (either way)?

**Fix**: Clarify semantics in documentation, or use only confidence as the single decision variable.

---

### M9: VerifiedRelationship Loses Original Reference for Normalized Input

**Severity**: MEDIUM
**File**: `relationships/verifier.py:518`
**Impact**: Cannot trace verified relationships back to their normalized source

```python
return VerifiedRelationship(
    ...
    original_relationship=rel if isinstance(rel, Relationship) else None,
)
```

If input is `NormalizedRelationship`, `original_relationship` is set to `None`. Downstream code cannot access the normalized form's metadata (counts, all descriptions, all window_indices).

**Fix**: Add `original_normalized_relationship` field or make `original_relationship` accept both types.

---

### L3: JSON Parsing Regex May Match Nested Objects

**Severity**: LOW
**File**: `relationships/verifier.py:455-472`
**Impact**: Rare edge case in malformed LLM responses

The fallback regex `re.search(r'\{.*\}', clean_response, re.DOTALL)` uses greedy matching which will capture the largest possible `{...}` span. If the LLM response contains multiple JSON objects, this may capture too much and produce invalid JSON.

---

## CAUSAL ANALYSIS FINDINGS

### H6: Failed Causal Analysis Silently Counted as Non-Causal

**Severity**: HIGH
**File**: `relationships/causal.py:203-222`
**Impact**: Error rate inflates non-causal count; no way to distinguish failures from genuine non-causal

```python
except Exception as e:
    logger.error(f"Error analyzing {source} -> {target}: {e}")
    stats["errors"] += 1

    # Create relationship with null causal attributes
    causal_rel = CausalRelationship(
        ...
        causal=CausalAttributes(
            is_causal=False,        # ← marked as non-causal
            reasoning=f"Error: {str(e)}",
        ),
    )
    causal_relationships.append(causal_rel)
    stats["non_causal_count"] += 1  # ← inflates non-causal count
```

If the LLM is unavailable or returns garbage for 20% of relationships, the results show 20% more non-causal relationships than reality. The `errors` counter tracks this separately but:
1. The causal_percentage stat is computed from `causal_count / total`, not `causal_count / (total - errors)`
2. The output CSV has no field to distinguish "analyzed as non-causal" from "analysis failed"

**Fix**:
- Don't count errors in `non_causal_count`
- Add an `error` or `analysis_failed` field to CausalRelationship
- Compute `causal_percentage` excluding errors

---

### H7: Causal Analysis Has No Batching — O(N) LLM Calls

**Severity**: HIGH (performance)
**File**: `relationships/causal.py:137`
**Impact**: For 100 relationships, makes 100 separate LLM calls (~100 seconds at 1s/call)

```python
for i, rel in enumerate(relationships):
    # ... each iteration makes one LLM call
    attrs = await self._analyze_single(
        source=source, target=target, relationship=rel_type,
        evidence=evidence, llm_provider=llm_provider,
    )
```

Compare with the verifier, which batches up to 20 relationships per call. The causal analyzer processes one at a time. For datasets with hundreds of relationships, this is a significant performance bottleneck.

The `batch_size` parameter (line 91) is misleading — it's only used for progress reporting, not actual batching.

**Fix**: Implement true batching like the verifier:
```python
async def _analyze_batch(self, relationships, llm_provider):
    # Format multiple relationships in one prompt
    # Parse array response
```

---

### M10: Causal Validation Defaults to Least-Informative Values

**Severity**: MEDIUM
**File**: `relationships/causal.py:352-357`
**Impact**: Invalid LLM responses systematically bias toward uncertainty

```python
if polarity not in VALID_POLARITIES:
    polarity = "neutral"           # ← least informative
if certainty not in VALID_CERTAINTIES:
    certainty = "possible"          # ← least certain
if explicit_vs_implicit not in VALID_EXPLICIT_IMPLICIT:
    explicit_vs_implicit = "implicit"  # ← least certain
```

If the LLM returns creative values like "mixed", "somewhat", or "inferred", these are silently converted to the least-informative/most-uncertain defaults. This creates a systematic bias toward uncertainty in the aggregate statistics.

**Fix**: Log warnings when defaulting:
```python
if polarity not in VALID_POLARITIES:
    logger.warning(f"Invalid polarity '{polarity}', defaulting to 'neutral'")
    polarity = "neutral"
```

---

### M11: Config Comparison Re-Reads Prompt File From Disk

**Severity**: MEDIUM
**File**: `relationships/causal.py:236`
**Impact**: Unnecessary disk I/O on every analyze() call

```python
config = {
    ...
    "prompt_template": "default" if self._prompt_template == self._load_default_prompt() else "custom",
}
```

`self._load_default_prompt()` reads the prompt file from disk. This is called every time `analyze()` runs, just to check if the template is the default one. Wasteful, especially if the file is on a slow filesystem.

**Fix**: Cache the comparison result in `__init__`:
```python
self._is_default_prompt = (prompt_template is None)
```

---

### L4: Progress Callback Fires Irregularly

**Severity**: LOW
**File**: `relationships/causal.py:225-230`
**Impact**: Inconsistent progress reporting

```python
# Only fires every batch_size iterations
if on_progress and (i + 1) % batch_size == 0:
    on_progress(i + 1, len(relationships))

# Final callback always fires
if on_progress:
    on_progress(len(relationships), len(relationships))
```

With `batch_size=10` and 15 relationships: progress fires at 10 and 15, skipping 5 updates. The user sees 66%→100% with no intermediate feedback.
