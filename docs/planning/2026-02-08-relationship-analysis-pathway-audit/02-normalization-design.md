# Normalization Design & Implementation Audit

**Scope**: `relationships/normalizer.py`, normalization pipeline design
**Key Context**: During the entity pathway audit, early normalization was identified as problematic because it merged entities that appeared in different contexts. The same concern applies here.

---

## D1: Early Normalization Loses Per-Participant and Per-Context Information

**Severity**: DESIGN ISSUE (HIGH impact)
**File**: `relationships/normalizer.py:402-468`
**Impact**: Per-participant relationship patterns are destroyed before analysis

The normalization step merges relationships with identical `(normalized_source, normalized_type, normalized_target)` tuples across ALL texts:

```python
# Line 417 - grouping key
key = (norm_source.lower(), norm_type.lower(), norm_target.lower())
grouped[key].append(rel)

# Lines 421-464 - merge all relationships with same key
count=len(rels),  # how many instances merged
window_indices=sorted(window_indices),  # aggregated
text_ids=sorted(text_ids),  # aggregated
```

This means:
- **Participant A** says "Technology → causes → Social change" (in education context)
- **Participant B** says "Technology → leads to → Social change" (in healthcare context)

After normalization: These become a single relationship with `count=2`, losing:
1. Which participant said what
2. The different contexts (education vs healthcare)
3. Individual descriptions/evidence

**Why this matters for relationships**: Unlike the entity scoring pathway (where normalization broke scoring), relationship normalization breaks:
- **Per-participant graph analysis**: Cannot build individual relationship networks
- **Group comparison**: Cannot compare relationship patterns between groups
- **Verification accuracy**: Merged relationships may reference wrong source text
- **Causal analysis**: Different contexts may have different causal attributions

**Recommendation**: The relationships pathway has a legitimate use case for normalization in **aggregate graph construction** (combining "causes" and "leads to" into one edge type). However, the normalization step should be:
1. **Optional** (it already is via `--no-normalize-entities` / `--no-normalize-types`)
2. **Clearly documented** as lossy and aggregate-only
3. **Positioned later in the pipeline** — after verification and causal analysis, just before graph construction
4. **Preserving per-text data** alongside the normalized form

**Current pipeline order**: detect → normalize → verify → causal → graph
**Recommended order**: detect → verify → causal → normalize (optional, for graph only) → graph

**Status: IMPLEMENTED** — The normalizer now accepts `CausalRelationship` and `NormalizedRelationship` inputs in addition to raw `Relationship` objects. The CLI `normalize` command uses the smart CSV reader that auto-detects input format. The unified CLI description documents the recommended pipeline order.

---

## H3: Confidence Always Hardcoded to 1.0

**Severity**: HIGH
**File**: `relationships/normalizer.py:460`
**Impact**: Misleading certainty signal — all normalizations appear perfect

```python
normalized.append(NormalizedRelationship(
    ...
    confidence=1.0,  # Could compute based on cluster similarity
    ...
))
```

Every normalized relationship gets `confidence=1.0` regardless of how questionable the entity clustering was. A marginal merge at threshold boundary (0.75 similarity) appears equally confident as an exact match (1.0 similarity).

The cluster's `avg_similarity` is computed and stored but never propagated to the relationship confidence.

**Fix**: Compute confidence from cluster similarity:
```python
source_cluster_sim = next(
    (c.avg_similarity for c in entity_clusters if rel.source in [m for m in c.members]),
    1.0
)
# confidence = min(source_sim, target_sim, type_sim)
```

---

## M5: LLM Canonical Label Generation Has Async Anti-Pattern

**Severity**: MEDIUM
**File**: `relationships/normalizer.py:310-320`
**Impact**: Potential crash in async contexts; deprecated Python API

```python
elif method == "llm" and llm_provider is not None:
    import asyncio
    try:
        loop = asyncio.get_event_loop()  # deprecated in 3.10+
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    canonical = loop.run_until_complete(
        self._generate_canonical_llm(members, llm_provider)
    )
```

Problems:
1. `asyncio.get_event_loop()` emits `DeprecationWarning` in Python 3.10+
2. `loop.run_until_complete()` raises `RuntimeError` if called from within an already-running event loop (which happens when the CLI calls this via `asyncio.run()`)
3. The same anti-pattern appears in `graph.py:1232-1251` for cluster label generation

**Fix**: Make `normalize()` an async method, or use `nest_asyncio` as a bridge.

---

## M6: Entity Frequency Counts Relationships, Not Text Occurrences

**Severity**: MEDIUM
**File**: `relationships/normalizer.py:231-244`
**Impact**: "Frequent" canonical method uses wrong frequency measure

```python
def _count_entities_and_types(self, relationships):
    entity_counts = defaultdict(int)
    for rel in relationships:
        entity_counts[rel.source] += 1  # counts as source
        entity_counts[rel.target] += 1  # counts as target
```

An entity appearing in 5 relationships has `count=5`, regardless of how many times it appeared in the source text. For the `canonical_method="frequent"` selection, this biases toward entities in dense relationship clusters rather than truly frequent mentions.

**Note**: This is acceptable for the relationships pipeline context (we care about relationship participation), but should be documented.

---

## M7: Case-Insensitive Grouping With Case-Dependent Display

**Severity**: MEDIUM
**File**: `relationships/normalizer.py:417, 424-427`
**Impact**: Display case of normalized relationships is non-deterministic

```python
# Grouping uses lowercase
key = (norm_source.lower(), norm_type.lower(), norm_target.lower())

# Display uses first relationship's mapping
first = rels[0]
norm_source = entity_mapping.get(first.source, first.source)  # depends on input order
```

If relationships arrive in different orders (e.g., due to async processing or different window orderings), the display case of the canonical form can change between runs.

**Fix**: Use the canonical form from the cluster directly rather than re-mapping from the first relationship.

---

## L2: Single-Item Clusters Get Empty Canonical Initially

**Severity**: LOW
**File**: `relationships/normalizer.py:257`
**Impact**: Confusing intermediate state for debugging

```python
# In _cluster_items
if len(items) == 1:
    return [EntityCluster(canonical="", members=items, avg_similarity=1.0)]
```

The canonical is set to empty string, then overwritten in `_assign_canonical_labels`. This is functionally correct but creates confusing intermediate state if anyone inspects cluster objects between these calls.
