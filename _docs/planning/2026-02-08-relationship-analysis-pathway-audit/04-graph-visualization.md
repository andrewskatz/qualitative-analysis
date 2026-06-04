# Graph Construction & Visualization Audit

**Scope**: `relationships/graph.py`

---

## H8: Causal Attributes Overwritten for Duplicate Edges

**Severity**: HIGH
**File**: `relationships/graph.py:203-211`
**Impact**: Only the last causal classification survives for edges with multiple instances

```python
# Add causal attributes if available
if is_causal and include_causal:
    causal = getattr(rel, "causal", None)
    if causal and causal.is_causal:
        edges_data[edge_key]["causal_attributes"] = {  # ← overwrites
            "is_causal": causal.is_causal,
            "polarity": causal.polarity,
            "certainty": causal.certainty,
            "explicit_vs_implicit": causal.explicit_vs_implicit,
        }
```

If edge `A→causes→B` appears 3 times with different causal attributes (e.g., once "positive/certain" and twice "negative/possible"), only the last one wins. The first two classifications are silently lost.

For aggregate graphs, the correct approach would be majority voting or averaging across instances.

**Fix**: Accumulate causal attributes and use majority voting:
```python
if "causal_votes" not in edges_data[edge_key]:
    edges_data[edge_key]["causal_votes"] = []
edges_data[edge_key]["causal_votes"].append({...})
# Then resolve via majority vote in a post-processing step
```

---

## H9: GEXF Export Drops Most Causal Attributes

**Severity**: HIGH
**File**: `relationships/graph.py:1481-1488`
**Impact**: Gephi users lose certainty, explicit/implicit, and evidence data

```python
if edge.causal_attributes:
    G[edge.source][edge.target]["is_causal"] = edge.causal_attributes.get("is_causal", False)
    G[edge.source][edge.target]["polarity"] = edge.causal_attributes.get("polarity", "")
    # ← certainty, explicit_vs_implicit, reasoning NOT exported
```

Only `is_causal` and `polarity` are written to GEXF. `certainty`, `explicit_vs_implicit`, and evidence are lost. Users importing into Gephi for analysis won't see these attributes.

**Fix**: Export all causal attributes:
```python
for attr in ["is_causal", "polarity", "certainty", "explicit_vs_implicit"]:
    G[edge.source][edge.target][attr] = edge.causal_attributes.get(attr, "")
```

---

## M12: Type Detection Based on First Element Only

**Severity**: MEDIUM
**File**: `relationships/graph.py:99-100`
**Impact**: Mixed-type lists produce incorrect behavior

```python
is_normalized = isinstance(relationships[0], NormalizedRelationship)
is_causal = isinstance(relationships[0], CausalRelationship)
```

If the list contains a mix of types (e.g., from merging multiple pipeline outputs), the first element determines behavior for all. No homogeneity check is performed.

Additionally, since `CausalRelationship` is NOT a subclass of `NormalizedRelationship`, a list of CausalRelationship objects will have `is_normalized=False`, causing the code to miss `original_source`/`original_target` fields that CausalRelationship also has.

**Fix**: Check all elements or use duck typing:
```python
is_causal = any(isinstance(r, CausalRelationship) for r in relationships)
is_normalized = any(isinstance(r, NormalizedRelationship) for r in relationships)
```

---

## M13: Self-Loops Silently Dropped

**Severity**: MEDIUM
**File**: `relationships/graph.py:123-124`
**Impact**: Self-referential relationships lost without notification

```python
if source_id == target_id:
    continue  # silently dropped
```

Self-referential relationships like "Technology → improves → Technology" (meaning technology bootstrapping) are silently discarded. There's no logging, counting, or option to include them.

In some qualitative research contexts, self-referential relationships are meaningful (e.g., "poverty → perpetuates → poverty").

**Fix**: Log dropped self-loops and add an `allow_self_loops` parameter:
```python
if source_id == target_id:
    if not allow_self_loops:
        self_loop_count += 1
        continue
```

---

## M14: Isolated Nodes Silently Dropped by build()

**Severity**: MEDIUM
**File**: `relationships/graph.py:239-240`
**Impact**: Entities mentioned in relationships below min_edge_weight disappear

```python
for node_id, data in nodes_data.items():
    if node_id not in G.nodes:
        continue  # Skip nodes without edges meeting min_edge_weight
```

When `min_edge_weight > 1`, entities that only appear in low-frequency relationships are excluded from both the graph AND the node list. The `filter()` method has `include_isolated` parameter, but `build()` always excludes.

**Fix**: Track filtered-out nodes in metrics:
```python
metrics_summary["filtered_nodes"] = len(nodes_data) - len(nodes)
```

---

## M15: Node Frequency Counts Relationship Participation, Not Text Mentions

**Severity**: MEDIUM
**File**: `relationships/graph.py:154`
**Impact**: Misleading frequency metric

```python
nodes_data[source_id]["frequency"] += count
```

For raw relationships, `count=1` per relationship. So `frequency` = number of relationships involving this entity. This is NOT how many times the entity was mentioned in the original text.

For normalized relationships, `count` is the number of merged instances, so frequency reflects relationship consolidation, not text frequency.

The metric label "frequency" implies text occurrence frequency but measures relationship participation.

**Fix**: Rename to `relationship_count` or add separate `text_mention_count` from entity extraction data.

---

## M16: Semantic Layout Hardcoded Embedding Model May Not Be Installed

**Severity**: MEDIUM
**File**: `relationships/graph.py:767, 855`
**Impact**: Semantic layout silently fails, falls back to spring layout

```python
def _compute_semantic_positions(
    self,
    node_labels,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",  # specific model
    ...
```

The default embedding model `Qwen/Qwen3-Embedding-0.6B` requires downloading ~1.2GB. If not installed, the method returns empty positions (line 789) and the caller falls back to spring layout (line 951) with only a warning.

**Fix**: Check model availability before attempting and suggest installation:
```python
if not _model_available(embedding_model):
    logger.warning(f"Embedding model '{embedding_model}' not available. "
                   f"Install with: pip install sentence-transformers && "
                   f"python -c 'from sentence_transformers import SentenceTransformer; "
                   f"SentenceTransformer(\"{embedding_model}\")'")
```

---

## L5: Hardcoded UMAP random_state=42

**Severity**: LOW
**File**: `relationships/graph.py:833`
**Impact**: Cannot perform sensitivity analysis with different seeds

```python
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=n_neighbors,
    min_dist=umap_min_dist,
    metric='cosine',
    random_state=42,  # hardcoded
)
```

Fixed seed is good for reproducibility by default, but the `to_png()` method doesn't expose a `random_state` parameter.

---

## L6: adjustText Not Installed Warning Buried

**Severity**: LOW
**File**: `relationships/graph.py:1323-1324`
**Impact**: Labels overlap significantly without adjustText; users may not notice warning

```python
except ImportError:
    logger.warning("adjustText not installed. Labels may overlap. Install with: pip install adjustText")
```

The warning only appears in log output, which users may not see. The resulting visualization will have overlapping labels that significantly reduce readability.

**Fix**: Add adjustText to optional dependencies in pyproject.toml under a `[viz]` extra, and add a visible note in the PNG title when labels are unadjusted.

---

## L7: Convex Hull Handles Degenerate Cases

**Severity**: LOW (positive finding)
**File**: `relationships/graph.py:1051-1074`
**Impact**: None — properly handled

The convex hull drawing correctly:
- Requires `len(points) >= 3` before attempting
- Wraps in try-except for QhullError (collinear points)
- Handles missing scipy gracefully

This is good defensive programming.
