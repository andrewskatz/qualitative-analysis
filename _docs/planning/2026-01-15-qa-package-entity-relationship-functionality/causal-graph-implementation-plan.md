# Causal Graph Improvements Implementation Plan

## Overview
Enhance `qa rel graph` to add causal attribute visualization, scope selection, and label management to achieve parity with the web app.

## Files to Modify

| File | Changes |
|------|---------|
| `qualitative-analysis/src/qualitative_analysis/relationships/graph.py` | Add edge styling, scope, labels, legend |
| `qualitative-analysis/src/qualitative_analysis/relationships_cli.py` | Add new CLI flags |

## Implementation Tasks

### Task 1: Add Color Constants
Add at top of `graph.py` after imports:

```python
# Causal edge colors (from web app conventions)
POLARITY_COLORS = {
    "positive": "#22c55e",  # Green
    "negative": "#ef4444",  # Red
    "neutral": "#94a3b8",   # Slate gray
    None: "#94a3b8",        # Default gray
}

# Certainty line styles
CERTAINTY_STYLES = {
    "certain": "solid",
    "likely": (0, (5, 3)),      # Dashed
    "possible": (0, (2, 2)),    # Dotted
    None: "solid",
}
```

### Task 2: Modify `to_png()` Method
Update the edge drawing section (around line 546-555) to use causal attributes:

1. Build edge color list based on polarity
2. Build edge style list based on certainty
3. Use `FancyArrowPatch` for individual edge styling (since `draw_networkx_edges` doesn't support per-edge line styles)
4. Add `color_by` parameter: `"polarity"`, `"type"`, `"none"`

### Task 3: Add Label Management
Add `labels` parameter to `to_png()`:
- `"all"` - show all labels (current)
- `"none"` - hide all labels
- `"top:N"` - only show top N nodes by degree
- `"pagerank:N"` - only show top N by PageRank

Implementation:
```python
def _filter_labels(self, labels: Dict, mode: str) -> Dict:
    if mode == "none":
        return {}
    if mode == "all":
        return labels
    if mode.startswith("top:"):
        n = int(mode.split(":")[1])
        top_nodes = sorted(self._graph_data.nodes, key=lambda x: x.degree, reverse=True)[:n]
        return {n.id: labels[n.id] for n in top_nodes if n.id in labels}
    if mode.startswith("pagerank:"):
        n = int(mode.split(":")[1])
        top_nodes = sorted(self._graph_data.nodes, key=lambda x: x.pagerank, reverse=True)[:n]
        return {n.id: labels[n.id] for n in top_nodes if n.id in labels}
    return labels
```

### Task 4: Add Legend Generation
Add method to generate legend showing:
- Polarity colors (if `color_by="polarity"`)
- Certainty line styles
- Node size scale

```python
def _draw_legend(self, ax, color_by: str, has_causal: bool):
    handles = []
    if color_by == "polarity" and has_causal:
        for polarity, color in POLARITY_COLORS.items():
            if polarity:
                handles.append(mpatches.Patch(color=color, label=f"{polarity.title()} causality"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
```

### Task 5: Add Scope Selection
Add `scope` parameter to `build()` and `to_png()`:
- `"aggregate"` - all texts combined (current default)
- `"individual"` - generate separate graph per text_id
- `"text:ID"` - filter to specific text_id

For individual scope, add new method:
```python
def build_individual_graphs(self, relationships, output_dir, ...):
    """Build and save one graph per text_id."""
    by_text = defaultdict(list)
    for rel in relationships:
        text_ids = getattr(rel, "text_ids", []) or [getattr(rel, "text_id", "unknown")]
        for tid in text_ids:
            by_text[tid].append(rel)

    for text_id, rels in by_text.items():
        self.build(rels, ...)
        self.to_png(output_dir / f"graph_{text_id}.png", ...)
```

### Task 6: Improve Semantic Layout

The current semantic layout exists but produces poor results. Improvements:

1. **Add node spacing/repulsion after UMAP**:
```python
def _apply_repulsion(self, positions: Dict, min_distance: float = 0.05, iterations: int = 50):
    """Apply force-based repulsion to prevent node overlap."""
    import numpy as np
    nodes = list(positions.keys())
    coords = np.array([positions[n] for n in nodes])

    for _ in range(iterations):
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                diff = coords[i] - coords[j]
                dist = np.linalg.norm(diff)
                if dist < min_distance and dist > 0:
                    force = (min_distance - dist) / 2
                    direction = diff / dist
                    coords[i] += direction * force
                    coords[j] -= direction * force

    return {n: (coords[i][0], coords[i][1]) for i, n in enumerate(nodes)}
```

2. **Add hierarchical layout option** for causal chains:
```python
elif layout == "hierarchical":
    # Use graphviz dot layout for directed acyclic structure
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
```

3. **Improve spring layout parameters**:
- Increase `k` (node spacing) based on node count
- Add `--layout-spacing` flag for user control

4. **Add community detection layout**:
```python
elif layout == "community":
    # Detect communities and position nodes by group
    communities = nx.community.louvain_communities(G.to_undirected())
    # Position communities in grid, nodes within community use spring
```

### Task 7: Update CLI Arguments
In `add_relationships_graph_args()`, add:

```python
parser.add_argument("--color-by", choices=["polarity", "type", "none"], default="polarity")
parser.add_argument("--labels", default="all", help="Label mode: all, none, top:N, pagerank:N")
parser.add_argument("--scope", default="aggregate", help="aggregate, individual, or text:ID")
parser.add_argument("--show-legend/--no-legend", default=True)
```

### Task 7: Update `run_relationships_graph()`
- Pass new parameters to graph builder
- Handle scope="individual" by calling `build_individual_graphs()`
- Handle scope="text:ID" by filtering relationships before build

## Implementation Order

1. **Add color constants**
2. **Modify edge coloring in to_png()** - most complex, add polarity/certainty styling
3. **Improve semantic layout** - add repulsion, improve spacing
4. **Add hierarchical layout option**
5. **Add label filtering**
6. **Add legend generation**
7. **Add scope filtering**
8. **Update CLI arguments**
9. **Test with existing causal CSV**

## Testing

Run after implementation:
```bash
# Test polarity coloring with semantic layout
qa rel graph tests/output/pipeline_test/run_20260119-1353/causal/combined_rows_relationships_edges_20260119-1353_causal.csv \
    --layout semantic \
    --color-by polarity \
    --labels top:30 \
    --show-legend \
    --output-dir ./tests/output/graph_improved

# Test hierarchical layout for causal chains
qa rel graph tests/output/pipeline_test/run_20260119-1353/causal/combined_rows_relationships_edges_20260119-1353_causal.csv \
    --layout hierarchical \
    --color-by polarity \
    --output-dir ./tests/output/graph_hierarchical

# Test individual scope
qa rel graph tests/output/pipeline_test/run_20260119-1353/causal/combined_rows_relationships_edges_20260119-1353_causal.csv \
    --scope individual \
    --output-dir ./tests/output/individual_graphs

# Compare layouts
qa rel graph tests/output/pipeline_test/run_20260119-1353/causal/combined_rows_relationships_edges_20260119-1353_causal.csv \
    --layout spring --output-dir ./tests/output/compare/spring
qa rel graph tests/output/pipeline_test/run_20260119-1353/causal/combined_rows_relationships_edges_20260119-1353_causal.csv \
    --layout semantic --output-dir ./tests/output/compare/semantic
```

## Verification Checklist

### Edge Styling
- [ ] Positive causal edges render green (#22c55e)
- [ ] Negative causal edges render red (#ef4444)
- [ ] Non-causal/neutral edges render gray (#94a3b8)
- [ ] Certain relationships show solid lines
- [ ] Possible relationships show dashed lines

### Layout
- [ ] Semantic layout produces less overlap than current version
- [ ] Hierarchical layout shows causal chains top-to-bottom
- [ ] Node repulsion prevents label overlap
- [ ] `--layout-spacing` adjusts node separation

### Labels & Legend
- [ ] `--labels top:20` only shows 20 node labels
- [ ] `--labels none` hides all labels
- [ ] Legend shows polarity color meanings
- [ ] Legend shows certainty line style meanings

### Scope
- [ ] `--scope individual` creates one PNG per text_id
- [ ] `--scope text:response_Alex` filters to single text
- [ ] Individual graphs maintain same styling as aggregate
