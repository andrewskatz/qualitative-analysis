# Causal Graph Improvements Plan

**Date:** 2026-01-20
**Package:** `qualitative-analysis`
**Component:** `qa rel graph`
**Priority:** High

---

## Executive Summary

The current `qa rel graph` command produces a basic network visualization that lacks the sophisticated styling, layout algorithms, and flexibility present in the web application. This plan outlines improvements to achieve feature parity and add CLI-specific enhancements.

### Current State Issues

1. **Layout**: Spring layout only, resulting in overlapping labels and poor readability
2. **Styling**: No causal attribute visualization (polarity, certainty, explicitness)
3. **Scope**: Only aggregate graphs; no per-participant/per-text graph support
4. **Node Sizing**: Basic degree-based sizing without dimension coloring
5. **Labels**: Cluttered, no smart label placement or filtering

---

## Part 1: Visual Styling Improvements

### 1.1 Edge Styling (Causal Attributes)

Align with web app conventions from `relationship-graph.tsx` and `graph_image_export_service.py`:

| Attribute | Visual Encoding | Values |
|-----------|----------------|--------|
| **Polarity** | Edge Color | Positive: `#22c55e` (green), Negative: `#ef4444` (red), Neutral: `#94a3b8` (slate) |
| **Certainty** | Line Style | Certain: solid, Likely: dashed (short), Possible: dashed (long) |
| **Explicitness** | Arrow Style | Explicit: filled arrow, Implicit: hollow arrow |
| **Weight** | Line Width | Range 1-6px scaled by edge count/weight |

**Implementation Notes:**
- Detect if input CSV has `is_causal`, `polarity`, `certainty`, `explicit_vs_implicit` columns
- Fall back to neutral gray if causal attributes not present
- Add `--color-by` flag: `polarity` (default), `type`, `none`

### 1.2 Node Styling

| Attribute | Visual Encoding | Source |
|-----------|----------------|--------|
| **Degree** | Node Size | 20-60px range based on in+out degree |
| **Dimension Scores** | Node Color | RGB blend if SETS scores available |
| **Uncertainty** | Border/Opacity | Double orange border or reduced opacity for high CV |
| **Centrality** | Label Priority | PageRank determines which labels shown |

**Dimension Colors (SETS Framework):**
```python
DIMENSION_COLORS = {
    'social': (255, 99, 132),       # Red/Pink
    'ecological': (75, 192, 192),   # Teal
    'technological': (54, 162, 235) # Blue
}
```

### 1.3 Legend

Add automatic legend generation showing:
- Polarity color meanings
- Line style meanings (certainty)
- Node size scale
- Dimension color key (if applicable)

---

## Part 2: Layout Algorithms

### 2.1 Current Layout Options

Enhance `--layout` flag with additional algorithms:

| Layout | Description | Best For |
|--------|-------------|----------|
| `spring` | Force-directed (current) | General use, <100 nodes |
| `semantic` | Embedding + UMAP | Semantically meaningful positioning |
| `circular` | Nodes on circle | Clear edge visibility |
| `hierarchical` | Top-down DAG | Causal chains |
| `community` | Cluster-based | Grouped entities |

### 2.2 Semantic Layout Implementation

Port from `figurative/domains/graph.py`:

```python
def compute_semantic_layout(nodes: List[str], embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Position nodes based on semantic similarity using embeddings + UMAP.

    Steps:
    1. Generate embeddings for all node labels
    2. Apply PCA for dimensionality reduction (if >50 dims)
    3. Project to 2D using UMAP
    4. Scale to visualization bounds
    """
    from sentence_transformers import SentenceTransformer
    import umap

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(nodes)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(nodes) - 1),
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    positions = reducer.fit_transform(embeddings)

    return {node: (x, y) for node, (x, y) in zip(nodes, positions)}
```

### 2.3 Label Placement

Implement smart label strategies:

| Strategy | Flag | Behavior |
|----------|------|----------|
| All labels | `--labels all` | Show all (current behavior) |
| Top-N by degree | `--labels top:20` | Only top 20 nodes by degree |
| Top-N by PageRank | `--labels pagerank:20` | Only top 20 by centrality |
| No labels | `--labels none` | Hide all labels |
| Numbered | `--labels numbered` | Numbers on nodes, legend on side |

Use `adjustText` library for label repulsion to avoid overlaps.

---

## Part 3: Individual vs Aggregate Graphs

### 3.1 Scope Selection

Add `--scope` flag:

| Scope | Flag | Description |
|-------|------|-------------|
| Aggregate | `--scope aggregate` | All texts combined (current default) |
| Individual | `--scope individual` | One graph per text_id |
| Filtered | `--scope text:ID` | Single specific text_id |

### 3.2 Individual Graph Output

When `--scope individual`:
- Create subdirectory per text_id
- Generate separate graph files for each
- Include text_id in filename: `graph_{text_id}.png`
- Summary file listing all generated graphs

### 3.3 Aggregate Graph Enhancements

For aggregate mode:
- Track edge frequency across texts (`text_count`)
- Scale edge width by frequency
- Add `--min-texts` filter: only show edges appearing in N+ texts
- Node tooltip/label can show "appears in X texts"

### 3.4 CSV Column Requirements

| Scope | Required Columns |
|-------|-----------------|
| Aggregate | `source`, `target`, `type` |
| Individual | `source`, `target`, `type`, `text_id` |
| Causal styling | + `is_causal`, `polarity`, `certainty`, `explicit_vs_implicit` |

---

## Part 4: CLI Interface Updates

### 4.1 New/Modified Flags

```
qa rel graph INPUT_CSV [OPTIONS]

Layout Options:
  --layout {spring,semantic,circular,hierarchical,community}
  --layout-seed INT          Random seed for reproducibility

Styling Options:
  --color-by {polarity,type,none}
  --node-size-by {degree,pagerank,uniform}
  --edge-width-by {weight,count,uniform}
  --show-legend / --no-legend

Label Options:
  --labels {all,none,numbered,top:N,pagerank:N}
  --label-size INT           Font size for labels (default: 8)

Scope Options:
  --scope {aggregate,individual}
  --scope text:TEXT_ID       Filter to specific text
  --min-texts INT            Min texts for edge to appear (aggregate only)
  --min-edge-weight FLOAT    Min weight threshold

Output Options:
  --output-dir PATH
  --output-format {png,svg,pdf,gexf,json,csv}
  --figsize W,H              Figure dimensions in inches
  --dpi INT                  Resolution for raster formats
```

### 4.2 Example Commands

```bash
# Basic aggregate graph with causal coloring
qa rel graph relationships_causal.csv --color-by polarity --layout spring

# Semantic layout with top entities labeled
qa rel graph relationships.csv --layout semantic --labels top:30

# Individual graphs per participant
qa rel graph relationships.csv --scope individual --output-dir ./participant_graphs

# High-quality export for publication
qa rel graph relationships.csv --layout semantic --figsize 16,12 --dpi 300 \
    --output-format svg,pdf --labels pagerank:25

# Filtered view of specific text
qa rel graph relationships.csv --scope text:response_Alex --layout spring
```

---

## Part 5: Implementation Roadmap

### Phase 1: Edge Styling (Priority: High)
- [ ] Detect causal columns in input CSV
- [ ] Implement polarity-based edge coloring
- [ ] Implement certainty-based line styles (solid/dashed)
- [ ] Add edge width scaling by weight/count
- [ ] Add `--color-by` flag

### Phase 2: Layout Improvements (Priority: High)
- [ ] Implement semantic layout using embeddings + UMAP
- [ ] Add circular layout option
- [ ] Improve spring layout parameters for less overlap
- [ ] Add `--layout-seed` for reproducibility

### Phase 3: Label Management (Priority: Medium)
- [ ] Implement `--labels` flag with multiple modes
- [ ] Integrate `adjustText` for label repulsion
- [ ] Add numbered labels with legend option

### Phase 4: Scope Selection (Priority: Medium)
- [ ] Add `--scope` flag parsing
- [ ] Implement individual graph generation loop
- [ ] Add text_id filtering
- [ ] Implement `--min-texts` aggregate filtering

### Phase 5: Node Styling (Priority: Medium)
- [ ] Add dimension-based RGB blending if scores available
- [ ] Implement uncertainty visualization (border/opacity)
- [ ] Add `--node-size-by` options

### Phase 6: Export & Polish (Priority: Low)
- [ ] Add SVG and PDF export options
- [ ] Generate automatic legends
- [ ] Add GEXF export for Gephi
- [ ] Performance optimization for large graphs

---

## Part 6: Technical Dependencies

### Required Packages
```
networkx>=3.0        # Graph operations (existing)
matplotlib>=3.7      # Visualization (existing)
sentence-transformers # Semantic embeddings
umap-learn           # Dimensionality reduction
adjustText           # Label placement (optional)
```

### File Modifications

| File | Changes |
|------|---------|
| `relationships/graph.py` | Add layout algorithms, edge styling, scope handling |
| `relationships_cli.py` | Add new CLI flags, update argument parser |
| `core/colors.py` | New file for shared color constants |
| `requirements.txt` | Add `umap-learn`, `adjustText` |

---

## Part 7: Acceptance Criteria

### Edge Styling
- [ ] Positive causal edges render green, negative red
- [ ] Certain relationships show solid lines, possible show dashed
- [ ] Edge width scales with weight/frequency

### Layout
- [ ] Semantic layout produces semantically meaningful clustering
- [ ] Labels do not overlap significantly
- [ ] Layout is reproducible with `--layout-seed`

### Scope
- [ ] `--scope individual` creates one graph per text_id
- [ ] `--scope text:ID` filters to single text
- [ ] Aggregate graphs show edge frequency

### Quality
- [ ] Publication-ready output at 300 DPI
- [ ] Legends auto-generated for styling attributes
- [ ] Consistent with web app visual conventions

---

## References

- Web app interactive graph: `frontend/app/components/relationship-graph.tsx`
- Web app export service: `backend/services/graph_image_export_service.py`
- Figurative domain graph: `qualitative-analysis/src/qualitative_analysis/figurative/domains/graph.py`
- Color utilities: `frontend/app/utils/colorMappers.ts`
