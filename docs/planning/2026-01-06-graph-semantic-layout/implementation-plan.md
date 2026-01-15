# Semantic Graph Layout Implementation Plan
**Date:** 2026-01-06  
**Status:** Draft

---

## Overview

Add a "Semantic" layout option to the Domain Graph that positions nodes based on embeddings projected to 2D, so semantically similar domains cluster together.

**Architecture:** Backend pre-computes 2D coordinates using UMAP on sentence embeddings, frontend applies as preset layout.

---

## Changes Summary

| Layer | File | Changes |
|-------|------|---------|
| Backend | `figurative_source_target_service.py` | Add `_compute_semantic_positions()` method |
| Backend | `figurative_source_target_schemas.py` | Add `x`, `y` to DomainGraphNode |
| Backend | Router endpoints | Pass positions through graph responses |
| Frontend | `api-client.ts` | Update DomainGraphNode type |
| Frontend | `DomainGraph.tsx` | Add "Semantic" layout option |

---

## Backend Changes

### 1. Schema: Add Position Fields

#### [figurative_source_target_schemas.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/models/figurative_source_target_schemas.py)

```python
class DomainGraphNode(PydanticBaseModel):
    id: str
    label: str
    domain_type: str  # "source", "target", "both"
    frequency: int
    as_source: int
    as_target: int
    raw_domains: Optional[List[str]] = None  # For normalized nodes
    x: Optional[float] = None  # Semantic layout position
    y: Optional[float] = None
```

### 2. Service: Compute Semantic Positions

#### [figurative_source_target_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py)

Add method to compute 2D positions:

```python
def _compute_semantic_positions(
    self,
    domain_labels: List[str],
    embedding_model: SentenceTransformer
) -> Dict[str, Tuple[float, float]]:
    """
    Compute 2D positions for domains using UMAP projection of embeddings.
    
    Returns dict mapping domain label -> (x, y) normalized to [0, 1000] range.
    """
    import umap
    
    if len(domain_labels) < 2:
        # Can't project with <2 points
        return {label: (500.0, 500.0) for label in domain_labels}
    
    # Generate embeddings
    embeddings = embedding_model.encode(domain_labels)
    
    # Project to 2D with UMAP
    n_neighbors = min(15, len(domain_labels) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='cosine',
        random_state=42  # Reproducible
    )
    coords_2d = reducer.fit_transform(embeddings)
    
    # Normalize to [0, 1000] range for Cytoscape
    x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
    y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
    
    positions = {}
    for i, label in enumerate(domain_labels):
        x = (coords_2d[i, 0] - x_min) / (x_max - x_min + 1e-6) * 1000
        y = (coords_2d[i, 1] - y_min) / (y_max - y_min + 1e-6) * 1000
        positions[label.lower()] = (float(x), float(y))
    
    return positions
```

### 3. Update Graph Methods

Modify `get_domain_graph()` and `get_normalized_domain_graph()`:

```python
def get_domain_graph(self, job_id: UUID, include_positions: bool = True) -> Dict:
    # ... existing code to build nodes/edges ...
    
    if include_positions:
        # Load embedding model and compute positions
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cpu')
        domain_labels = [node["label"] for node in nodes]
        positions = self._compute_semantic_positions(domain_labels, embedding_model)
        
        # Add positions to nodes
        for node in nodes:
            pos = positions.get(node["label"].lower(), (500, 500))
            node["x"] = pos[0]
            node["y"] = pos[1]
    
    return {"nodes": nodes, "edges": edges, "stats": stats}
```

### 4. Add Dependency

```bash
pip install umap-learn
```

Add to `requirements.txt` / `pyproject.toml`.

---

## Frontend Changes

### 1. Update Types

#### [api-client.ts](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/api-client.ts)

```typescript
export type DomainGraphNode = {
  id: string;
  label: string;
  domain_type: "source" | "target" | "both";
  frequency: number;
  as_source: number;
  as_target: number;
  raw_domains?: string[];
  x?: number;  // Semantic layout position
  y?: number;
};
```

### 2. Add Semantic Layout Option

#### [DomainGraph.tsx](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/DomainGraph.tsx)

Update layout state and options:

```typescript
const [layout, setLayout] = useState<"fcose" | "dagre" | "semantic">("fcose");

// In layout change effect:
useEffect(() => {
  if (!cyRef.current) return;
  
  if (layout === "semantic") {
    // Check if nodes have positions
    const hasPositions = nodes.some(n => n.x !== undefined && n.y !== undefined);
    if (!hasPositions) {
      console.warn("Semantic layout requires pre-computed positions");
      return;
    }
    
    // Use preset layout with backend-provided positions
    cyRef.current.layout({
      name: "preset",
      positions: (node: cytoscape.NodeSingular) => {
        const nodeData = nodes.find(n => n.id === node.id());
        return {
          x: nodeData?.x ?? 500,
          y: nodeData?.y ?? 500
        };
      },
      fit: true,
      padding: 30,
      animate: true,
      animationDuration: 500,
    }).run();
  } else if (layout === "fcose") {
    // ... existing fcose config
  } else {
    // ... existing dagre config
  }
}, [layout, nodes]);
```

Update dropdown:

```tsx
<Select value={layout} onValueChange={(v) => setLayout(v as "fcose" | "dagre" | "semantic")}>
  <SelectTrigger className="w-[130px] h-8">
    <SelectValue />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="fcose">Force Layout</SelectItem>
    <SelectItem value="dagre">Hierarchical</SelectItem>
    <SelectItem value="semantic">Semantic</SelectItem>
  </SelectContent>
</Select>
```

---

## Verification Plan

### Unit Tests

Add test for position computation:

```python
def test_compute_semantic_positions():
    service = FigurativeSourceTargetService(db)
    embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cpu')
    
    domains = ["war", "conflict", "battle", "love", "romance", "affection"]
    positions = service._compute_semantic_positions(domains, embedding_model)
    
    # Check all domains have positions
    assert len(positions) == 6
    
    # Check positions are in range
    for label, (x, y) in positions.items():
        assert 0 <= x <= 1000
        assert 0 <= y <= 1000
    
    # Check war-related terms are closer together than war-love
    war_pos = positions["war"]
    conflict_pos = positions["conflict"]
    love_pos = positions["love"]
    
    war_conflict_dist = ((war_pos[0] - conflict_pos[0])**2 + (war_pos[1] - conflict_pos[1])**2)**0.5
    war_love_dist = ((war_pos[0] - love_pos[0])**2 + (war_pos[1] - love_pos[1])**2)**0.5
    
    assert war_conflict_dist < war_love_dist  # war closer to conflict than love
```

### Manual Testing

1. Load a job with normalized domains
2. View Domain Graph or Normalized Graph
3. Select "Semantic" layout from dropdown
4. Verify semantically similar domains cluster together
5. Compare with Force/Hierarchical layouts

---

## Performance Considerations

- **Embedding generation:** ~0.5s for 100 domains (CPU)
- **UMAP projection:** ~1-2s for 100 domains
- **Caching:** Consider caching positions in job/run metadata if regeneration is slow

## Decisions

1. **Cache positions?** → **Yes**, store in job's `graph_data` column to avoid recomputation
2. **Support un-normalized graphs?** → **Yes**, semantic layout works on both raw and normalized graphs

---

## Caching Strategy

Store computed positions in the job's `graph_data` JSONB column:

```python
# After computing positions, cache in job metadata
job.graph_data = job.graph_data or {}
job.graph_data["semantic_positions"] = {
    "raw": positions_dict,  # For raw domain graph
    "normalized": {},       # Populated when normalized graph requested
    "computed_at": datetime.utcnow().isoformat()
}
self.db.commit()
```

**Cache invalidation:**
- Raw positions: recompute if results are added/removed
- Normalized positions: recompute when normalization is re-run

**Retrieval:**
```python
def get_domain_graph(self, job_id: UUID) -> Dict:
    job = self.get_job(job_id)
    
    # Check cache first
    cached = job.graph_data.get("semantic_positions", {}).get("raw")
    if cached:
        positions = cached
    else:
        # Compute and cache
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cpu')
        domain_labels = [node["label"] for node in nodes]
        positions = self._compute_semantic_positions(domain_labels, embedding_model)
        
        # Store in cache
        job.graph_data = job.graph_data or {}
        job.graph_data.setdefault("semantic_positions", {})["raw"] = positions
        self.db.commit()
    
    # Add positions to nodes
    for node in nodes:
        pos = positions.get(node["label"].lower(), (500, 500))
        node["x"] = pos[0]
        node["y"] = pos[1]
```
