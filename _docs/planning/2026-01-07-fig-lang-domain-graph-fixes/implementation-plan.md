# Domain Graph Abstraction Level Selector - Implementation Plan

**Date:** 2026-01-07  
**Feature:** Raw Domain Graph Abstraction Level Selection  
**Estimated Effort:** 4-6 hours

---

## Objective

Add an abstraction level dropdown to the **raw Domain Graph** (not Normalized Graph) that allows users to select which domain abstraction level (`specific`, `moderate`, `abstract`) to visualize. Currently, the raw domain graph uses the `moderate` level implicitly.

---

## Background

### Multi-Level Domain Structure

Each figurative language result has domains extracted at three abstraction levels:

```json
{
  "source_domain": "military conflict",  // moderate (default)
  "source_domain_levels": {
    "specific": "trench warfare",
    "moderate": "military conflict",
    "abstract": "conflict"
  },
  "target_domain": "love",
  "target_domain_levels": {
    "specific": "romantic love",
    "moderate": "love",
    "abstract": "emotion"
  }
}
```

### Current State

- **Raw Domain Graph:** Uses `result.source_domain` (moderate level)
- **Normalized Graph:** Uses the abstraction level selected during normalization (configurable per run)

---

## Proposed Changes

### 1. Backend API Changes

#### [figurative_source_target.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/routers/figurative_source_target.py#L191-L220)

Add `abstraction_level` query parameter to `GET /jobs/{job_id}/graph`:

```python
@router.get(
    "/jobs/{job_id}/graph",
    response_model=DomainGraphResponse,
    summary="Get domain graph for a source/target analysis job"
)
async def get_domain_graph(
    job_id: UUID,
    abstraction_level: Optional[str] = Query(
        "moderate", 
        regex="^(specific|moderate|abstract)$",
        description="Domain abstraction level to use for graph"
    ),
    service: FigurativeSourceTargetService = Depends(get_service)
):
    # ... existing checks ...
    graph_data = service.get_domain_graph(job_id, abstraction_level=abstraction_level)
```

---

### 2. Backend Service Changes

#### [figurative_source_target_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L837-L854)

Update `get_domain_graph()` and `_generate_domain_graph()` to accept abstraction level:

```python
def get_domain_graph(
    self, 
    job_id: UUID, 
    abstraction_level: str = "moderate"
) -> Optional[Dict[str, Any]]:
    """Get domain graph for a job, generating if not cached or level differs."""
    job = self.get_job(job_id)
    if not job:
        return None
    
    # Check if cached graph matches requested level
    if job.graph_data:
        cached_level = job.graph_data.get("abstraction_level", "moderate")
        if cached_level == abstraction_level:
            return job.graph_data
    
    # Generate fresh graph for requested level
    if job.status == FigurativeSourceTargetStatus.COMPLETED.value:
        graph_data = self._generate_domain_graph(job_id, abstraction_level)
        # Only cache "moderate" level (default)
        if abstraction_level == "moderate":
            job.graph_data = graph_data
            self.db.commit()
        return graph_data
    
    return None
```

Update `_generate_domain_graph()` to extract domains at selected level:

```python
def _generate_domain_graph(
    self, 
    job_id: UUID, 
    abstraction_level: str = "moderate"
) -> Dict[str, Any]:
    """Generate domain graph data from results, including semantic positions."""
    # ... existing setup ...
    
    for result in results:
        # Get source domain at selected abstraction level
        source = result.source_domain
        if abstraction_level != "moderate" and result.raw_response:
            levels = result.raw_response.get("source_domain_levels", {})
            source = levels.get(abstraction_level) or source
        
        # Get target domain at selected abstraction level
        target = result.target_domain
        if abstraction_level != "moderate" and result.raw_response:
            levels = result.raw_response.get("target_domain_levels", {})
            target = levels.get(abstraction_level) or target
        
        # ... rest of aggregation logic ...
    
    # Add abstraction_level to returned data for cache validation
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {...},
        "abstraction_level": abstraction_level
    }
```

---

### 3. Frontend API Client Changes

#### [api-client.ts](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/api-client.ts#L2160-L2168)

Update `getFigurativeSourceTargetGraph()` to accept abstraction level:

```typescript
export async function getFigurativeSourceTargetGraph(
  jobId: string,
  abstractionLevel: string = "moderate"
): Promise<DomainGraphData> {
  const r = await fetch(
    `${API_BASE}/figurative-analysis/source-target/jobs/${jobId}/graph?abstraction_level=${abstractionLevel}`,
    { cache: "no-store" }
  );
  if (!r.ok) throw new Error(`Get domain graph failed: ${r.status}`);
  return r.json();
}
```

---

### 4. Frontend Panel Changes

#### [FigurativeSourceTargetPanel.tsx](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/FigurativeSourceTargetPanel.tsx)

Add state and dropdown in the Graph tab:

```tsx
// Add state
const [graphAbstractionLevel, setGraphAbstractionLevel] = 
  useState<"specific" | "moderate" | "abstract">("moderate");

// Update loadGraph to pass abstraction level
const loadGraph = useCallback(async () => {
  const data = await getFigurativeSourceTargetGraph(job.id, graphAbstractionLevel);
  setGraphData(data);
}, [job.id, graphAbstractionLevel]);

// Trigger reload when level changes
useEffect(() => {
  if (activeTab === "graph" && job.status === "completed") {
    loadGraph();
  }
}, [graphAbstractionLevel, activeTab, job.status, loadGraph]);
```

Add dropdown in the Graph tab toolbar (near the layout toggle):

```tsx
<Select value={graphAbstractionLevel} onValueChange={(v) => setGraphAbstractionLevel(v as any)}>
  <SelectTrigger className="w-[140px]">
    <SelectValue placeholder="Domain Level" />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="specific">Specific</SelectItem>
    <SelectItem value="moderate">Moderate</SelectItem>
    <SelectItem value="abstract">Abstract</SelectItem>
  </SelectContent>
</Select>
```

---

## Edge Cases

1. **Missing level data:** Some results may lack `source_domain_levels`. Fall back to `source_domain` (moderate).
2. **Cache management:** Only cache "moderate" level graph; regenerate others on demand.
3. **Semantic positions:** Positions are recomputed when abstraction level changes (different domain labels = different embeddings).

---

## Verification Plan

### Existing Tests (Regression Check)
```bash
cd "/Users/akatz4/Documents/ak fac/research/projects/entity-id-app-v2"
pytest tests/backend_api/test_figurative_source_target.py -v
```

### Manual Verification
1. Start backend: `uvicorn backend.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to a completed source/target job with multi-level domain data
4. Open Domain Graph tab
5. Verify dropdown appears next to layout toggle
6. Select each abstraction level and verify:
   - Graph updates with appropriate domains
   - Node labels change (e.g., "trench warfare" vs "military conflict" vs "conflict")
   - Semantic layout still works (positions recomputed)

---

## Acceptance Criteria

- [ ] Dropdown appears in Domain Graph toolbar (not Normalized Graph)
- [ ] Selecting a level triggers graph refresh
- [ ] Node labels reflect the selected abstraction level
- [ ] Semantic layout works correctly with the selected level
- [ ] Graceful fallback when level data is missing
- [ ] Existing tests pass (no regressions)
