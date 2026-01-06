# Domain Mapping Workflow Audit

**Date:** 2026-01-05  
**Auditor:** AI Assistant  
**Scope:** Source/target domain mapping and domain normalization workflow for figurative language analysis

---

## Executive Summary

This audit examines the domain mapping processes for metaphors and analogies in the figurative language analysis workflow. The implementation is largely well-designed and functional, following good software engineering practices. However, several areas for improvement were identified across code efficiency, linguistic accuracy, and user experience.

### Key Findings Summary

| Category | Severity | Count |
|----------|----------|-------|
| 🔴 High Priority | Potential bugs/logic issues | 3 |
| 🟡 Medium Priority | Performance/efficiency | 4 |
| 🟢 Low Priority | Enhancements | 5 |

---

## 1. Architecture Overview

The domain mapping workflow consists of two main phases:

### Phase 1: Source/Target Domain Identification
```
[Figurative Batch Complete] 
        ↓
User triggers "Source/Target Domain Analysis"
        ↓
Filter to metaphors/analogies/extended_metaphors
        ↓
For each instance: LLM identifies source + target domains
        ↓
Store results + generate domain graph
```

### Phase 2: Domain Normalization (Optional)
```
[Source/Target Analysis Complete]
        ↓
User triggers "Normalize Domains"
        ↓
Cluster semantically similar domains via embeddings
        ↓
Generate canonical labels (LLM or representative)
        ↓
Store normalization mapping
```

**Key Files:**
- Backend Service: [figurative_source_target_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py)
- Router: [figurative_source_target.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/routers/figurative_source_target.py)
- Frontend Panel: [FigurativeSourceTargetPanel.tsx](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/FigurativeSourceTargetPanel.tsx)
- Domain Graph: [DomainGraph.tsx](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/DomainGraph.tsx)

---

## 2. High Priority Issues

### 2.1 🔴 Domain Normalization Mapping Key Mismatch

**Location:** [figurative_source_target_service.py:812-822](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L812-L822)

**Problem:** The normalization mapping stores raw domain labels as keys, but when normalizing for the graph, the code normalizes domains with `.lower().strip()` before lookup.

```python
# In normalize_domains():
source_mapping = {}
for cluster in source_clusters:
    for member in cluster["members"]:
        source_mapping[member] = cluster["canonical"]  # Keys are raw

# In get_normalized_domain_graph():
source_lower = source.lower().strip()  # Normalized key
canonical_source = source_mapping.get(source_lower, source_lower.upper())  # May not match!
```

**Impact:** If the original domain was "Money/Currency" but stored as-is, looking up "money/currency" will fail and return a default instead of the canonical label.

**Recommendation:** Ensure consistent key normalization when building and querying the mapping.

---

### 2.2 🔴 Duplicate Representative Label Calculation Logic

**Location:** Multiple methods in [figurative_source_target_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py)

**Problem:** The same logic for calculating the most representative cluster member (highest average cosine similarity) is duplicated in three places:

1. `_generate_canonical_labels_representative()` (lines 924-962)
2. `_get_representative_label()` (lines 1051-1077) 
3. `_cluster_domains()` calculates avg_similarity but doesn't reuse it

**Impact:** Code maintainability issues; changes to the algorithm must be made in multiple places.

**Recommendation:** Extract into a single utility function:
```python
def _find_most_central_member(
    self, 
    members: List[str], 
    embeddings: np.ndarray
) -> Tuple[int, float]:
    """Find the member with highest average similarity to others."""
    # Single implementation
```

---

### 2.3 🔴 Missing Window Context Fallback in Prompt

**Location:** [figurative_source_target_service.py:446](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L446)

**Problem:** When `window_text` is empty/None, the prompt uses "No additional context available." However, for **v2** prompts that heavily rely on contextual analysis, this may reduce accuracy significantly.

```python
prompt = prompt_template.format(
    figurative_type=instance_data["figurative_type"],
    figurative_text=instance_data["figurative_text"],
    window_text=instance_data["window_text"] or "No additional context available.",
)
```

**Impact:** The v2 prompt asks the LLM to identify "what is actually being talked about in context" but may receive no context.

**Recommendation:** 
1. Add a warning log when context is missing
2. Consider falling back to the full original text if window is unavailable
3. Consider adjusting v2 prompt instructions when context is missing

---

## 3. Medium Priority Issues

### 3.1 🟡 Embedding Model Loaded Repeatedly

**Location:** [figurative_source_target_service.py:780](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L780)

**Problem:** The embedding model is loaded fresh for every normalization request:
```python
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cpu')
```

**Impact:** Loading the model takes several seconds and consumes memory unnecessarily for repeated requests.

**Recommendation:** Cache the embedding model at service or module level:
```python
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cpu')
    return _embedding_model
```

---

### 3.2 🟡 Synchronous LLM Calls in Normalization

**Location:** [figurative_source_target_service.py:706-858](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L706-L858)

**Problem:** The `normalize_domains()` method makes synchronous LLM calls when generating canonical labels (one per cluster). This blocks the main thread and makes the endpoint slow.

**Impact:** If there are 20 clusters, this means 20 sequential LLM calls, potentially taking 20+ seconds.

**Recommendation:** 
1. Make `_generate_canonical_labels_llm()` async
2. Use asyncio.gather() to parallelize LLM calls
3. Or convert to background task like `run_analysis()`

---

### 3.3 🟡 Hardcoded Embedding Model

**Location:** [figurative_source_target_service.py:780](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L780)

**Problem:** The embedding model is hardcoded:
```python
embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device='cpu')
```

**Impact:** Users cannot choose a different embedding model that may be more appropriate for their domain (e.g., domain-specific embeddings for medical or legal texts).

**Recommendation:** Make embedding model configurable as a parameter with a sensible default.

---

### 3.4 🟡 Missing Test Coverage for Normalization

**Location:** [test_figurative_source_target.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/tests/backend_api/test_figurative_source_target.py)

**Problem:** Current test file has 9 tests covering job CRUD and graph generation, but **no tests** for:
- Domain normalization clustering
- Canonical label generation
- Normalized graph generation
- Normalization with different conservativeness levels

**Impact:** The most complex logic in the service (normalization) is untested.

**Recommendation:** Add unit tests for:
- `_cluster_domains()` with known embeddings
- `_generate_canonical_labels_representative()`
- Key mismatch scenarios
- Edge cases (single domain, all identical domains)

---

## 4. Low Priority Improvements

### 4.1 🟢 Inconsistent Prompt Versioning Strategy

**Problem:** Three prompt versions exist (v1, v2, v2-numeric) with subtle differences:

| Version | Confidence Format | Extra Fields |
|---------|------------------|--------------|
| v1 | Numeric (0.0-1.0) | None |
| v2 | Categorical ("high/medium/low") | reasoning, conventionality, ambiguity_notes |
| v2-numeric | Numeric (0.0-1.0) | reasoning, conventionality, ambiguity_notes |

The code parses categorical confidence to numeric, but the v2/v2-numeric split seems unnecessary.

**Recommendation:** Consider consolidating to v1 and v2 only, with numeric confidence expected from both.

---

### 4.2 🟢 Graph Edge ID Format Potentially Conflicting

**Location:** [figurative_source_target_service.py:664](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L664)

**Problem:** Edge IDs are generated as:
```python
"id": f"{source}-->{target}"
```

If domain names contain `-->`, this could create parsing ambiguity.

**Recommendation:** Use a delimiter that's less likely to appear in text, like `|||` or a UUID.

---

### 4.3 🟢 Limited Context from Window Stack

**Problem:** The source/target analysis only uses the immediate window text, even though the figurative language detection uses a rolling summary buffer for context.

**Impact:** Extended metaphors or metaphors that rely on document-level theme may be misidentified.

**Recommendation:** Consider including rolling summaries or expanded window context from the parent batch analysis.

---

### 4.4 🟢 No Caching of Normalized Graph

**Problem:** While the base graph is cached on the job (`job.graph_data`), the normalized graph is regenerated on every request.

**Location:** [figurative_source_target_service.py:1114-1204](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py#L1114-L1204)

**Recommendation:** Cache normalized graph after normalization is run, similar to base graph.

---

### 4.5 🟢 Domain Labels Not Normalized Consistently Across Result Storage

**Problem:** Raw domain labels from LLM responses are stored as-is in results:
```python
source_domain=parsed.get("source_domain"),  # Could be "Money", "money", "MONEY"
```

The graph generation normalizes with `.lower().strip()` but stored results may show inconsistent casing.

**Recommendation:** Consider normalizing labels at storage time, or add a display_domain vs canonical_domain distinction.

---

## 5. Strengths of Current Implementation

✅ **Well-structured service architecture** - Clean separation between service, router, and frontend  
✅ **Multiple prompt versions** - Supports A/B testing and iteration  
✅ **Configurable clustering** - Conservativeness presets make it user-friendly  
✅ **Good error handling** - Graceful fallbacks when LLM calls fail  
✅ **Comprehensive schema design** - Pydantic models are well-defined with validation  
✅ **Domain graph visualization** - Cytoscape.js integration is sophisticated  

---

## 6. Recommended Priority Order

1. **Fix normalization mapping key mismatch** (2.1) - Bug causing incorrect normalization
2. **Add normalization tests** (3.4) - De-risk future changes
3. **Extract duplicate similarity logic** (2.2) - Improve maintainability
4. **Cache embedding model** (3.1) - Easy performance win
5. **Async normalization LLM calls** (3.2) - Better UX for large datasets
6. **Add missing context warning** (2.3) - Improve analysis quality

---

## 7. Testing Recommendations

To verify any fixes, the following test commands can be run:

### Existing Tests
```bash
cd /Users/akatz4/Documents/ak\ fac/research/projects/entity-id-app-v2
pytest tests/backend_api/test_figurative_source_target.py -v
```

### Manual Testing
1. Create a figurative language batch with metaphors
2. Run source/target domain analysis
3. Trigger domain normalization with different conservativeness levels
4. Verify normalized graph shows merged domains correctly
5. Check that domains with different casing are properly normalized

---

## Appendix: File References

| Component | Path | Lines |
|-----------|------|-------|
| Service | `backend/services/figurative_source_target_service.py` | 1205 |
| Router | `backend/routers/figurative_source_target.py` | 367 |
| Schemas | `backend/models/figurative_source_target_schemas.py` | 523 |
| Tests | `tests/backend_api/test_figurative_source_target.py` | 352 |
| Frontend Panel | `frontend/app/components/FigurativeSourceTargetPanel.tsx` | 983 |
| Domain Graph | `frontend/app/components/DomainGraph.tsx` | ~400 |
| Design Doc | `docs/planning/2025-12-25-fig-lang-source-target-labeling/02-design-document.md` | 609 |
| Normalization Spec | `docs/planning/2025-12-25-fig-lang-source-target-labeling/04-domain-normalization-feature.md` | 207 |
