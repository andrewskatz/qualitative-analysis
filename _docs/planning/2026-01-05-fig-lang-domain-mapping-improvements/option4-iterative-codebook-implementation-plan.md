# Implementation Plan: Iterative Codebook Approach for Domain Normalization

**Date:** 2026-01-05  
**Status:** Draft  
**Priority:** 2 (Implement After Option 1)  
**Feature:** Replace simple clustering normalization with iterative LLM-assisted codebook generation and application

---

## 1. Problem Statement

The current domain normalization uses agglomerative clustering to group similar domains and then generates canonical labels. This has limitations:

1. **No deduplication logic**: Similar codes may be added to different clusters
2. **Post-hoc clustering**: Labels are generated after the fact, not during analysis
3. **No semantic consideration of similarity to existing codes**: Each cluster is processed independently

## 2. Proposed Solution: Two-Phase Codebook Approach

Based on the user's design, implement a sophisticated codebook generation and application workflow:

### Phase 1: Codebook Generation (from existing domains)

```
[Source/Target Domains from Analysis]
        ↓
1. Generate embeddings for all unique domains
        ↓
2. Cluster domains (AgglomerativeClustering)
        ↓
3. For each cluster:
   a. LLM suggests candidate code(s) for the cluster
   b. For each candidate:
      - RAG: Find top-K most similar codes already in codebook
      - LLM decides: Accept (new concept) or Reject (redundant)
   c. If accepted, add to codebook
        ↓
4. Final codebook: List of canonical codes
```

### Phase 2: Codebook Application

```
[Codebook from Phase 1]
        ↓
For each original source/target domain:
   a. RAG: Find top-K most similar codes from codebook
   b. LLM picks the best matching code
   c. Map: original_domain → canonical_code
        ↓
[Normalized Results]
```

---

## 3. Existing Infrastructure Analysis

### Available Patterns (Can Reuse)

| Component | Location | What It Does |
|-----------|----------|--------------|
| Clustering | [entity_consolidation_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/entity_consolidation_service.py#L33-L97) | AgglomerativeClustering with cosine distance |
| Batch Similarity | [entity_consolidation_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/entity_consolidation_service.py#L199-L243) | `compute_batch_similarity()` for top-K lookup |
| Embedding Model | Same file | Cached `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")` |
| LLM Service | [llm_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/llm_service.py) | JSON response generation |

### Not Existing (Must Build)

- Codebook data model (storing codes with embeddings)
- Iterative accept/reject decision logic
- RAG-based code matching for application
- Codebook management API endpoints
- Frontend codebook review UI

---

## 4. Proposed Changes

### 4.1 New Database Model

Create new table for storing codebooks:

```python
# backend/models/codebook_schemas.py (NEW FILE)

class Codebook(Base):
    """A collection of canonical codes for domain normalization."""
    __tablename__ = "codebooks"
    
    id = Column(UUID, primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Context: what job generated this
    source_job_id = Column(UUID, ForeignKey("figurative_source_target_jobs.id"))
    codebook_type = Column(String(50))  # "source_domains" or "target_domains"
    
    # Configuration used
    config = Column(JSONB)  # {similarity_threshold, top_k, model, etc.}
    
    # Status
    status = Column(String(20), default="pending")  # pending, generating, complete
    
    created_at = Column(DateTime, default=datetime.utcnow)


class CodebookEntry(Base):
    """Single code in a codebook."""
    __tablename__ = "codebook_entries"
    
    id = Column(UUID, primary_key=True, default=uuid4)
    codebook_id = Column(UUID, ForeignKey("codebooks.id", ondelete="CASCADE"))
    
    # The canonical code label
    code = Column(String(255), nullable=False)
    
    # Embedding for similarity search (stored as array)
    embedding = Column(ARRAY(Float))  # Or use pgvector if installed
    
    # Source cluster that generated this code
    source_cluster = Column(JSONB)  # {members: [...], avg_similarity: 0.85}
    
    # LLM reasoning for accepting this code
    reasoning = Column(Text)
    
    # Similar codes that were considered during deduplication
    compared_to = Column(JSONB)  # [{code: "X", similarity: 0.7}, ...]
    
    created_at = Column(DateTime, default=datetime.utcnow)
```

---

### 4.2 New Service: CodebookService

```python
# backend/services/codebook_service.py (NEW FILE)

class CodebookService:
    """Service for iterative codebook generation and application."""
    
    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = EntityConsolidationService()  # Reuse
        self.llm_service = None  # Initialized per-request
    
    # =========================================================================
    # Phase 1: Codebook Generation
    # =========================================================================
    
    async def generate_codebook(
        self,
        job_id: UUID,
        codebook_type: Literal["source_domains", "target_domains"],
        config: CodebookGenerationConfig
    ) -> Codebook:
        """
        Generate a codebook from domain labels in a source/target analysis job.
        
        Steps:
        1. Extract unique domains from job results
        2. Cluster domains by embedding similarity
        3. For each cluster, propose candidate codes
        4. For each candidate, check against existing codebook (accept/reject)
        5. Build final codebook
        """
        pass
    
    def _cluster_domains(
        self, 
        domains: List[str], 
        threshold: float
    ) -> List[List[str]]:
        """Cluster domains using existing EntityConsolidationService pattern."""
        # Reuse AgglomerativeClustering logic
        pass
    
    async def _propose_code_for_cluster(
        self,
        cluster_members: List[str],
        model: str,
        platform: str
    ) -> str:
        """LLM generates candidate code for a cluster."""
        prompt = f"""<task>
Given these semantically related domain labels, suggest a single canonical code (1-3 words, ALL CAPS) that captures the shared concept.

Domain labels:
{chr(10).join(f"- {m}" for m in cluster_members)}

Respond in JSON: {{"code": "YOUR_CODE"}}
</task>"""
        # Call LLM
        pass
    
    async def _should_accept_code(
        self,
        candidate_code: str,
        existing_codes: List[CodebookEntry],
        top_k: int,
        model: str,
        platform: str
    ) -> Tuple[bool, str]:
        """
        LLM decides whether to accept candidate or if it's redundant.
        
        Returns:
            (should_accept, reasoning)
        """
        # Step 1: Find top-K similar existing codes
        if not existing_codes:
            return True, "First code in codebook"
        
        existing_labels = [e.code for e in existing_codes]
        similarities = self.embedding_service.compute_batch_similarity(
            candidate_code, existing_labels
        )
        
        # Get top-K
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        similar_codes = [
            {"code": existing_labels[i], "similarity": similarities[i]}
            for i in sorted_indices
        ]
        
        # Step 2: LLM decides
        prompt = f"""<task>
You are building a codebook for qualitative research. Decide if this new code should be added.

Candidate code: "{candidate_code}"

Most similar existing codes:
{chr(10).join(f"- {c['code']} (similarity: {c['similarity']:.2f})" for c in similar_codes)}

If the candidate represents a DISTINCT concept not covered by existing codes, respond:
{{"accept": true, "reasoning": "Why this is a new concept"}}

If the candidate is REDUNDANT with an existing code, respond:
{{"accept": false, "reasoning": "Which existing code covers this", "merge_with": "EXISTING_CODE"}}
</task>"""
        # Call LLM, parse response
        pass
    
    # =========================================================================
    # Phase 2: Codebook Application
    # =========================================================================
    
    async def apply_codebook(
        self,
        job_id: UUID,
        codebook_id: UUID,
        codebook_type: Literal["source_domains", "target_domains"],
        top_k: int = 5,
        model: str = None,
        platform: str = "ollama"
    ) -> Dict[str, str]:
        """
        Apply a codebook to normalize domain labels.
        
        For each original domain:
        1. Find top-K most similar codes from codebook
        2. LLM picks the best match
        3. Return mapping: original -> canonical
        """
        pass
    
    async def _pick_best_code(
        self,
        original_domain: str,
        candidate_codes: List[Dict[str, Any]],  # [{code, similarity}, ...]
        model: str,
        platform: str
    ) -> str:
        """LLM picks the best matching code from candidates."""
        prompt = f"""<task>
Match this domain label to the most appropriate code from the codebook.

Original domain: "{original_domain}"

Candidate codes (from most to least similar):
{chr(10).join(f"- {c['code']} (similarity: {c['similarity']:.2f})" for c in candidate_codes)}

Pick the code that best represents the concept. If none fit well, pick the closest.

Respond in JSON: {{"selected_code": "CODE_NAME", "reasoning": "why"}}
</task>"""
        # Call LLM, return selected code
        pass
```

---

### 4.3 API Endpoints

```python
# backend/routers/codebook.py (NEW FILE)

router = APIRouter(prefix="/codebooks", tags=["codebooks"])

@router.post("/generate")
async def generate_codebook(
    job_id: UUID,
    codebook_type: Literal["source_domains", "target_domains"],
    config: CodebookGenerationConfig,
    background_tasks: BackgroundTasks
):
    """Generate a new codebook from job results."""
    pass

@router.get("/{codebook_id}")
async def get_codebook(codebook_id: UUID):
    """Get codebook with all entries."""
    pass

@router.post("/{codebook_id}/apply")
async def apply_codebook(
    codebook_id: UUID,
    job_id: UUID,
    config: CodebookApplicationConfig
):
    """Apply codebook to normalize domains in a job."""
    pass

@router.get("/{codebook_id}/entries")
async def list_entries(codebook_id: UUID):
    """List all codes in a codebook."""
    pass

@router.patch("/{codebook_id}/entries/{entry_id}")
async def update_entry(codebook_id: UUID, entry_id: UUID, update: CodeEntryUpdate):
    """Allow user to edit a codebook entry."""
    pass
```

---

### 4.4 Frontend Components

#### CodebookGenerationDialog.tsx (NEW)

- Trigger from source/target results panel
- Configure: clustering threshold, top-K for dedup, model
- Show progress during generation

#### CodebookReviewPanel.tsx (NEW)

- Display generated codebook with all entries
- Show cluster members that generated each code
- Allow manual editing of codes
- Export codebook as JSON/CSV

#### CodebookApplicationDialog.tsx (NEW)

- Select codebook to apply
- Configure top-K for matching
- Show preview of mappings before applying

---

## 5. Verification Plan

### 5.1 Unit Tests

Add to `tests/backend_api/test_codebook.py` (NEW):

```python
def test_cluster_domains():
    """Test domain clustering produces expected groupings."""
    
def test_should_accept_code_first_code():
    """First code should always be accepted."""
    
def test_should_accept_code_distinct_concept():
    """Distinct concepts should be accepted."""
    
def test_should_reject_redundant_code():
    """Redundant codes should be rejected."""
    
def test_apply_codebook_mapping():
    """Test that codebook application produces correct mappings."""
```

### 5.2 Integration Tests

```bash
# Run after implementation
pytest tests/backend_api/test_codebook.py -v
```

### 5.3 Manual Testing

1. Complete a source/target domain analysis with 50+ instances
2. Trigger "Generate Codebook" for source domains
3. Watch generation progress (should see accept/reject decisions)
4. Review generated codebook - verify no redundant codes
5. Apply codebook - verify mappings are sensible
6. Compare results with existing clustering normalization

---

## 6. Implementation Phases

### Phase 1: Backend Core (8-10 hours)
- [ ] Create database migration for codebook tables
- [ ] Implement `CodebookService` with generation logic
- [ ] Implement codebook application logic
- [ ] Add API endpoints
- [ ] Write unit tests

### Phase 2: Frontend Integration (6-8 hours)
- [ ] Create CodebookGenerationDialog
- [ ] Create CodebookReviewPanel
- [ ] Create CodebookApplicationDialog
- [ ] Integrate with FigurativeSourceTargetPanel
- [ ] Add progress indicators

### Phase 3: Polish (2-4 hours)
- [ ] Add export functionality
- [ ] Add manual code editing
- [ ] Performance optimization for large datasets
- [ ] Documentation

---

## 7. Dependencies

- Existing: `sentence-transformers`, `sklearn`, `numpy`
- Existing: `EntityConsolidationService` (reuse patterns)
- New: Consider `pgvector` extension for faster embedding search (optional)

---

## 8. Comparison with Current Normalization

| Aspect | Current Approach | Codebook Approach |
|--------|-----------------|-------------------|
| Deduplication | None (clusters processed independently) | LLM checks each code against existing |
| Consistency | Varies by cluster | Builds consistent vocabulary iteratively |
| Transparency | Cluster → LLM label (opaque) | Full reasoning for each decision |
| Reusability | Per-job only | Codebook can be saved and reapplied |
| User Control | Limited (threshold only) | Review/edit codes before application |
| Computational Cost | Lower | Higher (more LLM calls) |

---

## 9. Open Questions for User

1. **Codebook persistence**: Should codebooks be sharable across projects/batches?
   - Recommendation: Start with job-specific, add sharing later

2. **Embedding storage**: Should we use pgvector for faster similarity search?
   - Recommendation: Start with in-memory numpy, upgrade if needed

3. **Batch application**: Should users be able to apply one codebook to multiple jobs?
   - Recommendation: Yes, this is a key advantage of the approach

4. **Code editing**: How much manual control should users have?
   - Recommendation: Allow editing code labels, adding/removing codes
