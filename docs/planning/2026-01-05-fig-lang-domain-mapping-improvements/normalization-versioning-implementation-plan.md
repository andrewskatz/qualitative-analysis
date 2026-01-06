# Normalization Versioning Implementation Plan

Add dedicated `DomainNormalizationRun` table to support multiple normalization runs per source/target job, with version selection in UI.

## Proposed Changes

### Database Model

#### [NEW] [figurative_source_target_schemas.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/models/figurative_source_target_schemas.py)

Add new SQLAlchemy model and Pydantic schemas:

```python
class DomainNormalizationRun(BaseModel):
    __tablename__ = "domain_normalization_runs"
    
    job_id = Column(PGUUID(as_uuid=True), ForeignKey("figurative_source_target_jobs.id", ondelete="CASCADE"))
    name = Column(String(255))
    status = Column(String(20))  # pending, running, completed, failed
    configuration = Column(JSONB)  # {conservativeness, threshold, cluster_mode, canonical_method, model, abstraction_level}
    source_mapping = Column(JSONB)  # {original: canonical}
    target_mapping = Column(JSONB)
    source_clusters = Column(JSONB)  # [{canonical, members, count, avg_similarity}]
    target_clusters = Column(JSONB)
    stats = Column(JSONB)  # {original_source_count, normalized_source_count, ...}
    is_active = Column(Boolean, default=False)  # Only one active per job
    error = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
```

---

#### [NEW] Migration

Create Alembic migration `e2f3a4b5c6d7_add_domain_normalization_runs.py`:
- Create `domain_normalization_runs` table
- Migrate existing `domain_normalization_map` data from jobs table to new runs
- Keep `domain_normalization_map` column for backward compatibility

---

### Backend Service

#### [figurative_source_target_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py)

| Method | Changes |
|--------|---------|
| `normalize_domains()` | Create `DomainNormalizationRun` record, return run ID |
| `get_normalization_runs()` | NEW: List all runs for a job |
| `get_normalization_run()` | NEW: Get single run by ID |
| `set_active_normalization_run()` | NEW: Set is_active flag |
| `delete_normalization_run()` | NEW: Delete a run |
| `get_normalized_domain_graph()` | Accept `run_id` parameter, use run's mappings |

---

### API Endpoints

#### [figurative_source_target.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/routers/figurative_source_target.py)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs/{job_id}/normalization-runs` | GET | List normalization runs |
| `/jobs/{job_id}/normalization-runs` | POST | Create new run (replaces normalize-domains) |
| `/normalization-runs/{run_id}` | GET | Get run details |
| `/normalization-runs/{run_id}/activate` | POST | Set as active run |
| `/normalization-runs/{run_id}` | DELETE | Delete run |
| `/normalization-runs/{run_id}/graph` | GET | Get normalized graph for specific run |

> [!NOTE]
> Keep existing `/normalize-domains` endpoint for backward compatibility, but have it create a run internally.

---

### Frontend

#### [api-client.ts](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/api-client.ts)

Add types and API functions:
- `DomainNormalizationRun` type
- `listNormalizationRuns()`, `createNormalizationRun()`, `activateNormalizationRun()`, etc.

#### [FigurativeSourceTargetPanel.tsx](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/FigurativeSourceTargetPanel.tsx)

- Add run selector dropdown (shows run name + config summary)
- Show run list with config details (threshold, level, method)
- Allow switching between runs
- Show active run indicator
- Fix normalized graph to use selected run's data

---

## Verification Plan

### Existing Tests
No existing unit tests for domain normalization. Will add new tests.

### New Tests
Add to `tests/backend_api/test_figurative_source_target.py`:

```python
def test_create_normalization_run(...)
def test_list_normalization_runs(...)
def test_activate_normalization_run(...)
def test_delete_normalization_run(...)
def test_normalized_graph_uses_run(...)
```

**Command to run:**
```bash
pytest tests/backend_api/test_figurative_source_target.py -v -k normalization
```

### Manual Verification
1. Start backend + frontend
2. Open a completed source/target job
3. Run normalization with different settings (e.g., moderate vs aggressive threshold)
4. Verify both runs appear in a dropdown/list
5. Switch between runs and verify table columns update
6. Switch to "Normalized Graph" tab and verify graph reflects selected run
