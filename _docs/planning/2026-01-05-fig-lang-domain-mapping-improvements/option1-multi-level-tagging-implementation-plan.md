# Implementation Plan: Multi-Level Domain Tagging

**Date:** 2026-01-05  
**Status:** Draft  
**Priority:** 1 (Implement First)  
**Feature:** Add multi-level abstraction tagging to source/target domain analysis

---

## 1. Problem Statement

The current domain mapping prompts ask the LLM for a single "moderate" abstraction level for source and target domains. This creates inconsistency because:

1. "Moderate" is subjective and varies by LLM interpretation
2. Different research questions require different abstraction levels
3. Aggregation is unreliable when abstraction levels vary across instances

## 2. Proposed Solution

Modify the LLM prompt to request domains at **three abstraction levels** in a single pass:

| Level | Description | Example ("Time is money") |
|-------|-------------|---------------------------|
| **Specific** | Concrete terms from text | "cash exchange", "currency" |
| **Moderate** | Conceptual category | "commerce", "economics" |
| **Abstract** | High-level cognitive frame | "valuable resource", "commodity" |

Users can then choose which level to use for analysis/aggregation.

---

## 3. Proposed Changes

### Backend Service

#### [figurative_source_target_service.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/services/figurative_source_target_service.py)

**3.1 Add new prompt version (v3-multilevel)**

Create a new prompt template that requests all three abstraction levels:

```python
# New v3-multilevel prompt (to be added around line 370)
if prompt_version == "v3-multilevel":
    return """<system>
You are an expert in conceptual metaphor theory. Extract source and target domains at THREE abstraction levels.

ABSTRACTION LEVELS:
- SPECIFIC: Precise terms from the text (e.g., "sprinting", "highway driving")
- MODERATE: Conceptual category labels (e.g., "running/athletics", "vehicle motion")  
- ABSTRACT: High-level cognitive frames (e.g., "physical movement", "progression")
</system>

<figurative_language>
Type: {figurative_type}
Expression: "{figurative_text}"
</figurative_language>

<context>
{window_text}
</context>

<instructions>
Analyze this {figurative_type} expression.

Respond with a JSON object:
{{
  "reasoning": "Brief analysis of the metaphor",
  "source_domain": {{
    "specific": "1-3 word label at specific level",
    "moderate": "1-3 word label at moderate level", 
    "abstract": "1-3 word label at abstract level"
  }},
  "target_domain": {{
    "specific": "1-3 word label at specific level",
    "moderate": "1-3 word label at moderate level",
    "abstract": "1-3 word label at abstract level"
  }},
  "mapping_explanation": "How source helps understand target (1-2 sentences)",
  "confidence": 0.85
}}
</instructions>"""
```

**3.2 Update result parsing (`_parse_response`)**

Modify to handle nested domain structure:

```python
# In _parse_response(), add logic to extract multi-level domains
if isinstance(result.get("source_domain"), dict):
    # Multi-level format
    source_levels = result["source_domain"]
    result["source_domain_specific"] = source_levels.get("specific")
    result["source_domain_moderate"] = source_levels.get("moderate")
    result["source_domain_abstract"] = source_levels.get("abstract")
    # Keep moderate as default for backward compatibility
    result["source_domain"] = source_levels.get("moderate", "")
```

**3.3 Update result storage**

Store all three levels in the result:

```python
result = FigurativeSourceTargetResult(
    # ... existing fields ...
    source_domain=parsed.get("source_domain"),  # Moderate (default)
    target_domain=parsed.get("target_domain"),
    raw_response={
        **parsed,
        "source_levels": parsed.get("source_domain") if isinstance(parsed.get("source_domain"), dict) else None,
        "target_levels": parsed.get("target_domain") if isinstance(parsed.get("target_domain"), dict) else None,
        "_raw_llm_response": response,
    },
)
```

---

### Database Schema

#### [figurative_source_target_schemas.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/models/figurative_source_target_schemas.py)

**3.4 Add optional multi-level columns (future migration)**

For now, store in `raw_response` JSON. In a future phase, consider adding:

```python
# Optional - could add in later migration if needed for indexing
source_domain_specific: Optional[str] = None
source_domain_abstract: Optional[str] = None
target_domain_specific: Optional[str] = None
target_domain_abstract: Optional[str] = None
```

> [!NOTE]
> For MVP, store multi-level data in `raw_response` to avoid migration. Add dedicated columns only if needed for querying/indexing.

---

### Frontend

#### [FigurativeSourceTargetPanel.tsx](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/FigurativeSourceTargetPanel.tsx)

**3.5 Add abstraction level selector**

Add a dropdown in the Statistics tab to choose which level to display:

```tsx
const [abstractionLevel, setAbstractionLevel] = useState<'specific' | 'moderate' | 'abstract'>('moderate');

// In Statistics tab
<Select value={abstractionLevel} onValueChange={setAbstractionLevel}>
  <SelectTrigger>
    <SelectValue placeholder="Abstraction Level" />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="specific">Specific</SelectItem>
    <SelectItem value="moderate">Moderate (Default)</SelectItem>
    <SelectItem value="abstract">Abstract</SelectItem>
  </SelectContent>
</Select>
```

**3.6 Update results table**

Show multi-level domains when available:

```tsx
// In table row
const getSourceDomain = (result: FigurativeSourceTargetResult) => {
  const levels = result.raw_response?.source_levels;
  if (levels && abstractionLevel !== 'moderate') {
    return levels[abstractionLevel] || result.source_domain;
  }
  return result.source_domain;
};
```

#### [FigurativeSourceTargetDialog.tsx](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/frontend/app/components/FigurativeSourceTargetDialog.tsx)

**3.7 Add v3-multilevel prompt version option**

Add new option to prompt version dropdown:

```tsx
<SelectItem value="v3-multilevel">V3 - Multi-Level Abstraction</SelectItem>
```

---

### API

#### [figurative_source_target.py](file:///Users/akatz4/Documents/ak%20fac/research/projects/entity-id-app-v2/backend/routers/figurative_source_target.py)

No changes needed - existing API structure supports the new prompt version.

---

## 4. Verification Plan

### 4.1 Existing Tests

Run existing tests to ensure no regressions:

```bash
cd "/Users/akatz4/Documents/ak fac/research/projects/entity-id-app-v2"
pytest tests/backend_api/test_figurative_source_target.py -v
```

### 4.2 New Unit Tests to Add

Add tests in `tests/backend_api/test_figurative_source_target.py`:

```python
def test_parse_multilevel_response(source_target_service):
    """Test parsing multi-level domain response format."""
    response = json.dumps({
        "source_domain": {
            "specific": "cash exchange",
            "moderate": "commerce",
            "abstract": "valuable resource"
        },
        "target_domain": {
            "specific": "work hours",
            "moderate": "time",
            "abstract": "limited resource"
        },
        "mapping_explanation": "Time is treated as commodity",
        "confidence": 0.9
    })
    
    parsed = source_target_service._parse_response(response)
    
    assert parsed["source_domain"] == "commerce"  # Default to moderate
    assert parsed["source_domain_specific"] == "cash exchange"
    assert parsed["source_domain_abstract"] == "valuable resource"

def test_backward_compatibility_flat_response(source_target_service):
    """Test that flat (non-multilevel) responses still work."""
    response = json.dumps({
        "source_domain": "money",
        "target_domain": "time",
        "mapping_explanation": "Time as commodity",
        "confidence": 0.9
    })
    
    parsed = source_target_service._parse_response(response)
    assert parsed["source_domain"] == "money"
    assert parsed.get("source_domain_specific") is None
```

### 4.3 Manual Testing

1. Start backend: `cd backend && uvicorn main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to a completed figurative language batch
4. Click "Source/Target Domain Analysis"
5. Select "V3 - Multi-Level Abstraction" prompt version
6. Run analysis
7. Verify results show all three abstraction levels
8. Toggle abstraction level selector and confirm statistics update

---

## 5. Implementation Order

1. Add v3-multilevel prompt to `figurative_source_target_service.py`
2. Update `_parse_response()` to handle nested domains
3. Update result storage to include levels in `raw_response`
4. Add unit tests for parsing
5. Run existing tests to confirm no regressions
6. Update frontend dialog to include new prompt version
7. Add abstraction level selector to results panel
8. Manual end-to-end testing

---

## 6. Rollback Strategy

- New prompt is additive (v3-multilevel alongside existing v1, v2, v2-numeric)
- No database migration required (data stored in existing `raw_response` JSON)
- Old prompt versions continue to work unchanged
- Frontend changes are behind the abstraction level selector (hidden if data not present)

---

## 7. Open Questions

1. Should we default new jobs to v3-multilevel or keep v1 as default?
   - **Recommendation**: Keep v1 as default for now, let users opt-in

2. Should the domain graph support switching abstraction levels?
   - **Recommendation**: Add in Phase 2 after core functionality is validated

3. Should normalization operate on one level or all levels?
   - **Recommendation**: Operate on whichever level user selects for analysis
