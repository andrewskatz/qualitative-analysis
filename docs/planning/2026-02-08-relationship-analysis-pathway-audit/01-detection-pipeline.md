# Detection Pipeline Audit

**Scope**: `relationships/detector.py`, `relationships/components/`, `relationships/prompts/`

---

## H1: Entity Context Buffer Persists Between detect() Calls

**Severity**: HIGH
**File**: `relationships/detector.py:88, 143, 151`
**Impact**: Entity extraction from one text pollutes subsequent texts

The `EntityBuffer` is initialized once in `__init__` and never reset between calls to `detect()`:

```python
# __init__ (line 88)
self.context_buffer = EntityBuffer(context_buffer_size)

# detect() - adds entities each window (line 143, 151)
self.context_buffer.add(entities)
```

When the CLI creates a single `RelationshipDetector` and calls `detect()` for each CSV row (`relationships_cli.py:260-286, 373`), entities from row 1 leak into row 2's extraction via the context buffer.

**Default is safe**: `context_buffer_size=0` disables the buffer. But if a user enables it with `--context-buffer-size N`, cross-text contamination occurs silently.

**Fix**: Reset the context buffer at the start of each `detect()` call:
```python
async def detect(self, text, current_entities=None):
    self.context_buffer = EntityBuffer(self.context_buffer.max_size)
    # ... rest of method
```

---

## H2: text_id Not Propagated to Relationship Objects

**Severity**: HIGH
**File**: `relationships/components/extractor.py`, `relationship_only_extractor.py`
**Impact**: Downstream pipeline stages that group by text_id receive empty strings

The extractors create `Relationship` objects with `window_index` but never set `text_id`:

```python
# relationship_only_extractor.py - creates relationships
Relationship(
    source=..., target=..., type=..., description=...,
    window_index=window_index,
    # text_id is never set → defaults to ""
)
```

The CLI compensates by writing `text_id` as a column when serializing to CSV (`relationships_cli.py:394`), but:
- **Python API users** who call `detect()` directly get empty `text_id` fields
- Verification (`verifier.py`) groups by `text_id` — empty strings cause all relationships to land in one group
- Graph building (`graph.py`) tracks `text_ids` for per-text analysis — empty strings produce `text_id=""` in outputs

**Fix**: Pass `text_id` into `detect()` and propagate to all created Relationship objects.

---

## M1: Entity Deduplication Is Case-Sensitive

**Severity**: MEDIUM
**File**: `relationships/detector.py:168, 181-190`
**Impact**: "Climate Change" and "climate change" treated as distinct entities

```python
# Line 168 - final dedup
dedup_entities = sorted({e for e in all_entities if e})  # set() is case-sensitive

# Lines 181-190 - _merge_entities
def _merge_entities(self, *lists):
    seen = set()
    for items in lists:
        for item in items:
            value = item.strip()
            if value and value not in seen:  # case-sensitive check
                merged.append(value)
                seen.add(value)
```

When different LLM calls produce "Technology" vs "technology", both survive deduplication. This inflates entity counts and creates duplicate nodes in downstream graph construction.

**Fix**: Use case-insensitive deduplication while preserving original casing:
```python
if value.lower() not in seen:
    merged.append(value)
    seen.add(value.lower())
```

---

## M2: JSON Parsing Inconsistency Across Extractors

**Severity**: MEDIUM
**Files**: `detector.py:207-227`, `components/extractor.py:72-116`, `components/entity_extractor.py`, `components/relationship_only_extractor.py`
**Impact**: Different fallback behaviors for the same type of LLM response

Each extractor implements its own JSON extraction logic with slightly different fallback chains:

| Component | Strategy |
|-----------|----------|
| `detector.py` (coref) | Strip markdown → find `{`/`}` → `json.loads` |
| `extractor.py` | Pydantic parse → strip markdown → find `{`/`}` → heuristic wrapping |
| `entity_extractor.py` | Pydantic `model_validate_json` only |
| `relationship_only_extractor.py` | Pydantic `model_validate_json` only |

The `extractor.py` has an especially fragile heuristic on line ~107: if the response contains `"entities"` but no `{`, it wraps the response in braces. This could match on responses that merely mention the word "entities" in natural language.

**Fix**: Create a shared `parse_llm_json(response, expected_keys)` utility in `core/` and use it consistently.

---

## M3: No Validation of Extracted Relationships

**Severity**: MEDIUM
**File**: `relationships/components/extractor.py`, `relationship_only_extractor.py`
**Impact**: LLM hallucinations pass through unfiltered

In two-pass mode, the LLM is given a list of known entities and asked to find relationships between them. But the extracted relationships are NOT validated against the entity list. If the LLM invents entities not in the provided list, they're accepted silently.

Similarly, no validation that:
- Source and target are non-empty
- Source ≠ target (self-loops)
- Relationship type is non-empty
- Description contains actual evidence

**Fix**: Add post-extraction validation:
```python
def _validate_relationship(rel, known_entities=None):
    if not rel.source.strip() or not rel.target.strip():
        return False
    if known_entities and rel.source not in known_entities:
        logger.warning(f"Unknown source entity: {rel.source}")
    return True
```

---

## M4: Coreference Resolution Loses Original Text

**Severity**: MEDIUM
**File**: `relationships/detector.py:98-101`
**Impact**: Cannot audit what changes coreference resolution made

```python
if self.coref_resolution and text:
    text = await self._resolve_coreferences(text)  # overwrites original

windows = self.text_processor.process(text)  # windows use modified text
```

The original text is overwritten with the resolved version. Window results (line 162) contain the resolved text, but the user has no way to see what substitutions were made or compare original vs resolved text.

**Fix**: Store both versions and optionally include the diff in metadata:
```python
original_text = text
if self.coref_resolution and text:
    text = await self._resolve_coreferences(text)
    metadata["coref_applied"] = text != original_text
```

---

## L1: Summary Context Check is Fragile

**Severity**: LOW
**File**: `relationships/detector.py:127-128`
**Impact**: Minor — relies on magic string comparison

```python
if self.include_summaries_in_prompt:
    summary_context = summary_buffer.get_formatted_context()
    if summary_context == "No prior context.":  # magic string
        summary_context = None
```

This depends on `SummaryBuffer.get_formatted_context()` returning exactly "No prior context." when empty. If that string changes, summaries won't be included in prompts.

**Fix**: Use a more robust check like `if summary_buffer.has_context()` or check `len(summary_buffer)`.
