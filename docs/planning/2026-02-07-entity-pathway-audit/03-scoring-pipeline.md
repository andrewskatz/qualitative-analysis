# Scoring Pipeline Audit

**Files:** detector.py, models.py, scorer.py, prompts/

---

## 1. Entity Detection (detector.py)

### JSON Extraction Fragility (Lines 144-151)

```python
start = content.find("{")
end = content.rfind("}")
if start != -1 and end != -1 and end > start:
    content = content[start:end + 1]
data = json.loads(content)
```

Uses "first `{` to last `}`" strategy. If the LLM response contains explanatory text with JSON-like structures before the actual payload, this greedy extraction can grab malformed content. For example:

```
I found {"count": 3} entities. Here's the result: {"entities_and_concepts": [...]}
```

Would extract `{"count": 3} entities. Here's the result: {"entities_and_concepts": [...]}` which is invalid JSON.

**Mitigation:** Try parsing from the *last* `{` that has a matching `}`, or use a streaming JSON parser.

### Case-Sensitive Entity Deduplication (Line 209)

```python
if entity not in all_entities:
    all_entities.append(entity)
```

"Renewable Energy" and "renewable energy" are treated as distinct entities. For qualitative research where participants may use inconsistent capitalization, this inflates entity counts. The consolidation step downstream can catch this, but it adds unnecessary noise to the intermediate data.

### Silent LLM Error Handling (Lines 199-201)

```python
except Exception as e:
    logger.error(f"Entity extraction failed for {text_id} window {idx}: {e}")
    extracted = []
```

All errors produce an empty entity list. There's no distinction between "LLM error" and "text genuinely has no entities." No retry logic, no error tracking in results. For a research tool, this silent failure could hide systematic extraction problems.

---

## 2. Prompt Design (prompts/)

### Entity Extraction Prompt (entity_extraction_v1.txt)

**Strengths:**
- Clear JSON output format with example
- Granularity guidance (prefer multi-word phrases)
- Instructions to avoid generic pronouns

**Weaknesses:**
- "Strongly implied" entities are ambiguous. Different LLM runs may infer different implications without concrete examples of what qualifies.
- "Only the most complete or canonical form" lacks operationalized rules. Is "John Smith, the city mayor" more canonical than "John Smith"?
- No negative examples showing what NOT to extract.
- No guidance on confidence or uncertainty.

### System Prompts (system_prompt_v1.txt, scoring_system_prompt_v1.txt)

Both files appear truncated or minimal (single-sentence role descriptions). They lack:
- Specific task framing
- Output format reinforcement
- Guardrails against common LLM failure modes

---

## 3. Entity Scoring Models (models.py)

### H3: CV Undefined at Zero Mean (Line 227)

```python
cv = std_dev / mean_val if mean_val > 0 else 0.0
```

Coefficient of variation is mathematically undefined when the mean is zero. Defaulting to 0.0 implies "no variability" when the truth is "undefined relative variability." This misleads downstream consumers. Should be `float('nan')` or a sentinel value.

**Additionally:** CV is designed for ratio-scale data on [0, infinity). For bounded [0, 100] data, relative standard error (`std_dev / scale_max`) may be more appropriate.

### H4: Mode Fabrication (Lines 211-212)

```python
modes = statistics.multimode(scores)
mode_val = float(statistics.median(modes))
```

With n=3 scoring runs and all unique values (e.g., [50, 75, 100]):
- `multimode` returns all three values
- `median` returns 75

This reports 75 as the "mode" even though no value actually repeats. The reported mode is fabricated and statistically meaningless. Should either:
- Only report mode when `len(modes) < len(scores)` (i.e., a true repeat exists)
- Report `None`/`NaN` when no true mode exists

### H5: Confidence Interval Clamping (Lines 217-224)

```python
ci_low = max(0, mean_val - margin)
ci_high = min(100, mean_val + margin)
```

Clamping CIs to [0, 100] creates:
- **Asymmetric intervals** (e.g., mean=5, margin=10 gives CI [0, 15] instead of [-5, 15])
- **False precision** near boundaries (narrow CI doesn't mean high confidence)
- **Coverage probability violation** (the stated 95% CI no longer has 95% coverage)

For bounded data, consider using the Wilson score interval or a Beta-based interval that respects bounds naturally.

### Missing Data Representation

Three different fallback values are used for missing dimension scores:
1. **scorer.py:505** - Skip (dimension absent from dict)
2. **models.py:336** - Default to 0.0 in `get_scores_as_tuple()`
3. **visualizer.py:406** - Default to 50

No explicit "missing" flag exists. An entity genuinely scored 0 is indistinguishable from a missing score in some contexts.

---

## 4. Entity Scoring (scorer.py)

### H1: Silent Skip of Missing Dimensions (Line 505)

```python
for dim in dimensions:
    dim_name = dim.name.lower()
    scores = []
    for run in runs:
        if dim_name in run.dimension_scores:
            scores.append(run.dimension_scores[dim_name]["score"])
    if not scores:
        continue  # <-- DIMENSION SILENTLY DROPPED
```

If one dimension fails to appear in ANY scoring run (e.g., the LLM consistently omits "Systemic"), that dimension is silently dropped from the EntityScore. Downstream code has no way to know this happened. The entity appears to have been scored on fewer dimensions than expected.

### H2: No Score Range Validation (Lines 465-472)

```python
if not isinstance(score, (int, float)):
    logger.warning(f"Invalid score for {dim_name}: {score}")
    continue
dimension_scores[dim_name] = {"score": int(score), "justification": justification}
```

LLM-generated scores are not validated against the [0, 100] scale. A score of 150 or -20 is accepted and propagated to all downstream calculations. This can:
- Distort mean/median/CI statistics
- Produce invalid ternary coordinates after normalization
- Cause out-of-range values in visualizations

**Fix:** Add range validation:
```python
if not (dim.scale_min <= score <= dim.scale_max):
    logger.warning(f"Score {score} for {dim_name} outside [{dim.scale_min}, {dim.scale_max}]")
    score = max(dim.scale_min, min(dim.scale_max, score))  # Clamp with warning
```

### M5: Single Justification for Aggregate (Lines 507-517)

When multiple runs are aggregated, only the justification from the run closest to the mean is preserved. This:
- Loses the reasoning from other runs
- Provides no indication of justification consistency/divergence across runs
- Selected justification may not reflect the actual uncertainty

### LLM Consistency Concerns

The default temperature of 0.3 is quite low for variability estimation. At this temperature, runs will be highly correlated, producing artificially tight CIs. The system treats multiple runs as independent samples, but they're drawn from a low-temperature distribution that:
- Underestimates true scoring uncertainty
- May not reveal genuine disagreement about entity classification
- Produces n=3 samples that are nearly identical, giving false confidence

Consider using higher temperature (0.7-1.0) for scoring runs to better capture the genuine uncertainty in the LLM's assessment, or use the current temperature but interpret the CI as a measure of LLM self-consistency rather than true uncertainty.
