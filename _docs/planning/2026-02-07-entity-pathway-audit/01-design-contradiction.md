# C1: Ternary Plots for Independent Dimensions - Information Loss and Remediation

**Severity:** MEDIUM (downgraded from CRITICAL after further analysis)
**Scope:** Primarily visualization; distance metric and statistical concerns remain separate
**Files:** visualizer.py, comparison_viz.py

---

## The Situation

The scoring model treats Social, Ecological, and Technological as **independent dimensions**, each ranging 0-100. The ternary plots normalize these to sum to 1 before display, projecting the scores onto a simplex.

**This normalization is a valid operation.** It answers the question: "What is the relative proportion of each dimension for this entity?" An entity scoring equally on all three dimensions landing at the center of the ternary plot is a correct representation of its *relative profile*.

However, normalization is **lossy** - it discards magnitude information. This means the ternary plot alone doesn't tell the full story.

---

## What's Lost: Magnitude Information

| Entity | Raw Scores (S, E, T) | Sum | Normalized | Ternary Position |
|--------|----------------------|-----|------------|-----------------|
| "water governance" | (80, 80, 80) | 240 | (0.33, 0.33, 0.33) | Center |
| "abstract concept" | (10, 10, 10) | 30 | (0.33, 0.33, 0.33) | Center |
| "solar panel" | (20, 10, 70) | 100 | (0.20, 0.10, 0.70) | Near T corner |
| "power grid" | (40, 20, 140) | 200 | (0.20, 0.10, 0.70) | Near T corner |

Entities with the same *profile shape* but very different *absolute strengths* map to the same point. The first pair are both balanced but differ in overall relevance; the second pair have the same skew but different magnitudes.

---

## Remediation: Multi-Channel Encoding

The fix is straightforward: **encode magnitude as point size** and **encode uncertainty as opacity**.

- **Position** (ternary coordinates) → relative dimensional profile
- **Size** (marker area) → mean score across dimensions (overall relevance strength)
- **Opacity** (alpha channel) → average coefficient of variation (scoring confidence)
- **Color** → primary dimension (existing behavior)

This restores the lost magnitude information without abandoning the ternary layout. A large, opaque point at center means "strongly and confidently relevant across all dimensions." A small, translucent point at center means "weakly and uncertainly relevant across all dimensions."

The plot subtitle should note: *"Position shows relative proportions; size shows overall score magnitude; opacity shows scoring confidence."*

---

## Remaining Concerns for Other Modules

The ternary visualization itself is defensible with the above fixes. However, other parts of the pipeline should be mindful of the independent-vs-compositional distinction:

- **Distance metrics:** Aitchison distance operates on compositional geometry and finds differences in *profile shape* but misses differences in *magnitude*. For comparing absolute scores, Euclidean distance is more appropriate. The default should be Euclidean.
- **CLR/ILR transforms:** Only meaningful for genuinely compositional data. Should carry a warning when applied to independent dimension scores.
- **Bayesian modeling:** If the Beta model is fit on normalized data, it models proportions rather than absolute judgments. These answer different research questions. Should document which is being used.
