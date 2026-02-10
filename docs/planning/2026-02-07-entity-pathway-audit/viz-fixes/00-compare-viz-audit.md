# Compare Pathway Visualization Fixes

**Date:** 2026-02-07
**Scope:** Distance heatmap, forest plot, similarity map in `comparison_viz.py`
**Related:** `04-visualization.md` (ternary plot issues, already addressed)

---

## Overview

Review of the three non-ternary comparison visualizations: distance heatmap, forest plot, and similarity map. Findings are prioritized by impact and implementation difficulty.

---

## 1. Distance Heatmap (`generate_distance_heatmap`, lines 802-997)

### H1: Dendrogram Branch Coloring Disabled (HIGH)
**Lines:** 880, 885
```python
dendro = dendrogram(linkage_matrix, ax=ax_dendro_top, orientation='top',
                    no_labels=True, color_threshold=0)
```
`color_threshold=0` makes all dendrogram branches the same color, defeating the purpose of showing hierarchical structure. Remove this argument to let scipy auto-select a meaningful threshold.

**Fix:** Delete `color_threshold=0` from both dendrogram calls.

### H2: Heatmap Text Contrast Heuristic is Wrong (HIGH)
**Line:** 968
```python
text_color = 'white' if value > ordered_matrix.max() * 0.6 else 'black'
```
The `RdYlBu_r` colormap goes dark-red → bright-yellow → dark-blue. Luminance is NOT monotonic with value, so the 60% threshold puts white text on bright yellow cells (unreadable) and black text on dark blue cells (also unreadable).

**Fix:** Use perceptual luminance from the colormap's actual RGBA output:
```python
rgba = cmap(norm(value))
luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
text_color = 'white' if luminance < 0.5 else 'black'
```

### M1: Cell Values Silently Vanish for n>15 (MEDIUM)
**Line:** 964
```python
if show_values and len(ordered_ids) <= 15:
```
For 16+ participants, distance values disappear with no indication. Could add a note in the title or log a message.

### M2: Fixed `.2f` Formatting Loses Precision (MEDIUM)
**Line:** 969
Small distances (e.g., 0.001) display as "0.00". Should use adaptive formatting.

### M3: Label Font Size Steps Discretely (MEDIUM)
**Line:** 945
Font size jumps 8→7→6 at cutoffs of 10/15 participants. A continuous formula would scale more gracefully.

### L1: Hardcoded Linkage Method (LOW)
**Line:** 876
Always uses `"average"` linkage. Could expose as parameter.

### L2: Diagonal Not Visually Distinguished (LOW)
Self-distance cells (always 0) look like any other low-distance cell.

---

## 2. Forest Plot (`generate_forest_plot`, lines 999-1212)

### H3: Axis Label Says "Score" But Shows Normalized Proportions (HIGH)
**Lines:** 1047-1053, 1122
Data is normalized to compositional (sum-to-1) before plotting, but the x-axis label says "Score" and the title doesn't mention normalization. Users will misinterpret proportions [0,1] as raw scores [0-100].

**Fix:** Add subtitle: "Scores normalized to proportions (sum = 1)" and change x-axis label to "Proportion".

### H4: Bootstrap Not Seeded — Irreproducible CIs (HIGH)
**Lines:** 1062-1066
```python
for _ in range(n_bootstrap):
    sample = np.random.choice(dim_scores, size=n_entities, replace=True)
```
No random seed. CIs differ between runs on the same data.

**Fix:** Use `rng = np.random.default_rng(42)` and `rng.choice()`.

### H5: CI Clamping Destroys Statistical Properties (HIGH)
**Lines:** 1075-1077
```python
ci_low = max(0.0, min(1.0, ci_low))
ci_high = max(0.0, min(1.0, ci_high))
```
For normalized data already in [0,1], bootstrap means are already in [0,1], so clamping is unnecessary. Remove it — if bootstrap means somehow exceed bounds, that indicates a bug worth surfacing, not hiding.

### M4: Equal-Distribution Reference Line Has Orphan Label (MEDIUM)
**Line:** 1125
```python
ax.axvline(x=1/n_dims, color='gray', linestyle='--', alpha=0.5, label='Equal')
```
The `label='Equal'` is set but no `ax.legend()` is called, so the label is never displayed. Should annotate the line with text instead.

### M5: Single-Entity Participants Indistinguishable (MEDIUM)
**Lines:** 1071-1073
Participants with n=1 show zero-width error bars — looks identical to very precise estimates. Should use a distinct marker or annotation.

### M6: Fixed Figure Size for Variable Participant Count (MEDIUM)
**Lines:** 1136-1138
`group_by="participant"` uses fixed `figsize=(12, 8)` regardless of participant count. 30 participants → 10×3 grid in the same space → tiny unreadable subplots.

**Fix:** Scale height: `figsize=(12, max(6, 2.5 * n_rows))`.

### L3: Hardcoded Bootstrap Count (LOW)
**Line:** 1062
`n_bootstrap=1000` with no way to configure.

---

## 3. Similarity Map (`generate_similarity_map`, lines 1214-1446)

### H6: Silent UMAP-to-MDS Fallback (HIGH)
**Lines:** 1255-1271
If UMAP fails (import error or runtime), silently falls back to MDS. Only logged, not printed. User may not realize they're seeing a different projection.

**Fix:** Print warning to stdout.

### H7: Most-Similar/Different Lines Suppressed With Groups (HIGH)
**Lines:** 1395-1421
When groups are provided, the most-similar and most-different pair annotations are completely suppressed. These are useful for understanding the distance structure.

**Fix:** Show these annotations regardless of grouping, or at minimum document the behavior.

### M8: UMAP Degenerate with n<4 (MEDIUM)
**Line:** 1261
`n_neighbors=min(15, n-1)` gives n_neighbors=2 for n=3 participants, producing degenerate embeddings. Should enforce minimum or skip UMAP for very small n.

### M9: normalized_stress="auto" Requires scikit-learn >= 1.2 (MEDIUM)
**Line:** 1278
Will raise `TypeError` on older scikit-learn versions.

### M10: No Label Collision Avoidance (MEDIUM)
**Line:** 1327
All labels offset by fixed (5, 5) points. Dense clusters produce unreadable overlapping text.

### M11: Inconsistent Label Truncation Lengths (MEDIUM)
**Line:** 1323
Truncates at 15 chars here vs 20 chars in forest plot vs 25 in participant subplots.

### L4: Hardcoded UMAP Hyperparameters (LOW)
**Lines:** 1258-1263
`min_dist=0.1` and `n_neighbors=15` with no tunability.

### L5: Groups with <3 Members Get No Visual Indicator (LOW)
**Line:** 1347
No convex hull for groups with <3 members, with no alternative visual indicator.

---

## Cross-Cutting Issues

### CC1: Dimension Colors Duplicated (HIGH)
Hex strings in `__init__` (lines 45-49) duplicate the RGB tuples used in ternary plots. Should be a single source of truth.

### CC2: Participant Color Palette Limited to 20 (MEDIUM)
Lines 52-57. Colors repeat for >20 participants (`i % len`), making them indistinguishable.

### CC3: Inconsistent Label Truncation (MEDIUM)
15 chars (similarity map), 20 chars (forest, heatmap), 25 chars (participant subplots).

### CC4: Save-or-Return Boilerplate (LOW)
Every `generate_*` method has ~15 identical lines for saving/returning. Could extract helper.

---

## Implementation Plan (Priority Order)

### Phase 1: Quick High-Impact Fixes
1. **H4** — Seed bootstrap RNG (1 line)
2. **H1** — Remove `color_threshold=0` (delete 1 arg × 2 calls)
3. **H3** — Forest plot subtitle + axis label fix
4. **H6** — Print warning on UMAP fallback

### Phase 2: Moderate-Effort Fixes
5. **H2** — Luminance-based text color in heatmap
6. **M4** — Annotate equal-distribution reference line
7. **M6** — Scale forest plot figure height
8. **M10** — Label collision avoidance in similarity map

### Phase 3: Future Improvements (not planned for this session)
- H5, H7, M1-M3, M5, M8-M9, M11
- CC1-CC4
- L1-L5
