# Visualization Audit

**Files:** visualizer.py, comparison_viz.py

---

## 1. Ternary Plot Issues

### H11: Coordinate Mapping Scrambled in visualizer.py (Lines 158-163)

```python
x, y = self._barycentric_to_cartesian(
    dim_values[2],  # top vertex
    dim_values[0],  # bottom-left
    dim_values[1]   # bottom-right
)
```

The dimension-to-vertex mapping uses indices `(2, 0, 1)`, meaning:
- Top vertex = dimension index 2 (e.g., "Technological")
- Bottom-left = dimension index 0 (e.g., "Social")
- Bottom-right = dimension index 1 (e.g., "Ecological")

This may or may not match the triangle labels drawn by `_draw_triangle()`. If the triangle labels are drawn in order `(0=top, 1=bottom-left, 2=bottom-right)`, the mapping is scrambled and points will appear in the wrong location relative to vertex labels.

### H12: Incompatible Coordinate Mapping in comparison_viz.py (Lines 232, 389)

```python
x, y = self._barycentric_to_cartesian(score[1], score[2], score[0])
```

Uses indices `(1, 2, 0)`, which is a DIFFERENT scrambling than visualizer.py's `(2, 0, 1)`. This means:
- Single-entity ternary plots (visualizer.py) and comparison ternary plots (comparison_viz.py) place the same entity at **different positions**
- Results from the two visualizations are not directly comparable
- Users comparing the two plots will see inconsistent positions for the same entities

**Fix:** Both files must use the same consistent mapping. Define the mapping once in a shared constant.

### Normalization Loses Magnitude (See also 01-design-contradiction.md)

Both files normalize scores to sum to 1 before plotting. This is a valid projection showing *relative proportions*, but it loses magnitude information: entities with raw scores [80,80,80] and [10,10,10] both map to the triangle centroid.

**Remediation (implemented):** Encode magnitude as point size (mean score) and uncertainty as opacity (CV). This restores the lost information while preserving the ternary layout. See 01-design-contradiction.md for full analysis.

### H13: Confidence Ellipses on Ternary Data (comparison_viz.py:415-440)

```python
cov = np.cov(xs_arr, ys_arr)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
```

Confidence ellipses are computed using standard Euclidean covariance on the Cartesian-projected ternary coordinates. Problems:
1. Ternary data lives on a 2-simplex, not in unrestricted 2D space
2. Ellipses can extend outside the triangle boundary, displaying impossible compositions
3. The covariance structure on a simplex is inherently constrained (points must sum to 1)
4. No check that the covariance matrix is positive-definite before eigendecomposition

For compositional data, use the Aitchison geometry to compute confidence regions (CLR-transform, compute ellipse, back-transform).

### M13: Hardcoded 3-Dimension Fallback (comparison_viz.py:146)

```python
return scores / total if total > 0 else np.ones(3) / 3
```

The fallback `np.ones(3)` assumes exactly 3 dimensions. Will fail silently for 2D or 4D scoring frameworks.

---

## 2. Uncertainty Visualization

### M14: Opacity Formula Issues (visualizer.py:175-181)

```python
alpha = max(0.3, min(0.95, 0.95 - (cv * 2)))
```

For CV = 0.5: `alpha = max(0.3, min(0.95, -0.05)) = 0.3`
For CV = 0.1: `alpha = max(0.3, min(0.95, 0.75)) = 0.75`
For CV = 0.0: `alpha = max(0.3, min(0.95, 0.95)) = 0.95`

The multiplier of 2 is arbitrary and causes the formula to hit the floor (0.3) at CV=0.325. Any CV above ~0.33 produces the same minimum opacity, losing discrimination between moderately and highly uncertain entities.

**Better formula:** `alpha = max(0.3, 1.0 - cv)` provides linear mapping from CV to opacity with a sensible floor.

### Bootstrap CIs Violate Compositional Constraint (comparison_viz.py:748-764)

```python
for _ in range(n_bootstrap):
    sample = np.random.choice(dim_scores, size=n_entities, replace=True)
    bootstrap_means.append(np.mean(sample))
ci_low = np.percentile(bootstrap_means, alpha / 2 * 100)
ci_high = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
```

Bootstrap is applied independently to each dimension's normalized scores. The resulting CIs can violate the compositional constraint (low bounds across dimensions might sum to > 1, or high bounds might sum to < 1). This produces impossible error bars on the forest plot.

---

## 3. Other Visualization Issues

### M15: Radar Chart Missing Data Default (visualizer.py:307)

```python
else:
    values.append(50)  # Default middle score
```

Missing dimensions silently default to 50 (the midpoint). This:
- Makes incomplete data appear valid
- Places missing dimensions at the scale midpoint, which could be misinterpreted as a genuine moderate score
- No visual indication that this value is imputed rather than observed

### Heatmap Text Color (comparison_viz.py:655)

```python
text_color = 'white' if value > ordered_matrix.max() * 0.6 else 'black'
```

Uses a fixed threshold relative to the global maximum. With color maps like RdYlBu_r, the perceptual luminance at 60% of max may not be where white text becomes more readable. A luminance-based threshold would be more robust.

### UMAP Neighbor Count (comparison_viz.py:945-950)

```python
n_neighbors=min(15, n - 1)
```

With n=3 participants: `n_neighbors = 2`, which produces degenerate UMAP embeddings. UMAP generally requires at least 3-5 neighbors for meaningful results. Add a minimum: `n_neighbors=max(3, min(15, n - 1))`.

### Label Collision Detection (visualizer.py:593-598)

```python
if px == x and py == y:
    continue
```

Exact floating-point equality comparison will almost never match. Should use a distance threshold:
```python
if abs(px - x) < 1e-6 and abs(py - y) < 1e-6:
    continue
```

---

## 4. Accessibility Concerns

### Color Palette

Default dimension colors:
- Social: `#1f77b4` (blue)
- Ecological: `#2ca02c` (green)
- Technological: `#d62728` (red)

The green/red combination is problematic for ~8% of males with deuteranopia or protanopia. Consider:
- Using a colorblind-safe palette (e.g., IBM Design palette, or Okabe-Ito)
- Adding pattern fills or markers in addition to color
- Providing a `--colorblind` flag

### Font Sizes

- Direct labels: 7px (visualizer.py:542) - very small, may be unreadable in print
- Heatmap labels: 6-7pt for >15 participants - illegible at standard sizes
- Consider auto-scaling font size based on number of data points

### Interactive vs. Static

The codebase supports both plotly (interactive HTML) and matplotlib (static PNG). The plotly path provides hover tooltips and zooming, which helps with dense plots. However, the matplotlib path is the primary codepath and lacks equivalent interactivity. Consider documenting when to prefer each output format.
