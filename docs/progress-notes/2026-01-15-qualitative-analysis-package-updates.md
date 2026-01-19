# Qualitative Analysis Package Progress Update

**Date:** 2026-01-15  
**Status:** Active Development

---

## Summary

Major enhancements to the `qualitative-analysis` CLI package, including a unified command structure, improved pipeline integration, and new semantic graph visualization features.

---

## Completed Work

### 1. Unified CLI Architecture
Consolidated three separate CLI entry points into a single `qa` command:

| Old Command | New Command |
|-------------|-------------|
| `qualitative-analysis` | `qa fig detect` |
| `qualitative-domains` | `qa fig map/normalize/graph` |
| `qualitative-relationships` | `qa rel detect` |

**Key changes:**
- Created `unified_cli.py` with nested subparsers
- Added aliases (`fig` for `figurative`, `rel` for `relationships`)
- Backward-compatible: old commands still work with deprecation warnings

### 2. Pipeline Integration Improvements
- **Nested output structure**: Results auto-organize into `run_*/map/map_*/`, `run_*/normalize/normalize_*/`, etc.
- **Auto-detection of columns**: `instance_text`, `window_text`, `text_id` columns detected automatically when chaining pipeline steps
- **Fixed bugs** in column mapping between detect → map → normalize

### 3. LLM Output Robustness
- **JSON structured output** for canonical label generation
- LLM now returns `{"reasoning": "...", "label": "..."}` for better debugging
- Fallback handling for malformed responses

### 4. Semantic Graph Layout
- **PCA → UMAP pipeline**: Reduces 1024D embeddings to 2D via PCA (retaining ~90% variance) then UMAP
- **`--layout semantic`** flag positions nodes by embedding similarity
- Significant improvement in layout stability and speed

### 5. Cluster-Based Graph Visualization (New)
Added `--cluster-labels` mode for cleaner graph visualization:

| Feature | Implementation |
|---------|---------------|
| Clustering | HDBSCAN on 2D semantic positions |
| Cluster labels | LLM-generated descriptive names |
| Visual encoding | **Colors** = cluster, **Shapes** = source/target/both |
| Label placement | `adjustText` library for collision avoidance |
| Regions | Convex hulls with translucent fill |

**New CLI flags:**
- `--cluster-labels` - Enable cluster-based visualization
- `--min-cluster-size N` - Minimum nodes per cluster
- `--noise-handling label|hide|other` - Handle outliers

---

## New Dependencies

Added to `[project.optional-dependencies.viz]`:
- `hdbscan>=0.8.0` - Density-based clustering
- `adjustText>=1.0.0` - Label collision avoidance

---

## Example Commands

```bash
# Full pipeline
qa fig detect input.csv --text-col text --output instances+windows
qa fig map output/run_*/instances.csv --multi-level
qa fig normalize output/run_*/map/*/mappings.csv --canonical-method llm
qa fig graph output/run_*/normalize/*/normalized.csv --visualize --cluster-labels --model gpt-oss:120b
```

---

## Next Steps

- [ ] Test cluster visualization on larger datasets
- [ ] Consider `--max-clusters` parameter for readability
- [ ] Explore concave hulls for tighter cluster boundaries
- [ ] Add interactive HTML output option
