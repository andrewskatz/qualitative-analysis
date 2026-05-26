# Multi-Model Scoring Experiment: Analysis Complete

**Date:** 2026-04-08
**Package:** `qualitative-analysis`
**Component:** Entity Scoring Pipeline — Factorial Experiment (scale x prompt x model)

---

## Summary

Completed a multi-model scoring experiment comparing 6 LLMs (plus 3 previously tested) across a 3x3 factorial design (scales: 0-100, 1-10, 1-5; prompt versions: v2/Score-then-Justify, v3/Justify-then-Score, v4/No CoT). Each condition used 3 scoring runs per entity on a shared sample of ~450 entities from the ABC simulation platform dataset.

**Models tested in this round:**
- qwen3.5:122b-a10b-q4_K_M (46.0 hrs total)
- qwen3.5:4b-q4_K_M (17.5 hrs)
- qwen3:4b-instruct-2507-q4_K_M (8.9 hrs)
- qwen2.5:7b-instruct-q4_K_M (7.6 hrs)
- glm-4.7-flash:q4_K_M (15.6 hrs)

**Previously tested:** qwen3:30b, qwen3:80b, qwen3.5:35b, gpt-oss:120b

All 45 new conditions (5 models x 9 conditions) completed successfully with 0 failures.

---

## Key Findings

### Universal Patterns (All Models)

1. **0-100 scale consistently best** — highest reliability (alpha 0.85+), maximum entropy, least ceiling compression across all models tested.

2. **Social dimension ceiling compression** — 51-83% of entities hit the ceiling on the 1-5 scale, universally. This is a property of the construct, not the model.

3. **Chain-of-thought deflates scores** — v4 (no CoT) inflates scores vs v2/v3 across all models (all p < 1e-05). CoT acts as a deflationary mechanism. v2 vs v3 order difference is small.

4. **v4 is ~2x faster** than CoT prompts across all models, with modest quality loss (Q drops ~0.02-0.05).

### Model Tier Rankings

| Tier | Models | Alpha Range | Mean Rank rho | Notes |
|------|--------|-------------|---------------|-------|
| **Tier 1** | qwen3-80b, qwen3.5-122b | 0.806-0.994 | 0.86-0.91 | Best quality, slowest |
| **Tier 2** | qwen3-30b, gpt-oss-120b | 0.764-0.989 | 0.85-0.91 | Good balance |
| **Tier 3** | qwen3.5-4b, qwen3.5-35b | 0.776-0.969 | 0.71-0.87 | Moderate |
| **Tier 4** | qwen3-4b-inst, qwen2.5-7b, glm-4.7-flash | 0.707-0.996 | 0.67-0.69 | Fast but unstable rankings |

**Critical finding:** Smaller models (Tier 4) show concerning rank instability across configurations (mean Spearman rho < 0.70), meaning entity rankings change substantially depending on scale/prompt choice. This undermines comparability.

### Recommended Configurations

- **Maximum quality:** qwen3-80b or qwen3.5-122b + 0-100 scale + v2 prompt
- **Production balanced:** qwen3-30b + 0-100 scale + v2 prompt (good alpha, reasonable speed)
- **Speed-sensitive:** qwen3-30b + 0-100 scale + v4 prompt (accept ~0.03 Q drop, 2x faster)

---

## Bug Fixes Applied

Fixed three bugs in `scripts/analyze_scoring_experiment.py` that surfaced with models producing duplicate entity rows:

1. **Rank displacement plot** — used raw `.values` without entity alignment; fixed to use `aligned_values()` inner join.
2. **Scale/prompt main effects** — `np.mean(combined, axis=0)` on ragged arrays from different entity counts; fixed with `aligned_mean_over_keys()` helper.
3. **All cross-condition operations** — duplicate entities caused many-to-many join inflation; added `_dedup()` helper that averages duplicate entity scores before joining.

---

## Next Steps

- [ ] Build cross-model comparison script for publication-ready figures
- [ ] Investigate social dimension ceiling compression mitigation strategies
- [ ] Consider whether smaller models are acceptable for any use case given rank instability
- [ ] Decide on final model + configuration for production pipeline

---

## Artifacts

- Experiment runner: `qualitative-analysis/scripts/run_scoring_experiment.py`
- Analysis script: `qualitative-analysis/scripts/analyze_scoring_experiment.py`
- Per-model results: `qualitative-analysis/output/scoring-experiment-<model>/analysis/`
- Each contains: `figures/`, `tables/`, `summary_report.md`
