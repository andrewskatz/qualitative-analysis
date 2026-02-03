# Model Specification: Bayesian Hierarchical Model for Entity Scores

**Date:** 2026-01-29
**Parent:** [00-overview.md](./00-overview.md)

---

## 1. Data Structure

Each observation is a single score for one dimension, one entity, one participant, one LLM run:

```
y_{d,i,j,k} = score for dimension d, entity i, participant j, run k
```

| Index | Symbol | Range | Description |
|-------|--------|-------|-------------|
| Dimension | d | 1..D (typically 3) | e.g., Social, Ecological, Technological |
| Entity | i | 1..I (~400 in current dataset) | Unique entity-context pairs |
| Participant | j | 1..J (15 in current dataset) | Interview participants |
| Run | k | 1..K (3 in current dataset) | Replicate LLM scoring runs |

**Total observations per dimension:** I × J × K (e.g., ~400 × 15 × 3 ≈ 18,000)

Note: Not every entity is scored by every participant. The model handles this naturally — missing data simply means fewer observations for that (entity, participant) cell.

---

## 2. Likelihood

Scores are bounded on [0, 100]. After rescaling to [0, 1], a Beta distribution is the natural choice:

```
y_{d,i,j,k} ~ Beta(μ_{d,i,j} · κ_{d,i,j}, (1 - μ_{d,i,j}) · κ_{d,i,j})
```

where:
- **μ_{d,i,j}** ∈ (0, 1) = mean score for dimension d, entity i, participant j
- **κ_{d,i,j}** > 0 = precision (concentration) parameter — controls how tightly runs cluster

This is the **mean-precision parameterization** of the Beta distribution:
- `α = μ · κ`
- `β = (1 - μ) · κ`
- `E[y] = μ`, `Var[y] = μ(1-μ) / (κ+1)`

### Why Beta over Normal?

| Property | Beta | Normal |
|----------|------|--------|
| Bounded support | Yes — (0,1) natural for proportions | No — can predict outside [0,1] |
| Heteroscedastic | Built-in — variance depends on mean | Requires explicit modeling |
| Skewness | Flexible — can be left/right skewed | Symmetric |
| Edge behavior | Can model scores near 0 or 100 | Awkward at boundaries |

### Handling Exact 0 and 100

Beta requires y ∈ (0, 1), not [0, 1]. If raw scores hit exactly 0 or 100:
- Apply a small squeeze: `y_transformed = (y * (N-1) + 0.5) / N` where N is sample size
- This is a standard approach (Smithson & Verkuilen, 2006) and has negligible impact with scores on a 0–100 scale

---

## 3. Hierarchical Structure

### 3.1 Three-Level Hierarchy

```
Level 3 (Entity):      θ_{d,i}     ~ population-level entity score
Level 2 (Participant):  μ_{d,i,j}  ~ participant-specific deviation
Level 1 (Run):          y_{d,i,j,k} ~ observed score (with run-level noise via κ)
```

### 3.2 Full Model (Per Dimension d)

Each dimension is modeled independently. For a single dimension d (subscript omitted for clarity):

```
# --- Likelihood ---
y_{i,j,k} ~ Beta(μ_{i,j} · κ, (1 - μ_{i,j}) · κ)

# --- Participant-level (non-centered) ---
μ_{i,j} = logit⁻¹(η_{i,j})
η_{i,j} = θ_i + group_effect_{g[j]} + z_{i,j} · σ_participant

z_{i,j} ~ Normal(0, 1)                    # standardized offset
σ_participant ~ HalfNormal(1)              # participant-level spread

# --- Group-level ---
group_effect_g ~ Normal(0, σ_group)        # deviation from entity mean by group
σ_group ~ HalfNormal(0.5)                 # group-level spread

# --- Entity-level (non-centered) ---
θ_i = μ_population + z_entity_i · σ_entity
z_entity_i ~ Normal(0, 1)                 # standardized entity offset
σ_entity ~ HalfNormal(2)                  # entity-level spread

# --- Population-level ---
μ_population ~ Normal(0, 1.5)             # population logit-mean (weakly informative)

# --- Precision ---
κ ~ Gamma(5, 0.1)                         # run-level precision (diffuse)
```

### 3.3 Non-Centered Parameterization

The model uses **non-centered parameterization** throughout, which is essential for:
- Large numbers of entities (~400) where centered parameterization causes funnel geometries
- Efficient MCMC sampling with NUTS/nutpie
- Avoiding divergences in PyMC

Standard (centered):
```
μ_{i,j} ~ Normal(θ_i, σ_participant)     # ← problematic with many i
```

Non-centered (used here):
```
z_{i,j} ~ Normal(0, 1)
μ_{i,j} = θ_i + z_{i,j} · σ_participant  # ← much better geometry
```

### 3.4 Group as Covariate

Groups (e.g., ecological, social, technological) enter as a **fixed effect at the entity level**, not as another random effect level. This is because:
- Groups are defined categories, not a random sample from a population of groups
- We want direct estimates of group differences
- With only 3 groups, a random effect would be poorly identified

```
η_{i,j} = θ_i + group_effect_{g[j]} + z_{i,j} · σ_participant
```

where `g[j]` maps participant j to their group. One group is the reference category (effect = 0), and `group_effect_g` for each other group represents the deviation from the reference.

---

## 4. Prior Justification

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| μ_population | Normal(0, 1.5) | On logit scale: covers ~[5%, 95%] of the unit interval |
| σ_entity | HalfNormal(2) | Entities can vary widely in how they are scored |
| group_effect_g | Normal(0, σ_group) | Centered at zero — groups may or may not differ |
| σ_group | HalfNormal(0.5) | Mildly regularizing — encourages modest group differences |
| σ_participant | HalfNormal(1) | Moderate participant variation expected |
| κ | Gamma(5, 0.1) | Mean of 50, allows wide range — run-level precision |

All priors are **weakly informative** — they regularize extreme values but let the data dominate with sufficient observations.

### Prior Predictive Check

Before fitting, the prior predictive distribution should be examined to ensure:
- Predicted scores cover the full [0, 100] range
- No prior places excessive mass at boundaries
- Group differences are plausible but not forced

---

## 5. Model Outputs

### 5.1 Posterior Estimates

| Output | Description | Use |
|--------|-------------|-----|
| `θ_i` (entity means) | Population-level score per entity per dimension | Shrinkage-adjusted entity profiles |
| `group_effect_g` | Group-level offset per dimension | How much each group deviates |
| `μ_{i,j}` (participant means) | Participant-specific entity scores | Individual-level inference |
| `σ_participant` | Participant spread | How much participants vary within a group |
| `σ_entity` | Entity spread | How much entities vary in the population |
| `σ_group` | Group spread | Magnitude of group differences |
| `κ` | Run precision | How consistent LLM runs are |

### 5.2 Derived Quantities

#### Group Contrasts
For each pair of groups (A, B) and each dimension d:
```
Δ_{A,B,d} = group_effect_A_d - group_effect_B_d
P(Δ_{A,B,d} > 0) = proportion of posterior samples where A > B
```

Reported as:
- Posterior mean difference with 95% HDI (Highest Density Interval)
- Probability of direction: P(group A scores higher than group B)
- ROPE analysis: P(|difference| < δ) for a chosen practical equivalence threshold δ

#### Variance Decomposition (ICC)
```
ICC_group = σ²_group / (σ²_group + σ²_entity + σ²_participant + σ²_run)
ICC_entity = σ²_entity / (σ²_group + σ²_entity + σ²_participant + σ²_run)
ICC_participant = σ²_participant / (σ²_group + σ²_entity + σ²_participant + σ²_run)
```

Answers: "What proportion of total variance in scores is attributable to each level?"

#### Entity-Level Shrinkage
Compare raw entity means to posterior estimates:
```
shrinkage_i = 1 - (posterior_sd_i / raw_sd_i)
```

Entities with few observations will be shrunk more toward the population mean — a key advantage of hierarchical modeling over raw averaging.

---

## 6. Diagnostics

### 6.1 Convergence Diagnostics

| Diagnostic | Target | Tool |
|-----------|--------|------|
| R-hat | < 1.01 | ArviZ `az.rhat()` |
| Effective sample size (ESS) | > 400 | ArviZ `az.ess()` |
| Divergences | 0 | PyMC `trace.sample_stats` |
| Tree depth | < max_treedepth | PyMC `trace.sample_stats` |
| Energy transition | No issues | ArviZ `az.plot_energy()` |

### 6.2 Model Checks

| Check | Method |
|-------|--------|
| Prior predictive | Sample from priors only, inspect predicted score distribution |
| Posterior predictive | Compare observed vs predicted score distributions (per group, per dimension) |
| Residual patterns | Examine residuals by entity, participant, group |
| LOO-CV | Leave-one-out cross-validation for model comparison (ArviZ `az.loo()`) |

---

## 7. Sampling Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sampler | nutpie (primary), NUTS (fallback) | nutpie is ~2x faster; NUTS is more established |
| Chains | 4 | Standard for R-hat computation |
| Tune | 1000 | Default; increase if divergences occur |
| Draws | 2000 | 8000 total posterior samples across chains |
| Target accept | 0.95 | Higher than default (0.8) for hierarchical models |
| Max tree depth | 12 | Increase from default 10 for deep hierarchies |
| Init | "jitter+adapt_diag" | PyMC default; robust initialization |

### Estimated Runtime
- ~400 entities, 15 participants, 3 runs, 3 dimensions
- **Per dimension:** 5–30 minutes (depending on convergence)
- **Total (3 dimensions):** 15–90 minutes
- Running dimensions in parallel (if memory permits) could reduce wall-clock time

---

## 8. Comparison to Existing Methods

The Bayesian model complements (does not replace) the distance-based pipeline:

| Question | Distance-based | Bayesian |
|----------|---------------|----------|
| Are groups different? | Permutation test p-value | Posterior P(Δ > 0) + ROPE |
| How different? | Effect size ratio | Posterior mean Δ with 95% HDI |
| On which dimensions? | Not directly | Per-dimension group contrasts |
| How reliable are scores? | Collapse runs to means | Explicit run-level variance (κ) |
| Which entities drive differences? | Not directly | Entity-level posterior comparisons |
| How much of the variance is between groups? | Not directly | ICC decomposition |

---

## References

- Smithson, M. & Verkuilen, J. (2006). "A Better Lemon Squeezer? Maximum-Likelihood Regression with Beta-Distributed Dependent Variables." *Psychological Methods*, 11(1), 54–71.
- Gelman, A. & Hill, J. (2007). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
- Bürkner, P.C. (2017). "brms: An R Package for Bayesian Multilevel Models Using Stan." *Journal of Statistical Software*, 80(1), 1–28.
- Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
- Vehtari, A., Gelman, A., & Gabry, J. (2017). "Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC." *Statistics and Computing*, 27(5), 1413–1432.
