# Multi-Participant Entity Score Comparison: Planning Document

**Date:** 2026-01-23
**Package:** `qualitative-analysis`
**Component:** `qa entity` extensions

---

## Overview

This document outlines the design for extending entity scoring functionality to support multi-participant comparisons. The goal is to enable researchers to:

1. Visualize how different participants score the same entities
2. Quantify differences between participants using rigorous statistical methods
3. Support both individual-level and group-level comparisons
4. Handle uncertainty from multiple LLM scoring runs

---

## Design Requirements

### Scale
- **Participants:** 2 to 100+ (potentially stratified into groups)
- **Entities:** Variable (typically 10-100 per analysis)
- **Dimensions:** 3+ (SETS framework as default, custom dimensions supported)

### Analysis Levels
| Level | Description | Example |
|-------|-------------|---------|
| **Individual** | Pairwise or n-way comparison of specific participants | "How does Participant A differ from Participant B?" |
| **Group** | Compare aggregate patterns across defined groups | "Do experts differ from novices?" |
| **Population** | Characterize overall variance and clustering | "Are there distinct 'types' of respondents?" |

### Pattern Focus
- **Overall patterns:** Multivariate compositional differences
- **Dimension-specific:** Which dimensions drive differences?

### Uncertainty Handling
- **Preferred approach:** Bayesian modeling throughout
- **Propagate uncertainty** from multiple LLM runs into all comparisons
- **Report credible intervals**, not just point estimates

---

## Statistical Methods Catalog

### 1. Distance Metrics for Compositional Data

#### 1.1 Earth Mover's Distance (Wasserstein Distance)

**Why it's relevant:**
- Measures the "work" required to transform one distribution into another
- Natural interpretation on the simplex (ternary space)
- Handles full distributions, not just point estimates
- Robust to outliers and captures distribution shape

**Applications:**
- Compare uncertainty distributions between participants
- Quantify how "far apart" two participants are in score space
- Compare aggregate group distributions

**Implementation:**
```python
from scipy.stats import wasserstein_distance
# For 1D: direct scipy
# For multivariate: use POT (Python Optimal Transport) library
import ot
emd = ot.emd2(weights_a, weights_b, cost_matrix)
```

**Variants:**
| Variant | Use Case |
|---------|----------|
| **1-Wasserstein (W₁)** | Standard EMD, interpretable as "work" |
| **2-Wasserstein (W₂)** | Squared cost, connects to Gaussian assumptions |
| **Sliced Wasserstein** | Fast approximation for high dimensions |
| **Sinkhorn distance** | Regularized EMD, computationally efficient |

#### 1.2 Aitchison Distance

**Why it's relevant:**
- The "correct" distance metric for compositional data
- Accounts for the relative nature of proportions
- Invariant to which component is used as reference

**Formula:**
```
d_A(x, y) = sqrt(sum((log(x_i/g(x)) - log(y_i/g(y)))^2))
where g(x) = geometric mean of x
```

**Implementation:**
```python
import numpy as np
def aitchison_distance(x, y):
    # CLR transform
    clr_x = np.log(x) - np.mean(np.log(x))
    clr_y = np.log(y) - np.mean(np.log(y))
    return np.sqrt(np.sum((clr_x - clr_y)**2))
```

#### 1.3 Distance Metric Comparison

| Metric | Pros | Cons | Best For |
|--------|------|------|----------|
| **EMD/Wasserstein** | Handles distributions, robust | Slower for large samples | Comparing uncertainty distributions |
| **Aitchison** | Proper for compositions | Point estimates only | Comparing mean positions |
| **Cosine similarity** | Intuitive, fast | Ignores magnitude | Profile similarity |
| **Jensen-Shannon** | Symmetric, bounded [0,1] | Requires discretization | Comparing probability distributions |

---

### 2. Multivariate Statistical Tests

#### 2.1 Bayesian Multivariate Models (Preferred)

**Hierarchical Model Structure:**
```
Score_ijk ~ Normal(μ_ij, σ²_run)        # Run-level variation
μ_ij ~ Normal(θ_i, Σ_participant)        # Participant-level
θ_i ~ Normal(μ_entity, Σ_entity)         # Entity-level
μ_entity ~ Normal(μ_0, Σ_0)              # Population prior

Where:
  i = entity
  j = participant
  k = LLM run
```

**Outputs:**
- Posterior distributions for each participant's "true" scores
- Credible intervals for participant differences
- Variance decomposition (how much variation is between-participant vs within-run?)
- Probability statements: P(Participant A > Participant B on dimension X)

**Implementation:** PyMC, Stan, or NumPyro

#### 2.2 Dirichlet-Based Models (For Compositional Data)

Since scores on a ternary plot are compositional, Dirichlet distributions are natural:

```
Scores_j ~ Dirichlet(α_j)                # Participant j's composition
α_j ~ LogNormal(μ_group, σ_group)        # Group-level concentration
```

**Benefits:**
- Proper model for simplex-constrained data
- Naturally handles the "sum to 100%" constraint
- Concentration parameter captures how "extreme" vs "balanced" scores are

#### 2.3 Frequentist Alternatives (For Reference)

| Test | Use Case | Assumptions |
|------|----------|-------------|
| **Hotelling's T²** | Compare 2 participants | Multivariate normality |
| **MANOVA** | Compare 3+ groups | Normality, homoscedasticity |
| **PERMANOVA** | Compare groups non-parametrically | Exchangeability |
| **Kruskal-Wallis** | Per-dimension comparison | Ordinal data |

---

### 3. Agreement & Reliability Metrics

#### 3.1 Intraclass Correlation Coefficient (ICC)

**Question answered:** "How much do participants agree overall?"

**Variants:**
| ICC Type | Model | Use Case |
|----------|-------|----------|
| **ICC(1,1)** | One-way random | Each participant rates different entities |
| **ICC(2,1)** | Two-way random | All participants rate all entities |
| **ICC(3,1)** | Two-way mixed | Specific participants, generalizable entities |

**Bayesian version:** Variance components from hierarchical model

#### 3.2 Krippendorff's Alpha

**Question answered:** "Multi-rater agreement across all dimensions"

- Works with any number of raters
- Handles missing data
- Can use interval, ordinal, or nominal scales

#### 3.3 Coefficient of Variation (Per Dimension)

**Question answered:** "How much spread is there for each dimension?"

```
CV_d = std(scores_d) / mean(scores_d)
```

---

### 4. Effect Size Quantification

#### 4.1 Standardized Effect Sizes

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Cohen's d** | (μ₁ - μ₂) / σ_pooled | 0.2 small, 0.5 medium, 0.8 large |
| **Hedges' g** | Bias-corrected Cohen's d | Better for small samples |
| **Glass's Δ** | (μ₁ - μ₂) / σ_control | When groups have different variance |

#### 4.2 Variance Explained

| Metric | Question Answered |
|--------|-------------------|
| **η² (eta-squared)** | What % of variance is between participants? |
| **ω² (omega-squared)** | Adjusted η² (less biased) |
| **R²** | How well do group memberships predict scores? |

#### 4.3 Bayesian Effect Sizes

- **Posterior probability of direction:** P(effect > 0)
- **Region of Practical Equivalence (ROPE):** P(effect within [-δ, +δ])
- **Bayes Factor:** Evidence ratio for difference vs no difference

---

### 5. Clustering & Similarity Analysis

#### 5.1 Participant Clustering

**Goal:** Identify natural groupings of participants based on scoring patterns

**Methods:**
| Method | Output | Best For |
|--------|--------|----------|
| **Hierarchical clustering** | Dendrogram | Visualizing nested groups |
| **K-means on CLR-transformed** | K clusters | Known number of groups |
| **Gaussian Mixture Model** | Soft assignments + uncertainty | Unknown groups, Bayesian |
| **HDBSCAN** | Density-based clusters | Unknown groups, outlier detection |

#### 5.2 Pairwise Similarity Matrix

Compute all pairwise distances (EMD or Aitchison), visualize as heatmap.

#### 5.3 Dimensionality Reduction

| Method | Output | Use Case |
|--------|--------|----------|
| **PCA on CLR scores** | 2D participant map | Linear relationships |
| **MDS on distance matrix** | 2D preserving distances | Non-linear |
| **t-SNE / UMAP** | 2D clusters | Finding groups visually |

---

## Visualization Specifications

### 1. Faceted Ternary Plots

**Layout:** Grid of ternary plots, one per participant (or group)

```
+-------------------+-------------------+-------------------+
|   Participant A   |   Participant B   |   Participant C   |
|        △          |        △          |        △          |
|      /   \        |      /   \        |      /   \        |
|     /  •  \       |     / •   \       |     /   • \       |
|    /___•___\      |    /___•___\      |    /___•___\      |
+-------------------+-------------------+-------------------+
|   Participant D   |   Participant E   |   Participant F   |
|        ...        |        ...        |        ...        |
+-------------------+-------------------+-------------------+
```

**Features:**
- Consistent axis scales across all panels
- Entity labels or numbers
- Optional: convex hull around each participant's entities
- Optional: centroid marker with confidence ellipse

### 2. Overlaid Ternary with Uncertainty

**Single ternary plot with all participants overlaid:**

- Each participant gets a distinct color
- Show uncertainty as confidence ellipses or contours
- Legend mapping colors to participants
- Option: connect same entity across participants with lines

### 3. Centroid Comparison Plot

**Show only centroids (mean positions) per participant:**

```
        Ecological
            △
           /|\
          / | \
         /  |  \
        / A •  B\
       /    |• C \
      /_____•_____\
   Tech    D     Social
```

**Features:**
- Centroid point per participant
- 95% credible region as ellipse (from Bayesian model)
- Color by group membership if applicable

### 4. Pairwise Distance Heatmap

```
         A    B    C    D    E
    A   [0]  .12  .34  .56  .23
    B   .12  [0]  .45  .67  .34
    C   .34  .45  [0]  .23  .56
    D   .56  .67  .23  [0]  .45
    E   .23  .34  .56  .45  [0]
```

**Features:**
- Symmetric matrix of pairwise EMD or Aitchison distances
- Color scale from similar (blue) to different (red)
- Hierarchical clustering dendrogram on margins

### 5. Dimension-Specific Comparison

**Forest plot or grouped bar chart:**

```
Social:      A |----●----|
             B |------●--|
             C |--●------|

Ecological:  A |------●--|
             B |----●----|
             C |--------●|

Tech:        A |--●------|
             B |----●----|
             C |------●--|
```

**Features:**
- Point estimate with credible interval per participant per dimension
- Easy to see which dimensions drive differences

### 6. Participant Similarity Map (MDS/UMAP)

**2D scatter plot of participants:**

```
          •A
    •B
              •C  •D

    •E      •F
              •G
```

**Features:**
- Each point is a participant
- Distance reflects score pattern similarity
- Color/shape by group membership
- Size by number of entities scored (optional)

---

## Implementation Plan

### Phase 1: Core Distance & Comparison Functions

**Priority:** High
**Effort:** Medium

| Task | Description | Dependencies |
|------|-------------|--------------|
| Implement Aitchison distance | Core compositional distance | numpy |
| Implement EMD wrapper | Wasserstein for distributions | scipy, POT |
| Pairwise distance matrix | All participant pairs | Above |
| Distance-based tests | PERMANOVA, etc. | scikit-bio or custom |

### Phase 2: Bayesian Hierarchical Model

**Priority:** High
**Effort:** High

| Task | Description | Dependencies |
|------|-------------|--------------|
| Design model specification | Hierarchical structure | PyMC or NumPyro |
| Implement model fitting | MCMC sampling | arviz for diagnostics |
| Extract posteriors | Participant effects, variance components | - |
| Comparison summaries | P(A > B), credible intervals | - |

### Phase 3: Visualizations

**Priority:** High
**Effort:** Medium

| Task | Description | Dependencies |
|------|-------------|--------------|
| Faceted ternary plots | Grid of participant ternaries | matplotlib, existing ternary code |
| Overlaid ternary | Multiple participants on one plot | matplotlib |
| Distance heatmap | Pairwise similarity matrix | seaborn |
| Forest plots | Per-dimension comparisons | matplotlib |
| Similarity map | MDS/UMAP of participants | scikit-learn, umap-learn |

### Phase 4: CLI Integration

**Priority:** Medium
**Effort:** Medium

| Task | Description |
|------|-------------|
| `qa entity compare` command | New subcommand for comparisons |
| Group specification | `--groups` flag for group definitions |
| Output formats | JSON summary + PNG visualizations |
| Report generation | Markdown summary of findings |

### Phase 5: Advanced Features

**Priority:** Low
**Effort:** Variable

| Task | Description |
|------|-------------|
| Dirichlet regression | Model covariates affecting composition |
| Longitudinal comparison | Same participant over time |
| Cross-entity comparison | Same participant, different entity types |

---

## Dependencies

### Required
| Package | Purpose | Install |
|---------|---------|---------|
| `numpy` | Core numerics | Already installed |
| `scipy` | Statistics, Wasserstein | Already installed |
| `matplotlib` | Visualization | Already installed |

### Recommended
| Package | Purpose | Install |
|---------|---------|---------|
| `POT` | Optimal transport / EMD | `pip install POT` |
| `pymc` | Bayesian modeling | `pip install pymc` |
| `arviz` | Bayesian diagnostics | `pip install arviz` |
| `seaborn` | Statistical visualization | `pip install seaborn` |

### Optional
| Package | Purpose | Install |
|---------|---------|---------|
| `scikit-bio` | PERMANOVA, distance tests | `pip install scikit-bio` |
| `numba` | JIT compilation for speed | `pip install numba` |

---

## Example Workflow

```bash
# 1. Score entities (already implemented)
qa entity score input.csv --dimensions sets --num-runs 5 --output-dir scored/

# 2. Compare participants (new)
qa entity compare scored/scores.csv \
    --method bayesian \
    --distance emd \
    --groups groups.json \
    --output-dir comparison/

# 3. Generate visualizations (new)
qa entity compare-viz scored/scores.csv \
    --type faceted-ternary \
    --participants "A,B,C,D" \
    --output comparison/faceted.png

qa entity compare-viz scored/scores.csv \
    --type distance-heatmap \
    --distance aitchison \
    --output comparison/heatmap.png
```

---

## Open Questions

1. **How to handle missing data?** (participant didn't score all entities)
   - Bayesian models handle naturally
   - Pairwise deletion for frequentist tests

2. **How to define "groups" for group-level comparison?**
   - JSON file mapping participant IDs to groups?
   - Column in input CSV?

3. **What's the minimum sample size for Bayesian models?**
   - Generally more flexible than frequentist
   - Need to test with small N (2-5 participants)

4. **Should we support weighted comparisons?**
   - Weight by number of entities scored?
   - Weight by confidence/uncertainty?

---

## References

- Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*
- Villani, C. (2009). *Optimal Transport: Old and New* (EMD theory)
- Gelman, A. et al. (2013). *Bayesian Data Analysis* (hierarchical models)
- Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for Compositional Data Analysis"
