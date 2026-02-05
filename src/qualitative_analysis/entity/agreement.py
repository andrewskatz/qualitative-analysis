"""
Krippendorff's Alpha for inter-rater agreement.

This module computes Krippendorff's Alpha to measure agreement among multiple
raters (e.g., LLM scoring runs or human participants) on interval-scale data.

Two analysis modes are supported:
- **Run-level**: Measures consistency across multiple LLM scoring runs per entity.
- **Participant-level**: Measures agreement across participants on entity scores.

Example::

    from qualitative_analysis.entity.agreement import compute_agreement

    results = compute_agreement(
        scores_df,
        dimensions=["social", "ecological", "technological"],
        mode="run_level",
    )
    for dim, result in results.items():
        print(f"{dim}: α = {result.alpha:.3f} ({result.interpretation})")
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AgreementResult:
    """Result of Krippendorff's Alpha computation for a single dimension."""

    alpha: float
    ci_lower: float
    ci_upper: float
    n_units: int
    n_raters: int
    mode: str  # "run_level" or "participant_level"
    dimension: str
    interpretation: str  # "poor", "fair", "moderate", "good", "excellent"

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "alpha": round(self.alpha, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "n_units": self.n_units,
            "n_raters": self.n_raters,
            "mode": self.mode,
            "dimension": self.dimension,
            "interpretation": self.interpretation,
        }


def _interpret_alpha(alpha: float) -> str:
    """
    Interpret Krippendorff's Alpha using standard thresholds.

    Krippendorff (2004) recommends:
    - α ≥ 0.80: Good reliability for drawing conclusions
    - 0.67 ≤ α < 0.80: Acceptable for tentative conclusions
    - α < 0.67: Unreliable

    We add finer granularity for practical use.
    """
    if alpha >= 0.90:
        return "excellent"
    elif alpha >= 0.80:
        return "good"
    elif alpha >= 0.67:
        return "fair"
    elif alpha >= 0.50:
        return "moderate"
    else:
        return "poor"


def krippendorff_alpha(
    reliability_data: np.ndarray,
    missing_value: float = np.nan,
) -> float:
    """
    Compute Krippendorff's Alpha for interval data.

    This is the core algorithm implementing Krippendorff's Alpha using the
    coincidence matrix approach with interval (squared difference) distance.

    Args:
        reliability_data: Matrix of shape (n_raters, n_units) where each cell
            contains a rater's score for a unit. Use NaN for missing values.
        missing_value: Value representing missing data (default: np.nan).

    Returns:
        Krippendorff's Alpha coefficient. Returns NaN if computation fails
        (e.g., insufficient data).

    References:
        Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability.
        https://repository.upenn.edu/asc_papers/43/
    """
    # Convert to float and handle missing values
    data = np.array(reliability_data, dtype=float)
    if not np.isnan(missing_value):
        data[data == missing_value] = np.nan

    n_raters, n_units = data.shape

    if n_raters < 2:
        raise ValueError("Krippendorff's Alpha requires at least 2 raters")

    if n_units < 2:
        raise ValueError("Krippendorff's Alpha requires at least 2 units")

    # Collect all pairwise coincidences within units
    # For each unit, count all pairs of values from different raters
    coincidences = []
    for u in range(n_units):
        unit_values = data[:, u]
        valid_values = unit_values[~np.isnan(unit_values)]
        n_valid = len(valid_values)
        if n_valid < 2:
            continue
        # All pairs of values within this unit
        for i in range(n_valid):
            for j in range(n_valid):
                if i != j:
                    coincidences.append((valid_values[i], valid_values[j]))

    if len(coincidences) == 0:
        logger.warning("No valid coincidences found for Krippendorff's Alpha")
        return np.nan

    coincidences = np.array(coincidences)

    # Observed disagreement: mean squared difference among coincidences
    # D_o = (1/n_c) * sum((c_i - c_j)^2) for all coincidence pairs
    observed_disagreement = np.mean((coincidences[:, 0] - coincidences[:, 1]) ** 2)

    # Expected disagreement: computed from all pairwise values in the dataset
    # Pool all non-missing values
    all_values = data[~np.isnan(data)]
    n_values = len(all_values)

    if n_values < 2:
        logger.warning("Insufficient values for expected disagreement computation")
        return np.nan

    # Expected disagreement: mean squared difference if values were random
    # D_e = (1/(n*(n-1))) * sum_c sum_c' (v_c - v_c')^2
    # This equals 2 * Var(all_values) for interval data
    expected_disagreement = 2 * np.var(all_values, ddof=0)

    if expected_disagreement == 0:
        # All values are identical — perfect agreement by definition
        return 1.0

    # Krippendorff's Alpha
    alpha = 1 - observed_disagreement / expected_disagreement

    return alpha


def bootstrap_ci(
    reliability_data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for Krippendorff's Alpha.

    Resamples units (columns) with replacement to estimate the sampling
    distribution of alpha.

    Args:
        reliability_data: Matrix of shape (n_raters, n_units).
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (default 0.95 for 95% CI).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (lower_bound, upper_bound) for the confidence interval.
    """
    rng = np.random.default_rng(random_seed)
    n_raters, n_units = reliability_data.shape

    alpha_samples = []
    for _ in range(n_bootstrap):
        # Resample units (columns) with replacement
        unit_indices = rng.choice(n_units, size=n_units, replace=True)
        resampled_data = reliability_data[:, unit_indices]

        try:
            alpha = krippendorff_alpha(resampled_data)
            if not np.isnan(alpha):
                alpha_samples.append(alpha)
        except ValueError:
            # Skip invalid bootstrap samples
            continue

    if len(alpha_samples) < 10:
        logger.warning(
            f"Only {len(alpha_samples)} valid bootstrap samples; CI may be unreliable"
        )
        if len(alpha_samples) == 0:
            return (np.nan, np.nan)

    alpha_arr = np.array(alpha_samples)
    lower_pct = (1 - confidence) / 2 * 100
    upper_pct = (1 + confidence) / 2 * 100

    return (np.percentile(alpha_arr, lower_pct), np.percentile(alpha_arr, upper_pct))


def _build_run_level_matrix(
    scores_df: pd.DataFrame,
    dimension: str,
    participant_col: str = "text_id",
    entity_col: str = "entity",
) -> np.ndarray:
    """
    Build reliability matrix for run-level agreement.

    Raters = LLM runs (columns like {dim}_run1, {dim}_run2, ...)
    Units = entity-participant pairs (each row in the DataFrame)

    Returns:
        Matrix of shape (n_runs, n_units).
    """
    # Find run columns for this dimension
    run_cols = sorted([
        c for c in scores_df.columns
        if c.startswith(f"{dimension}_run")
    ])

    if len(run_cols) == 0:
        raise ValueError(
            f"No run-level columns found for dimension '{dimension}'. "
            f"Expected columns like '{dimension}_run1', '{dimension}_run2', etc."
        )

    # Each row is one unit; each run column is one rater
    reliability_matrix = scores_df[run_cols].values.T  # (n_runs, n_units)

    return reliability_matrix


def _build_participant_level_matrix(
    scores_df: pd.DataFrame,
    dimension: str,
    participant_col: str = "text_id",
    entity_col: str = "entity",
) -> np.ndarray:
    """
    Build reliability matrix for participant-level agreement.

    Raters = participants
    Units = entities (aggregated across all rows for each entity)

    Returns:
        Matrix of shape (n_participants, n_entities).
    """
    mean_col = f"{dimension}_mean"

    if mean_col not in scores_df.columns:
        raise ValueError(
            f"Column '{mean_col}' not found. "
            f"Participant-level agreement requires mean score columns."
        )

    # Pivot: rows=participants, columns=entities, values=mean scores
    pivot = scores_df.pivot_table(
        index=participant_col,
        columns=entity_col,
        values=mean_col,
        aggfunc="mean",  # In case of duplicates
    )

    # Matrix: (n_participants, n_entities)
    reliability_matrix = pivot.values

    return reliability_matrix


def compute_agreement(
    scores_df: pd.DataFrame,
    dimensions: List[str],
    mode: str = "run_level",
    participant_col: str = "text_id",
    entity_col: str = "entity",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, AgreementResult]:
    """
    Compute Krippendorff's Alpha for multiple dimensions.

    Args:
        scores_df: DataFrame with scored entities. Must contain either:
            - Run-level columns ({dim}_run1, {dim}_run2, ...) for run_level mode
            - Mean columns ({dim}_mean) for participant_level mode
        dimensions: List of dimension names (e.g., ["social", "ecological"]).
        mode: "run_level" (LLM run consistency) or "participant_level" (inter-participant agreement).
        participant_col: Column identifying participants.
        entity_col: Column identifying entities.
        n_bootstrap: Number of bootstrap samples for CI.
        confidence: Confidence level for CI.
        random_seed: Random seed for bootstrap reproducibility.

    Returns:
        Dict mapping dimension name to AgreementResult.

    Raises:
        ValueError: If mode is invalid or required columns are missing.
    """
    if mode not in ("run_level", "participant_level"):
        raise ValueError(f"mode must be 'run_level' or 'participant_level', got '{mode}'")

    results = {}

    for dim in dimensions:
        try:
            if mode == "run_level":
                reliability_matrix = _build_run_level_matrix(
                    scores_df, dim, participant_col, entity_col
                )
            else:
                reliability_matrix = _build_participant_level_matrix(
                    scores_df, dim, participant_col, entity_col
                )

            n_raters, n_units = reliability_matrix.shape

            if n_units < 10:
                logger.warning(
                    f"Dimension '{dim}': only {n_units} units — "
                    f"Krippendorff's Alpha may be unstable with few units."
                )

            # Compute alpha
            alpha = krippendorff_alpha(reliability_matrix)

            # Bootstrap CI
            ci_lower, ci_upper = bootstrap_ci(
                reliability_matrix,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                random_seed=random_seed,
            )

            interpretation = _interpret_alpha(alpha) if not np.isnan(alpha) else "undefined"

            results[dim] = AgreementResult(
                alpha=alpha,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_units=n_units,
                n_raters=n_raters,
                mode=mode,
                dimension=dim,
                interpretation=interpretation,
            )

            logger.info(
                f"Agreement [{dim}] ({mode}): α = {alpha:.3f} "
                f"[{ci_lower:.3f}, {ci_upper:.3f}] — {interpretation}"
            )

        except Exception as e:
            logger.error(f"Failed to compute agreement for dimension '{dim}': {e}")
            results[dim] = AgreementResult(
                alpha=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                n_units=0,
                n_raters=0,
                mode=mode,
                dimension=dim,
                interpretation="error",
            )

    return results
