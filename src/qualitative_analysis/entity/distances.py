"""
Distance metrics for entity score comparison.

Provides distance and similarity metrics for comparing entity score vectors
across participants. Includes both standard metrics (Euclidean, cosine) and
compositional metrics (Aitchison, Wasserstein).

Metric choice guidance:
    - **euclidean** (default): Treats each dimension as an independent axis.
      Appropriate when absolute score levels matter (e.g., Social/Ecological/
      Technological scores where an entity can score 80/80/80).
    - **cosine**: Measures angular similarity; ignores magnitude. Useful when
      only the *profile shape* matters, not absolute levels.
    - **aitchison**: The proper metric for **compositional data** (parts of a
      whole that sum to a constant). Uses CLR transform internally.
      **WARNING:** Do NOT use for independent dimension scores — normalizing
      to sum-to-1 destroys absolute score information and manufactures
      artificial negative correlations.
    - **emd / wasserstein**: Earth Mover's Distance. Useful for comparing
      full score *distributions* (not just means) across participants.

Example usage:
    from qualitative_analysis.entity.distances import (
        compute_pairwise_distances,
        cosine_distance,
    )

    distance_matrix, ids = compute_pairwise_distances(
        participant_scores, metric="euclidean"
    )
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Distance Metrics for Compositional Data
# =============================================================================


def clr_transform(x: np.ndarray) -> np.ndarray:
    """
    Centered Log-Ratio (CLR) transformation for compositional data.

    Transforms compositional data from the simplex to unconstrained space,
    enabling standard statistical operations.

    .. warning::
        This transform normalizes vectors to sum to 1, which means it is only
        appropriate for **compositional data** (parts of a whole). If your
        dimension scores are independent (e.g., an entity can score 80/80/80),
        use Euclidean or cosine distance on raw scores instead. Applying CLR
        to non-compositional data destroys absolute score-level information
        and manufactures artificial negative correlations between dimensions.

    Args:
        x: Compositional data array, shape (n_samples, n_components) or (n_components,).
           Values should be positive and ideally sum to 1 (or constant).

    Returns:
        CLR-transformed data with same shape as input.

    Notes:
        - Zeros are handled by adding a small epsilon before transformation.
        - The geometric mean is computed per sample (row).
    """
    x = np.asarray(x, dtype=np.float64)

    # Handle zeros by adding small epsilon
    epsilon = 1e-10
    x = np.clip(x, epsilon, None)

    # Normalize to proportions if not already
    if x.ndim == 1:
        x = x / x.sum()
        log_x = np.log(x)
        return log_x - np.mean(log_x)
    else:
        x = x / x.sum(axis=1, keepdims=True)
        log_x = np.log(x)
        return log_x - np.mean(log_x, axis=1, keepdims=True)


def ilr_transform(x: np.ndarray) -> np.ndarray:
    """
    Isometric Log-Ratio (ILR) transformation for compositional data.

    Transforms D-dimensional compositional data to (D-1)-dimensional
    unconstrained space while preserving distances.

    Args:
        x: Compositional data array, shape (n_samples, n_components) or (n_components,).

    Returns:
        ILR-transformed data with shape (n_samples, n_components-1) or (n_components-1,).
    """
    x = np.asarray(x, dtype=np.float64)

    # Handle zeros
    epsilon = 1e-10
    x = np.clip(x, epsilon, None)

    # Normalize
    if x.ndim == 1:
        x = x / x.sum()
        D = len(x)
    else:
        x = x / x.sum(axis=1, keepdims=True)
        D = x.shape[1]

    # Construct Helmert subcomposition matrix
    # This creates an orthonormal basis for the simplex
    log_x = np.log(x)

    if x.ndim == 1:
        ilr = np.zeros(D - 1)
        for i in range(D - 1):
            ilr[i] = (1.0 / np.sqrt((i + 1) * (i + 2))) * (
                np.sum(log_x[: i + 1]) - (i + 1) * log_x[i + 1]
            )
    else:
        ilr = np.zeros((x.shape[0], D - 1))
        for i in range(D - 1):
            ilr[:, i] = (1.0 / np.sqrt((i + 1) * (i + 2))) * (
                np.sum(log_x[:, : i + 1], axis=1) - (i + 1) * log_x[:, i + 1]
            )

    return ilr


def aitchison_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Aitchison distance between two compositional vectors.

    The Aitchison distance is the proper distance metric for compositional data,
    accounting for the relative nature of proportions.

    Args:
        x: First compositional vector, shape (n_components,).
        y: Second compositional vector, shape (n_components,).

    Returns:
        Aitchison distance (non-negative float).

    Notes:
        d_A(x, y) = ||clr(x) - clr(y)||_2

        This is equivalent to the Euclidean distance in CLR-transformed space.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")

    clr_x = clr_transform(x)
    clr_y = clr_transform(y)

    return float(np.sqrt(np.sum((clr_x - clr_y) ** 2)))


def aitchison_distance_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Aitchison distance matrix.

    Args:
        X: Compositional data matrix, shape (n_samples, n_components).

    Returns:
        Distance matrix, shape (n_samples, n_samples).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    # Transform all at once
    clr_X = clr_transform(X)

    # Compute pairwise Euclidean distances in CLR space
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((clr_X[i] - clr_X[j]) ** 2))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def wasserstein_distance_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute 1-Wasserstein (Earth Mover's) distance between two 1D distributions.

    Args:
        x: First distribution samples.
        y: Second distribution samples.

    Returns:
        Wasserstein distance.
    """
    from scipy.stats import wasserstein_distance as scipy_wasserstein

    return float(scipy_wasserstein(x, y))


def wasserstein_distance_compositional(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "sliced",
    n_projections: int = 100,
) -> float:
    """
    Compute Wasserstein distance between two compositional distributions.

    For multivariate compositional data, we use either:
    - Sliced Wasserstein: Fast approximation via random 1D projections
    - Exact EMD: Requires POT library, slower but exact

    Args:
        x: First distribution, shape (n_samples_x, n_components) or (n_components,).
        y: Second distribution, shape (n_samples_y, n_components) or (n_components,).
        method: "sliced" for sliced Wasserstein, "exact" for exact EMD.
        n_projections: Number of random projections for sliced method.

    Returns:
        Wasserstein distance (non-negative float).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Handle 1D case (single sample per distribution)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)

    # Work in CLR space for proper compositional treatment
    clr_x = clr_transform(x)
    clr_y = clr_transform(y)

    if method == "sliced":
        return _sliced_wasserstein(clr_x, clr_y, n_projections)
    elif method == "exact":
        return _exact_wasserstein(clr_x, clr_y)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sliced' or 'exact'.")


def _sliced_wasserstein(x: np.ndarray, y: np.ndarray, n_projections: int = 100) -> float:
    """
    Sliced Wasserstein distance approximation.

    Projects high-dimensional distributions onto random 1D directions
    and averages the 1D Wasserstein distances.
    """
    from scipy.stats import wasserstein_distance as scipy_wasserstein

    d = x.shape[1]
    distances = []

    # Generate random projection directions (unit vectors)
    rng = np.random.default_rng(42)
    for _ in range(n_projections):
        direction = rng.standard_normal(d)
        direction = direction / np.linalg.norm(direction)

        # Project both distributions
        proj_x = x @ direction
        proj_y = y @ direction

        # 1D Wasserstein
        distances.append(scipy_wasserstein(proj_x, proj_y))

    return float(np.mean(distances))


def _exact_wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    """
    Exact Wasserstein distance using optimal transport.

    Requires the POT (Python Optimal Transport) library.
    """
    try:
        import ot
    except ImportError:
        logger.warning(
            "POT library not installed. Falling back to sliced Wasserstein. "
            "Install with: pip install POT"
        )
        return _sliced_wasserstein(x, y)

    n_x, n_y = len(x), len(y)

    # Uniform weights
    weights_x = np.ones(n_x) / n_x
    weights_y = np.ones(n_y) / n_y

    # Cost matrix (squared Euclidean distances)
    cost = ot.dist(x, y, metric="sqeuclidean")

    # Compute EMD
    emd_value = ot.emd2(weights_x, weights_y, cost)

    return float(np.sqrt(emd_value))  # Return W2 distance


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        Cosine similarity in range [-1, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)

    if norm_x == 0 or norm_y == 0:
        return 0.0

    return float(np.dot(x, y) / (norm_x * norm_y))


def cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute cosine distance between two vectors.

    Args:
        x: First vector.
        y: Second vector.

    Returns:
        Cosine distance in range [0, 2].
    """
    return 1.0 - cosine_similarity(x, y)


# =============================================================================
# Pairwise Distance Computation
# =============================================================================


def compute_pairwise_distances(
    scores: Dict[str, np.ndarray],
    metric: str = "euclidean",
    **kwargs,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distance matrix between participants.

    Args:
        scores: Dictionary mapping participant IDs to score arrays.
                Each array should be shape (n_entities, n_dimensions) or
                (n_dimensions,) for single entity mean scores.
        metric: Distance metric to use:
                - "euclidean": Standard Euclidean distance (default)
                - "cosine": Cosine distance
                - "aitchison": Aitchison distance (compositional — only
                  appropriate when dimensions sum to a constant)
                - "emd" or "wasserstein": Earth Mover's Distance
        **kwargs: Additional arguments passed to distance function.

    Returns:
        Tuple of (distance_matrix, participant_ids).
        distance_matrix is shape (n_participants, n_participants).
    """
    participant_ids = list(scores.keys())
    n = len(participant_ids)

    # Select distance function
    if metric == "aitchison":
        dist_fn = lambda x, y: aitchison_distance(x.mean(axis=0) if x.ndim > 1 else x,
                                                   y.mean(axis=0) if y.ndim > 1 else y)
    elif metric in ("emd", "wasserstein"):
        dist_fn = lambda x, y: wasserstein_distance_compositional(x, y, **kwargs)
    elif metric == "cosine":
        dist_fn = lambda x, y: cosine_distance(x.mean(axis=0) if x.ndim > 1 else x,
                                                y.mean(axis=0) if y.ndim > 1 else y)
    elif metric == "euclidean":
        dist_fn = lambda x, y: float(np.linalg.norm(
            (x.mean(axis=0) if x.ndim > 1 else x) -
            (y.mean(axis=0) if y.ndim > 1 else y)
        ))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute pairwise distances
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            pid_i = participant_ids[i]
            pid_j = participant_ids[j]

            d = dist_fn(scores[pid_i], scores[pid_j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    logger.info(f"Computed {n}x{n} pairwise distance matrix using {metric} metric")

    return dist_matrix, participant_ids
