"""
Multi-participant comparison module for entity scores.

Provides distance metrics, statistical tests, and comparison utilities
for analyzing how different participants score entities across dimensions.

Supports:
- Compositional distance metrics (Aitchison, EMD/Wasserstein)
- Pairwise participant similarity matrices
- Group-level comparisons
- Bayesian hierarchical modeling (future)

Example usage:
    from qualitative_analysis.entity.comparison import (
        aitchison_distance,
        wasserstein_distance_compositional,
        compute_pairwise_distances,
        ParticipantComparison,
    )

    # Compute distance between two score vectors
    dist = aitchison_distance(scores_a, scores_b)

    # Compute all pairwise distances
    distance_matrix = compute_pairwise_distances(
        participant_scores,
        metric="aitchison"
    )

    # Full comparison analysis
    comparison = ParticipantComparison(scored_entities_df)
    results = comparison.analyze(method="pairwise", metric="emd")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    np.random.seed(42)  # For reproducibility
    for _ in range(n_projections):
        direction = np.random.randn(d)
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
    metric: str = "aitchison",
    **kwargs,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute pairwise distance matrix between participants.

    Args:
        scores: Dictionary mapping participant IDs to score arrays.
                Each array should be shape (n_entities, n_dimensions) or
                (n_dimensions,) for single entity mean scores.
        metric: Distance metric to use:
                - "aitchison": Aitchison distance (compositional)
                - "emd" or "wasserstein": Earth Mover's Distance
                - "cosine": Cosine distance
                - "euclidean": Standard Euclidean distance
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


# =============================================================================
# Participant Comparison Class
# =============================================================================


@dataclass
class ComparisonResult:
    """Results from participant comparison analysis."""

    # Distance matrix
    distance_matrix: np.ndarray
    participant_ids: List[str]
    metric: str

    # Summary statistics
    mean_distance: float
    median_distance: float
    min_distance: float
    max_distance: float

    # Most similar and different pairs
    most_similar_pair: Tuple[str, str, float]
    most_different_pair: Tuple[str, str, float]

    # Aggregation mode used
    aggregate: str = "mean"

    # Optional: clustering results
    cluster_labels: Optional[np.ndarray] = None
    n_clusters: Optional[int] = None

    # Optional: dimension-specific results
    dimension_comparisons: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "metric": self.metric,
            "aggregate": self.aggregate,
            "n_participants": len(self.participant_ids),
            "participant_ids": self.participant_ids,
            "summary": {
                "mean_distance": self.mean_distance,
                "median_distance": self.median_distance,
                "min_distance": self.min_distance,
                "max_distance": self.max_distance,
            },
            "most_similar_pair": {
                "participants": list(self.most_similar_pair[:2]),
                "distance": self.most_similar_pair[2],
            },
            "most_different_pair": {
                "participants": list(self.most_different_pair[:2]),
                "distance": self.most_different_pair[2],
            },
            "n_clusters": self.n_clusters,
            "distance_matrix": self.distance_matrix.tolist(),
        }


@dataclass
class GroupComparisonResult:
    """Results from group-level comparison analysis.

    This dataclass holds statistics for comparing pre-defined groups of participants,
    including within-group distances, between-group distances, and group centroids.

    Attributes:
        groups: Dictionary mapping group names to lists of participant IDs.
        metric: Distance metric used (e.g., "aitchison", "emd").
        aggregate: Aggregation mode ("mean" or "distribution").
        within_group_distances: Per-group distance statistics.
        between_group_distances: Pairwise group distance statistics.
        group_centroids: Mean score vector per group.
        excluded_participants: Participants not assigned to any group.
        permutation_test_p: Optional p-value from permutation test.
        effect_size: Optional effect size (between/within ratio).
    """

    groups: Dict[str, List[str]]
    metric: str
    aggregate: str

    # Within-group statistics: {group_name: {"mean": x, "median": y, ...}}
    within_group_distances: Dict[str, Dict[str, float]]

    # Between-group statistics: {"group1_vs_group2": {"mean": x, ...}}
    between_group_distances: Dict[str, Dict[str, float]]

    # Group centroids: {group_name: [dim1, dim2, dim3]}
    group_centroids: Dict[str, List[float]]

    # Participants excluded from analysis (not in any group)
    excluded_participants: List[str] = field(default_factory=list)

    # Optional: statistical test results
    permutation_test_p: Optional[float] = None
    effect_size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "metric": self.metric,
            "aggregate": self.aggregate,
            "groups": self.groups,
            "group_sizes": {name: len(members) for name, members in self.groups.items()},
            "within_group_distances": self.within_group_distances,
            "between_group_distances": self.between_group_distances,
            "group_centroids": self.group_centroids,
        }

        if self.excluded_participants:
            result["excluded_participants"] = self.excluded_participants

        if self.permutation_test_p is not None:
            result["permutation_test_p"] = self.permutation_test_p

        if self.effect_size is not None:
            result["effect_size"] = self.effect_size

        return result

    def summary_str(self) -> str:
        """Generate human-readable summary string."""
        lines = [
            f"Group Comparison Summary ({self.metric}, {self.aggregate}):",
            f"  Groups: {', '.join(f'{name} (n={len(members)})' for name, members in self.groups.items())}",
            "",
            "  Within-group distances:",
        ]

        for group_name, stats in self.within_group_distances.items():
            lines.append(f"    {group_name}: mean={stats.get('mean', 0):.4f}, median={stats.get('median', 0):.4f}")

        lines.append("")
        lines.append("  Between-group distances:")

        for pair_name, stats in self.between_group_distances.items():
            lines.append(f"    {pair_name}: mean={stats.get('mean', 0):.4f}")

        if self.effect_size is not None:
            lines.append("")
            lines.append(f"  Effect size (between/within ratio): {self.effect_size:.2f}")

        if self.excluded_participants:
            lines.append("")
            lines.append(f"  Excluded participants (not in any group): {len(self.excluded_participants)}")

        return "\n".join(lines)


class ParticipantComparison:
    """
    Analyze and compare entity scores across multiple participants.

    This class provides methods for computing distances between participants,
    identifying clusters of similar participants, and statistical comparisons.

    Example:
        comparison = ParticipantComparison()
        comparison.load_scores(scores_csv_path)
        results = comparison.compute_distances(metric="aitchison")
        comparison.plot_heatmap(results, output_path="heatmap.png")
    """

    def __init__(self):
        self.scores_by_participant: Dict[str, np.ndarray] = {}
        self.dimension_names: List[str] = []
        self.entity_names: List[str] = []

    def load_scores(
        self,
        scores_path: Union[str, Path],
        participant_col: str = "text_id",
        entity_col: str = "entity",
        dimension_pattern: str = "{dim}_mean",
        dimensions: Optional[List[str]] = None,
    ) -> None:
        """
        Load scored entities from CSV file.

        Args:
            scores_path: Path to scored entities CSV (from `qa entity score`).
            participant_col: Column name for participant/text IDs.
            entity_col: Column name for entity names.
            dimension_pattern: Pattern for dimension columns, with {dim} placeholder.
            dimensions: List of dimension names. If None, auto-detected.
        """
        import csv

        scores_path = Path(scores_path)

        with open(scores_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            raise ValueError(f"No data in {scores_path}")

        # Auto-detect dimensions if not provided
        if dimensions is None:
            dimensions = []
            for col in rows[0].keys():
                if col.endswith("_mean"):
                    dim_name = col.replace("_mean", "")
                    dimensions.append(dim_name)

        self.dimension_names = dimensions
        logger.info(f"Detected dimensions: {dimensions}")

        # Group by participant
        participant_scores: Dict[str, List[np.ndarray]] = {}
        entity_set = set()

        for row in rows:
            participant = row.get(participant_col, "unknown")
            entity = row.get(entity_col, "unknown")
            entity_set.add(entity)

            # Extract dimension scores
            score_vector = []
            for dim in dimensions:
                col_name = dimension_pattern.format(dim=dim)
                try:
                    score = float(row.get(col_name, 0))
                except (ValueError, TypeError):
                    score = 0.0
                score_vector.append(score)

            if participant not in participant_scores:
                participant_scores[participant] = []
            participant_scores[participant].append(np.array(score_vector))

        # Convert to arrays
        for participant, scores_list in participant_scores.items():
            self.scores_by_participant[participant] = np.array(scores_list)

        self.entity_names = sorted(entity_set)

        logger.info(
            f"Loaded scores for {len(self.scores_by_participant)} participants, "
            f"{len(self.entity_names)} entities, {len(self.dimension_names)} dimensions"
        )

    def compute_distances(
        self,
        metric: str = "aitchison",
        aggregate: str = "mean",
        **kwargs,
    ) -> ComparisonResult:
        """
        Compute pairwise distances between all participants.

        Args:
            metric: Distance metric ("aitchison", "emd", "cosine", "euclidean").
            aggregate: How to aggregate multiple entities per participant:
                - "mean": Compare centroids (single point per participant).
                  Works with all metrics. Fast but loses distribution info.
                - "distribution": Compare full entity distributions per participant.
                  Only meaningful for EMD metric. Captures spread and shape.
            **kwargs: Additional arguments for distance function.

        Returns:
            ComparisonResult with distance matrix and summary statistics.

        Notes:
            For EMD with aggregate="distribution", the distance measures how much
            "work" is needed to transform one participant's entity distribution
            into another's. This captures differences in spread, clustering, and
            overall distribution shape that centroid comparison misses.

            Example: Two participants might have identical centroids but very
            different distributions - one with tightly clustered entities, another
            with widely spread entities. EMD with distribution mode captures this.
        """
        if not self.scores_by_participant:
            raise ValueError("No scores loaded. Call load_scores() first.")

        # Normalize aggregate parameter
        if aggregate in ("distribution", "none"):
            aggregate_mode = "distribution"
        elif aggregate == "mean":
            aggregate_mode = "mean"
        else:
            raise ValueError(
                f"Unknown aggregate method: {aggregate}. "
                "Use 'mean' (centroid) or 'distribution' (full, EMD only)."
            )

        # Warn if using distribution mode with non-EMD metric
        if aggregate_mode == "distribution" and metric not in ("emd", "wasserstein"):
            logger.warning(
                f"aggregate='distribution' is only meaningful with EMD metric. "
                f"Using metric='{metric}' will still aggregate to means internally. "
                f"Consider using metric='emd' for distribution-level comparison."
            )

        # Prepare scores based on aggregation method
        if aggregate_mode == "mean":
            scores = {
                pid: arr.mean(axis=0) for pid, arr in self.scores_by_participant.items()
            }
        else:
            # Pass full distributions for EMD
            scores = self.scores_by_participant

        # Compute distance matrix
        dist_matrix, participant_ids = compute_pairwise_distances(
            scores, metric=metric, **kwargs
        )

        # Extract upper triangle (excluding diagonal)
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]

        # Find most similar and different pairs
        n = len(participant_ids)
        min_idx = np.argmin(upper_tri)
        max_idx = np.argmax(upper_tri)

        # Convert flat index to (i, j) pair
        triu_indices = np.triu_indices(n, k=1)

        min_i, min_j = triu_indices[0][min_idx], triu_indices[1][min_idx]
        max_i, max_j = triu_indices[0][max_idx], triu_indices[1][max_idx]

        return ComparisonResult(
            distance_matrix=dist_matrix,
            participant_ids=participant_ids,
            metric=metric,
            aggregate=aggregate_mode,
            mean_distance=float(np.mean(upper_tri)),
            median_distance=float(np.median(upper_tri)),
            min_distance=float(np.min(upper_tri)),
            max_distance=float(np.max(upper_tri)),
            most_similar_pair=(
                participant_ids[min_i],
                participant_ids[min_j],
                float(dist_matrix[min_i, min_j]),
            ),
            most_different_pair=(
                participant_ids[max_i],
                participant_ids[max_j],
                float(dist_matrix[max_i, max_j]),
            ),
        )

    def cluster_participants(
        self,
        result: ComparisonResult,
        method: str = "hierarchical",
        n_clusters: Optional[int] = None,
        **kwargs,
    ) -> ComparisonResult:
        """
        Cluster participants based on their distance matrix.

        Args:
            result: ComparisonResult from compute_distances().
            method: Clustering method ("hierarchical", "kmeans", "hdbscan").
            n_clusters: Number of clusters (required for hierarchical/kmeans).
            **kwargs: Additional arguments for clustering algorithm.

        Returns:
            Updated ComparisonResult with cluster labels.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        dist_matrix = result.distance_matrix

        if method == "hierarchical":
            # Convert to condensed form for scipy
            condensed = squareform(dist_matrix)
            linkage_matrix = linkage(condensed, method="ward")

            if n_clusters is None:
                n_clusters = max(2, len(result.participant_ids) // 5)

            labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
            result.cluster_labels = labels - 1  # 0-indexed
            result.n_clusters = n_clusters

        elif method == "hdbscan":
            try:
                import hdbscan

                clusterer = hdbscan.HDBSCAN(metric="precomputed", **kwargs)
                labels = clusterer.fit_predict(dist_matrix)
                result.cluster_labels = labels
                result.n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            except ImportError:
                logger.warning("hdbscan not installed, using hierarchical clustering")
                return self.cluster_participants(result, method="hierarchical", n_clusters=n_clusters)

        else:
            raise ValueError(f"Unknown clustering method: {method}")

        logger.info(f"Clustered participants into {result.n_clusters} groups")
        return result

    def get_participant_centroid(self, participant_id: str) -> np.ndarray:
        """Get the mean score vector for a participant."""
        if participant_id not in self.scores_by_participant:
            raise ValueError(f"Unknown participant: {participant_id}")
        return self.scores_by_participant[participant_id].mean(axis=0)

    def get_all_centroids(self) -> Dict[str, np.ndarray]:
        """Get centroids for all participants."""
        return {pid: arr.mean(axis=0) for pid, arr in self.scores_by_participant.items()}

    def set_groups(self, groups: Dict[str, List[str]]) -> List[str]:
        """
        Define participant groups for comparison.

        Args:
            groups: Dict mapping group names to participant ID lists.
                    e.g., {"control": ["pid1", "pid2"], "treatment": ["pid3", "pid4"]}

        Returns:
            List of participant IDs that were excluded (not found in loaded data).

        Raises:
            ValueError: If groups is empty or contains invalid data.
        """
        if not groups:
            raise ValueError("Groups dictionary cannot be empty")

        self.groups = {}
        excluded = []

        for group_name, member_ids in groups.items():
            valid_members = []
            for pid in member_ids:
                if pid in self.scores_by_participant:
                    valid_members.append(pid)
                else:
                    excluded.append(pid)
                    logger.warning(f"Participant '{pid}' in group '{group_name}' not found in loaded data")

            if valid_members:
                self.groups[group_name] = valid_members
            else:
                logger.warning(f"Group '{group_name}' has no valid participants after filtering")

        if not self.groups:
            raise ValueError("No valid groups after filtering. Check participant IDs.")

        # Warn about small groups
        for group_name, members in self.groups.items():
            if len(members) < 2:
                logger.warning(
                    f"Group '{group_name}' has only {len(members)} participant(s). "
                    "Within-group distance cannot be computed meaningfully."
                )

        logger.info(f"Set {len(self.groups)} groups: {list(self.groups.keys())}")
        return excluded

    def compute_group_distances(
        self,
        metric: str = "aitchison",
        aggregate: str = "mean",
        **kwargs,
    ) -> GroupComparisonResult:
        """
        Compute within-group and between-group distances.

        This method computes pairwise distances within each group and between
        groups, providing statistics useful for comparing group-level patterns.

        Args:
            metric: Distance metric ("aitchison", "emd", "cosine", "euclidean").
            aggregate: How to aggregate entities per participant:
                - "mean": Compare centroids (single point per participant).
                - "distribution": Compare full entity distributions (EMD only).
            **kwargs: Additional arguments for distance function.

        Returns:
            GroupComparisonResult with within-group, between-group distances,
            and group centroids.

        Raises:
            ValueError: If groups have not been set via set_groups().
        """
        if not hasattr(self, 'groups') or not self.groups:
            raise ValueError("No groups defined. Call set_groups() first.")

        # Normalize aggregate parameter
        if aggregate in ("distribution", "none"):
            aggregate_mode = "distribution"
        elif aggregate == "mean":
            aggregate_mode = "mean"
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        # Warn if using distribution mode with non-EMD metric
        if aggregate_mode == "distribution" and metric not in ("emd", "wasserstein"):
            logger.warning(
                f"aggregate='distribution' is only meaningful with EMD metric. "
                f"Using metric='{metric}' will compare centroids internally."
            )

        # First, compute full pairwise distance matrix for all participants in groups
        all_grouped_pids = []
        for members in self.groups.values():
            all_grouped_pids.extend(members)
        all_grouped_pids = list(set(all_grouped_pids))  # Dedupe

        # Identify excluded participants
        all_pids = set(self.scores_by_participant.keys())
        grouped_pids = set(all_grouped_pids)
        excluded_participants = list(all_pids - grouped_pids)

        # Prepare scores for distance computation
        if aggregate_mode == "mean":
            scores = {
                pid: self.scores_by_participant[pid].mean(axis=0)
                for pid in all_grouped_pids
            }
        else:
            scores = {
                pid: self.scores_by_participant[pid]
                for pid in all_grouped_pids
            }

        # Compute full pairwise distance matrix
        dist_matrix, pid_order = compute_pairwise_distances(scores, metric=metric, **kwargs)

        # Create pid -> index mapping
        pid_to_idx = {pid: i for i, pid in enumerate(pid_order)}

        # Compute within-group distances
        within_group_distances = {}
        for group_name, members in self.groups.items():
            if len(members) < 2:
                # Can't compute pairwise distances with < 2 members
                within_group_distances[group_name] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "n_pairs": 0,
                }
                continue

            # Extract pairwise distances within this group
            within_dists = []
            for i, pid_i in enumerate(members):
                for pid_j in members[i + 1:]:
                    idx_i = pid_to_idx[pid_i]
                    idx_j = pid_to_idx[pid_j]
                    within_dists.append(dist_matrix[idx_i, idx_j])

            within_group_distances[group_name] = {
                "mean": float(np.mean(within_dists)),
                "median": float(np.median(within_dists)),
                "min": float(np.min(within_dists)),
                "max": float(np.max(within_dists)),
                "n_pairs": len(within_dists),
            }

        # Compute between-group distances
        between_group_distances = {}
        group_names = list(self.groups.keys())

        for i, group_i in enumerate(group_names):
            for group_j in group_names[i + 1:]:
                members_i = self.groups[group_i]
                members_j = self.groups[group_j]

                # All pairwise distances between groups
                between_dists = []
                for pid_i in members_i:
                    for pid_j in members_j:
                        idx_i = pid_to_idx[pid_i]
                        idx_j = pid_to_idx[pid_j]
                        between_dists.append(dist_matrix[idx_i, idx_j])

                pair_key = f"{group_i}_vs_{group_j}"
                between_group_distances[pair_key] = {
                    "mean": float(np.mean(between_dists)),
                    "median": float(np.median(between_dists)),
                    "min": float(np.min(between_dists)),
                    "max": float(np.max(between_dists)),
                    "n_pairs": len(between_dists),
                }

        # Compute group centroids
        group_centroids = {}
        for group_name, members in self.groups.items():
            member_centroids = [
                self.scores_by_participant[pid].mean(axis=0)
                for pid in members
            ]
            group_centroid = np.mean(member_centroids, axis=0)
            group_centroids[group_name] = group_centroid.tolist()

        # Compute effect size (ratio of mean between-group to mean within-group)
        all_within = []
        for stats in within_group_distances.values():
            if stats["n_pairs"] > 0:
                all_within.append(stats["mean"])

        all_between = []
        for stats in between_group_distances.values():
            all_between.append(stats["mean"])

        effect_size = None
        if all_within and all_between:
            mean_within = np.mean(all_within)
            mean_between = np.mean(all_between)
            if mean_within > 0:
                effect_size = float(mean_between / mean_within)

        result = GroupComparisonResult(
            groups=self.groups,
            metric=metric,
            aggregate=aggregate_mode,
            within_group_distances=within_group_distances,
            between_group_distances=between_group_distances,
            group_centroids=group_centroids,
            excluded_participants=excluded_participants,
            effect_size=effect_size,
        )

        logger.info(f"Computed group distances for {len(self.groups)} groups")
        return result


# =============================================================================
# Convenience Functions
# =============================================================================


# =============================================================================
# Comparison Visualizations
# =============================================================================


class ComparisonVisualizer:
    """
    Visualization utilities for multi-participant comparisons.

    Provides faceted ternary plots, overlaid ternary plots, and
    distance heatmaps for comparing entity scores across participants.
    """

    def __init__(self, comparison: ParticipantComparison):
        """
        Initialize with a loaded ParticipantComparison.

        Args:
            comparison: ParticipantComparison with loaded scores.
        """
        self.comparison = comparison
        self.dimension_colors = {
            "social": "#1f77b4",       # Blue
            "ecological": "#2ca02c",   # Green
            "technological": "#d62728" # Red
        }

        # Extended color palette for participants
        self.participant_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        ]

    def _barycentric_to_cartesian(
        self,
        a: float,
        b: float,
        c: float
    ) -> Tuple[float, float]:
        """
        Convert barycentric coordinates to cartesian coordinates.

        Args:
            a: Coordinate for top vertex
            b: Coordinate for bottom-left vertex
            c: Coordinate for bottom-right vertex

        Returns:
            Tuple of (x, y) cartesian coordinates
        """
        # Equilateral triangle vertices
        vertices = np.array([
            [0, 0],               # Bottom-left
            [1, 0],               # Bottom-right
            [0.5, np.sqrt(3)/2]   # Top
        ])

        x = a * vertices[2, 0] + b * vertices[0, 0] + c * vertices[1, 0]
        y = a * vertices[2, 1] + b * vertices[0, 1] + c * vertices[1, 1]

        return x, y

    def _draw_triangle(
        self,
        ax,
        dimension_names: List[str],
        show_labels: bool = True,
        show_grid: bool = True,
    ) -> None:
        """Draw ternary triangle with optional labels and grid."""
        import matplotlib.pyplot as plt

        # Triangle outline
        vertices = np.array([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [0, 0]
        ])
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=1.5)

        if show_labels and len(dimension_names) >= 3:
            label_offset = 0.06

            # Bottom-left (dimension 2)
            ax.text(0, -label_offset, dimension_names[2][:4].upper(),
                    ha='center', va='top', fontsize=8, fontweight='bold')

            # Bottom-right (dimension 0)
            ax.text(1, -label_offset, dimension_names[0][:4].upper(),
                    ha='center', va='top', fontsize=8, fontweight='bold')

            # Top (dimension 1)
            ax.text(0.5, np.sqrt(3)/2 + label_offset, dimension_names[1][:4].upper(),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        if show_grid:
            for i in [2, 4, 6, 8]:
                alpha = 0.2

                # Lines parallel to each edge
                x1, y1 = self._barycentric_to_cartesian(1-i/10, i/10, 0)
                x2, y2 = self._barycentric_to_cartesian(0, i/10, 1-i/10)
                ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

                x1, y1 = self._barycentric_to_cartesian(i/10, 0, 1-i/10)
                x2, y2 = self._barycentric_to_cartesian(i/10, 1-i/10, 0)
                ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

                x1, y1 = self._barycentric_to_cartesian(0, 1-i/10, i/10)
                x2, y2 = self._barycentric_to_cartesian(1-i/10, 0, i/10)
                ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to sum to 1 for ternary plotting."""
        scores = np.asarray(scores, dtype=np.float64)
        scores = np.clip(scores, 0, None)  # Ensure non-negative

        if scores.ndim == 1:
            total = scores.sum()
            return scores / total if total > 0 else np.ones(3) / 3
        else:
            totals = scores.sum(axis=1, keepdims=True)
            totals = np.where(totals == 0, 1, totals)
            return scores / totals

    def generate_faceted_ternary(
        self,
        output_path: Optional[Union[str, Path]] = None,
        dimension_names: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        max_cols: int = 4,
        marker_size: int = 60,
        show_centroid: bool = True,
        show_convex_hull: bool = False,
        title: Optional[str] = None,
    ):
        """
        Generate faceted ternary plots, one per participant.

        Creates a grid of ternary plots showing each participant's entity
        scores, allowing visual comparison of scoring patterns.

        Args:
            output_path: Path to save PNG file (optional).
            dimension_names: List of 3 dimension names. Uses loaded if None.
            participants: Specific participants to include. All if None.
            figsize: Figure size. Auto-calculated if None.
            max_cols: Maximum columns in grid.
            marker_size: Size of entity markers.
            show_centroid: Show centroid marker for each participant.
            show_convex_hull: Draw convex hull around entities.
            title: Overall plot title.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # Validate dimensions
        dims = dimension_names or self.comparison.dimension_names
        if len(dims) != 3:
            raise ValueError(f"Faceted ternary requires exactly 3 dimensions, got {len(dims)}")

        # Select participants
        if participants:
            pids = [p for p in participants if p in self.comparison.scores_by_participant]
        else:
            pids = list(self.comparison.scores_by_participant.keys())

        if not pids:
            raise ValueError("No participants to plot")

        n_participants = len(pids)
        n_cols = min(max_cols, n_participants)
        n_rows = (n_participants + n_cols - 1) // n_cols

        # Calculate figure size
        if figsize is None:
            fig_width = 3.5 * n_cols
            fig_height = 3.5 * n_rows + (0.5 if title else 0)
            figsize = (fig_width, fig_height)

        fig = plt.figure(figsize=figsize)

        # Add space for title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)

        for idx, pid in enumerate(pids):
            row = idx // n_cols
            col = idx % n_cols

            ax = fig.add_subplot(gs[row, col])

            # Get scores for this participant
            scores = self.comparison.scores_by_participant[pid]
            normalized = self._normalize_scores(scores)

            # Convert to cartesian coordinates
            xs, ys = [], []
            for score in normalized:
                x, y = self._barycentric_to_cartesian(score[1], score[2], score[0])
                xs.append(x)
                ys.append(y)

            xs = np.array(xs)
            ys = np.array(ys)

            # Draw triangle
            self._draw_triangle(ax, dims, show_labels=True, show_grid=True)

            # Draw convex hull if requested
            if show_convex_hull and len(xs) >= 3:
                try:
                    from scipy.spatial import ConvexHull

                    points = np.column_stack([xs, ys])
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close hull
                    ax.fill(hull_points[:, 0], hull_points[:, 1],
                           alpha=0.1, color='gray')
                    ax.plot(hull_points[:, 0], hull_points[:, 1],
                           'k--', alpha=0.3, linewidth=1)
                except Exception as e:
                    logger.debug(f"Could not compute convex hull for {pid}: {e}")

            # Determine primary dimension colors for each entity
            colors = []
            for score in scores:
                primary_idx = np.argmax(score)
                dim_name = dims[primary_idx].lower()
                colors.append(self.dimension_colors.get(dim_name, '#7f7f7f'))

            # Plot entities
            ax.scatter(xs, ys, s=marker_size, c=colors,
                      edgecolors='white', linewidths=0.5, alpha=0.7)

            # Plot centroid
            if show_centroid:
                centroid_scores = self._normalize_scores(scores.mean(axis=0))
                cx, cy = self._barycentric_to_cartesian(
                    centroid_scores[1], centroid_scores[2], centroid_scores[0]
                )
                ax.scatter([cx], [cy], s=150, c='black', marker='X',
                          edgecolors='white', linewidths=2, zorder=10)

            # Configure subplot
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
            ax.set_aspect('equal')
            ax.axis('off')

            # Truncate long participant IDs
            display_name = pid[:20] + "..." if len(pid) > 20 else pid
            ax.set_title(f"{display_name}\n(n={len(scores)})", fontsize=9, pad=5)

        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved faceted ternary plot to {output_path}")
            return None

        return fig

    def generate_overlaid_ternary(
        self,
        output_path: Optional[Union[str, Path]] = None,
        dimension_names: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        marker_size: int = 80,
        show_centroids: bool = True,
        show_legend: bool = True,
        connect_same_entities: bool = False,
        title: str = "Multi-Participant Entity Comparison",
    ):
        """
        Generate overlaid ternary plot with all participants.

        Shows all participants on a single ternary plot with different
        colors, making it easy to see overall patterns and differences.

        Args:
            output_path: Path to save PNG file (optional).
            dimension_names: List of 3 dimension names.
            participants: Specific participants to include.
            figsize: Figure size.
            marker_size: Size of entity markers.
            show_centroids: Show centroid marker for each participant.
            show_legend: Show participant legend.
            connect_same_entities: Draw lines connecting same entity across participants.
            title: Plot title.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # Validate dimensions
        dims = dimension_names or self.comparison.dimension_names
        if len(dims) != 3:
            raise ValueError(f"Ternary plot requires exactly 3 dimensions, got {len(dims)}")

        # Select participants
        if participants:
            pids = [p for p in participants if p in self.comparison.scores_by_participant]
        else:
            pids = list(self.comparison.scores_by_participant.keys())

        if not pids:
            raise ValueError("No participants to plot")

        # Create figure with legend space
        if show_legend:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])
            legend_ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            legend_ax = None

        # Draw triangle
        self._draw_triangle(ax, dims, show_labels=True, show_grid=True)

        # Track entities for connecting lines
        entity_positions: Dict[str, List[Tuple[float, float, str]]] = {}

        legend_handles = []
        legend_labels = []

        for pid_idx, pid in enumerate(pids):
            color = self.participant_colors[pid_idx % len(self.participant_colors)]

            scores = self.comparison.scores_by_participant[pid]
            normalized = self._normalize_scores(scores)

            # Convert to cartesian
            xs, ys = [], []
            for i, score in enumerate(normalized):
                x, y = self._barycentric_to_cartesian(score[1], score[2], score[0])
                xs.append(x)
                ys.append(y)

                # Track for connecting lines
                if connect_same_entities and i < len(self.comparison.entity_names):
                    entity = self.comparison.entity_names[i]
                    if entity not in entity_positions:
                        entity_positions[entity] = []
                    entity_positions[entity].append((x, y, pid))

            # Plot entities
            scatter = ax.scatter(xs, ys, s=marker_size, c=[color], alpha=0.6,
                               edgecolors='white', linewidths=0.5, label=pid)

            # Plot centroid
            if show_centroids:
                centroid_scores = self._normalize_scores(scores.mean(axis=0))
                cx, cy = self._barycentric_to_cartesian(
                    centroid_scores[1], centroid_scores[2], centroid_scores[0]
                )
                ax.scatter([cx], [cy], s=200, c=[color], marker='X',
                          edgecolors='black', linewidths=2, zorder=10)

            # For legend
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=10))
            legend_labels.append(f"{pid} (n={len(scores)})")

        # Draw connecting lines between same entities
        if connect_same_entities:
            for entity, positions in entity_positions.items():
                if len(positions) > 1:
                    for i in range(len(positions) - 1):
                        x1, y1, _ = positions[i]
                        x2, y2, _ = positions[i + 1]
                        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)

        # Configure plot
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=15)

        # Add legend
        if show_legend and legend_ax:
            legend_ax.legend(legend_handles, legend_labels,
                           loc='center left', fontsize=9,
                           title="Participants", title_fontsize=10,
                           frameon=True, framealpha=0.9)

        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved overlaid ternary plot to {output_path}")
            return None

        return fig

    def generate_distance_heatmap(
        self,
        result: ComparisonResult,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8),
        show_values: bool = True,
        show_dendrogram: bool = True,
        cmap: str = "RdYlBu_r",
        title: Optional[str] = None,
        groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Generate distance matrix heatmap with optional dendrogram or group ordering.

        Args:
            result: ComparisonResult with distance matrix.
            output_path: Path to save PNG file (optional).
            figsize: Figure size.
            show_values: Show distance values in cells.
            show_dendrogram: Show hierarchical clustering dendrogram.
            cmap: Colormap name.
            title: Plot title.
            groups: Optional dict mapping group names to participant IDs for ordering.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt

        n = len(result.participant_ids)

        # If groups are provided, order by group membership instead of dendrogram
        if groups:
            # Build ordered list: all members of group1, then group2, etc.
            ordered_ids = []
            group_boundaries = []  # Store indices where groups change

            for group_name, members in groups.items():
                group_start = len(ordered_ids)
                for pid in members:
                    if pid in result.participant_ids:
                        ordered_ids.append(pid)
                if len(ordered_ids) > group_start:
                    group_boundaries.append((group_start, len(ordered_ids), group_name))

            # Add any participants not in groups at the end
            for pid in result.participant_ids:
                if pid not in ordered_ids:
                    ordered_ids.append(pid)

            # Build index mapping
            order = [result.participant_ids.index(pid) for pid in ordered_ids]
            ordered_matrix = result.distance_matrix[np.ix_(order, order)]

            fig, ax_heatmap = plt.subplots(figsize=figsize)

        elif show_dendrogram and n >= 3:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform

            fig = plt.figure(figsize=figsize)

            # Create grid for dendrogram + heatmap
            gs = fig.add_gridspec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1],
                                 hspace=0.02, wspace=0.02)

            ax_dendro_top = fig.add_subplot(gs[0, 1])
            ax_dendro_left = fig.add_subplot(gs[1, 0])
            ax_heatmap = fig.add_subplot(gs[1, 1])

            # Compute linkage
            condensed = squareform(result.distance_matrix)
            linkage_matrix = linkage(condensed, method="average")

            # Top dendrogram
            dendro = dendrogram(linkage_matrix, ax=ax_dendro_top, orientation='top',
                              no_labels=True, color_threshold=0)
            ax_dendro_top.axis('off')

            # Left dendrogram
            dendrogram(linkage_matrix, ax=ax_dendro_left, orientation='left',
                      no_labels=True, color_threshold=0)
            ax_dendro_left.axis('off')

            # Reorder matrix by dendrogram
            order = dendro['leaves']
            ordered_matrix = result.distance_matrix[np.ix_(order, order)]
            ordered_ids = [result.participant_ids[i] for i in order]
            group_boundaries = []

        else:
            fig, ax_heatmap = plt.subplots(figsize=figsize)
            ordered_matrix = result.distance_matrix
            ordered_ids = result.participant_ids
            order = list(range(n))
            group_boundaries = []

        # Plot heatmap
        im = ax_heatmap.imshow(ordered_matrix, cmap=cmap, aspect='equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label(f'{result.metric.capitalize()} Distance', fontsize=10)

        # Draw group boundaries if groups are defined
        if groups and group_boundaries:
            for start, end, group_name in group_boundaries:
                # Draw rectangle around group block
                rect_width = end - start
                ax_heatmap.axhline(y=start - 0.5, xmin=0, xmax=1, color='black', linewidth=2)
                ax_heatmap.axhline(y=end - 0.5, xmin=0, xmax=1, color='black', linewidth=2)
                ax_heatmap.axvline(x=start - 0.5, ymin=0, ymax=1, color='black', linewidth=2)
                ax_heatmap.axvline(x=end - 0.5, ymin=0, ymax=1, color='black', linewidth=2)

            # Add group labels on the right side
            for start, end, group_name in group_boundaries:
                mid = (start + end) / 2
                ax_heatmap.annotate(
                    group_name,
                    xy=(len(ordered_ids), mid),
                    xytext=(5, 0),
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold',
                    va='center',
                    ha='left',
                )

        # Add labels
        ax_heatmap.set_xticks(range(len(ordered_ids)))
        ax_heatmap.set_yticks(range(len(ordered_ids)))

        # Truncate long labels
        x_labels = [pid[:15] + "..." if len(pid) > 15 else pid for pid in ordered_ids]
        y_labels = [pid[:15] + "..." if len(pid) > 15 else pid for pid in ordered_ids]

        ax_heatmap.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax_heatmap.set_yticklabels(y_labels, fontsize=8)

        # Add values to cells
        if show_values and len(ordered_ids) <= 15:
            for i in range(len(ordered_ids)):
                for j in range(len(ordered_ids)):
                    value = ordered_matrix[i, j]
                    text_color = 'white' if value > ordered_matrix.max() * 0.6 else 'black'
                    ax_heatmap.text(j, i, f'{value:.2f}',
                                   ha='center', va='center',
                                   color=text_color, fontsize=7)

        # Title
        plot_title = title or f"Participant Distance Matrix ({result.metric.capitalize()})"
        if show_dendrogram and n >= 3 and not groups:
            fig.suptitle(plot_title, fontsize=12, fontweight='bold', y=0.98)
        else:
            ax_heatmap.set_title(plot_title, fontsize=12, pad=10)

        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved distance heatmap to {output_path}")
            return None

        return fig

    def generate_forest_plot(
        self,
        result: Optional["ComparisonResult"] = None,
        output_path: Optional[Union[str, Path]] = None,
        dimension_names: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        ci_level: float = 0.95,
        group_by: str = "dimension",
        title: Optional[str] = None,
    ):
        """
        Generate forest plot showing per-dimension scores with confidence intervals.

        Displays each participant's mean score with bootstrapped confidence intervals
        for each dimension, allowing visual comparison of scoring patterns.

        Args:
            result: ComparisonResult (optional, not used but kept for consistency).
            output_path: Path to save PNG file (optional).
            dimension_names: List of dimension names to display.
            participants: Specific participants to include.
            figsize: Figure size.
            ci_level: Confidence interval level (default 0.95 = 95% CI).
            group_by: How to organize plot - "dimension" (default) or "participant".
            title: Plot title.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt

        # Get dimensions
        dims = dimension_names or self.comparison.dimension_names
        n_dims = len(dims)

        # Select participants
        if participants:
            pids = [p for p in participants if p in self.comparison.scores_by_participant]
        else:
            pids = list(self.comparison.scores_by_participant.keys())

        if not pids:
            raise ValueError("No participants to plot")

        n_participants = len(pids)

        # Compute means and CIs for each participant-dimension pair
        # Normalize scores compositionally (sum to 1) for meaningful comparison
        stats = {}  # {(pid, dim_idx): (mean, ci_low, ci_high)}

        for pid in pids:
            raw_scores = self.comparison.scores_by_participant[pid]  # shape: (n_entities, n_dims)
            # Normalize each entity's scores to sum to 1
            scores = self._normalize_scores(raw_scores)
            n_entities = len(scores)

            for dim_idx in range(n_dims):
                dim_scores = scores[:, dim_idx]
                mean_val = np.mean(dim_scores)

                # Bootstrap confidence interval
                if n_entities >= 2:
                    n_bootstrap = 1000
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        sample = np.random.choice(dim_scores, size=n_entities, replace=True)
                        bootstrap_means.append(np.mean(sample))

                    alpha = 1 - ci_level
                    ci_low = np.percentile(bootstrap_means, alpha / 2 * 100)
                    ci_high = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
                else:
                    # Single entity - no CI
                    ci_low = ci_high = mean_val

                # Clamp CIs to valid [0, 1] range for compositional data
                ci_low = max(0.0, min(1.0, ci_low))
                ci_high = max(0.0, min(1.0, ci_high))

                stats[(pid, dim_idx)] = (mean_val, ci_low, ci_high)

        # Create figure
        if group_by == "dimension":
            # One subplot per dimension, participants on y-axis
            fig, axes = plt.subplots(1, n_dims, figsize=figsize, sharey=True)
            if n_dims == 1:
                axes = [axes]

            for dim_idx, ax in enumerate(axes):
                dim_name = dims[dim_idx] if dim_idx < len(dims) else f"Dim {dim_idx}"

                y_positions = list(range(n_participants))
                means = []
                ci_lows = []
                ci_highs = []

                for pid in pids:
                    mean_val, ci_low, ci_high = stats[(pid, dim_idx)]
                    means.append(mean_val)
                    ci_lows.append(ci_low)
                    ci_highs.append(ci_high)

                # Convert to error bar format (ensure non-negative)
                errors = [
                    [max(0, m - cl) for m, cl in zip(means, ci_lows)],
                    [max(0, ch - m) for m, ch in zip(ci_highs, means)]
                ]

                # Color by dimension
                color = self.dimension_colors.get(dim_name.lower(), self.participant_colors[dim_idx % len(self.participant_colors)])

                ax.errorbar(
                    means, y_positions,
                    xerr=errors,
                    fmt='o',
                    color=color,
                    capsize=3,
                    capthick=1.5,
                    markersize=6,
                    linewidth=1.5
                )

                ax.set_xlabel('Score', fontsize=10)
                ax.set_title(dim_name.capitalize(), fontsize=11, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.axvline(x=1/n_dims, color='gray', linestyle='--', alpha=0.5, label='Equal')
                ax.grid(axis='x', alpha=0.3)

                if dim_idx == 0:
                    ax.set_yticks(y_positions)
                    # Truncate long labels
                    y_labels = [pid[:20] + "..." if len(pid) > 20 else pid for pid in pids]
                    ax.set_yticklabels(y_labels, fontsize=9)

        else:
            # group_by == "participant": One subplot per participant, dimensions on y-axis
            n_cols = min(3, n_participants)
            n_rows = (n_participants + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
            axes = np.array(axes).flatten()

            for pid_idx, pid in enumerate(pids):
                ax = axes[pid_idx]

                y_positions = list(range(n_dims))
                means = []
                ci_lows = []
                ci_highs = []

                for dim_idx in range(n_dims):
                    mean_val, ci_low, ci_high = stats[(pid, dim_idx)]
                    means.append(mean_val)
                    ci_lows.append(ci_low)
                    ci_highs.append(ci_high)

                errors = [
                    [max(0, m - cl) for m, cl in zip(means, ci_lows)],
                    [max(0, ch - m) for m, ch in zip(ci_highs, means)]
                ]

                # Color by dimension
                colors = [self.dimension_colors.get(dims[i].lower(), self.participant_colors[i]) for i in range(n_dims)]

                for i, (y, m, err_l, err_h, c) in enumerate(zip(y_positions, means, errors[0], errors[1], colors)):
                    ax.errorbar(
                        m, y,
                        xerr=[[err_l], [err_h]],
                        fmt='o',
                        color=c,
                        capsize=3,
                        capthick=1.5,
                        markersize=6,
                        linewidth=1.5
                    )

                ax.set_xlim(0, 1)
                ax.axvline(x=1/n_dims, color='gray', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                ax.set_title(pid[:25] + "..." if len(pid) > 25 else pid, fontsize=10)

                if pid_idx % n_cols == 0:
                    ax.set_yticks(y_positions)
                    y_labels = [d[:10].capitalize() for d in dims]
                    ax.set_yticklabels(y_labels, fontsize=9)

                if pid_idx >= (n_rows - 1) * n_cols:
                    ax.set_xlabel('Score', fontsize=10)

            # Hide unused subplots
            for idx in range(n_participants, len(axes)):
                axes[idx].axis('off')

        # Overall title
        plot_title = title or f"Dimension Scores by Participant ({int(ci_level*100)}% CI)"
        fig.suptitle(plot_title, fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved forest plot to {output_path}")
            return None

        return fig

    def generate_similarity_map(
        self,
        result: "ComparisonResult",
        output_path: Optional[Union[str, Path]] = None,
        method: str = "mds",
        figsize: Tuple[int, int] = (10, 8),
        show_labels: bool = True,
        title: Optional[str] = None,
        groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Generate 2D similarity map using MDS or UMAP dimensionality reduction.

        Projects participants into 2D space based on their distance matrix,
        so similar participants appear close together.

        Args:
            result: ComparisonResult with distance_matrix.
            output_path: Path to save PNG file (optional).
            method: Dimensionality reduction method - "mds" (default) or "umap".
            figsize: Figure size.
            show_labels: Show participant ID labels.
            title: Plot title.
            groups: Optional dict mapping group names to participant IDs for coloring.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from sklearn.manifold import MDS

        if result is None or result.distance_matrix is None:
            raise ValueError("ComparisonResult with distance_matrix required")

        n = len(result.participant_ids)
        if n < 2:
            raise ValueError("Need at least 2 participants for similarity map")

        # Compute 2D embedding
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    metric="precomputed",
                    n_neighbors=min(15, n - 1),
                    min_dist=0.1,
                    random_state=42
                )
                embedding = reducer.fit_transform(result.distance_matrix)
            except ImportError:
                logger.warning("UMAP not available, falling back to MDS")
                method = "mds"

        if method == "mds":
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                normalized_stress="auto"
            )
            embedding = mds.fit_transform(result.distance_matrix)

        # Build group color mapping if groups provided
        group_colors = {}
        pid_to_group = {}
        if groups:
            group_names = list(groups.keys())
            for g_idx, (group_name, members) in enumerate(groups.items()):
                group_colors[group_name] = self.participant_colors[g_idx % len(self.participant_colors)]
                for pid in members:
                    pid_to_group[pid] = group_name

        # Create figure with optional legend for groups
        if groups:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], figure=fig)
            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])
            legend_ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            legend_ax = None

        # Plot participants
        for i, pid in enumerate(result.participant_ids):
            if groups and pid in pid_to_group:
                group_name = pid_to_group[pid]
                color = group_colors[group_name]
            else:
                color = self.participant_colors[i % len(self.participant_colors)]

            ax.scatter(
                embedding[i, 0], embedding[i, 1],
                s=150,
                c=color,
                alpha=0.7,
                edgecolors='white',
                linewidth=1.5,
                zorder=2
            )

            if show_labels:
                # Truncate long labels
                label = pid[:15] + "..." if len(pid) > 15 else pid
                ax.annotate(
                    label,
                    (embedding[i, 0], embedding[i, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                    zorder=3
                )

        # Draw convex hulls around groups
        if groups:
            try:
                from scipy.spatial import ConvexHull

                for group_name, members in groups.items():
                    # Get indices of members in this group
                    group_indices = [
                        result.participant_ids.index(pid)
                        for pid in members
                        if pid in result.participant_ids
                    ]

                    if len(group_indices) >= 3:
                        points = embedding[group_indices]
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])  # Close hull
                        ax.fill(
                            hull_points[:, 0], hull_points[:, 1],
                            alpha=0.15,
                            color=group_colors[group_name],
                            zorder=0
                        )
                        ax.plot(
                            hull_points[:, 0], hull_points[:, 1],
                            '--',
                            alpha=0.4,
                            color=group_colors[group_name],
                            linewidth=1.5,
                            zorder=1
                        )
            except Exception as e:
                logger.debug(f"Could not draw convex hulls: {e}")

        # Style
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')

        # Add legend for groups
        if groups and legend_ax:
            legend_handles = []
            legend_labels = []
            for group_name, members in groups.items():
                n_members = len([m for m in members if m in result.participant_ids])
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=group_colors[group_name],
                              markersize=12, alpha=0.7)
                )
                legend_labels.append(f"{group_name} (n={n_members})")
            legend_ax.legend(
                legend_handles, legend_labels,
                loc='center left',
                fontsize=10,
                title="Groups",
                title_fontsize=11,
                frameon=True,
                framealpha=0.9
            )
        elif not groups:
            # Add distance annotations for closest/furthest pairs
            if n >= 2:
                most_similar = result.most_similar_pair
                most_different = result.most_different_pair

                if most_similar:
                    idx1 = result.participant_ids.index(most_similar[0])
                    idx2 = result.participant_ids.index(most_similar[1])
                    ax.plot(
                        [embedding[idx1, 0], embedding[idx2, 0]],
                        [embedding[idx1, 1], embedding[idx2, 1]],
                        'g-', alpha=0.5, linewidth=2, zorder=1,
                        label=f'Most similar: {most_similar[2]:.3f}'
                    )

                if most_different:
                    idx1 = result.participant_ids.index(most_different[0])
                    idx2 = result.participant_ids.index(most_different[1])
                    ax.plot(
                        [embedding[idx1, 0], embedding[idx2, 0]],
                        [embedding[idx1, 1], embedding[idx2, 1]],
                        'r--', alpha=0.5, linewidth=2, zorder=1,
                        label=f'Most different: {most_different[2]:.3f}'
                    )

                ax.legend(loc='best', fontsize=9)

        # Title
        aggregate_mode = getattr(result, 'aggregate', 'mean')
        mode_label = "centroid" if aggregate_mode == "mean" else "distribution"
        plot_title = title or f"Participant Similarity Map ({result.metric.capitalize()}, {mode_label}, {method.upper()})"
        ax.set_title(plot_title, fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved similarity map to {output_path}")
            return None

        return fig


def compare_participants(
    scores_path: Union[str, Path],
    metric: str = "aitchison",
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> ComparisonResult:
    """
    Convenience function for quick participant comparison.

    Args:
        scores_path: Path to scored entities CSV.
        metric: Distance metric to use.
        output_dir: Optional output directory for results.
        **kwargs: Additional arguments passed to comparison.

    Returns:
        ComparisonResult with distance matrix and summary.
    """
    comparison = ParticipantComparison()
    comparison.load_scores(scores_path)
    result = comparison.compute_distances(metric=metric, **kwargs)

    if output_dir:
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        with open(output_dir / "comparison_results.json", "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Saved comparison results to {output_dir}")

    return result
