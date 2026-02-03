"""
Multi-participant comparison module for entity scores.

Provides statistical comparison utilities for analyzing how different
participants score entities across dimensions.

Supports:
- Distance metrics: Euclidean (default), cosine, Aitchison (compositional), EMD/Wasserstein
- Pairwise participant similarity matrices
- Group-level comparisons with permutation testing

Note on metric choice:
    The default metric is **euclidean**, which treats each dimension as an
    independent axis. Use ``aitchison`` only when scores are truly compositional
    (parts of a whole that sum to a constant). For independent dimension scores
    (e.g., Social/Ecological/Technological where an entity can score 80/80/80),
    Aitchison distance is inappropriate because it normalizes vectors to sum to 1,
    destroying absolute score-level information.

Distance functions are defined in ``distances.py`` and re-exported here for
backward compatibility. Visualization is in ``comparison_viz.py``.

Example usage:
    from qualitative_analysis.entity.comparison import (
        ParticipantComparison,
        compute_pairwise_distances,
    )

    # Compute all pairwise distances (Euclidean by default)
    distance_matrix = compute_pairwise_distances(
        participant_scores,
        metric="euclidean"
    )

    # Full comparison analysis
    comparison = ParticipantComparison(scored_entities_df)
    results = comparison.analyze(method="pairwise", metric="euclidean")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Re-export distance functions for backward compatibility.
# Code that does ``from qualitative_analysis.entity.comparison import aitchison_distance``
# will continue to work.
from qualitative_analysis.entity.distances import (  # noqa: F401
    clr_transform,
    ilr_transform,
    aitchison_distance,
    aitchison_distance_matrix,
    wasserstein_distance_1d,
    wasserstein_distance_compositional,
    cosine_similarity,
    cosine_distance,
    compute_pairwise_distances,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclasses
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


# =============================================================================
# Participant Comparison Class
# =============================================================================


class ParticipantComparison:
    """
    Analyze and compare entity scores across multiple participants.

    This class provides methods for computing distances between participants,
    identifying clusters of similar participants, and statistical comparisons.

    Example:
        comparison = ParticipantComparison()
        comparison.load_scores(scores_csv_path)
        results = comparison.compute_distances(metric="euclidean")
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
        metric: str = "euclidean",
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
        metric: str = "euclidean",
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

    def run_permutation_test(
        self,
        metric: str = "euclidean",
        aggregate: str = "mean",
        n_permutations: int = 1000,
        random_seed: Optional[int] = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run permutation test for group comparison significance.

        Tests the null hypothesis that group labels are exchangeable (i.e., no
        true difference between groups). Shuffles group labels and recomputes
        the effect size (between/within ratio) to build a null distribution.

        Args:
            metric: Distance metric ("aitchison", "emd", "cosine", "euclidean").
            aggregate: Aggregation mode ("mean" or "distribution").
            n_permutations: Number of permutations (default 1000).
            random_seed: Random seed for reproducibility (default 42).
            **kwargs: Additional arguments for distance function.

        Returns:
            Dictionary with:
                - observed_effect_size: The actual effect size from the data
                - p_value: Proportion of permutations with effect size >= observed
                - n_permutations: Number of permutations run
                - null_distribution: List of effect sizes from permutations

        Raises:
            ValueError: If groups have not been set via set_groups().
        """
        if not hasattr(self, 'groups') or not self.groups:
            raise ValueError("No groups defined. Call set_groups() first.")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Get all participants in groups
        all_pids = []
        group_sizes = []
        for group_name, members in self.groups.items():
            all_pids.extend(members)
            group_sizes.append(len(members))

        n_participants = len(all_pids)

        # Normalize aggregate parameter
        if aggregate in ("distribution", "none"):
            aggregate_mode = "distribution"
        else:
            aggregate_mode = "mean"

        # Prepare scores for distance computation
        if aggregate_mode == "mean":
            scores = {
                pid: self.scores_by_participant[pid].mean(axis=0)
                for pid in all_pids
            }
        else:
            scores = {
                pid: self.scores_by_participant[pid]
                for pid in all_pids
            }

        # Compute full pairwise distance matrix (once)
        dist_matrix, pid_order = compute_pairwise_distances(scores, metric=metric, **kwargs)
        pid_to_idx = {pid: i for i, pid in enumerate(pid_order)}

        def compute_effect_size_from_labels(group_labels: Dict[str, List[str]]) -> Optional[float]:
            """Compute effect size given group label assignments."""
            # Within-group distances
            all_within = []
            for group_name, members in group_labels.items():
                if len(members) < 2:
                    continue
                within_dists = []
                for i, pid_i in enumerate(members):
                    for pid_j in members[i + 1:]:
                        idx_i = pid_to_idx[pid_i]
                        idx_j = pid_to_idx[pid_j]
                        within_dists.append(dist_matrix[idx_i, idx_j])
                if within_dists:
                    all_within.append(np.mean(within_dists))

            # Between-group distances
            all_between = []
            group_names = list(group_labels.keys())
            for i, group_i in enumerate(group_names):
                for group_j in group_names[i + 1:]:
                    members_i = group_labels[group_i]
                    members_j = group_labels[group_j]
                    between_dists = []
                    for pid_i in members_i:
                        for pid_j in members_j:
                            idx_i = pid_to_idx[pid_i]
                            idx_j = pid_to_idx[pid_j]
                            between_dists.append(dist_matrix[idx_i, idx_j])
                    if between_dists:
                        all_between.append(np.mean(between_dists))

            # Compute effect size
            if all_within and all_between:
                mean_within = np.mean(all_within)
                mean_between = np.mean(all_between)
                if mean_within > 0:
                    return mean_between / mean_within
            return None

        # Compute observed effect size
        observed_effect_size = compute_effect_size_from_labels(self.groups)

        if observed_effect_size is None:
            logger.warning("Could not compute observed effect size")
            return {
                "observed_effect_size": None,
                "p_value": None,
                "n_permutations": n_permutations,
                "null_distribution": [],
            }

        # Run permutations
        null_distribution = []
        group_names = list(self.groups.keys())

        for _ in range(n_permutations):
            # Shuffle participant IDs
            shuffled_pids = np.random.permutation(all_pids).tolist()

            # Assign to groups based on original group sizes
            permuted_groups = {}
            start_idx = 0
            for i, group_name in enumerate(group_names):
                end_idx = start_idx + group_sizes[i]
                permuted_groups[group_name] = shuffled_pids[start_idx:end_idx]
                start_idx = end_idx

            # Compute effect size for this permutation
            perm_effect_size = compute_effect_size_from_labels(permuted_groups)
            if perm_effect_size is not None:
                null_distribution.append(perm_effect_size)

        # Compute p-value (proportion of permutations >= observed)
        null_distribution = np.array(null_distribution)
        p_value = float(np.mean(null_distribution >= observed_effect_size))

        logger.info(
            f"Permutation test: observed effect size = {observed_effect_size:.3f}, "
            f"p-value = {p_value:.4f} ({n_permutations} permutations)"
        )

        return {
            "observed_effect_size": float(observed_effect_size),
            "p_value": p_value,
            "n_permutations": n_permutations,
            "null_distribution": null_distribution.tolist(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def compare_participants(
    scores_path: Union[str, Path],
    metric: str = "euclidean",
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
