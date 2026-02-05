"""
Participant clustering and PCA for entity score analysis.

This module provides methods for discovering natural groupings among participants
based on their entity scoring patterns, as well as dimensionality reduction via PCA.

Clustering methods:
- **Hierarchical clustering**: Ward's method with dendrogram visualization
- **K-means**: With elbow method and silhouette analysis for optimal k
- **HDBSCAN**: Density-based clustering (optional dependency)

PCA:
- Reduce dimensionality of participant score vectors
- Optional CLR transform for compositional data
- Biplot and scree plot visualizations

Example::

    from qualitative_analysis.entity.comparison import ParticipantComparison
    from qualitative_analysis.entity.clustering import cluster_participants, pca_on_scores

    comparison = ParticipantComparison()
    comparison.load_scores("scored_entities.csv", dimensions)

    # Cluster participants
    result = cluster_participants(comparison, method="hierarchical", n_clusters=3)

    # PCA
    pca_result = pca_on_scores(comparison, n_components=2)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from qualitative_analysis.entity.comparison import ParticipantComparison

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Result of participant clustering."""

    method: str  # "hierarchical", "kmeans", "hdbscan"
    labels: np.ndarray  # Cluster assignments (0-indexed, -1 for noise in HDBSCAN)
    participant_ids: List[str]
    n_clusters: int
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "labels": self.labels.tolist(),
            "participant_ids": self.participant_ids,
            "n_clusters": self.n_clusters,
            "quality_metrics": {k: round(v, 4) for k, v in self.quality_metrics.items()},
        }


@dataclass
class PCAResult:
    """Result of PCA on participant scores."""

    components: np.ndarray  # (n_components, n_dims) - loadings
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    transformed_scores: np.ndarray  # (n_participants, n_components)
    participant_ids: List[str]
    dimension_names: List[str]
    used_clr: bool

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_components": len(self.explained_variance_ratio),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cumulative_variance": self.cumulative_variance.tolist(),
            "participant_ids": self.participant_ids,
            "dimension_names": self.dimension_names,
            "used_clr": self.used_clr,
            "loadings": {
                f"PC{i+1}": {
                    dim: round(self.components[i, j], 4)
                    for j, dim in enumerate(self.dimension_names)
                }
                for i in range(len(self.explained_variance_ratio))
            },
        }


def _get_participant_centroids(comparison: "ParticipantComparison") -> np.ndarray:
    """
    Extract centroid matrix from ParticipantComparison.

    Returns:
        Matrix of shape (n_participants, n_dims) with mean scores.
    """
    participant_ids = list(comparison.scores_by_participant.keys())
    centroids = np.array([
        comparison.scores_by_participant[pid].mean(axis=0)
        for pid in participant_ids
    ])
    return centroids, participant_ids


def cluster_hierarchical(
    distance_matrix: np.ndarray,
    participant_ids: List[str],
    n_clusters: Optional[int] = None,
    linkage_method: str = "ward",
    max_k: int = 10,
) -> ClusteringResult:
    """
    Hierarchical clustering using scipy.

    Args:
        distance_matrix: Square distance matrix (n_participants x n_participants).
        participant_ids: List of participant IDs corresponding to matrix rows.
        n_clusters: Number of clusters. If None, auto-select using silhouette scores.
        linkage_method: Linkage method ("ward", "average", "complete", "single").
        max_k: Maximum k to consider for auto-selection.

    Returns:
        ClusteringResult with cluster labels and linkage matrix in metadata.
    """
    n = len(participant_ids)

    # Convert to condensed form for scipy
    condensed = squareform(distance_matrix)

    # Compute linkage
    if linkage_method == "ward":
        # Ward requires Euclidean distances; use sqrt of squared distances
        # For other methods, use the distance matrix directly
        linkage_matrix = linkage(condensed, method="ward")
    else:
        linkage_matrix = linkage(condensed, method=linkage_method)

    # Auto-select k using silhouette scores if not provided
    if n_clusters is None:
        best_k = 2
        best_silhouette = -1

        for k in range(2, min(max_k + 1, n)):
            labels = fcluster(linkage_matrix, k, criterion="maxclust") - 1
            if len(set(labels)) < 2:
                continue
            try:
                sil = silhouette_score(distance_matrix, labels, metric="precomputed")
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_k = k
            except ValueError:
                continue

        n_clusters = best_k
        logger.info(f"Auto-selected k={n_clusters} (silhouette={best_silhouette:.3f})")

    labels = fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1

    # Quality metrics
    quality_metrics = {}
    if len(set(labels)) >= 2:
        try:
            quality_metrics["silhouette"] = silhouette_score(
                distance_matrix, labels, metric="precomputed"
            )
        except ValueError:
            pass

    return ClusteringResult(
        method="hierarchical",
        labels=labels,
        participant_ids=participant_ids,
        n_clusters=n_clusters,
        quality_metrics=quality_metrics,
        metadata={
            "linkage_matrix": linkage_matrix,
            "linkage_method": linkage_method,
        },
    )


def cluster_kmeans(
    score_matrix: np.ndarray,
    participant_ids: List[str],
    n_clusters: Optional[int] = None,
    max_k: int = 10,
    random_seed: int = 42,
) -> ClusteringResult:
    """
    K-means clustering using sklearn.

    Args:
        score_matrix: Matrix of shape (n_participants, n_dims) with scores.
        participant_ids: List of participant IDs.
        n_clusters: Number of clusters. If None, use elbow + silhouette analysis.
        max_k: Maximum k to consider for auto-selection.
        random_seed: Random seed for reproducibility.

    Returns:
        ClusteringResult with cluster labels and elbow data in metadata.
    """
    n = len(participant_ids)

    # Compute elbow data for all k
    inertias = []
    silhouettes = []

    for k in range(2, min(max_k + 1, n)):
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = kmeans.fit_predict(score_matrix)
        inertias.append({"k": k, "inertia": kmeans.inertia_})

        if len(set(labels)) >= 2:
            sil = silhouette_score(score_matrix, labels)
            silhouettes.append({"k": k, "silhouette": sil})

    # Auto-select k using silhouette score if not provided
    if n_clusters is None and silhouettes:
        best = max(silhouettes, key=lambda x: x["silhouette"])
        n_clusters = best["k"]
        logger.info(
            f"Auto-selected k={n_clusters} (silhouette={best['silhouette']:.3f})"
        )
    elif n_clusters is None:
        n_clusters = 2

    # Final fit
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(score_matrix)

    # Quality metrics
    quality_metrics = {}
    if len(set(labels)) >= 2:
        quality_metrics["silhouette"] = silhouette_score(score_matrix, labels)
        quality_metrics["calinski_harabasz"] = calinski_harabasz_score(
            score_matrix, labels
        )
    quality_metrics["inertia"] = kmeans.inertia_

    return ClusteringResult(
        method="kmeans",
        labels=labels,
        participant_ids=participant_ids,
        n_clusters=n_clusters,
        quality_metrics=quality_metrics,
        metadata={
            "elbow_data": inertias,
            "silhouette_data": silhouettes,
            "centroids": kmeans.cluster_centers_.tolist(),
        },
    )


def cluster_hdbscan(
    distance_matrix: np.ndarray,
    participant_ids: List[str],
    min_cluster_size: int = 3,
    min_samples: Optional[int] = None,
    **kwargs,
) -> ClusteringResult:
    """
    HDBSCAN clustering (optional dependency).

    Args:
        distance_matrix: Square distance matrix.
        participant_ids: List of participant IDs.
        min_cluster_size: Minimum cluster size.
        min_samples: Minimum samples for core point (default: min_cluster_size).
        **kwargs: Additional arguments passed to HDBSCAN.

    Returns:
        ClusteringResult with cluster labels and probabilities in metadata.

    Raises:
        ImportError: If hdbscan is not installed.
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError(
            "hdbscan is not installed. Install with: pip install hdbscan "
            "or install the package with [viz] extras."
        )

    if min_samples is None:
        min_samples = min_cluster_size

    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        **kwargs,
    )
    labels = clusterer.fit_predict(distance_matrix)

    # Number of clusters (excluding noise label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Quality metrics
    quality_metrics = {}
    if n_clusters >= 2:
        # Filter out noise for silhouette
        non_noise_mask = labels != -1
        if non_noise_mask.sum() > 1 and len(set(labels[non_noise_mask])) >= 2:
            quality_metrics["silhouette"] = silhouette_score(
                distance_matrix[non_noise_mask][:, non_noise_mask],
                labels[non_noise_mask],
                metric="precomputed",
            )

    quality_metrics["n_noise"] = int((labels == -1).sum())

    return ClusteringResult(
        method="hdbscan",
        labels=labels,
        participant_ids=participant_ids,
        n_clusters=n_clusters,
        quality_metrics=quality_metrics,
        metadata={
            "probabilities": clusterer.probabilities_.tolist(),
            "min_cluster_size": min_cluster_size,
        },
    )


def cluster_participants(
    comparison: "ParticipantComparison",
    method: str = "hierarchical",
    metric: str = "euclidean",
    n_clusters: Optional[int] = None,
    **kwargs,
) -> ClusteringResult:
    """
    Cluster participants from a ParticipantComparison instance.

    This is a convenience wrapper that extracts participant data from the
    comparison object and dispatches to the appropriate clustering method.

    Args:
        comparison: ParticipantComparison instance with loaded scores.
        method: Clustering method ("hierarchical", "kmeans", "hdbscan").
        metric: Distance metric for computing participant distances.
        n_clusters: Number of clusters (auto-selected if None).
        **kwargs: Additional arguments passed to the clustering function.

    Returns:
        ClusteringResult with cluster assignments.
    """
    centroids, participant_ids = _get_participant_centroids(comparison)

    if method == "kmeans":
        return cluster_kmeans(
            centroids, participant_ids, n_clusters=n_clusters, **kwargs
        )

    # For hierarchical and hdbscan, compute distance matrix first
    if metric == "euclidean":
        distance_matrix = squareform(pdist(centroids, metric="euclidean"))
    elif metric == "cosine":
        distance_matrix = squareform(pdist(centroids, metric="cosine"))
    else:
        # Use the comparison's compute_distances method for other metrics
        from qualitative_analysis.entity.distances import compute_pairwise_distances

        distance_matrix = compute_pairwise_distances(
            centroids, metric=metric, aggregate=None
        )

    if method == "hierarchical":
        return cluster_hierarchical(
            distance_matrix, participant_ids, n_clusters=n_clusters, **kwargs
        )
    elif method == "hdbscan":
        return cluster_hdbscan(distance_matrix, participant_ids, **kwargs)
    else:
        raise ValueError(f"Unknown clustering method: {method}")


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------


def pca_on_scores(
    comparison: "ParticipantComparison",
    n_components: Optional[int] = None,
    use_clr: bool = False,
    standardize: bool = True,
) -> PCAResult:
    """
    PCA on participant mean score vectors.

    Args:
        comparison: ParticipantComparison instance with loaded scores.
        n_components: Number of components to retain. If None, keep all.
        use_clr: Apply CLR (centered log-ratio) transform before PCA.
            Only appropriate for compositional data where scores sum to a constant.
            WARNING: Entity scores are typically NOT compositional.
        standardize: Z-score standardize features before PCA.

    Returns:
        PCAResult with loadings, scores, and explained variance.
    """
    centroids, participant_ids = _get_participant_centroids(comparison)
    dimension_names = comparison.dimension_names

    if n_components is None:
        n_components = centroids.shape[1]

    # Optional CLR transform
    if use_clr:
        logger.warning(
            "CLR transform is being applied. This is only appropriate for "
            "compositional data where scores represent parts of a whole. "
            "Entity dimension scores are typically independent, not compositional."
        )
        from qualitative_analysis.entity.distances import clr_transform

        centroids = np.array([clr_transform(row) for row in centroids])

    # Optional standardization
    if standardize:
        scaler = StandardScaler()
        centroids = scaler.fit_transform(centroids)

    # Fit PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(centroids)

    return PCAResult(
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
        transformed_scores=transformed,
        participant_ids=participant_ids,
        dimension_names=dimension_names,
        used_clr=use_clr,
    )


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------


def plot_dendrogram(
    result: ClusteringResult,
    output_path: Path,
    figsize: Tuple[float, float] = (12, 6),
    truncate_mode: Optional[str] = None,
    p: int = 30,
) -> None:
    """
    Generate dendrogram from hierarchical clustering result.

    Args:
        result: ClusteringResult from hierarchical clustering.
        output_path: Path to save the figure.
        figsize: Figure dimensions.
        truncate_mode: Truncation mode for large trees ("level", "lastp", None).
        p: Parameter for truncation.
    """
    import matplotlib.pyplot as plt

    if result.method != "hierarchical":
        raise ValueError("Dendrogram requires hierarchical clustering result")

    linkage_matrix = result.metadata.get("linkage_matrix")
    if linkage_matrix is None:
        raise ValueError("Linkage matrix not found in result metadata")

    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=result.participant_ids,
        ax=ax,
        truncate_mode=truncate_mode,
        p=p,
        leaf_rotation=90,
        leaf_font_size=8,
    )

    ax.set_xlabel("Participant")
    ax.set_ylabel("Distance")
    ax.set_title(f"Hierarchical Clustering Dendrogram (k={result.n_clusters})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved dendrogram to {output_path}")


def plot_elbow(
    result: ClusteringResult,
    output_path: Path,
    figsize: Tuple[float, float] = (10, 4),
) -> None:
    """
    Plot elbow curve and silhouette scores from k-means result.

    Args:
        result: ClusteringResult from k-means clustering.
        output_path: Path to save the figure.
        figsize: Figure dimensions.
    """
    import matplotlib.pyplot as plt

    if result.method != "kmeans":
        raise ValueError("Elbow plot requires k-means clustering result")

    elbow_data = result.metadata.get("elbow_data", [])
    silhouette_data = result.metadata.get("silhouette_data", [])

    if not elbow_data:
        raise ValueError("Elbow data not found in result metadata")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Elbow plot
    ks = [d["k"] for d in elbow_data]
    inertias = [d["inertia"] for d in elbow_data]

    axes[0].plot(ks, inertias, "b-o", markersize=6)
    axes[0].axvline(result.n_clusters, color="r", linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    # Silhouette plot
    if silhouette_data:
        ks_sil = [d["k"] for d in silhouette_data]
        sils = [d["silhouette"] for d in silhouette_data]

        axes[1].plot(ks_sil, sils, "g-o", markersize=6)
        axes[1].axvline(result.n_clusters, color="r", linestyle="--", alpha=0.7)
        axes[1].set_xlabel("Number of Clusters (k)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Analysis")
    else:
        axes[1].text(
            0.5, 0.5, "No silhouette data", ha="center", va="center", fontsize=12
        )
        axes[1].set_title("Silhouette Analysis")

    plt.suptitle(f"K-Means Cluster Selection (selected k={result.n_clusters})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved elbow plot to {output_path}")


def plot_cluster_scatter(
    result: ClusteringResult,
    score_matrix: np.ndarray,
    output_path: Path,
    dimension_names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    2D scatter plot of clustered participants using PCA projection.

    Args:
        result: ClusteringResult with cluster assignments.
        score_matrix: Matrix of shape (n_participants, n_dims).
        output_path: Path to save the figure.
        dimension_names: Names of dimensions for axis labels.
        figsize: Figure dimensions.
    """
    import matplotlib.pyplot as plt

    # Project to 2D using PCA
    if score_matrix.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(StandardScaler().fit_transform(score_matrix))
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    else:
        coords = score_matrix
        xlabel = dimension_names[0] if dimension_names else "Dim 1"
        ylabel = dimension_names[1] if dimension_names and len(dimension_names) > 1 else "Dim 2"

    fig, ax = plt.subplots(figsize=figsize)

    # Color by cluster
    unique_labels = sorted(set(result.labels))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_labels))))

    for i, label in enumerate(unique_labels):
        mask = result.labels == label
        label_name = f"Cluster {label}" if label >= 0 else "Noise"
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i % len(colors)]],
            label=label_name,
            alpha=0.7,
            s=80,
        )

        # Label points
        for j, pid in enumerate(result.participant_ids):
            if mask[j]:
                ax.annotate(
                    pid,
                    (coords[j, 0], coords[j, 1]),
                    fontsize=8,
                    alpha=0.7,
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Participant Clusters ({result.method}, k={result.n_clusters})")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved cluster scatter to {output_path}")


def plot_biplot(
    result: PCAResult,
    output_path: Path,
    pc_x: int = 0,
    pc_y: int = 1,
    groups: Optional[Dict[str, List[str]]] = None,
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """
    PCA biplot showing participant scores and dimension loadings.

    Args:
        result: PCAResult from pca_on_scores().
        output_path: Path to save the figure.
        pc_x: Index of PC for x-axis (default 0 = PC1).
        pc_y: Index of PC for y-axis (default 1 = PC2).
        groups: Optional dict mapping group name to list of participant IDs.
        figsize: Figure dimensions.
    """
    import matplotlib.pyplot as plt

    scores = result.transformed_scores
    loadings = result.components
    var_explained = result.explained_variance_ratio

    fig, ax = plt.subplots(figsize=figsize)

    # Plot participant scores
    if groups:
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for i, (group_name, members) in enumerate(groups.items()):
            mask = [pid in members for pid in result.participant_ids]
            ax.scatter(
                scores[mask, pc_x],
                scores[mask, pc_y],
                c=[colors[i]],
                label=group_name,
                alpha=0.7,
                s=80,
            )
    else:
        ax.scatter(
            scores[:, pc_x], scores[:, pc_y], alpha=0.7, s=80, c="steelblue"
        )

    # Label points
    for i, pid in enumerate(result.participant_ids):
        ax.annotate(pid, (scores[i, pc_x], scores[i, pc_y]), fontsize=8, alpha=0.7)

    # Plot loading vectors
    # Scale loadings for visibility
    scale = max(np.abs(scores[:, [pc_x, pc_y]]).max() * 0.8 / np.abs(loadings[[pc_x, pc_y], :]).max(), 1)

    for j, dim in enumerate(result.dimension_names):
        ax.arrow(
            0, 0,
            loadings[pc_x, j] * scale,
            loadings[pc_y, j] * scale,
            head_width=0.05 * scale,
            head_length=0.03 * scale,
            fc="red",
            ec="red",
            alpha=0.8,
        )
        ax.text(
            loadings[pc_x, j] * scale * 1.1,
            loadings[pc_y, j] * scale * 1.1,
            dim,
            color="red",
            fontsize=10,
            fontweight="bold",
        )

    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    ax.set_xlabel(f"PC{pc_x + 1} ({var_explained[pc_x]*100:.1f}% variance)")
    ax.set_ylabel(f"PC{pc_y + 1} ({var_explained[pc_y]*100:.1f}% variance)")
    ax.set_title("PCA Biplot" + (" (CLR-transformed)" if result.used_clr else ""))

    if groups:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved biplot to {output_path}")


def plot_scree(
    result: PCAResult,
    output_path: Path,
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """
    Scree plot of explained variance per component.

    Args:
        result: PCAResult from pca_on_scores().
        output_path: Path to save the figure.
        figsize: Figure dimensions.
    """
    import matplotlib.pyplot as plt

    n_components = len(result.explained_variance_ratio)
    components = np.arange(1, n_components + 1)

    fig, ax = plt.subplots(figsize=figsize)

    # Bar chart of individual variance
    ax.bar(
        components,
        result.explained_variance_ratio * 100,
        alpha=0.7,
        label="Individual",
    )

    # Line plot of cumulative variance
    ax.plot(
        components,
        result.cumulative_variance * 100,
        "r-o",
        markersize=6,
        label="Cumulative",
    )

    ax.axhline(80, color="gray", linestyle="--", alpha=0.5)
    ax.text(n_components, 81, "80%", va="bottom", ha="right", fontsize=9, color="gray")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA Scree Plot")
    ax.set_xticks(components)
    ax.legend(loc="center right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved scree plot to {output_path}")
