"""
Tests for participant clustering and PCA.

Tests cover:
- Hierarchical clustering
- K-means clustering
- HDBSCAN (optional dependency)
- PCA dimensionality reduction
- Visualization functions
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qualitative_analysis.entity.clustering import (
    ClusteringResult,
    PCAResult,
    cluster_hierarchical,
    cluster_kmeans,
    pca_on_scores,
)


class TestHierarchicalClustering:
    """Tests for hierarchical clustering."""

    @pytest.fixture
    def well_separated_data(self):
        """Create data with two clearly separated clusters."""
        rng = np.random.default_rng(42)

        # Cluster 1: centered at [20, 20, 20]
        cluster1 = rng.normal([20, 20, 20], 5, size=(5, 3))

        # Cluster 2: centered at [80, 80, 80]
        cluster2 = rng.normal([80, 80, 80], 5, size=(5, 3))

        centroids = np.vstack([cluster1, cluster2])
        participant_ids = [f"p{i}" for i in range(10)]

        # Compute Euclidean distance matrix
        from scipy.spatial.distance import squareform, pdist
        distance_matrix = squareform(pdist(centroids, metric="euclidean"))

        return distance_matrix, participant_ids, centroids

    def test_finds_two_clusters(self, well_separated_data):
        """Should correctly identify two well-separated clusters."""
        distance_matrix, participant_ids, _ = well_separated_data

        result = cluster_hierarchical(
            distance_matrix, participant_ids, n_clusters=2
        )

        assert isinstance(result, ClusteringResult)
        assert result.method == "hierarchical"
        assert result.n_clusters == 2
        assert len(result.labels) == 10
        assert set(result.labels) == {0, 1}

        # Check that clusters are separated correctly
        # First 5 should be in one cluster, last 5 in another
        labels_first5 = set(result.labels[:5])
        labels_last5 = set(result.labels[5:])
        assert len(labels_first5) == 1
        assert len(labels_last5) == 1
        assert labels_first5 != labels_last5

    def test_auto_n_clusters(self, well_separated_data):
        """Auto-selection should find reasonable number of clusters."""
        distance_matrix, participant_ids, _ = well_separated_data

        result = cluster_hierarchical(
            distance_matrix, participant_ids, n_clusters=None, max_k=5
        )

        # Should auto-select 2 clusters for well-separated data
        assert result.n_clusters >= 2
        assert result.n_clusters <= 5

    def test_stores_linkage_matrix(self, well_separated_data):
        """Should store linkage matrix in metadata for dendrogram."""
        distance_matrix, participant_ids, _ = well_separated_data

        result = cluster_hierarchical(
            distance_matrix, participant_ids, n_clusters=2
        )

        assert "linkage_matrix" in result.metadata
        linkage = result.metadata["linkage_matrix"]
        assert linkage.shape[0] == len(participant_ids) - 1  # n-1 merges

    def test_quality_metrics(self, well_separated_data):
        """Should compute silhouette score."""
        distance_matrix, participant_ids, _ = well_separated_data

        result = cluster_hierarchical(
            distance_matrix, participant_ids, n_clusters=2
        )

        assert "silhouette" in result.quality_metrics
        # Good separation should give high silhouette
        assert result.quality_metrics["silhouette"] > 0.5


class TestKMeansClustering:
    """Tests for k-means clustering."""

    @pytest.fixture
    def kmeans_data(self):
        """Create data suitable for k-means (not distance matrix)."""
        rng = np.random.default_rng(42)

        # Three clusters
        cluster1 = rng.normal([20, 80], 5, size=(4, 2))
        cluster2 = rng.normal([50, 50], 5, size=(4, 2))
        cluster3 = rng.normal([80, 20], 5, size=(4, 2))

        score_matrix = np.vstack([cluster1, cluster2, cluster3])
        participant_ids = [f"p{i}" for i in range(12)]

        return score_matrix, participant_ids

    def test_finds_known_clusters(self, kmeans_data):
        """Should recover known cluster structure."""
        score_matrix, participant_ids = kmeans_data

        result = cluster_kmeans(
            score_matrix, participant_ids, n_clusters=3, random_seed=42
        )

        assert isinstance(result, ClusteringResult)
        assert result.method == "kmeans"
        assert result.n_clusters == 3
        assert len(set(result.labels)) == 3

    def test_stores_elbow_data(self, kmeans_data):
        """Should store elbow data for visualization."""
        score_matrix, participant_ids = kmeans_data

        result = cluster_kmeans(
            score_matrix, participant_ids, n_clusters=3, random_seed=42
        )

        assert "elbow_data" in result.metadata
        assert "silhouette_data" in result.metadata
        assert len(result.metadata["elbow_data"]) > 0

    def test_auto_n_clusters(self, kmeans_data):
        """Auto-selection using silhouette should work."""
        score_matrix, participant_ids = kmeans_data

        result = cluster_kmeans(
            score_matrix, participant_ids, n_clusters=None, max_k=6, random_seed=42
        )

        # Should pick something reasonable (2-4 for this data)
        assert 2 <= result.n_clusters <= 6

    def test_quality_metrics(self, kmeans_data):
        """Should compute multiple quality metrics."""
        score_matrix, participant_ids = kmeans_data

        result = cluster_kmeans(
            score_matrix, participant_ids, n_clusters=3, random_seed=42
        )

        assert "silhouette" in result.quality_metrics
        assert "calinski_harabasz" in result.quality_metrics
        assert "inertia" in result.quality_metrics


class TestHDBSCAN:
    """Tests for HDBSCAN clustering (optional dependency)."""

    def test_optional_dependency(self):
        """Should handle missing hdbscan gracefully."""
        # Import the function
        from qualitative_analysis.entity.clustering import cluster_hdbscan

        # If hdbscan is installed, it should work
        # If not, it should raise ImportError with helpful message
        rng = np.random.default_rng(42)
        data = rng.random((10, 10))
        pids = [f"p{i}" for i in range(10)]

        try:
            result = cluster_hdbscan(data, pids, min_cluster_size=3)
            assert isinstance(result, ClusteringResult)
        except ImportError as e:
            assert "hdbscan" in str(e).lower()


class TestPCA:
    """Tests for PCA dimensionality reduction."""

    @pytest.fixture
    def mock_comparison(self):
        """Create a mock ParticipantComparison-like object."""

        class MockComparison:
            def __init__(self):
                self.scores_by_participant = {
                    "p1": np.array([[10, 20, 30], [15, 25, 35]]),
                    "p2": np.array([[40, 50, 60], [45, 55, 65]]),
                    "p3": np.array([[70, 80, 90], [75, 85, 95]]),
                    "p4": np.array([[20, 30, 40], [25, 35, 45]]),
                }
                self.dimension_names = ["social", "eco", "tech"]

        return MockComparison()

    def test_pca_output_structure(self, mock_comparison):
        """PCA result should have correct structure."""
        result = pca_on_scores(mock_comparison, n_components=2)

        assert isinstance(result, PCAResult)
        assert result.components.shape == (2, 3)  # 2 components, 3 dims
        assert len(result.explained_variance_ratio) == 2
        assert len(result.cumulative_variance) == 2
        assert result.transformed_scores.shape == (4, 2)  # 4 participants, 2 PCs
        assert len(result.participant_ids) == 4

    def test_explained_variance_sums_correctly(self, mock_comparison):
        """Cumulative variance should be correct."""
        result = pca_on_scores(mock_comparison, n_components=3)

        # Cumulative should sum correctly
        expected_cumulative = np.cumsum(result.explained_variance_ratio)
        np.testing.assert_array_almost_equal(
            result.cumulative_variance, expected_cumulative
        )

        # Total variance should be <= 1
        assert result.cumulative_variance[-1] <= 1.0 + 1e-10

    def test_clr_flag_applies_transform(self, mock_comparison):
        """CLR flag should apply the transform."""
        result_no_clr = pca_on_scores(mock_comparison, use_clr=False)
        result_clr = pca_on_scores(mock_comparison, use_clr=True)

        assert result_no_clr.used_clr is False
        assert result_clr.used_clr is True

        # Results should differ
        assert not np.allclose(
            result_no_clr.transformed_scores, result_clr.transformed_scores
        )

    def test_to_dict_serialization(self, mock_comparison):
        """PCAResult.to_dict() should be JSON-serializable."""
        import json

        result = pca_on_scores(mock_comparison, n_components=2)
        result_dict = result.to_dict()

        # Should not raise
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0

        # Check expected keys
        assert "n_components" in result_dict
        assert "explained_variance_ratio" in result_dict
        assert "loadings" in result_dict


class TestClusteringResultSerialization:
    """Tests for ClusteringResult serialization."""

    def test_to_dict(self):
        """ClusteringResult.to_dict() should produce valid dict."""
        result = ClusteringResult(
            method="hierarchical",
            labels=np.array([0, 0, 1, 1]),
            participant_ids=["p1", "p2", "p3", "p4"],
            n_clusters=2,
            quality_metrics={"silhouette": 0.75},
            metadata={"linkage_matrix": np.array([[1, 2, 0.5, 2]])},
        )

        d = result.to_dict()

        assert d["method"] == "hierarchical"
        assert d["labels"] == [0, 0, 1, 1]
        assert d["n_clusters"] == 2
        assert d["quality_metrics"]["silhouette"] == 0.75


class TestVisualizationFunctions:
    """Tests for visualization functions."""

    @pytest.fixture
    def hierarchical_result(self):
        """Create a hierarchical clustering result with linkage matrix."""
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import pdist

        rng = np.random.default_rng(42)
        data = rng.random((6, 3))
        linkage_matrix = linkage(pdist(data), method="ward")

        return ClusteringResult(
            method="hierarchical",
            labels=np.array([0, 0, 0, 1, 1, 1]),
            participant_ids=[f"p{i}" for i in range(6)],
            n_clusters=2,
            quality_metrics={"silhouette": 0.6},
            metadata={"linkage_matrix": linkage_matrix, "linkage_method": "ward"},
        )

    @pytest.fixture
    def kmeans_result(self):
        """Create a k-means result with elbow data."""
        return ClusteringResult(
            method="kmeans",
            labels=np.array([0, 0, 1, 1, 2, 2]),
            participant_ids=[f"p{i}" for i in range(6)],
            n_clusters=3,
            quality_metrics={"silhouette": 0.5, "inertia": 100.0},
            metadata={
                "elbow_data": [
                    {"k": 2, "inertia": 200.0},
                    {"k": 3, "inertia": 100.0},
                    {"k": 4, "inertia": 80.0},
                ],
                "silhouette_data": [
                    {"k": 2, "silhouette": 0.4},
                    {"k": 3, "silhouette": 0.5},
                    {"k": 4, "silhouette": 0.45},
                ],
            },
        )

    def test_plot_dendrogram_creates_file(self, hierarchical_result):
        """plot_dendrogram should create output file."""
        from qualitative_analysis.entity.clustering import plot_dendrogram

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dendrogram.png"
            plot_dendrogram(hierarchical_result, output_path)
            assert output_path.exists()

    def test_plot_elbow_creates_file(self, kmeans_result):
        """plot_elbow should create output file."""
        from qualitative_analysis.entity.clustering import plot_elbow

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "elbow.png"
            plot_elbow(kmeans_result, output_path)
            assert output_path.exists()

    def test_plot_dendrogram_rejects_non_hierarchical(self, kmeans_result):
        """plot_dendrogram should reject non-hierarchical results."""
        from qualitative_analysis.entity.clustering import plot_dendrogram

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dendrogram.png"
            with pytest.raises(ValueError, match="hierarchical"):
                plot_dendrogram(kmeans_result, output_path)
