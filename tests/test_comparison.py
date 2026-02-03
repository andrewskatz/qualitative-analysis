"""
Tests for comparison module: distance metrics, group comparison, and permutation tests.

Covers:
- H6: Distance metric correctness with hand-computed expected values
- H7: Permutation test and group comparison logic
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from qualitative_analysis.entity.comparison import (
    ParticipantComparison,
    clr_transform,
    ilr_transform,
    aitchison_distance,
    aitchison_distance_matrix,
    cosine_similarity,
    cosine_distance,
    compute_pairwise_distances,
    wasserstein_distance_compositional,
)


# =============================================================================
# CLR Transform Tests
# =============================================================================


class TestCLRTransform:
    """Test centered log-ratio transform with known expected values."""

    def test_uniform_vector_returns_zeros(self):
        """CLR of a uniform vector should be all zeros (equal proportions)."""
        x = np.array([1.0, 1.0, 1.0])
        result = clr_transform(x)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_uniform_vector_any_scale(self):
        """CLR of [k, k, k] should be zeros for any positive k (scale invariant)."""
        for k in [0.5, 10.0, 100.0, 1000.0]:
            x = np.array([k, k, k])
            result = clr_transform(x)
            np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_known_2d_vector(self):
        """CLR of [3, 1] should yield known values."""
        x = np.array([3.0, 1.0])
        # Normalize: [0.75, 0.25]
        # log: [ln(0.75), ln(0.25)]
        # geometric mean of proportions: exp(mean(log(props))) = sqrt(0.75 * 0.25) = sqrt(0.1875)
        # CLR = log(prop) - mean(log(prop))
        log_props = np.log(np.array([0.75, 0.25]))
        expected = log_props - np.mean(log_props)
        result = clr_transform(x)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_clr_sums_to_zero(self):
        """CLR-transformed values should sum to zero."""
        x = np.array([50.0, 30.0, 20.0])
        result = clr_transform(x)
        assert abs(np.sum(result)) < 1e-10

    def test_batch_2d_input(self):
        """CLR should work on batched (n_samples, n_components) arrays."""
        X = np.array([
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
        ])
        result = clr_transform(X)
        assert result.shape == (2, 3)
        # First row: uniform -> zeros
        np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-10)
        # Both rows should sum to zero
        np.testing.assert_allclose(result.sum(axis=1), [0.0, 0.0], atol=1e-10)

    def test_normalizes_to_proportions(self):
        """CLR normalizes input to sum to 1 before transformation.
        This means [80, 80, 80] and [20, 20, 20] produce identical CLR values.
        (Documented behavior — see audit finding C2.)
        """
        a = clr_transform(np.array([80.0, 80.0, 80.0]))
        b = clr_transform(np.array([20.0, 20.0, 20.0]))
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_zero_handling(self):
        """Zeros should be replaced by epsilon; no NaN or inf in output."""
        x = np.array([0.0, 1.0, 1.0])
        result = clr_transform(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# =============================================================================
# ILR Transform Tests
# =============================================================================


class TestILRTransform:
    """Test isometric log-ratio transform."""

    def test_output_dimension_reduction(self):
        """ILR reduces D components to D-1."""
        x = np.array([0.5, 0.3, 0.2])
        result = ilr_transform(x)
        assert result.shape == (2,)

    def test_batch_dimension_reduction(self):
        """ILR on (n, D) produces (n, D-1)."""
        X = np.array([
            [0.5, 0.3, 0.2],
            [0.1, 0.6, 0.3],
        ])
        result = ilr_transform(X)
        assert result.shape == (2, 2)

    def test_uniform_produces_zeros(self):
        """ILR of uniform vector should be all zeros."""
        x = np.array([1.0, 1.0, 1.0])
        result = ilr_transform(x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)


# =============================================================================
# Aitchison Distance Tests
# =============================================================================


class TestAitchisonDistance:
    """Test Aitchison distance with known expected values."""

    def test_identical_vectors_zero_distance(self):
        """Same proportions should give zero distance."""
        x = np.array([50.0, 30.0, 20.0])
        assert aitchison_distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_proportionally_equal_zero_distance(self):
        """Vectors with same proportions but different scales have zero distance."""
        x = np.array([6.0, 3.0, 1.0])
        y = np.array([60.0, 30.0, 10.0])
        assert aitchison_distance(x, y) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        """d(x, y) == d(y, x)."""
        x = np.array([50.0, 30.0, 20.0])
        y = np.array([20.0, 50.0, 30.0])
        assert aitchison_distance(x, y) == pytest.approx(aitchison_distance(y, x))

    def test_triangle_inequality(self):
        """d(x, z) <= d(x, y) + d(y, z)."""
        x = np.array([50.0, 30.0, 20.0])
        y = np.array([20.0, 50.0, 30.0])
        z = np.array([10.0, 10.0, 80.0])
        assert aitchison_distance(x, z) <= aitchison_distance(x, y) + aitchison_distance(y, z) + 1e-10

    def test_non_negative(self):
        """Distance should always be non-negative."""
        x = np.array([50.0, 30.0, 20.0])
        y = np.array([10.0, 10.0, 80.0])
        assert aitchison_distance(x, y) >= 0

    def test_shape_mismatch_raises(self):
        """Different-shaped vectors should raise ValueError."""
        x = np.array([50.0, 30.0, 20.0])
        y = np.array([50.0, 50.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            aitchison_distance(x, y)

    def test_known_value_2d(self):
        """Hand-computed Aitchison distance for 2-component vectors."""
        # x = [3, 1], y = [1, 3]
        # Normalize: x -> [0.75, 0.25], y -> [0.25, 0.75]
        # CLR(x) = [ln(0.75)-mean, ln(0.25)-mean] where mean = (ln(0.75)+ln(0.25))/2
        # CLR(y) = [ln(0.25)-mean, ln(0.75)-mean] — mirror of CLR(x)
        # d = ||CLR(x) - CLR(y)|| = 2*|ln(0.75)-mean_ln|
        x = np.array([3.0, 1.0])
        y = np.array([1.0, 3.0])
        # Manually: ln(3) - ln(1) = ln(3) for the log-ratio, so d = sqrt(2) * |ln(3)|
        # Actually: d_A(x,y) = sqrt(sum((clr(x)-clr(y))^2))
        # clr(x) = [ln(3/gmean(3,1)), ln(1/gmean(3,1))] where gmean = sqrt(3)
        # = [ln(3/sqrt(3)), ln(1/sqrt(3))] = [ln(sqrt(3)), ln(1/sqrt(3))] = [0.5*ln3, -0.5*ln3]
        # clr(y) = [-0.5*ln3, 0.5*ln3]
        # diff = [ln3, -ln3], ||diff|| = ln3 * sqrt(2)
        expected = np.log(3) * np.sqrt(2)
        assert aitchison_distance(x, y) == pytest.approx(expected, rel=1e-8)


class TestAitchisonDistanceMatrix:
    """Test pairwise Aitchison distance matrix."""

    def test_diagonal_is_zero(self):
        """Diagonal of distance matrix should be zero."""
        X = np.array([
            [50.0, 30.0, 20.0],
            [20.0, 50.0, 30.0],
            [10.0, 10.0, 80.0],
        ])
        dm = aitchison_distance_matrix(X)
        np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-10)

    def test_symmetric(self):
        """Distance matrix should be symmetric."""
        X = np.array([
            [50.0, 30.0, 20.0],
            [20.0, 50.0, 30.0],
            [10.0, 10.0, 80.0],
        ])
        dm = aitchison_distance_matrix(X)
        np.testing.assert_allclose(dm, dm.T, atol=1e-10)

    def test_consistent_with_pairwise(self):
        """Matrix entries should match individual pairwise computations."""
        X = np.array([
            [50.0, 30.0, 20.0],
            [20.0, 50.0, 30.0],
        ])
        dm = aitchison_distance_matrix(X)
        expected = aitchison_distance(X[0], X[1])
        assert dm[0, 1] == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Cosine Distance Tests
# =============================================================================


class TestCosineDistance:
    """Test cosine similarity/distance with known expected values."""

    def test_identical_vectors_similarity_one(self):
        """Identical vectors have cosine similarity 1."""
        x = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(x, x) == pytest.approx(1.0, abs=1e-10)

    def test_identical_vectors_distance_zero(self):
        """Identical vectors have cosine distance 0."""
        x = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have cosine similarity 0, distance 1."""
        x = np.array([1.0, 0.0])
        y = np.array([0.0, 1.0])
        assert cosine_similarity(x, y) == pytest.approx(0.0, abs=1e-10)
        assert cosine_distance(x, y) == pytest.approx(1.0, abs=1e-10)

    def test_opposite_vectors(self):
        """Opposite vectors have cosine similarity -1, distance 2."""
        x = np.array([1.0, 0.0])
        y = np.array([-1.0, 0.0])
        assert cosine_similarity(x, y) == pytest.approx(-1.0, abs=1e-10)
        assert cosine_distance(x, y) == pytest.approx(2.0, abs=1e-10)

    def test_zero_vector_returns_zero_similarity(self):
        """Zero vector should return 0 similarity (not crash)."""
        x = np.array([0.0, 0.0, 0.0])
        y = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(x, y) == 0.0

    def test_scaled_vectors_same_similarity(self):
        """Cosine similarity is scale-invariant."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 4.0, 6.0])
        assert cosine_similarity(x, y) == pytest.approx(1.0, abs=1e-10)

    def test_known_45_degree_angle(self):
        """cos(45 degrees) = sqrt(2)/2 ~ 0.7071."""
        x = np.array([1.0, 0.0])
        y = np.array([1.0, 1.0])
        expected = 1.0 / np.sqrt(2)
        assert cosine_similarity(x, y) == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Euclidean Distance Tests (via compute_pairwise_distances)
# =============================================================================


class TestEuclideanDistance:
    """Test Euclidean distance via compute_pairwise_distances."""

    def test_known_distance(self):
        """Euclidean distance between [0,0,0] and [3,4,0] is 5."""
        scores = {
            "A": np.array([0.0, 0.0, 0.0]),
            "B": np.array([3.0, 4.0, 0.0]),
        }
        dm, pids = compute_pairwise_distances(scores, metric="euclidean")
        assert dm[0, 1] == pytest.approx(5.0, abs=1e-10)

    def test_identical_zero_distance(self):
        """Same score vector gives zero distance."""
        scores = {
            "A": np.array([50.0, 50.0, 50.0]),
            "B": np.array([50.0, 50.0, 50.0]),
        }
        dm, _ = compute_pairwise_distances(scores, metric="euclidean")
        assert dm[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_distinguishes_absolute_levels(self):
        """Unlike Aitchison, Euclidean distinguishes [80,80,80] from [20,20,20]."""
        scores = {
            "A": np.array([80.0, 80.0, 80.0]),
            "B": np.array([20.0, 20.0, 20.0]),
        }
        dm, _ = compute_pairwise_distances(scores, metric="euclidean")
        expected = np.sqrt(3 * 60.0**2)  # sqrt(3 * 3600) = 60*sqrt(3)
        assert dm[0, 1] == pytest.approx(expected, rel=1e-10)
        assert dm[0, 1] > 0  # Non-zero, unlike Aitchison for proportional vectors

    def test_2d_array_uses_mean(self):
        """When passed (n_entities, n_dims), should compute mean first."""
        scores = {
            "A": np.array([[10.0, 20.0], [30.0, 40.0]]),  # mean = [20, 30]
            "B": np.array([[20.0, 30.0]]),                  # mean = [20, 30]
        }
        dm, _ = compute_pairwise_distances(scores, metric="euclidean")
        assert dm[0, 1] == pytest.approx(0.0, abs=1e-10)


# =============================================================================
# Wasserstein Distance Tests
# =============================================================================


class TestWassersteinDistance:
    """Test Wasserstein/EMD distance."""

    def test_identical_distributions_zero(self):
        """Same distribution should give zero distance."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        d = wasserstein_distance_compositional(x, x.copy())
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        """Wasserstein distance should be non-negative."""
        x = np.array([[50.0, 30.0, 20.0], [60.0, 20.0, 20.0]])
        y = np.array([[20.0, 50.0, 30.0], [10.0, 80.0, 10.0]])
        d = wasserstein_distance_compositional(x, y)
        assert d >= 0

    def test_single_point_distributions(self):
        """For single-point distributions, distance should be non-zero for different points."""
        x = np.array([50.0, 30.0, 20.0])
        y = np.array([20.0, 50.0, 30.0])
        d = wasserstein_distance_compositional(x, y)
        assert d > 0


# =============================================================================
# compute_pairwise_distances Tests
# =============================================================================


class TestComputePairwiseDistances:
    """Test the dispatch function for pairwise distance computation."""

    def test_unknown_metric_raises(self):
        """Unknown metric should raise ValueError."""
        scores = {"A": np.array([1.0, 2.0]), "B": np.array([3.0, 4.0])}
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_pairwise_distances(scores, metric="nonexistent")

    def test_diagonal_always_zero(self):
        """Diagonal of distance matrix should be zero for any metric."""
        scores = {
            "A": np.array([50.0, 30.0, 20.0]),
            "B": np.array([20.0, 50.0, 30.0]),
            "C": np.array([10.0, 10.0, 80.0]),
        }
        for metric in ["euclidean", "cosine", "aitchison"]:
            dm, _ = compute_pairwise_distances(scores, metric=metric)
            np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-10,
                                       err_msg=f"Diagonal not zero for {metric}")

    def test_symmetric_for_all_metrics(self):
        """Distance matrix should be symmetric for all metrics."""
        scores = {
            "A": np.array([50.0, 30.0, 20.0]),
            "B": np.array([20.0, 50.0, 30.0]),
            "C": np.array([10.0, 10.0, 80.0]),
        }
        for metric in ["euclidean", "cosine", "aitchison"]:
            dm, _ = compute_pairwise_distances(scores, metric=metric)
            np.testing.assert_allclose(dm, dm.T, atol=1e-10,
                                       err_msg=f"Not symmetric for {metric}")

    def test_returns_correct_participant_order(self):
        """Returned participant IDs should match input keys."""
        scores = {
            "Alice": np.array([1.0, 2.0, 3.0]),
            "Bob": np.array([4.0, 5.0, 6.0]),
        }
        dm, pids = compute_pairwise_distances(scores, metric="euclidean")
        assert set(pids) == {"Alice", "Bob"}
        assert dm.shape == (2, 2)

    def test_default_metric_is_euclidean(self):
        """Default metric should be euclidean (changed from aitchison per audit C2)."""
        scores = {
            "A": np.array([80.0, 80.0, 80.0]),
            "B": np.array([20.0, 20.0, 20.0]),
        }
        dm_default, _ = compute_pairwise_distances(scores)
        dm_euclidean, _ = compute_pairwise_distances(scores, metric="euclidean")
        np.testing.assert_allclose(dm_default, dm_euclidean)
        # Euclidean should be non-zero (Aitchison would be zero here)
        assert dm_default[0, 1] > 0


# =============================================================================
# Group Comparison Tests (H7)
# =============================================================================


def _make_comparison_with_groups():
    """Helper: create a ParticipantComparison with controlled data and groups."""
    comp = ParticipantComparison()

    # Group A: high social scores [80, 20, 20]
    # Group B: high ecological scores [20, 80, 20]
    rng = np.random.default_rng(42)
    comp.scores_by_participant = {}
    comp.dimension_names = ["social", "ecological", "technological"]

    for i in range(5):
        # Group A: social-leaning
        comp.scores_by_participant[f"a_{i}"] = np.array([
            [80 + rng.normal(0, 3), 20 + rng.normal(0, 3), 20 + rng.normal(0, 3)]
            for _ in range(10)
        ])
        # Group B: ecological-leaning
        comp.scores_by_participant[f"b_{i}"] = np.array([
            [20 + rng.normal(0, 3), 80 + rng.normal(0, 3), 20 + rng.normal(0, 3)]
            for _ in range(10)
        ])

    comp.entity_names = [f"entity_{j}" for j in range(10)]

    groups = {
        "social_group": [f"a_{i}" for i in range(5)],
        "eco_group": [f"b_{i}" for i in range(5)],
    }
    comp.set_groups(groups)
    return comp


class TestGroupComparison:
    """Test compute_group_distances with known group structure."""

    def test_between_greater_than_within(self):
        """With clearly separated groups, between-group distance > within-group distance."""
        comp = _make_comparison_with_groups()
        result = comp.compute_group_distances(metric="euclidean")

        # Between-group mean should be much larger than within-group means
        between = list(result.between_group_distances.values())[0]["mean"]
        within_a = result.within_group_distances["social_group"]["mean"]
        within_b = result.within_group_distances["eco_group"]["mean"]

        assert between > within_a
        assert between > within_b

    def test_effect_size_greater_than_one(self):
        """Clearly separated groups should have effect size > 1."""
        comp = _make_comparison_with_groups()
        result = comp.compute_group_distances(metric="euclidean")
        assert result.effect_size is not None
        assert result.effect_size > 1.0

    def test_group_centroids_correct_direction(self):
        """Group A centroid should have higher social, Group B higher ecological."""
        comp = _make_comparison_with_groups()
        result = comp.compute_group_distances(metric="euclidean")

        centroid_a = result.group_centroids["social_group"]
        centroid_b = result.group_centroids["eco_group"]

        # Social dimension (index 0): group A > group B
        assert centroid_a[0] > centroid_b[0]
        # Ecological dimension (index 1): group B > group A
        assert centroid_b[1] > centroid_a[1]

    def test_within_group_distances_structure(self):
        """Within-group distances should have expected keys."""
        comp = _make_comparison_with_groups()
        result = comp.compute_group_distances(metric="euclidean")

        for group_name in ["social_group", "eco_group"]:
            stats = result.within_group_distances[group_name]
            assert "mean" in stats
            assert "median" in stats
            assert "min" in stats
            assert "max" in stats
            assert "n_pairs" in stats
            # 5 choose 2 = 10 pairs
            assert stats["n_pairs"] == 10

    def test_single_member_group_zero_within(self):
        """A group with 1 member should have 0 within-group distance."""
        comp = ParticipantComparison()
        comp.scores_by_participant = {
            "solo": np.array([[50.0, 50.0, 50.0]]),
            "p1": np.array([[80.0, 20.0, 20.0]]),
            "p2": np.array([[70.0, 30.0, 20.0]]),
        }
        comp.dimension_names = ["s", "e", "t"]
        comp.entity_names = ["e1"]
        comp.set_groups({"loners": ["solo"], "pair": ["p1", "p2"]})

        result = comp.compute_group_distances(metric="euclidean")
        assert result.within_group_distances["loners"]["n_pairs"] == 0
        assert result.within_group_distances["loners"]["mean"] == 0.0

    def test_no_groups_raises(self):
        """compute_group_distances without set_groups should raise."""
        comp = ParticipantComparison()
        comp.scores_by_participant = {"A": np.array([1.0, 2.0])}
        with pytest.raises(ValueError, match="No groups defined"):
            comp.compute_group_distances()


# =============================================================================
# Permutation Test Tests (H7)
# =============================================================================


class TestPermutationTest:
    """Test permutation test correctness."""

    def test_significant_with_separated_groups(self):
        """Clearly separated groups should give p < 0.05."""
        comp = _make_comparison_with_groups()
        result = comp.run_permutation_test(
            metric="euclidean",
            n_permutations=999,
            random_seed=42,
        )

        assert result["p_value"] < 0.05
        assert result["observed_effect_size"] is not None
        assert result["observed_effect_size"] > 1.0

    def test_null_distribution_length(self):
        """Null distribution should have n_permutations entries."""
        comp = _make_comparison_with_groups()
        result = comp.run_permutation_test(
            metric="euclidean",
            n_permutations=100,
            random_seed=42,
        )
        assert len(result["null_distribution"]) == 100

    def test_not_significant_with_random_labels(self):
        """Random group assignments (no real difference) should give p > 0.05 (usually)."""
        comp = ParticipantComparison()
        rng = np.random.default_rng(123)
        comp.dimension_names = ["d1", "d2", "d3"]

        # All participants drawn from same distribution
        for i in range(10):
            comp.scores_by_participant[f"p_{i}"] = np.array([
                rng.normal(50, 10, size=3) for _ in range(5)
            ])
        comp.entity_names = [f"e_{j}" for j in range(5)]

        # Arbitrary group split
        comp.set_groups({
            "g1": [f"p_{i}" for i in range(5)],
            "g2": [f"p_{i}" for i in range(5, 10)],
        })

        result = comp.run_permutation_test(
            metric="euclidean",
            n_permutations=999,
            random_seed=42,
        )
        # With no real difference, p should typically be > 0.05
        # Use a relaxed threshold to avoid flaky tests
        assert result["p_value"] > 0.01

    def test_preserves_group_sizes(self):
        """Permutation should maintain original group sizes."""
        comp = _make_comparison_with_groups()
        # Just verify we get a result without error — the internal logic
        # uses group_sizes to slice the shuffled array
        result = comp.run_permutation_test(
            metric="euclidean",
            n_permutations=50,
            random_seed=42,
        )
        assert result["observed_effect_size"] is not None
        assert result["p_value"] is not None

    def test_reproducible_with_seed(self):
        """Same seed should produce identical results."""
        comp = _make_comparison_with_groups()
        r1 = comp.run_permutation_test(metric="euclidean", n_permutations=100, random_seed=99)
        r2 = comp.run_permutation_test(metric="euclidean", n_permutations=100, random_seed=99)
        assert r1["p_value"] == r2["p_value"]
        assert r1["observed_effect_size"] == r2["observed_effect_size"]

    def test_no_groups_raises(self):
        """Permutation test without groups should raise."""
        comp = ParticipantComparison()
        comp.scores_by_participant = {"A": np.array([1.0, 2.0])}
        with pytest.raises(ValueError, match="No groups defined"):
            comp.run_permutation_test()


# =============================================================================
# ParticipantComparison.load_scores Tests
# =============================================================================


class TestLoadScores:
    """Test CSV data loading into ParticipantComparison."""

    def _write_csv(self, path: Path, rows: list[dict]):
        fieldnames = rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_basic_loading(self, tmp_path):
        """Load a simple scored CSV and verify structure."""
        csv_path = tmp_path / "scores.csv"
        self._write_csv(csv_path, [
            {"text_id": "p1", "entity": "e1", "social_mean": "80", "ecological_mean": "20", "technological_mean": "50"},
            {"text_id": "p1", "entity": "e2", "social_mean": "60", "ecological_mean": "40", "technological_mean": "30"},
            {"text_id": "p2", "entity": "e1", "social_mean": "30", "ecological_mean": "70", "technological_mean": "50"},
        ])

        comp = ParticipantComparison()
        comp.load_scores(csv_path, dimensions=["social", "ecological", "technological"])

        assert "p1" in comp.scores_by_participant
        assert "p2" in comp.scores_by_participant
        assert comp.scores_by_participant["p1"].shape == (2, 3)
        assert comp.scores_by_participant["p2"].shape == (1, 3)

    def test_dimension_auto_detection(self, tmp_path):
        """Dimensions should be auto-detected from *_mean columns."""
        csv_path = tmp_path / "scores.csv"
        self._write_csv(csv_path, [
            {"text_id": "p1", "entity": "e1", "alpha_mean": "50", "beta_mean": "60"},
        ])

        comp = ParticipantComparison()
        comp.load_scores(csv_path)

        assert set(comp.dimension_names) == {"alpha", "beta"}
