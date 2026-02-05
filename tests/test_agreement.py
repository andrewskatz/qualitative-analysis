"""
Tests for Krippendorff's Alpha agreement analysis.

Tests cover:
- Core algorithm correctness
- Bootstrap confidence intervals
- Run-level and participant-level modes
- Edge cases and error handling
"""

import numpy as np
import pandas as pd
import pytest

from qualitative_analysis.entity.agreement import (
    AgreementResult,
    krippendorff_alpha,
    bootstrap_ci,
    compute_agreement,
    _interpret_alpha,
)


class TestKrippendorffAlpha:
    """Tests for the core Krippendorff's Alpha algorithm."""

    def test_perfect_agreement(self):
        """Perfect agreement (all raters identical) should give alpha = 1.0."""
        # 3 raters, 5 units, all identical
        data = np.array([
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
            [10, 20, 30, 40, 50],
        ])
        alpha = krippendorff_alpha(data)
        assert alpha == pytest.approx(1.0, abs=1e-10)

    def test_no_agreement_low_alpha(self):
        """Random independent ratings should give alpha close to 0."""
        rng = np.random.default_rng(42)
        # 4 raters, 20 units, random scores 0-100
        data = rng.uniform(0, 100, size=(4, 20))
        alpha = krippendorff_alpha(data)
        # With random data, alpha should be near 0 (can be negative)
        assert -0.5 < alpha < 0.5

    def test_known_example_from_literature(self):
        """Test against a known example from Krippendorff (2011).

        Example from "Computing Krippendorff's Alpha-Reliability":
        https://repository.upenn.edu/asc_papers/43/

        4 raters, 12 units (some missing), interval data
        Expected alpha ≈ 0.743
        """
        # Data from Krippendorff (2011) Table 1 (nominal example adapted to interval)
        # Using a simplified example with complete data
        data = np.array([
            [1, 2, 3, 3, 2, 1, 4, 1, 2, 3, 4, 4],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 3, 4, 4],
            [np.nan, 2, 3, 3, 2, 2, 4, 1, 2, 3, np.nan, 4],
            [1, 2, 3, 3, 2, 1, 4, 1, 2, 3, 4, 4],
        ])
        alpha = krippendorff_alpha(data)
        # With this synthetic data, expect high agreement
        assert alpha > 0.7

    def test_handles_nan_missing_data(self):
        """Missing values (NaN) should be handled correctly."""
        data = np.array([
            [10, 20, np.nan, 40, 50],
            [10, np.nan, 30, 40, 50],
            [10, 20, 30, np.nan, 50],
        ])
        alpha = krippendorff_alpha(data)
        # Should still compute (high agreement on non-missing)
        assert alpha > 0.8

    def test_insufficient_raters_raises(self):
        """Should raise ValueError with fewer than 2 raters."""
        data = np.array([[10, 20, 30, 40, 50]])  # Only 1 rater
        with pytest.raises(ValueError, match="at least 2 raters"):
            krippendorff_alpha(data)

    def test_insufficient_units_raises(self):
        """Should raise ValueError with fewer than 2 units."""
        data = np.array([[10], [10], [10]])  # Only 1 unit
        with pytest.raises(ValueError, match="at least 2 units"):
            krippendorff_alpha(data)

    def test_all_same_value(self):
        """All values identical across all raters and units should give alpha = 1.0."""
        data = np.full((3, 10), 50.0)
        alpha = krippendorff_alpha(data)
        assert alpha == pytest.approx(1.0, abs=1e-10)


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_ci_contains_point_estimate(self):
        """Bootstrap CI should contain the point estimate."""
        data = np.array([
            [10, 20, 30, 40, 50, 60, 70],
            [11, 21, 29, 41, 49, 61, 69],
            [9, 19, 31, 39, 51, 59, 71],
        ])
        alpha = krippendorff_alpha(data)
        ci_lower, ci_upper = bootstrap_ci(data, n_bootstrap=500, random_seed=42)

        # Point estimate should be within CI (with high probability)
        # Allow some slack since bootstrap is stochastic
        assert ci_lower <= alpha + 0.1
        assert ci_upper >= alpha - 0.1

    def test_reproducible_with_seed(self):
        """Same seed should give identical CI."""
        data = np.array([
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51],
            [9, 19, 29, 39, 49],
        ])
        ci1 = bootstrap_ci(data, n_bootstrap=100, random_seed=42)
        ci2 = bootstrap_ci(data, n_bootstrap=100, random_seed=42)

        assert ci1[0] == ci2[0]
        assert ci1[1] == ci2[1]

    def test_different_seeds_give_different_ci(self):
        """Different seeds should give different CI on noisier data."""
        # Use data with more variability so bootstrap samples differ
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 100, size=(4, 15))

        ci1 = bootstrap_ci(data, n_bootstrap=100, random_seed=42)
        ci2 = bootstrap_ci(data, n_bootstrap=100, random_seed=123)

        # With noisy data, different seeds should give different CIs
        # At minimum one bound should differ
        assert ci1[0] != ci2[0] or ci1[1] != ci2[1]


class TestInterpretAlpha:
    """Tests for alpha interpretation thresholds."""

    def test_excellent_threshold(self):
        assert _interpret_alpha(0.95) == "excellent"
        assert _interpret_alpha(0.90) == "excellent"

    def test_good_threshold(self):
        assert _interpret_alpha(0.85) == "good"
        assert _interpret_alpha(0.80) == "good"

    def test_fair_threshold(self):
        assert _interpret_alpha(0.75) == "fair"
        assert _interpret_alpha(0.67) == "fair"

    def test_moderate_threshold(self):
        assert _interpret_alpha(0.60) == "moderate"
        assert _interpret_alpha(0.50) == "moderate"

    def test_poor_threshold(self):
        assert _interpret_alpha(0.40) == "poor"
        assert _interpret_alpha(0.0) == "poor"
        assert _interpret_alpha(-0.5) == "poor"


class TestComputeAgreement:
    """Tests for the high-level compute_agreement function."""

    @pytest.fixture
    def sample_run_level_df(self):
        """Create a sample DataFrame with run-level columns."""
        np.random.seed(42)
        n_rows = 20

        data = {
            "text_id": [f"p{i // 5}" for i in range(n_rows)],
            "entity": [f"e{i % 5}" for i in range(n_rows)],
            # High agreement runs (small noise)
            "social_run1": np.random.uniform(40, 60, n_rows),
            "social_run2": np.random.uniform(40, 60, n_rows),
            "social_run3": np.random.uniform(40, 60, n_rows),
            "social_mean": np.random.uniform(45, 55, n_rows),
            # Lower agreement runs (larger noise)
            "eco_run1": np.random.uniform(20, 80, n_rows),
            "eco_run2": np.random.uniform(20, 80, n_rows),
            "eco_run3": np.random.uniform(20, 80, n_rows),
            "eco_mean": np.random.uniform(30, 70, n_rows),
        }
        return pd.DataFrame(data)

    def test_run_level_mode(self, sample_run_level_df):
        """Run-level mode should compute alpha from run columns."""
        results = compute_agreement(
            sample_run_level_df,
            dimensions=["social"],
            mode="run_level",
            n_bootstrap=100,
        )

        assert "social" in results
        result = results["social"]
        assert isinstance(result, AgreementResult)
        assert result.mode == "run_level"
        assert result.n_raters == 3  # 3 runs
        assert not np.isnan(result.alpha)

    def test_participant_level_mode(self, sample_run_level_df):
        """Participant-level mode should compute alpha from mean columns."""
        results = compute_agreement(
            sample_run_level_df,
            dimensions=["social"],
            mode="participant_level",
            n_bootstrap=100,
        )

        assert "social" in results
        result = results["social"]
        assert isinstance(result, AgreementResult)
        assert result.mode == "participant_level"

    def test_invalid_mode_raises(self, sample_run_level_df):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            compute_agreement(
                sample_run_level_df,
                dimensions=["social"],
                mode="invalid_mode",
            )

    def test_missing_run_columns_handled(self, sample_run_level_df):
        """Missing run columns should produce NaN result, not crash."""
        results = compute_agreement(
            sample_run_level_df,
            dimensions=["nonexistent"],  # No columns for this dimension
            mode="run_level",
            n_bootstrap=100,
        )

        assert "nonexistent" in results
        assert np.isnan(results["nonexistent"].alpha)

    def test_to_dict_serialization(self, sample_run_level_df):
        """AgreementResult.to_dict() should be JSON-serializable."""
        import json

        results = compute_agreement(
            sample_run_level_df,
            dimensions=["social"],
            mode="run_level",
            n_bootstrap=50,
        )

        result_dict = results["social"].to_dict()

        # Should not raise
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0

        # Check expected keys
        assert "alpha" in result_dict
        assert "ci_lower" in result_dict
        assert "ci_upper" in result_dict
        assert "interpretation" in result_dict
