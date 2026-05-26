"""
Tests for Bayesian hierarchical model module.

Tests data preparation, model building, run-level serialization,
and round-trip score persistence.
"""

import json
import csv
import os
import statistics
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qualitative_analysis.entity.models import (
    DimensionScore,
    EntityScore,
    EntityScoreResult,
    DimensionDefinition,
)
from qualitative_analysis.entity.bayesian import (
    prepare_beta_data,
    check_pymc_available,
    validate_bayesian_runtime,
)

for env_key, env_value in {
    "ARVIZ_DATA": "/tmp/arviz_data",
    "MPLCONFIGDIR": "/tmp/mplconfig",
    "XDG_CACHE_HOME": "/tmp",
}.items():
    os.environ.setdefault(env_key, env_value)

Path(os.environ["ARVIZ_DATA"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _bayesian_runtime_available() -> bool:
    ready, _ = validate_bayesian_runtime()
    return ready


_BAYESIAN_RUNTIME_AVAILABLE, _BAYESIAN_RUNTIME_MESSAGE = validate_bayesian_runtime()
_BAYESIAN_RUNTIME_REASON = _BAYESIAN_RUNTIME_MESSAGE or "PyMC/ArviZ runtime not available"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scores_df(
    n_entities: int = 5,
    n_participants: int = 3,
    n_runs: int = 3,
    dimensions: List[str] = None,
    groups: Dict[str, List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic scored DataFrame with run-level columns.

    Returns a DataFrame matching the format produced by `qa entity score`,
    including `{dim}_run{k}`, `{dim}_mean`, `{dim}_std`, `text_id`,
    `entity`, `context`, and `group` columns.
    """
    rng = np.random.RandomState(seed)
    dims = dimensions or ["social", "ecological", "technological"]

    if groups is None:
        pids = [f"p{i+1:02d}" for i in range(n_participants)]
        groups = {"A": pids}
    else:
        pids = [pid for members in groups.values() for pid in members]

    pid_to_group = {}
    for gname, members in groups.items():
        for pid in members:
            pid_to_group[pid] = gname

    entities = [f"entity_{i+1}" for i in range(n_entities)]

    rows = []
    for pid in pids:
        for ent in entities:
            row = {
                "text_id": pid,
                "entity": ent,
                "context": f"context for {ent}",
                "group": pid_to_group[pid],
                "num_runs": n_runs,
            }
            for dim in dims:
                runs = rng.randint(10, 91, size=n_runs).tolist()
                for k, score in enumerate(runs, start=1):
                    row[f"{dim}_run{k}"] = score
                row[f"{dim}_mean"] = round(statistics.mean(runs), 2)
                row[f"{dim}_std"] = round(
                    statistics.stdev(runs) if len(runs) > 1 else 0.0, 2
                )
                row[f"{dim}_cv"] = round(
                    (statistics.stdev(runs) / statistics.mean(runs))
                    if len(runs) > 1 and statistics.mean(runs) > 0
                    else 0.0,
                    4,
                )
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def basic_df():
    """Small 5-entity, 3-participant DataFrame with run columns."""
    groups = {"A": ["p01", "p02"], "B": ["p03"]}
    return _make_scores_df(n_entities=5, groups=groups)


@pytest.fixture
def single_participant_df():
    """Edge-case: single participant."""
    groups = {"only_group": ["p01"]}
    return _make_scores_df(n_entities=3, groups=groups, n_runs=2)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: prepare_beta_data
# ---------------------------------------------------------------------------


class TestPrepareBetaData:
    """Tests for the prepare_beta_data helper."""

    def test_basic_shape(self, basic_df):
        """Output has correct dimensions and required keys."""
        dims = ["social", "ecological", "technological"]
        data = prepare_beta_data(basic_df, dims)

        # 5 entities * 3 participants * 3 runs * 3 dims = 135
        assert len(data["long_df"]) == 5 * 3 * 3 * 3
        assert data["n_entities"] == 5
        assert data["n_participants"] == 3
        assert data["n_groups"] == 2
        assert data["n_runs"] == 3
        assert set(data["long_df"].columns).issuperset(
            {"entity", "participant", "group", "run", "dimension", "score", "y",
             "entity_idx", "participant_idx", "group_idx"}
        )

    def test_rescaling_bounds(self, basic_df):
        """Rescaled y values must be strictly in (0, 1)."""
        dims = ["social", "ecological", "technological"]
        data = prepare_beta_data(basic_df, dims)
        y = data["long_df"]["y"]
        assert (y > 0).all(), "All y must be > 0"
        assert (y < 1).all(), "All y must be < 1"

    def test_auto_detect_runs(self, basic_df):
        """n_runs should be auto-detected when not supplied."""
        dims = ["social"]
        data = prepare_beta_data(basic_df, dims, n_runs=None)
        assert data["n_runs"] == 3

    def test_explicit_runs(self, basic_df):
        """Passing n_runs explicitly should work."""
        dims = ["social"]
        data = prepare_beta_data(basic_df, dims, n_runs=3)
        assert data["n_runs"] == 3

    def test_missing_run_columns_raises(self, basic_df):
        """Should raise if expected run columns don't exist."""
        dims = ["nonexistent_dimension"]
        with pytest.raises(ValueError, match="No run-level columns found"):
            prepare_beta_data(basic_df, dims)

    def test_entity_names_are_plain_str(self, basic_df):
        """Entity/participant/group name lists must be plain Python str."""
        dims = ["social"]
        data = prepare_beta_data(basic_df, dims)
        for name in data["entity_names"]:
            assert type(name) is str  # noqa: E721
        for name in data["participant_names"]:
            assert type(name) is str  # noqa: E721
        for name in data["group_names"]:
            assert type(name) is str  # noqa: E721

    def test_index_arrays_valid(self, basic_df):
        """Integer index arrays should have valid ranges."""
        dims = ["social"]
        data = prepare_beta_data(basic_df, dims)
        long_df = data["long_df"]

        assert long_df["entity_idx"].min() >= 0
        assert long_df["entity_idx"].max() < data["n_entities"]
        assert long_df["participant_idx"].min() >= 0
        assert long_df["participant_idx"].max() < data["n_participants"]
        assert long_df["group_idx"].min() >= 0
        assert long_df["group_idx"].max() < data["n_groups"]

    def test_participant_to_group_mapping(self, basic_df):
        """participant_to_group dict should map every participant to an int."""
        dims = ["social"]
        data = prepare_beta_data(basic_df, dims)
        assert len(data["participant_to_group"]) == data["n_participants"]
        for pid, gid in data["participant_to_group"].items():
            assert isinstance(gid, int)
            assert 0 <= gid < data["n_groups"]

    def test_nan_scores_skipped(self, basic_df):
        """Rows with NaN run scores should be dropped."""
        dims = ["social"]
        df = basic_df.copy()
        # Set one run score to NaN
        df.loc[0, "social_run1"] = np.nan
        data = prepare_beta_data(df, dims)
        # Should have one fewer observation
        expected = 5 * 3 * 3 - 1  # one NaN removed
        assert len(data["long_df"]) == expected

    def test_single_participant(self, single_participant_df):
        """Should work with a single participant."""
        dims = ["social", "ecological", "technological"]
        data = prepare_beta_data(single_participant_df, dims)
        assert data["n_participants"] == 1
        assert data["n_groups"] == 1

    def test_numeric_participant_ids_are_supported(self):
        """Numeric participant identifiers should be normalized without mapping failures."""
        df = pd.DataFrame(
            [
                {
                    "text_id": 101,
                    "entity": "entity_1",
                    "group": "A",
                    "social_run1": 50,
                    "social_mean": 50,
                },
                {
                    "text_id": 202,
                    "entity": "entity_1",
                    "group": "B",
                    "social_run1": 60,
                    "social_mean": 60,
                },
            ]
        )

        data = prepare_beta_data(df, ["social"], n_runs=1)

        assert data["participant_names"] == ["101", "202"]
        assert list(sorted(data["participant_to_group"].keys())) == ["101", "202"]
        assert data["long_df"]["participant_idx"].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# Tests: EntityScore run-level serialization
# ---------------------------------------------------------------------------


class TestRunLevelSerialization:
    """Tests for run-level score columns in EntityScore.to_dict()."""

    def test_run_columns_present(self):
        """to_dict() should produce {dim}_run{k} columns."""
        scores = [75, 80, 70]
        dim_score = DimensionScore.from_scores("social", scores)
        entity_score = EntityScore(
            entity="test_entity",
            text_id="p01",
            context="test context",
            dimension_scores={"social": dim_score},
            num_runs=3,
        )
        flat = entity_score.to_dict()
        assert flat["social_run1"] == 75
        assert flat["social_run2"] == 80
        assert flat["social_run3"] == 70

    def test_run_columns_multiple_dimensions(self):
        """Run columns should exist for every dimension."""
        dims = {}
        for dim_name, scores in [("social", [60, 65, 70]), ("ecological", [40, 45, 50])]:
            dims[dim_name] = DimensionScore.from_scores(dim_name, scores)

        entity_score = EntityScore(
            entity="multi_dim",
            text_id="p01",
            context="ctx",
            dimension_scores=dims,
            num_runs=3,
        )
        flat = entity_score.to_dict()

        for dim_name in ["social", "ecological"]:
            for k in range(1, 4):
                assert f"{dim_name}_run{k}" in flat, f"Missing {dim_name}_run{k}"

    def test_empty_scores_no_crash(self):
        """Entity with empty scores should not crash."""
        dim_score = DimensionScore.from_scores("social", [])
        entity_score = EntityScore(
            entity="empty",
            text_id="p01",
            context="",
            dimension_scores={"social": dim_score},
            num_runs=0,
        )
        flat = entity_score.to_dict()
        # No run columns should be present
        assert "social_run1" not in flat
        assert "social_mean" in flat

    def test_csv_round_trip(self, tmp_dir):
        """Scores written to CSV should preserve run-level columns."""
        scores = [75, 80, 70]
        dim_score = DimensionScore.from_scores("social", scores)
        entity_score = EntityScore(
            entity="roundtrip",
            text_id="p01",
            context="ctx",
            dimension_scores={"social": dim_score},
            num_runs=3,
        )

        csv_path = tmp_dir / "test_scores.csv"
        flat = entity_score.to_flat_dict()
        fieldnames = list(flat.keys())

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(flat)

        # Read back
        df = pd.read_csv(csv_path)
        assert int(df.iloc[0]["social_run1"]) == 75
        assert int(df.iloc[0]["social_run2"]) == 80
        assert int(df.iloc[0]["social_run3"]) == 70

    def test_json_round_trip(self, tmp_dir):
        """Scores saved to JSON and loaded back should reconstruct run scores."""
        from qualitative_analysis.entity.scorer import EntityScorer

        scores_social = [75, 80, 70]
        scores_eco = [40, 45, 50]

        dims_defs = [
            DimensionDefinition(name="Social", description="Social factors"),
            DimensionDefinition(name="Ecological", description="Ecological factors"),
        ]

        entity_score = EntityScore(
            entity="roundtrip",
            text_id="p01",
            context="ctx",
            dimension_scores={
                "social": DimensionScore.from_scores("social", scores_social),
                "ecological": DimensionScore.from_scores("ecological", scores_eco),
            },
            num_runs=3,
        )

        result = EntityScoreResult(
            scores=[entity_score],
            dimensions=dims_defs,
        )

        json_path = tmp_dir / "test_scores.json"
        scorer = EntityScorer()
        scorer.save(result, json_path)

        # Load back
        loaded = scorer.load(json_path)
        assert len(loaded.scores) == 1
        loaded_score = loaded.scores[0]
        assert loaded_score.dimension_scores["social"].scores == scores_social
        assert loaded_score.dimension_scores["ecological"].scores == scores_eco


# ---------------------------------------------------------------------------
# Tests: BayesianEntityModel.build_model (requires PyMC)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _BAYESIAN_RUNTIME_AVAILABLE,
    reason=_BAYESIAN_RUNTIME_REASON,
)
class TestBayesianModelBuild:
    """Tests for model construction (requires PyMC)."""

    def test_model_builds_without_error(self, basic_df):
        """build_model should produce a PyMC model without errors."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {"A": ["p01", "p02"], "B": ["p03"]}
        model = BayesianEntityModel(
            scores_df=basic_df,
            dimensions=["social"],
            groups=groups,
        )
        pm_model = model.build_model("social")
        assert pm_model is not None

    def test_model_has_expected_variables(self, basic_df):
        """Model should contain all expected random variables."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {"A": ["p01", "p02"], "B": ["p03"]}
        model = BayesianEntityModel(
            scores_df=basic_df,
            dimensions=["social"],
            groups=groups,
        )
        pm_model = model.build_model("social")

        free_var_names = {v.name for v in pm_model.free_RVs}
        observed_var_names = {v.name for v in pm_model.observed_RVs}

        expected_free = {
            "mu_pop", "sigma_entity", "z_entity",
            "sigma_group", "group_effect",
            "sigma_participant", "z_participant",
            "kappa",
        }
        expected_observed = {"y_obs"}
        assert expected_free.issubset(free_var_names), f"Missing free vars: {expected_free - free_var_names}"
        assert expected_observed.issubset(observed_var_names), f"Missing observed vars: {expected_observed - observed_var_names}"

    def test_model_coords_shape(self, basic_df):
        """Model coordinate dimensions should match data shapes."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {"A": ["p01", "p02"], "B": ["p03"]}
        model = BayesianEntityModel(
            scores_df=basic_df,
            dimensions=["social"],
            groups=groups,
        )
        pm_model = model.build_model("social")

        coords = pm_model.coords
        assert len(coords["entity"]) == 5
        assert len(coords["participant"]) == 3
        assert len(coords["group"]) == 2

    def test_prior_predictive(self, basic_df):
        """Prior predictive check should sample without errors."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {"A": ["p01", "p02"], "B": ["p03"]}
        model = BayesianEntityModel(
            scores_df=basic_df,
            dimensions=["social"],
            groups=groups,
        )
        pp = model.prior_predictive_check("social", draws=10)
        assert pp is not None

    def test_invalid_dimension_raises(self, basic_df):
        """Requesting a dimension not in the data should raise."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {"A": ["p01", "p02"], "B": ["p03"]}
        model = BayesianEntityModel(
            scores_df=basic_df,
            dimensions=["social"],
            groups=groups,
        )
        with pytest.raises(ValueError, match="No observations found"):
            model.build_model("nonexistent")


# ---------------------------------------------------------------------------
# Tests: check_pymc_available
# ---------------------------------------------------------------------------


class TestCheckPymcAvailable:
    """Tests for the import check utility."""

    def test_returns_bool(self):
        result = check_pymc_available()
        assert isinstance(result, bool)

    def test_false_when_missing(self):
        """Should return False when pymc is not importable."""
        with patch("importlib.util.find_spec", return_value=None):
            assert check_pymc_available() is False


# ---------------------------------------------------------------------------
# Tests: Integration — known ground truth recovery
# ---------------------------------------------------------------------------


def _make_ground_truth_df(
    n_entities: int = 8,
    n_participants_per_group: int = 4,
    n_runs: int = 3,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Generate synthetic data with known group-level differences.

    Group "high" participants score ~75 on social, ~40 on ecological.
    Group "low"  participants score ~40 on social, ~75 on ecological.

    Both groups score ~55 on technological (no difference).

    Returns a DataFrame in the scored format with run-level columns.
    """
    rng = np.random.RandomState(seed)

    groups = {
        "high": [f"p_high_{i}" for i in range(n_participants_per_group)],
        "low": [f"p_low_{i}" for i in range(n_participants_per_group)],
    }

    # Group-level true means per dimension (on 0-100 scale)
    group_means = {
        "high": {"social": 75, "ecological": 40, "technological": 55},
        "low": {"social": 40, "ecological": 75, "technological": 55},
    }

    entities = [f"ent_{i}" for i in range(n_entities)]
    dims = ["social", "ecological", "technological"]

    rows = []
    for gname, pids in groups.items():
        for pid in pids:
            for ent in entities:
                row = {
                    "text_id": pid,
                    "entity": ent,
                    "context": f"context for {ent}",
                    "group": gname,
                    "num_runs": n_runs,
                }
                for dim in dims:
                    mu = group_means[gname][dim]
                    # Add entity-level variation (sd=5) and run-level noise (sd=3)
                    entity_offset = rng.normal(0, 5)
                    runs = []
                    for k in range(1, n_runs + 1):
                        score = mu + entity_offset + rng.normal(0, 3)
                        score = max(1, min(99, round(score)))
                        runs.append(score)
                        row[f"{dim}_run{k}"] = score
                    row[f"{dim}_mean"] = round(np.mean(runs), 2)
                    row[f"{dim}_std"] = round(np.std(runs, ddof=1) if len(runs) > 1 else 0.0, 2)
                rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def ground_truth_df():
    """Synthetic dataset with known group differences."""
    return _make_ground_truth_df()


@pytest.mark.skipif(
    not _BAYESIAN_RUNTIME_AVAILABLE,
    reason=_BAYESIAN_RUNTIME_REASON,
)
class TestBayesianIntegration:
    """
    Integration tests: fit the full Bayesian model on synthetic data
    with known ground truth and verify recovered group contrasts.
    """

    def test_group_contrast_direction_social(self, ground_truth_df):
        """
        'high' group should score higher on social than 'low' group.

        Expected: contrast high-low > 0 on social, with P(direction) > 0.95.
        """
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        contrasts = model.compute_group_contrasts()
        social_contrasts = contrasts["social"]

        # Groups are sorted alphabetically: ["high", "low"]
        pair_key = "high_vs_low"
        assert pair_key in social_contrasts, f"Expected {pair_key}, got {list(social_contrasts.keys())}"

        c = social_contrasts[pair_key]
        # high scores ~75, low scores ~40 on social => positive difference
        assert c["mean_diff"] > 10, (
            f"Expected high group to score >10 pts higher on social, got {c['mean_diff']:.1f}"
        )
        assert c["p_a_gt_b"] > 0.95, (
            f"Expected P(high > low) > 0.95, got {c['p_a_gt_b']:.3f}"
        )

    def test_group_contrast_direction_ecological(self, ground_truth_df):
        """
        'low' group should score higher on ecological than 'high' group.

        Expected: contrast high-low < 0 on ecological, with P(direction) > 0.95.
        """
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["ecological"],
            groups=groups,
        )
        model.fit("ecological", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        contrasts = model.compute_group_contrasts()
        c = contrasts["ecological"]["high_vs_low"]

        # high scores ~40, low scores ~75 on ecological => negative difference
        assert c["mean_diff"] < -10, (
            f"Expected high group to score >10 pts lower on ecological, got {c['mean_diff']:.1f}"
        )
        assert c["p_b_gt_a"] > 0.95, (
            f"Expected P(low > high) > 0.95, got {c['p_b_gt_a']:.3f}"
        )

    def test_no_difference_on_technological(self, ground_truth_df):
        """
        Both groups score ~55 on technological — should find no meaningful difference.

        Expected: mean difference close to 0, large ROPE probability.
        """
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["technological"],
            groups=groups,
        )
        model.fit("technological", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        contrasts = model.compute_group_contrasts(rope_delta=5.0)
        c = contrasts["technological"]["high_vs_low"]

        # Both groups at ~55 => difference should be small
        assert abs(c["mean_diff"]) < 15, (
            f"Expected small difference on technological, got {c['mean_diff']:.1f}"
        )
        # ROPE probability should be non-trivial (>0.20)
        assert c["p_in_rope"] > 0.20, (
            f"Expected substantial ROPE probability, got {c['p_in_rope']:.3f}"
        )

    def test_icc_entity_dominates(self, ground_truth_df):
        """
        With 8 entities per participant, entity-level variance should be
        a meaningful component of total variance.
        """
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        icc = model.compute_icc()
        social_icc = icc["social"]

        # Run-level variance should be the smallest component
        assert social_icc["icc_run"]["mean"] < 0.30, (
            f"Expected run-level ICC < 0.30, got {social_icc['icc_run']['mean']:.3f}"
        )
        # At least some variance should be at entity or group level
        combined = social_icc["icc_entity"]["mean"] + social_icc["icc_group"]["mean"]
        assert combined > 0.20, (
            f"Expected entity+group ICC > 0.20, got {combined:.3f}"
        )

    def test_convergence_on_ground_truth(self, ground_truth_df):
        """Model should converge on well-separated synthetic data."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        result = model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        # With well-separated data, even 2 chains / 500 draws should converge
        assert result.diagnostics["divergences"] == 0, (
            f"Expected 0 divergences, got {result.diagnostics['divergences']}"
        )
        assert result.diagnostics["rhat_max"] < 1.05, (
            f"Expected R-hat < 1.05, got {result.diagnostics['rhat_max']:.4f}"
        )


# ---------------------------------------------------------------------------
# Audit Fix Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _BAYESIAN_RUNTIME_AVAILABLE,
    reason=_BAYESIAN_RUNTIME_REASON,
)
class TestAuditFixes:
    """Tests validating the fixes from the 2026-02-17 Bayesian audit."""

    def test_hdi_keys_consistent(self, ground_truth_df):
        """H3: All output dicts should use hdi_lower/hdi_upper, not hdi_3%/hdi_97%."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        # Check summarize_posteriors
        summaries = model.summarize_posteriors()
        for dim, summary in summaries.items():
            for var_name, var_data in summary["parameters"].items():
                assert "hdi_lower" in var_data, f"Missing hdi_lower in {var_name}"
                assert "hdi_upper" in var_data, f"Missing hdi_upper in {var_name}"
                assert "hdi_3%" not in var_data, f"Old key hdi_3% found in {var_name}"
                assert "hdi_97%" not in var_data, f"Old key hdi_97% found in {var_name}"
            for gname, gdata in summary["group_effects"].items():
                assert "hdi_lower_logit" in gdata, f"Missing hdi_lower_logit for {gname}"
                assert "hdi_lower_score" in gdata, f"Missing hdi_lower_score for {gname}"

        # Check group contrasts
        contrasts = model.compute_group_contrasts()
        for dim, dim_data in contrasts.items():
            for pair_key, c in dim_data.items():
                assert "hdi_lower" in c, f"Missing hdi_lower in contrast {pair_key}"
                assert "hdi_upper" in c, f"Missing hdi_upper in contrast {pair_key}"
                assert "hdi_prob" in c, f"Missing hdi_prob in contrast {pair_key}"
                assert "hdi_3%" not in c, f"Old key hdi_3% found in contrast {pair_key}"

        # Check ICC
        icc = model.compute_icc()
        for dim, icc_data in icc.items():
            for component in ["icc_group", "icc_entity", "icc_participant", "icc_run"]:
                assert "hdi_lower" in icc_data[component], f"Missing hdi_lower in {component}"
                assert "hdi_upper" in icc_data[component], f"Missing hdi_upper in {component}"
                assert "hdi_3%" not in icc_data[component], f"Old key in {component}"

    def test_icc_sums_to_approximately_one(self, ground_truth_df):
        """H1: ICC components should sum to approximately 1.0."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        icc = model.compute_icc()
        social_icc = icc["social"]
        total = sum(social_icc[comp]["mean"] for comp in
                     ["icc_group", "icc_entity", "icc_participant", "icc_run"])
        assert abs(total - 1.0) < 0.05, (
            f"ICC components should sum to ~1.0, got {total:.4f}"
        )

    def test_shrinkage_bounds(self, ground_truth_df):
        """M2: Shrinkage values should be clamped to [0, 100]."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        shrinkage = model.compute_shrinkage()
        for dim, records in shrinkage.items():
            for rec in records:
                assert 0 <= rec["shrinkage_pct"] <= 100, (
                    f"Shrinkage {rec['shrinkage_pct']} out of [0, 100] for {rec['entity']}"
                )
                # shrinkage_note field should exist
                assert "shrinkage_note" in rec

    def test_boundary_scores_warning(self):
        """M6: Boundary scores (0 or 100) should trigger a warning."""
        from qualitative_analysis.entity.bayesian import prepare_beta_data

        df = _make_scores_df(n_entities=3, n_participants=2, n_runs=2)
        # Inject boundary scores
        for col in [c for c in df.columns if c.startswith("social_run")]:
            df.at[df.index[0], col] = 0.0
            df.at[df.index[1], col] = 100.0

        with patch("qualitative_analysis.entity.bayesian.logger") as mock_logger:
            data = prepare_beta_data(df, ["social"])
            mock_logger.warning.assert_called()
            warning_msg = str(mock_logger.warning.call_args)
            assert "boundary" in warning_msg.lower()

    def test_marginal_contrast_output(self, ground_truth_df):
        """H2: Marginal contrasts should produce valid output with 'marginal' type."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        marginal = model.compute_marginal_group_contrasts()
        assert "social" in marginal
        for pair_key, c in marginal["social"].items():
            assert c["contrast_type"] == "marginal"
            assert "mean_diff" in c
            assert "hdi_lower" in c
            assert "p_a_gt_b" in c

    def test_logit_scale_contrast(self, ground_truth_df):
        """L4: Logit-scale contrasts should have 'logit' scale label."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        contrasts = model.compute_group_contrasts(scale="logit")
        for pair_key, c in contrasts["social"].items():
            assert c["scale"] == "logit"

        # Invalid scale should raise
        with pytest.raises(ValueError, match="scale must be"):
            model.compute_group_contrasts(scale="invalid")

    def test_model_cached_in_result(self, ground_truth_df):
        """L5: BayesianModelResult should cache the PyMC model."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        result = model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)
        assert result.model is not None, "Model should be cached in BayesianModelResult"

    def test_summarize_posteriors_structure(self, ground_truth_df):
        """Verify structure of summarize_posteriors output."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        summaries = model.summarize_posteriors()
        assert "social" in summaries
        s = summaries["social"]
        assert "parameters" in s
        assert "group_effects" in s
        assert "diagnostics" in s
        # Check parameter keys
        for var in ["mu_pop", "sigma_entity", "kappa"]:
            assert var in s["parameters"]
            p = s["parameters"][var]
            assert "mean" in p
            assert "std" in p
            assert "hdi_lower" in p
            assert "hdi_upper" in p
            assert "hdi_prob" in p
            assert p["hdi_prob"] == 0.94

    def test_contrast_includes_scale_field(self, ground_truth_df):
        """L4: Default contrasts should include a 'scale' field."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "high": [f"p_high_{i}" for i in range(4)],
            "low": [f"p_low_{i}" for i in range(4)],
        }
        model = BayesianEntityModel(
            scores_df=ground_truth_df,
            dimensions=["social"],
            groups=groups,
        )
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        contrasts = model.compute_group_contrasts()
        for pair_key, c in contrasts["social"].items():
            assert "scale" in c
            assert c["scale"] == "probability"


# ---------------------------------------------------------------------------
# Tests: Custom scale (1-10) support
# ---------------------------------------------------------------------------


def _make_custom_scale_df(
    scale_min: int = 1,
    scale_max: int = 10,
    n_entities: int = 5,
    n_participants_per_group: int = 3,
    n_runs: int = 3,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Generate synthetic data on a custom scale (default 1-10).

    Group "alpha" scores ~7.5 on social, ~4.0 on ecological.
    Group "beta"  scores ~4.0 on social, ~7.5 on ecological.
    """
    rng = np.random.RandomState(seed)

    groups = {
        "alpha": [f"p_alpha_{i}" for i in range(n_participants_per_group)],
        "beta": [f"p_beta_{i}" for i in range(n_participants_per_group)],
    }

    group_means = {
        "alpha": {"social": 7.5, "ecological": 4.0, "technological": 5.5},
        "beta": {"social": 4.0, "ecological": 7.5, "technological": 5.5},
    }

    entities = [f"ent_{i}" for i in range(n_entities)]
    dims = ["social", "ecological", "technological"]

    rows = []
    for gname, pids in groups.items():
        for pid in pids:
            for ent in entities:
                row = {
                    "text_id": pid,
                    "entity": ent,
                    "context": f"context for {ent}",
                    "group": gname,
                    "num_runs": n_runs,
                }
                for dim in dims:
                    mu = group_means[gname][dim]
                    entity_offset = rng.normal(0, 0.5)
                    runs = []
                    for k in range(1, n_runs + 1):
                        score = mu + entity_offset + rng.normal(0, 0.3)
                        score = max(scale_min + 0.1, min(scale_max - 0.1, round(score, 1)))
                        runs.append(score)
                        row[f"{dim}_run{k}"] = score
                    row[f"{dim}_mean"] = round(np.mean(runs), 2)
                    row[f"{dim}_std"] = round(np.std(runs, ddof=1) if len(runs) > 1 else 0.0, 2)
                rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def custom_scale_df():
    """Synthetic dataset on 1-10 scale."""
    return _make_custom_scale_df()


class TestCustomScale:
    """Tests that the pipeline handles non-default score scales correctly."""

    def test_prepare_beta_data_custom_scale(self, custom_scale_df):
        """S&V squeeze should normalize 1-10 scores to (0, 1)."""
        dims = ["social", "ecological", "technological"]
        data = prepare_beta_data(custom_scale_df, dims, scale_min=1, scale_max=10)
        y = data["long_df"]["y"]
        assert (y > 0).all(), "All y must be > 0"
        assert (y < 1).all(), "All y must be < 1"

    def test_prepare_beta_data_default_scale_unchanged(self, basic_df):
        """Default scale (0-100) should produce identical results to before."""
        dims = ["social", "ecological", "technological"]
        data_default = prepare_beta_data(basic_df, dims)
        data_explicit = prepare_beta_data(basic_df, dims, scale_min=0, scale_max=100)
        pd.testing.assert_frame_equal(data_default["long_df"], data_explicit["long_df"])

    def test_scale_config_properties(self):
        """ScaleConfig should compute correct derived values."""
        from qualitative_analysis.entity.models import ScaleConfig

        sc = ScaleConfig(scale_min=1, scale_max=10)
        assert sc.scale_range == 9
        assert sc.midpoint == 5.5
        assert abs(sc.rope_default() - 0.45) < 1e-10
        assert sc.fraction(1) == 0.0
        assert sc.fraction(10) == 1.0
        assert abs(sc.fraction(5.5) - 0.5) < 1e-10
        assert abs(sc.from_fraction(0.5) - 5.5) < 1e-10

    def test_dimension_score_ci_clamping_custom_scale(self):
        """CI should clamp to custom scale bounds, not hardcoded 0/100."""
        score = DimensionScore.from_scores(
            dimension="social",
            scores=[2, 3, 2, 3, 2],
            scale_min=1,
            scale_max=10,
        )
        assert score.confidence_interval_95[0] >= 1
        assert score.confidence_interval_95[1] <= 10

    def test_boundary_detection_custom_scale(self, custom_scale_df):
        """Boundary detection should use custom scale bounds, not 0/100."""
        # Add boundary scores at scale_min=1
        df = custom_scale_df.copy()
        df.iloc[0, df.columns.get_loc("social_run1")] = 1.0

        dims = ["social"]
        # Verify it doesn't crash and boundary detection uses the right values
        data = prepare_beta_data(df, dims, scale_min=1, scale_max=10)
        assert data is not None
        # The score=1.0 (boundary) should still be present in long_df
        assert len(data["long_df"]) > 0

    @pytest.mark.skipif(
        not _BAYESIAN_RUNTIME_AVAILABLE,
        reason=_BAYESIAN_RUNTIME_REASON,
    )
    def test_bayesian_model_custom_scale(self, custom_scale_df):
        """BayesianEntityModel should accept and use custom scale."""
        from qualitative_analysis.entity.bayesian import BayesianEntityModel

        groups = {
            "alpha": [f"p_alpha_{i}" for i in range(3)],
            "beta": [f"p_beta_{i}" for i in range(3)],
        }
        model = BayesianEntityModel(
            scores_df=custom_scale_df,
            dimensions=["social"],
            groups=groups,
            scale_min=1,
            scale_max=10,
        )
        assert model.scale_min == 1
        assert model.scale_max == 10
        assert model.scale_range == 9

        # Fit and verify posterior means are on the 1-10 scale
        model.fit("social", chains=2, draws=500, tune=300, sampler="nuts", random_seed=42)

        summary = model.summarize_posteriors()
        for gname, group_info in summary["social"]["group_effects"].items():
            mean_score = group_info["mean_score"]
            assert 1 <= mean_score <= 10, (
                f"Group {gname} mean {mean_score} should be on 1-10 scale"
            )
