"""
Bayesian hierarchical model for entity score comparison.

Implements a Beta-likelihood hierarchical model with:
- Entity-level random effects (non-centered)
- Participant-level random effects (non-centered)
- Group fixed effects
- Run-level precision via Beta concentration parameter

Each dimension is modeled independently. The model supports:
- Posterior group contrasts with HDI
- Per-dimension inference and P(direction)
- Variance decomposition (ICC)
- Entity-level shrinkage estimates
- ROPE analysis for practical significance

References:
    - Smithson & Verkuilen (2006), "A Better Lemon Squeezer"
    - Gelman & Hill (2007), Data Analysis Using Regression and
      Multilevel/Hierarchical Models
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_pymc_available() -> bool:
    """Check if PyMC and required Bayesian dependencies are installed."""
    try:
        import pymc  # noqa: F401
        import arviz  # noqa: F401
        return True
    except ImportError:
        return False


def _require_pymc():
    """Raise ImportError with install instructions if PyMC is not available."""
    if not check_pymc_available():
        raise ImportError(
            "Bayesian modeling requires PyMC and ArviZ. Install with:\n"
            '  pip install -e ".[bayes]"\n'
            "Or:\n"
            "  pip install 'pymc>=5.21' arviz nutpie"
        )


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def prepare_beta_data(
    scores_df: pd.DataFrame,
    dimensions: List[str],
    participant_col: str = "text_id",
    entity_col: str = "entity",
    group_col: str = "group",
    n_runs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare run-level data for Beta-likelihood Bayesian modeling.

    Reads ``{dim}_run{k}`` columns from the scored DataFrame, reshapes to
    long format (one row per observation = entity x participant x run),
    rescales scores from [0, 100] to the open interval (0, 1), and builds
    integer index arrays required by PyMC.

    Args:
        scores_df: DataFrame with scored entities. Must contain columns
            ``{dim}_run1``, ``{dim}_run2``, ... for each dimension.
        dimensions: List of dimension names (e.g. ["social", "ecological", "technological"]).
        participant_col: Column identifying participants.
        entity_col: Column identifying entities.
        group_col: Column identifying group membership.
        n_runs: Number of LLM scoring runs. If None, auto-detected from columns.

    Returns:
        Dictionary with keys:
            - ``long_df``: DataFrame in long format with columns
              [entity, participant, group, run, dimension, score, y,
               entity_idx, participant_idx, group_idx]
            - ``entity_names``: ordered list of entity names
            - ``participant_names``: ordered list of participant IDs
            - ``group_names``: ordered list of group labels
            - ``n_entities``: int
            - ``n_participants``: int
            - ``n_groups``: int
            - ``n_runs``: int
            - ``entity_map``: dict mapping entity name -> integer index
            - ``participant_map``: dict mapping participant ID -> integer index
            - ``group_map``: dict mapping group label -> integer index
            - ``participant_to_group``: dict mapping participant -> group index
    """

    df = scores_df.copy()

    # Auto-detect number of runs
    if n_runs is None:
        sample_dim = dimensions[0]
        run_cols = [c for c in df.columns if c.startswith(f"{sample_dim}_run")]
        n_runs = len(run_cols)
        if n_runs == 0:
            raise ValueError(
                f"No run-level columns found for dimension '{sample_dim}'. "
                f"Expected columns like '{sample_dim}_run1', '{sample_dim}_run2', etc. "
                "Re-score with the updated scorer to generate run-level data."
            )

    # Verify all dimensions have run columns
    for dim in dimensions:
        for k in range(1, n_runs + 1):
            col = f"{dim}_run{k}"
            if col not in df.columns:
                raise ValueError(f"Missing expected column: {col}")

    # Build long-format records: one row per (entity, participant, run, dimension)
    records = []
    for _, row in df.iterrows():
        entity = row[entity_col]
        participant = row[participant_col]
        group = row[group_col]
        for dim in dimensions:
            for k in range(1, n_runs + 1):
                score = row[f"{dim}_run{k}"]
                if pd.isna(score):
                    continue
                records.append({
                    "entity": entity,
                    "participant": participant,
                    "group": group,
                    "run": k - 1,
                    "dimension": dim,
                    "score": float(score),
                })

    long_df = pd.DataFrame(records)

    # Rescale [0, 100] -> (0, 1) using Smithson & Verkuilen (2006) squeeze.
    # N is computed per dimension (the sample size of the variable being modeled),
    # since each dimension is fit independently.
    eps = 1e-6
    dim_counts = long_df.groupby("dimension")["score"].transform("count")
    long_df["y"] = (long_df["score"] * (dim_counts - 1) + 0.5) / (dim_counts * 100.0)

    # Clip to ensure strictly within (0, 1)
    long_df["y"] = long_df["y"].clip(eps, 1.0 - eps)

    # Integer-encode indices (convert to plain Python str for PyMC/nutpie compat)
    entity_names = sorted(str(x) for x in long_df["entity"].unique())
    participant_names = sorted(str(x) for x in long_df["participant"].unique())
    group_names = sorted(str(x) for x in long_df["group"].unique())

    entity_map = {e: i for i, e in enumerate(entity_names)}
    participant_map = {p: i for i, p in enumerate(participant_names)}
    group_map = {g: i for i, g in enumerate(group_names)}

    long_df["entity_idx"] = long_df["entity"].map(entity_map).astype(int)
    long_df["participant_idx"] = long_df["participant"].map(participant_map).astype(int)
    long_df["group_idx"] = long_df["group"].map(group_map).astype(int)

    # Build participant -> group mapping
    participant_to_group = {}
    for _, row in df[[participant_col, group_col]].drop_duplicates().iterrows():
        pid = row[participant_col]
        gid = group_map[row[group_col]]
        participant_to_group[pid] = gid

    return {
        "long_df": long_df,
        "entity_names": entity_names,
        "participant_names": participant_names,
        "group_names": group_names,
        "n_entities": len(entity_names),
        "n_participants": len(participant_names),
        "n_groups": len(group_names),
        "n_runs": n_runs,
        "participant_to_group": participant_to_group,
        "entity_map": entity_map,
        "participant_map": participant_map,
        "group_map": group_map,
    }


# ---------------------------------------------------------------------------
# BayesianEntityModel
# ---------------------------------------------------------------------------

@dataclass
class BayesianModelResult:
    """Container for results from a single dimension's Bayesian model fit."""
    dimension: str
    trace: Any  # arviz.InferenceData
    diagnostics: Dict[str, Any]
    converged: bool


class BayesianEntityModel:
    """
    Bayesian hierarchical model for entity score comparison.

    For each dimension, fits a three-level hierarchical Beta model:
        Entity -> Participant (with group effect) -> Run

    Uses non-centered parameterization throughout for efficient MCMC
    sampling with large numbers of entities.

    Example::

        model = BayesianEntityModel(scores_df, dimensions, groups)
        results = model.fit_all_dimensions()
        contrasts = model.compute_group_contrasts()
        model.save_results(output_dir)
    """

    def __init__(
        self,
        scores_df: pd.DataFrame,
        dimensions: List[str],
        groups: Dict[str, List[str]],
        participant_col: str = "text_id",
        entity_col: str = "entity",
        group_col: str = "group",
        n_runs: Optional[int] = None,
    ):
        """
        Initialize the Bayesian model.

        Args:
            scores_df: DataFrame with scored entities including run-level columns.
            dimensions: List of dimension names to model.
            groups: Dict mapping group name -> list of participant IDs.
            participant_col: Column name for participant IDs.
            entity_col: Column name for entities.
            group_col: Column name for group membership.
            n_runs: Number of scoring runs (auto-detected if None).
        """
        _require_pymc()

        self.dimensions = dimensions
        self.groups = groups
        self.participant_col = participant_col
        self.entity_col = entity_col
        self.group_col = group_col

        # Prepare data
        self.data = prepare_beta_data(
            scores_df, dimensions, participant_col, entity_col, group_col, n_runs
        )

        # Store results per dimension
        self.results: Dict[str, BayesianModelResult] = {}

        logger.info(
            f"BayesianEntityModel initialized: "
            f"{self.data['n_entities']} entities, "
            f"{self.data['n_participants']} participants, "
            f"{self.data['n_groups']} groups, "
            f"{self.data['n_runs']} runs, "
            f"{len(dimensions)} dimensions"
        )

    def build_model(self, dimension: str) -> Any:
        """
        Build a PyMC model for a single dimension.

        The model structure (on logit scale):
            y_{i,j,k} ~ Beta(mu_{i,j} * kappa, (1 - mu_{i,j}) * kappa)
            logit(mu_{i,j}) = theta_i + group_effect_{g[j]} + z_{i,j} * sigma_participant
            theta_i = mu_population + z_entity_i * sigma_entity

        Args:
            dimension: Name of the dimension to model.

        Returns:
            PyMC Model object.
        """
        import pymc as pm

        dim_df = self.data["long_df"][
            self.data["long_df"]["dimension"] == dimension
        ].copy()

        if len(dim_df) == 0:
            raise ValueError(f"No observations found for dimension '{dimension}'")

        n_entities = self.data["n_entities"]
        n_participants = self.data["n_participants"]
        n_groups = self.data["n_groups"]

        # Extract observation-level index arrays
        entity_idx = dim_df["entity_idx"].values
        participant_idx = dim_df["participant_idx"].values
        group_idx = dim_df["group_idx"].values
        y_obs = dim_df["y"].values

        # Build participant -> group mapping array
        participant_group = np.zeros(n_participants, dtype=int)
        for pid, gid in self.data["participant_to_group"].items():
            pidx = self.data["participant_map"].get(pid)
            if pidx is not None:
                participant_group[pidx] = gid

        # Identify which (entity, participant) pairs exist in the data
        # For non-centered participant effects, we need a unique index
        ep_pairs = dim_df[["entity_idx", "participant_idx"]].drop_duplicates()
        ep_pairs = ep_pairs.sort_values(["entity_idx", "participant_idx"]).reset_index(drop=True)
        ep_map = {
            (row["entity_idx"], row["participant_idx"]): i
            for i, row in ep_pairs.iterrows()
        }
        n_ep = len(ep_pairs)
        ep_entity = ep_pairs["entity_idx"].values.astype(int)
        ep_participant = ep_pairs["participant_idx"].values.astype(int)

        # Map each observation to its (entity, participant) pair index
        obs_ep_idx = np.array([
            ep_map[(e, p)] for e, p in zip(entity_idx, participant_idx)
        ])

        # Ensure coordinates are plain Python lists (not ArrowStringArray)
        coords = {
            "entity": [str(e) for e in self.data["entity_names"]],
            "participant": [str(p) for p in self.data["participant_names"]],
            "group": [str(g) for g in self.data["group_names"]],
            "ep_pair": np.arange(n_ep),
            "obs": np.arange(len(y_obs)),
        }

        with pm.Model(coords=coords) as model:
            # --- Population-level ---
            mu_pop = pm.Normal("mu_pop", mu=0, sigma=1.5)

            # --- Entity-level (non-centered) ---
            sigma_entity = pm.HalfNormal("sigma_entity", sigma=2)
            z_entity = pm.Normal("z_entity", mu=0, sigma=1, dims="entity")
            theta = pm.Deterministic(
                "theta", mu_pop + z_entity * sigma_entity, dims="entity"
            )

            # --- Group effects ---
            sigma_group = pm.HalfNormal("sigma_group", sigma=0.5)
            group_effect = pm.Normal(
                "group_effect", mu=0, sigma=sigma_group, dims="group"
            )

            # --- Participant-level (non-centered) ---
            sigma_participant = pm.HalfNormal("sigma_participant", sigma=1)
            z_participant = pm.Normal(
                "z_participant", mu=0, sigma=1, dims="ep_pair"
            )

            # --- Linear predictor for each (entity, participant) pair ---
            eta_ep = (
                theta[ep_entity]
                + group_effect[participant_group[ep_participant]]
                + z_participant * sigma_participant
            )
            mu_ep = pm.math.invlogit(eta_ep)

            # Map to observation level
            mu_obs = mu_ep[obs_ep_idx]

            # --- Run-level precision ---
            kappa = pm.Gamma("kappa", alpha=5, beta=0.1)

            # --- Likelihood ---
            pm.Beta(
                "y_obs",
                alpha=mu_obs * kappa,
                beta=(1 - mu_obs) * kappa,
                observed=y_obs,
                dims="obs",
            )

        # Store metadata on the model for later use
        model._dimension = dimension
        model._dim_df = dim_df
        model._ep_pairs = ep_pairs
        model._ep_map = ep_map
        model._obs_ep_idx = obs_ep_idx
        model._participant_group = participant_group

        return model

    def fit(
        self,
        dimension: str,
        chains: int = 4,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.95,
        random_seed: int = 42,
        sampler: str = "nutpie",
    ) -> BayesianModelResult:
        """
        Fit the model for a single dimension via MCMC.

        Args:
            dimension: Dimension name to fit.
            chains: Number of MCMC chains.
            draws: Number of posterior draws per chain.
            tune: Number of tuning steps per chain.
            target_accept: Target acceptance rate for NUTS.
            random_seed: Random seed for reproducibility.
            sampler: Sampler backend: "nutpie" (faster) or "nuts" (standard).

        Returns:
            BayesianModelResult with trace and diagnostics.
        """
        import pymc as pm
        import arviz as az

        logger.info(f"Building model for dimension '{dimension}'...")
        model = self.build_model(dimension)

        logger.info(
            f"Sampling dimension '{dimension}': "
            f"{chains} chains, {draws} draws, {tune} tune, "
            f"target_accept={target_accept}, sampler={sampler}"
        )

        with model:
            sampler_kwargs = dict(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
                return_inferencedata=True,
            )

            if sampler == "nutpie":
                try:
                    import nutpie  # noqa: F401
                    sampler_kwargs["nuts_sampler"] = "nutpie"
                    logger.info("Using nutpie sampler")
                except ImportError:
                    logger.warning(
                        "nutpie not installed, falling back to default NUTS sampler"
                    )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                try:
                    trace = pm.sample(**sampler_kwargs)
                except RuntimeError as e:
                    if sampler_kwargs.get("nuts_sampler") == "nutpie":
                        logger.warning(
                            f"nutpie sampler failed ({e}), "
                            "falling back to PyMC NUTS"
                        )
                        sampler_kwargs.pop("nuts_sampler", None)
                        trace = pm.sample(**sampler_kwargs)
                    else:
                        raise

        # Run diagnostics
        diagnostics = self._run_diagnostics(trace, dimension)

        result = BayesianModelResult(
            dimension=dimension,
            trace=trace,
            diagnostics=diagnostics,
            converged=diagnostics["converged"],
        )
        self.results[dimension] = result

        status = "CONVERGED" if result.converged else "WARNING: convergence issues"
        logger.info(f"Dimension '{dimension}' sampling complete: {status}")

        return result

    def fit_all_dimensions(self, **kwargs) -> Dict[str, BayesianModelResult]:
        """
        Fit the model for all dimensions sequentially.

        Args:
            **kwargs: Passed to ``fit()`` (chains, draws, tune, etc.).

        Returns:
            Dict mapping dimension name -> BayesianModelResult.
        """
        for dim in self.dimensions:
            self.fit(dim, **kwargs)
        return self.results

    def _run_diagnostics(self, trace, dimension: str) -> Dict[str, Any]:
        """
        Compute convergence diagnostics for a fitted trace.

        Args:
            trace: ArviZ InferenceData object.
            dimension: Dimension name (for logging).

        Returns:
            Dict with rhat_max, ess_bulk_min, ess_tail_min, divergences, converged.
        """
        import arviz as az

        var_names = [
            "mu_pop", "sigma_entity", "sigma_group",
            "sigma_participant", "kappa", "group_effect",
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)

            # R-hat
            rhat = az.rhat(trace, var_names=var_names)
            rhat_vals = []
            for var in var_names:
                if var in rhat:
                    vals = rhat[var].values
                    if np.ndim(vals) == 0:
                        rhat_vals.append(float(vals))
                    else:
                        rhat_vals.extend(vals.flatten().tolist())
            rhat_max = max(rhat_vals) if rhat_vals else float("nan")

            # ESS
            ess_bulk = az.ess(trace, var_names=var_names, method="bulk")
            ess_tail = az.ess(trace, var_names=var_names, method="tail")

            ess_bulk_vals = []
            ess_tail_vals = []
            for var in var_names:
                if var in ess_bulk:
                    vals = ess_bulk[var].values
                    if np.ndim(vals) == 0:
                        ess_bulk_vals.append(float(vals))
                    else:
                        ess_bulk_vals.extend(vals.flatten().tolist())
                if var in ess_tail:
                    vals = ess_tail[var].values
                    if np.ndim(vals) == 0:
                        ess_tail_vals.append(float(vals))
                    else:
                        ess_tail_vals.extend(vals.flatten().tolist())

            ess_bulk_min = min(ess_bulk_vals) if ess_bulk_vals else 0.0
            ess_tail_min = min(ess_tail_vals) if ess_tail_vals else 0.0

        # Divergences
        divergences = 0
        if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats:
            divergences = int(trace.sample_stats["diverging"].values.sum())

        converged = (
            rhat_max < 1.01
            and ess_bulk_min > 400
            and ess_tail_min > 400
            and divergences == 0
        )

        diag = {
            "dimension": dimension,
            "rhat_max": round(rhat_max, 4),
            "ess_bulk_min": round(ess_bulk_min, 1),
            "ess_tail_min": round(ess_tail_min, 1),
            "divergences": divergences,
            "converged": converged,
        }

        logger.info(
            f"  Diagnostics [{dimension}]: R-hat max={diag['rhat_max']}, "
            f"ESS bulk min={diag['ess_bulk_min']}, "
            f"ESS tail min={diag['ess_tail_min']}, "
            f"divergences={divergences}"
        )

        return diag

    # ------------------------------------------------------------------
    # Posterior summaries
    # ------------------------------------------------------------------

    def summarize_posteriors(self) -> Dict[str, Any]:
        """
        Summarize posterior distributions for all fitted dimensions.

        Returns:
            Dict with per-dimension summaries including group means,
            variance parameters, and entity-level estimates.
        """
        import arviz as az

        summaries = {}
        for dim, result in self.results.items():
            trace = result.trace
            post = trace.posterior

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)

                # Scalar parameters
                scalar_vars = [
                    "mu_pop", "sigma_entity", "sigma_group",
                    "sigma_participant", "kappa",
                ]
                scalar_summary = {}
                for var in scalar_vars:
                    if var in post:
                        samples = post[var].values.flatten()
                        scalar_summary[var] = {
                            "mean": round(float(np.mean(samples)), 4),
                            "std": round(float(np.std(samples)), 4),
                            "hdi_3%": round(float(np.percentile(samples, 3)), 4),
                            "hdi_97%": round(float(np.percentile(samples, 97)), 4),
                        }

                # Group effects (logit scale)
                group_effects = {}
                if "group_effect" in post:
                    for i, gname in enumerate(self.data["group_names"]):
                        samples = post["group_effect"].values[:, :, i].flatten()
                        group_effects[gname] = {
                            "mean_logit": round(float(np.mean(samples)), 4),
                            "std_logit": round(float(np.std(samples)), 4),
                            "hdi_3%_logit": round(float(np.percentile(samples, 3)), 4),
                            "hdi_97%_logit": round(float(np.percentile(samples, 97)), 4),
                        }
                        # Also compute on probability scale (0-100)
                        mu_pop_samples = post["mu_pop"].values.flatten()
                        # Group mean on probability scale
                        group_logit = mu_pop_samples + samples
                        group_prob = 1 / (1 + np.exp(-group_logit)) * 100
                        group_effects[gname]["mean_score"] = round(float(np.mean(group_prob)), 2)
                        group_effects[gname]["hdi_3%_score"] = round(float(np.percentile(group_prob, 3)), 2)
                        group_effects[gname]["hdi_97%_score"] = round(float(np.percentile(group_prob, 97)), 2)

            summaries[dim] = {
                "parameters": scalar_summary,
                "group_effects": group_effects,
                "diagnostics": result.diagnostics,
            }

        return summaries

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def compute_group_contrasts(
        self, rope_delta: float = 5.0
    ) -> Dict[str, Any]:
        """
        Compute posterior group contrasts for all dimensions.

        For each pair of groups, computes:
        - Posterior mean difference (on 0-100 score scale)
        - 95% HDI of the difference
        - P(direction): probability that group A > group B
        - ROPE analysis: proportion of posterior within [-delta, +delta]

        Args:
            rope_delta: ROPE half-width on the 0-100 scale (default 5.0).

        Returns:
            Dict with per-dimension, per-pair contrast summaries.
        """
        contrasts = {}

        for dim, result in self.results.items():
            post = result.trace.posterior
            group_names = self.data["group_names"]
            n_groups = len(group_names)

            dim_contrasts = {}
            for i in range(n_groups):
                for j in range(i + 1, n_groups):
                    g_a = group_names[i]
                    g_b = group_names[j]

                    # Get group effect samples
                    eff_a = post["group_effect"].values[:, :, i].flatten()
                    eff_b = post["group_effect"].values[:, :, j].flatten()
                    mu_pop = post["mu_pop"].values.flatten()

                    # Convert to probability scale (0-100)
                    prob_a = 1 / (1 + np.exp(-(mu_pop + eff_a))) * 100
                    prob_b = 1 / (1 + np.exp(-(mu_pop + eff_b))) * 100
                    delta = prob_a - prob_b

                    pair_key = f"{g_a}_vs_{g_b}"
                    dim_contrasts[pair_key] = {
                        "group_a": g_a,
                        "group_b": g_b,
                        "mean_diff": round(float(np.mean(delta)), 3),
                        "std_diff": round(float(np.std(delta)), 3),
                        "hdi_3%": round(float(np.percentile(delta, 3)), 3),
                        "hdi_97%": round(float(np.percentile(delta, 97)), 3),
                        "p_a_gt_b": round(float(np.mean(delta > 0)), 4),
                        "p_b_gt_a": round(float(np.mean(delta < 0)), 4),
                        "rope_delta": rope_delta,
                        "p_in_rope": round(
                            float(np.mean(np.abs(delta) < rope_delta)), 4
                        ),
                        "p_above_rope": round(
                            float(np.mean(delta > rope_delta)), 4
                        ),
                        "p_below_rope": round(
                            float(np.mean(delta < -rope_delta)), 4
                        ),
                    }

            contrasts[dim] = dim_contrasts

        return contrasts

    def compute_icc(self) -> Dict[str, Any]:
        """
        Compute intraclass correlation coefficients (variance decomposition).

        Decomposes total variance into:
        - Between-group (sigma_group^2)
        - Between-entity (sigma_entity^2)
        - Between-participant (sigma_participant^2)
        - Within-run (derived from kappa)

        Returns:
            Dict with per-dimension ICC values.
        """
        icc_results = {}

        for dim, result in self.results.items():
            post = result.trace.posterior

            # Extract variance components (on logit scale)
            sig_group = post["sigma_group"].values.flatten()
            sig_entity = post["sigma_entity"].values.flatten()
            sig_participant = post["sigma_participant"].values.flatten()
            kappa_vals = post["kappa"].values.flatten()

            var_group = sig_group ** 2
            var_entity = sig_entity ** 2
            var_participant = sig_participant ** 2
            # Run-level variance from Beta: Var(y) ~ mu(1-mu)/(kappa+1)
            # On logit scale, approximate as pi^2 / (3 * kappa)
            var_run = (np.pi ** 2) / (3.0 * kappa_vals)

            total_var = var_group + var_entity + var_participant + var_run

            icc_group = var_group / total_var
            icc_entity = var_entity / total_var
            icc_participant = var_participant / total_var
            icc_run = var_run / total_var

            icc_results[dim] = {
                "icc_group": {
                    "mean": round(float(np.mean(icc_group)), 4),
                    "hdi_3%": round(float(np.percentile(icc_group, 3)), 4),
                    "hdi_97%": round(float(np.percentile(icc_group, 97)), 4),
                },
                "icc_entity": {
                    "mean": round(float(np.mean(icc_entity)), 4),
                    "hdi_3%": round(float(np.percentile(icc_entity, 3)), 4),
                    "hdi_97%": round(float(np.percentile(icc_entity, 97)), 4),
                },
                "icc_participant": {
                    "mean": round(float(np.mean(icc_participant)), 4),
                    "hdi_3%": round(float(np.percentile(icc_participant, 3)), 4),
                    "hdi_97%": round(float(np.percentile(icc_participant, 97)), 4),
                },
                "icc_run": {
                    "mean": round(float(np.mean(icc_run)), 4),
                    "hdi_3%": round(float(np.percentile(icc_run, 3)), 4),
                    "hdi_97%": round(float(np.percentile(icc_run, 97)), 4),
                },
                "variance_components": {
                    "sigma_group": round(float(np.mean(sig_group)), 4),
                    "sigma_entity": round(float(np.mean(sig_entity)), 4),
                    "sigma_participant": round(float(np.mean(sig_participant)), 4),
                    "kappa": round(float(np.mean(kappa_vals)), 4),
                },
            }

        return icc_results

    def compute_shrinkage(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Compare raw entity means to posterior (shrunken) estimates.

        Entities with few observations are shrunk more toward the population
        mean — a key advantage of hierarchical modeling.

        Returns:
            Dict with per-dimension shrinkage data (entity, raw_mean,
            posterior_mean, posterior_sd, shrinkage_pct).
        """
        shrinkage_results = {}

        for dim, result in self.results.items():
            post = result.trace.posterior
            dim_df = self.data["long_df"][
                self.data["long_df"]["dimension"] == dim
            ]

            # Raw entity means (on 0-100 scale)
            raw_means = dim_df.groupby("entity")["score"].mean()

            # Posterior entity means (theta on logit scale -> probability scale)
            theta_samples = post["theta"].values  # (chains, draws, n_entities)
            theta_mean = theta_samples.mean(axis=(0, 1))  # (n_entities,)
            theta_sd = theta_samples.std(axis=(0, 1))

            # Convert to probability scale (0-100)
            posterior_prob = 1 / (1 + np.exp(-theta_mean)) * 100
            # Delta method: Var(g(θ)) ≈ g'(θ_mean)² * Var(θ)
            # where g = sigmoid, g' = sigmoid(θ)(1 - sigmoid(θ))
            sigmoid_mean = 1 / (1 + np.exp(-theta_mean))
            posterior_sd_prob = theta_sd * sigmoid_mean * (1 - sigmoid_mean) * 100

            entity_names = self.data["entity_names"]
            records = []
            for i, ename in enumerate(entity_names):
                raw = raw_means.get(ename, np.nan)
                post_mean = posterior_prob[i]
                post_sd = posterior_sd_prob[i]

                # Shrinkage: how much the posterior moved from the raw
                if not np.isnan(raw) and raw != 0:
                    shrinkage_pct = round(
                        abs(post_mean - raw) / abs(raw - 50) * 100
                        if abs(raw - 50) > 1 else 0.0, 2
                    )
                else:
                    shrinkage_pct = 0.0

                records.append({
                    "entity": ename,
                    "raw_mean": round(float(raw), 2) if not np.isnan(raw) else None,
                    "posterior_mean": round(float(post_mean), 2),
                    "posterior_sd": round(float(post_sd), 2),
                    "shrinkage_pct": shrinkage_pct,
                    "n_observations": int(
                        dim_df[dim_df["entity"] == ename].shape[0]
                    ),
                })

            shrinkage_results[dim] = records

        return shrinkage_results

    def prior_predictive_check(self, dimension: str, draws: int = 500) -> Any:
        """
        Generate prior predictive samples.

        Args:
            dimension: Dimension to check.
            draws: Number of prior predictive draws.

        Returns:
            ArviZ InferenceData with prior predictive samples.
        """
        import pymc as pm

        model = self.build_model(dimension)
        with model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                prior_pred = pm.sample_prior_predictive(draws=draws)
        return prior_pred

    def posterior_predictive_check(self, dimension: str) -> Any:
        """
        Generate posterior predictive samples for a fitted dimension.

        Args:
            dimension: Dimension to check (must be already fitted).

        Returns:
            ArviZ InferenceData with posterior predictive samples.
        """
        import pymc as pm

        if dimension not in self.results:
            raise ValueError(f"Dimension '{dimension}' not yet fitted")

        model = self.build_model(dimension)
        trace = self.results[dimension].trace

        with model:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                ppc = pm.sample_posterior_predictive(trace)
        return ppc

    # ------------------------------------------------------------------
    # Output / serialization
    # ------------------------------------------------------------------

    def save_results(
        self,
        output_dir: Union[str, Path],
        rope_delta: float = 5.0,
        save_trace: bool = False,
    ) -> None:
        """
        Save all Bayesian results to the output directory.

        Creates:
            - model_summary.json
            - group_contrasts.json
            - icc_decomposition.json
            - shrinkage_estimates.csv

        Args:
            output_dir: Output directory path.
            rope_delta: ROPE half-width for group contrasts.
            save_trace: If True, save InferenceData as netCDF files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Model summary
        summaries = self.summarize_posteriors()
        with open(output_dir / "model_summary.json", "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        logger.info(f"Saved model summary to {output_dir / 'model_summary.json'}")

        # Group contrasts
        contrasts = self.compute_group_contrasts(rope_delta=rope_delta)
        with open(output_dir / "group_contrasts.json", "w") as f:
            json.dump(contrasts, f, indent=2, default=str)
        logger.info(f"Saved group contrasts to {output_dir / 'group_contrasts.json'}")

        # ICC decomposition
        icc = self.compute_icc()
        with open(output_dir / "icc_decomposition.json", "w") as f:
            json.dump(icc, f, indent=2, default=str)
        logger.info(f"Saved ICC decomposition to {output_dir / 'icc_decomposition.json'}")

        # Shrinkage estimates
        shrinkage = self.compute_shrinkage()
        import csv
        for dim, records in shrinkage.items():
            csv_path = output_dir / f"shrinkage_estimates_{dim}.csv"
            if records:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    writer.writeheader()
                    writer.writerows(records)
                logger.info(f"Saved shrinkage estimates to {csv_path}")

        # Save traces as netCDF (optional)
        if save_trace:
            trace_dir = output_dir / "trace"
            trace_dir.mkdir(exist_ok=True)
            for dim, result in self.results.items():
                nc_path = trace_dir / f"dim_{dim}.nc"
                result.trace.to_netcdf(str(nc_path))
                logger.info(f"Saved trace to {nc_path}")


# ---------------------------------------------------------------------------
# BayesianVisualizer
# ---------------------------------------------------------------------------

class BayesianVisualizer:
    """
    Visualization tools for Bayesian model results.

    Produces publication-quality figures from BayesianEntityModel outputs:
    - Group contrast forest plots
    - Posterior density plots
    - ICC decomposition bar charts
    - Shrinkage scatter plots
    - Posterior predictive check plots
    """

    def __init__(self, model: BayesianEntityModel) -> None:
        """
        Initialize the visualizer.

        Args:
            model: A fitted BayesianEntityModel instance.
        """
        self.model = model
        # Use consistent color palette matching ComparisonVisualizer
        self._group_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]

    def _get_group_color(self, idx: int) -> str:
        """Return a hex color string for the group at the given index."""
        return self._group_colors[idx % len(self._group_colors)]

    def plot_group_contrasts(
        self,
        output_path: Union[str, Path],
        rope_delta: float = 5.0,
        figsize: Tuple[float, float] = (10, 8),
    ) -> None:
        """
        Generate a forest plot showing per-dimension group contrasts with HDI.

        Args:
            output_path: Path to save the figure.
            rope_delta: ROPE half-width (shaded region on plot).
            figsize: Figure dimensions.
        """
        import matplotlib.pyplot as plt

        contrasts = self.model.compute_group_contrasts(rope_delta=rope_delta)
        dimensions = list(contrasts.keys())
        n_dims = len(dimensions)

        # Collect all pair keys (consistent across dimensions)
        pair_keys = list(contrasts[dimensions[0]].keys())
        n_pairs = len(pair_keys)

        fig, axes = plt.subplots(1, n_dims, figsize=figsize, sharey=True)
        if n_dims == 1:
            axes = [axes]

        for d_idx, dim in enumerate(dimensions):
            ax = axes[d_idx]
            dim_data = contrasts[dim]

            y_positions = np.arange(n_pairs)

            for p_idx, pair_key in enumerate(pair_keys):
                c = dim_data[pair_key]
                mean_diff = c["mean_diff"]
                hdi_lo = c["hdi_3%"]
                hdi_hi = c["hdi_97%"]

                color = self._get_group_color(p_idx)
                label = f"{c['group_a']} vs {c['group_b']}"

                ax.errorbar(
                    mean_diff, p_idx,
                    xerr=[[mean_diff - hdi_lo], [hdi_hi - mean_diff]],
                    fmt="o", color=color, capsize=5, markersize=8,
                    label=label, linewidth=2,
                )

            # ROPE region
            ax.axvspan(-rope_delta, rope_delta, alpha=0.1, color="gray",
                       label=f"ROPE (±{rope_delta})")
            ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

            ax.set_yticks(y_positions)
            ax.set_yticklabels([
                f"{dim_data[pk]['group_a']} vs\n{dim_data[pk]['group_b']}"
                for pk in pair_keys
            ])
            ax.set_xlabel("Difference (0-100 scale)")
            ax.set_title(dim.capitalize())
            ax.legend(loc="best", fontsize=8)

        fig.suptitle("Group Contrasts: Posterior Differences with 94% HDI",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved group contrast forest plot to {output_path}")

    def plot_posterior_densities(
        self,
        output_path: Union[str, Path],
        figsize: Tuple[float, float] = (12, 4),
    ) -> None:
        """
        Plot overlaid posterior density per group per dimension.

        Shows the distribution of group-level mean scores on the 0-100 scale.

        Args:
            output_path: Path to save the figure.
            figsize: Figure dimensions.
        """
        import matplotlib.pyplot as plt

        group_names = self.model.data["group_names"]
        dimensions = list(self.model.results.keys())
        n_dims = len(dimensions)

        fig, axes = plt.subplots(1, n_dims, figsize=figsize, sharey=True)
        if n_dims == 1:
            axes = [axes]

        for d_idx, dim in enumerate(dimensions):
            ax = axes[d_idx]
            post = self.model.results[dim].trace.posterior

            mu_pop = post["mu_pop"].values.flatten()

            for g_idx, gname in enumerate(group_names):
                eff = post["group_effect"].values[:, :, g_idx].flatten()
                # Convert to probability scale (0-100)
                group_prob = 1 / (1 + np.exp(-(mu_pop + eff))) * 100

                color = self._get_group_color(g_idx)
                ax.hist(group_prob, bins=50, alpha=0.4, color=color,
                        label=gname, density=True)
                ax.axvline(np.mean(group_prob), color=color, linewidth=2,
                           linestyle="--")

            ax.set_xlabel("Score (0-100)")
            if d_idx == 0:
                ax.set_ylabel("Density")
            ax.set_title(dim.capitalize())
            ax.legend(fontsize=8)

        fig.suptitle("Posterior Group Mean Distributions",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved posterior densities to {output_path}")

    def plot_icc_decomposition(
        self,
        output_path: Union[str, Path],
        figsize: Tuple[float, float] = (8, 5),
    ) -> None:
        """
        Generate a stacked bar chart showing variance decomposition (ICC).

        Args:
            output_path: Path to save the figure.
            figsize: Figure dimensions.
        """
        import matplotlib.pyplot as plt

        icc = self.model.compute_icc()
        dimensions = list(icc.keys())

        levels = ["icc_group", "icc_entity", "icc_participant", "icc_run"]
        level_labels = ["Between-group", "Between-entity",
                        "Between-participant", "Within-run"]
        colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(dimensions))
        width = 0.5
        bottom = np.zeros(len(dimensions))

        for level, label, color in zip(levels, level_labels, colors):
            values = [icc[dim][level]["mean"] for dim in dimensions]
            ax.bar(x, values, width, bottom=bottom, label=label, color=color)
            bottom += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in dimensions])
        ax.set_ylabel("Proportion of Total Variance")
        ax.set_title("Variance Decomposition (ICC) by Dimension",
                      fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved ICC decomposition plot to {output_path}")

    def plot_shrinkage(
        self,
        output_path: Union[str, Path],
        figsize: Tuple[float, float] = (8, 8),
    ) -> None:
        """
        Scatter plot of raw vs posterior entity means with 45-degree reference.

        Points farther from the diagonal show more shrinkage toward the
        population mean.

        Args:
            output_path: Path to save the figure.
            figsize: Figure dimensions.
        """
        import matplotlib.pyplot as plt

        shrinkage = self.model.compute_shrinkage()
        dimensions = list(shrinkage.keys())
        n_dims = len(dimensions)

        fig, axes = plt.subplots(1, n_dims, figsize=figsize)
        if n_dims == 1:
            axes = [axes]

        for d_idx, dim in enumerate(dimensions):
            ax = axes[d_idx]
            records = shrinkage[dim]

            raw = [r["raw_mean"] for r in records if r["raw_mean"] is not None]
            post = [r["posterior_mean"] for r in records if r["raw_mean"] is not None]
            n_obs = [r["n_observations"] for r in records if r["raw_mean"] is not None]

            scatter = ax.scatter(raw, post, c=n_obs, cmap="viridis",
                                 alpha=0.5, s=15, edgecolors="none")
            ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1)

            ax.set_xlabel("Raw Mean (0-100)")
            if d_idx == 0:
                ax.set_ylabel("Posterior Mean (0-100)")
            ax.set_title(dim.capitalize())
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_aspect("equal")

            plt.colorbar(scatter, ax=ax, label="N observations", shrink=0.7)

        fig.suptitle("Shrinkage: Raw vs Posterior Entity Means",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved shrinkage plot to {output_path}")

    def plot_trace(
        self,
        output_path: Union[str, Path],
        dimension: str,
        figsize: Tuple[float, float] = (14, 10),
    ) -> None:
        """
        Generate trace plots for key parameters (diagnostic visualization).

        Args:
            output_path: Path to save the figure.
            dimension: Dimension to plot traces for.
            figsize: Figure dimensions.
        """
        import arviz as az
        import matplotlib.pyplot as plt

        if dimension not in self.model.results:
            raise ValueError(f"Dimension '{dimension}' not fitted")

        trace = self.model.results[dimension].trace
        var_names = [
            "mu_pop", "sigma_entity", "sigma_group",
            "sigma_participant", "kappa", "group_effect",
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            az.plot_trace(trace, var_names=var_names, figsize=figsize)

        plt.suptitle(f"Trace Plots: {dimension.capitalize()}",
                      fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved trace plots to {output_path}")

    def plot_posterior_predictive(
        self,
        output_path: Union[str, Path],
        figsize: Tuple[float, float] = (12, 4),
    ) -> None:
        """
        Generate posterior predictive check plot for all fitted dimensions.

        Shows observed score distribution overlaid with replicated distributions
        from the posterior. Good calibration means the replicated data envelopes
        the observed data.

        Args:
            output_path: Path to save the figure.
            figsize: Figure dimensions (width per panel, height).
        """
        import matplotlib.pyplot as plt

        dimensions = list(self.model.results.keys())
        n_dims = len(dimensions)

        fig_w = figsize[0] * n_dims / 3 if n_dims != 3 else figsize[0]
        fig, axes = plt.subplots(1, n_dims, figsize=(fig_w, figsize[1]))
        if n_dims == 1:
            axes = [axes]

        for d_idx, dim in enumerate(dimensions):
            ax = axes[d_idx]

            # Get observed data for this dimension
            long_df = self.model.data["long_df"]
            obs_mask = long_df["dimension"] == dim
            obs_scores = long_df.loc[obs_mask, "score"].values  # original 0-100 scale

            # Generate posterior predictive samples
            try:
                ppc = self.model.posterior_predictive_check(dim)
                # Extract replicated y values, convert back to 0-100 scale
                if hasattr(ppc, "posterior_predictive") and "y_obs" in ppc.posterior_predictive:
                    rep_samples = ppc.posterior_predictive["y_obs"].values
                    # rep_samples shape: (chains, draws, n_obs)
                    # Flatten chains/draws, take a few replicates for overlay
                    n_chains, n_draws = rep_samples.shape[:2]
                    flat = rep_samples.reshape(n_chains * n_draws, -1)

                    # Plot a subset of replicated datasets (thin lines)
                    n_rep = min(100, flat.shape[0])
                    rep_indices = np.linspace(0, flat.shape[0] - 1, n_rep, dtype=int)
                    for r_idx in rep_indices:
                        rep_scores = flat[r_idx] * 100  # back to 0-100
                        ax.hist(
                            rep_scores, bins=30, range=(0, 100),
                            alpha=0.03, color="#1f77b4", density=True,
                        )

                    # Plot observed
                    ax.hist(
                        obs_scores, bins=30, range=(0, 100),
                        alpha=0.8, color="#d62728", density=True,
                        linewidth=1.5, histtype="step", label="Observed",
                    )

                    ax.set_xlabel("Score (0-100)")
                    if d_idx == 0:
                        ax.set_ylabel("Density")
                    ax.set_title(dim.capitalize())
                    ax.legend(fontsize=8)
                else:
                    ax.text(
                        0.5, 0.5, "PPC data\nnot available",
                        ha="center", va="center", transform=ax.transAxes,
                    )
                    ax.set_title(dim.capitalize())
            except Exception as e:
                logger.warning(f"Posterior predictive check failed for {dim}: {e}")
                ax.text(
                    0.5, 0.5, f"PPC failed:\n{e}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_title(dim.capitalize())

        fig.suptitle(
            "Posterior Predictive Check: Observed vs Replicated",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved posterior predictive check plot to {output_path}")

    def plot_group_ternary(
        self,
        output_path: Union[str, Path],
        dimension_names: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (12, 10),
        n_samples: int = 200,
    ) -> None:
        """
        Generate ternary plot with group credible region ellipses.

        Shows group-level posterior means as centroids with 95% credible
        region ellipses computed from the posterior distribution.

        Requires exactly 3 fitted dimensions.

        Args:
            output_path: Path to save the figure.
            dimension_names: Optional list of 3 dimension names (order matters
                for axis assignment). Defaults to fitted dimensions.
            figsize: Figure dimensions.
            n_samples: Number of posterior samples to draw for ellipse estimation.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms

        fitted_dims = list(self.model.results.keys())
        dims = dimension_names or fitted_dims

        if len(dims) != 3:
            logger.warning(
                f"Ternary plot requires exactly 3 dimensions, got {len(dims)}. Skipping."
            )
            return

        for d in dims:
            if d not in self.model.results:
                logger.warning(f"Dimension '{d}' not fitted. Skipping ternary plot.")
                return

        group_names = self.model.data["group_names"]

        # Extract posterior group means on probability scale (0-100) for each dimension
        # Shape: (n_samples, n_groups, 3_dimensions)
        group_scores = {}
        for g_idx, gname in enumerate(group_names):
            dim_samples = []
            for dim in dims:
                post = self.model.results[dim].trace.posterior
                mu_pop = post["mu_pop"].values.flatten()
                eff = post["group_effect"].values[:, :, g_idx].flatten()
                prob = 1 / (1 + np.exp(-(mu_pop + eff))) * 100
                dim_samples.append(prob)
            # dim_samples[i] has shape (n_total_samples,)
            group_scores[gname] = np.column_stack(dim_samples)  # (n_samples, 3)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Draw triangle
        vertices = np.array([
            [0, 0], [1, 0], [0.5, np.sqrt(3) / 2], [0, 0]
        ])
        ax.plot(vertices[:, 0], vertices[:, 1], "k-", linewidth=2)

        # Grid
        for i in range(1, 10):
            alpha = 0.3 if i % 2 == 0 else 0.15
            for start_a, end_a in [
                ((1 - i / 10, i / 10, 0), (0, i / 10, 1 - i / 10)),
                ((i / 10, 0, 1 - i / 10), (i / 10, 1 - i / 10, 0)),
                ((0, 1 - i / 10, i / 10), (1 - i / 10, 0, i / 10)),
            ]:
                x1, y1 = self._bary_to_cart(*start_a)
                x2, y2 = self._bary_to_cart(*end_a)
                ax.plot([x1, x2], [y1, y2], "k:", alpha=alpha, linewidth=0.5)

        # Labels
        label_offset = 0.08
        ax.text(0, -label_offset, dims[2].capitalize(),
                ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(1, -label_offset, dims[0].capitalize(),
                ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(0.5, np.sqrt(3) / 2 + label_offset, dims[1].capitalize(),
                ha="center", va="bottom", fontsize=12, fontweight="bold")

        # For each group, plot posterior samples and credible ellipse
        for g_idx, gname in enumerate(group_names):
            color = self._get_group_color(g_idx)
            scores = group_scores[gname]

            # Subsample for plotting
            idx_sub = np.random.choice(len(scores), min(n_samples, len(scores)), replace=False)
            sub = scores[idx_sub]

            # Normalize to sum-to-1 for ternary
            totals = sub.sum(axis=1, keepdims=True)
            totals = np.where(totals == 0, 1, totals)
            normed = sub / totals

            # Convert to cartesian
            xs, ys = [], []
            for row in normed:
                x, y = self._bary_to_cart(row[1], row[2], row[0])
                xs.append(x)
                ys.append(y)
            xs, ys = np.array(xs), np.array(ys)

            # Plot posterior samples as small dots
            ax.scatter(xs, ys, s=5, alpha=0.15, color=color)

            # Compute centroid
            cx, cy = np.mean(xs), np.mean(ys)
            ax.scatter(cx, cy, s=200, marker="X", color=color,
                       edgecolors="black", linewidths=1.5, zorder=10,
                       label=gname)

            # 95% credible ellipse
            cov = np.cov(xs, ys)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # 95% confidence: chi2(2) = 5.991
            chi2_val = 5.991
            width = 2 * np.sqrt(chi2_val * eigenvalues[0])
            height = 2 * np.sqrt(chi2_val * eigenvalues[1])
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

            ellipse = Ellipse(
                xy=(cx, cy), width=width, height=height, angle=angle,
                facecolor=color, alpha=0.15, edgecolor=color,
                linewidth=2, linestyle="--",
            )
            ax.add_patch(ellipse)

        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.set_title(
            "Group Posterior Centroids with 95% Credible Regions",
            fontsize=14, fontweight="bold", pad=20,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved group ternary plot to {output_path}")

    @staticmethod
    def _bary_to_cart(a: float, b: float, c: float) -> Tuple[float, float]:
        """
        Convert barycentric coordinates to 2D cartesian.

        Args:
            a: Weight for top vertex.
            b: Weight for bottom-left vertex.
            c: Weight for bottom-right vertex.

        Returns:
            (x, y) cartesian coordinates.
        """
        vertices = np.array([
            [0, 0],               # Bottom-left
            [1, 0],               # Bottom-right
            [0.5, np.sqrt(3) / 2] # Top
        ])
        x = a * vertices[2, 0] + b * vertices[0, 0] + c * vertices[1, 0]
        y = a * vertices[2, 1] + b * vertices[0, 1] + c * vertices[1, 1]
        return x, y

    def generate_all(
        self,
        output_dir: Union[str, Path],
        rope_delta: float = 5.0,
    ) -> None:
        """
        Generate all visualizations and save to output directory.

        Args:
            output_dir: Directory to save all plots.
            rope_delta: ROPE half-width for contrast plots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_group_contrasts(
            output_dir / "group_contrast_forest.png",
            rope_delta=rope_delta,
        )
        self.plot_posterior_densities(
            output_dir / "posterior_densities.png",
        )
        self.plot_icc_decomposition(
            output_dir / "icc_decomposition.png",
        )
        self.plot_shrinkage(
            output_dir / "shrinkage_plot.png",
        )

        # Posterior predictive check
        try:
            self.plot_posterior_predictive(
                output_dir / "posterior_predictive_check.png",
            )
        except Exception as e:
            logger.warning(f"Posterior predictive check plot failed: {e}")

        # Group ternary with credible regions (only if 3 dimensions)
        fitted_dims = list(self.model.results.keys())
        if len(fitted_dims) == 3:
            try:
                self.plot_group_ternary(
                    output_dir / "group_ternary_credible.png",
                )
            except Exception as e:
                logger.warning(f"Group ternary plot failed: {e}")

        # Trace plots per dimension
        for dim in self.model.results:
            self.plot_trace(
                output_dir / f"trace_{dim}.png",
                dimension=dim,
            )

        logger.info(f"All Bayesian visualizations saved to {output_dir}")
