"""
Entity analysis module for the qualitative-analysis package.

This package exposes scoring, comparison, detection, visualization, and
optional Bayesian/consolidation helpers. Heavier optional modules are loaded
on demand so lightweight imports do not trigger matplotlib, embedding, or
Bayesian dependency initialization.
"""

from importlib import import_module
from importlib.util import find_spec
from typing import Any, Dict, Tuple

from qualitative_analysis.entity.models import (
    ConsolidationResult,
    DimensionDefinition,
    DimensionScore,
    DimensionSet,
    EntityConsolidation,
    EntityScore,
    EntityScoreResult,
    SingleRunScore,
)
from qualitative_analysis.entity.scorer import EntityScorer, score_entities
from qualitative_analysis.entity.comparison import (
    ComparisonResult,
    GroupComparisonResult,
    ParticipantComparison,
    aitchison_distance,
    clr_transform,
    compare_participants,
    compute_pairwise_distances,
    ilr_transform,
    wasserstein_distance_compositional,
)
from qualitative_analysis.entity.agreement import (
    AgreementResult,
    compute_agreement,
    krippendorff_alpha,
)
from qualitative_analysis.entity.clustering import (
    ClusteringResult,
    PCAResult,
    cluster_hierarchical,
    cluster_kmeans,
    cluster_participants,
    pca_on_scores,
)
from qualitative_analysis.entity.detector import (
    EntityDetectionResult,
    EntityDetector,
)

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "EntityVisualizer": ("qualitative_analysis.entity.visualizer", "EntityVisualizer"),
    "EntityConsolidator": ("qualitative_analysis.entity.consolidator", "EntityConsolidator"),
    "consolidate_entities": ("qualitative_analysis.entity.consolidator", "consolidate_entities"),
    "ComparisonVisualizer": ("qualitative_analysis.entity.comparison_viz", "ComparisonVisualizer"),
    "cluster_hdbscan": ("qualitative_analysis.entity.clustering", "cluster_hdbscan"),
    "BayesianEntityModel": ("qualitative_analysis.entity.bayesian", "BayesianEntityModel"),
    "BayesianVisualizer": ("qualitative_analysis.entity.bayesian", "BayesianVisualizer"),
    "BayesianModelResult": ("qualitative_analysis.entity.bayesian", "BayesianModelResult"),
    "check_pymc_available": ("qualitative_analysis.entity.bayesian", "check_pymc_available"),
    "validate_bayesian_runtime": ("qualitative_analysis.entity.bayesian", "validate_bayesian_runtime"),
    "prepare_beta_data": ("qualitative_analysis.entity.bayesian", "prepare_beta_data"),
}

__all__ = [
    "DimensionDefinition",
    "DimensionSet",
    "DimensionScore",
    "SingleRunScore",
    "EntityScore",
    "EntityScoreResult",
    "EntityConsolidation",
    "ConsolidationResult",
    "EntityScorer",
    "score_entities",
    "ParticipantComparison",
    "ComparisonResult",
    "GroupComparisonResult",
    "aitchison_distance",
    "wasserstein_distance_compositional",
    "compute_pairwise_distances",
    "compare_participants",
    "clr_transform",
    "ilr_transform",
    "AgreementResult",
    "compute_agreement",
    "krippendorff_alpha",
    "ClusteringResult",
    "PCAResult",
    "cluster_participants",
    "cluster_hierarchical",
    "cluster_kmeans",
    "pca_on_scores",
    "EntityDetector",
    "EntityDetectionResult",
    "EntityVisualizer",
    "EntityConsolidator",
    "consolidate_entities",
    "ComparisonVisualizer",
]

if find_spec("hdbscan") is not None:
    __all__.append("cluster_hdbscan")

if find_spec("pymc") is not None and find_spec("arviz") is not None:
    __all__.extend(
        [
            "BayesianEntityModel",
            "BayesianVisualizer",
            "BayesianModelResult",
            "check_pymc_available",
            "validate_bayesian_runtime",
            "prepare_beta_data",
        ]
    )


def __getattr__(name: str) -> Any:
    """Lazily import optional/heavy entity helpers on first access."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise AttributeError(
            f"module {__name__!r} could not load optional attribute {name!r}"
        ) from exc

    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazily available attributes in interactive environments."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_IMPORTS))
