"""
Entity analysis module for the qualitative-analysis package.

This module provides functionality for:
- Entity consolidation/deduplication using semantic similarity
- Multi-dimensional entity scoring with uncertainty quantification
- Entity visualization (ternary plots, radar charts, etc.)

Example usage:
    from qualitative_analysis.entity import EntityConsolidator, EntityScorer
    from qualitative_analysis.entity.models import DimensionDefinition

    # Consolidate similar entities
    consolidator = EntityConsolidator()
    clusters = consolidator.consolidate(entities, threshold=0.85)

    # Score entities on dimensions
    dimensions = [
        DimensionDefinition(
            name="Social",
            description="Degree of social/human factors",
            min_anchor="No social component",
            max_anchor="Fundamentally social"
        ),
        # ... more dimensions
    ]
    scorer = EntityScorer(llm_provider, dimensions)
    scores = await scorer.score_entities(entities, contexts)
"""

from qualitative_analysis.entity.models import (
    DimensionDefinition,
    DimensionSet,
    DimensionScore,
    SingleRunScore,
    EntityScore,
    EntityScoreResult,
    EntityConsolidation,
    ConsolidationResult,
)
from qualitative_analysis.entity.scorer import EntityScorer, score_entities
from qualitative_analysis.entity.visualizer import EntityVisualizer
from qualitative_analysis.entity.consolidator import EntityConsolidator, consolidate_entities
from qualitative_analysis.entity.comparison import (
    ParticipantComparison,
    ComparisonResult,
    GroupComparisonResult,
    aitchison_distance,
    wasserstein_distance_compositional,
    compute_pairwise_distances,
    compare_participants,
    clr_transform,
    ilr_transform,
)
from qualitative_analysis.entity.comparison_viz import ComparisonVisualizer

# Agreement analysis (Krippendorff's Alpha)
from qualitative_analysis.entity.agreement import (
    AgreementResult,
    compute_agreement,
    krippendorff_alpha,
)

# Clustering and PCA
from qualitative_analysis.entity.clustering import (
    ClusteringResult,
    PCAResult,
    cluster_participants,
    cluster_hierarchical,
    cluster_kmeans,
    pca_on_scores,
)

# Entity detection
from qualitative_analysis.entity.detector import (
    EntityDetector,
    EntityDetectionResult,
)

# HDBSCAN (optional — requires hdbscan package)
try:
    from qualitative_analysis.entity.clustering import cluster_hdbscan
except ImportError:
    pass

# Bayesian modeling (optional — requires pymc, arviz, nutpie)
try:
    from qualitative_analysis.entity.bayesian import (
        BayesianEntityModel,
        BayesianVisualizer,
        BayesianModelResult,
        check_pymc_available,
        prepare_beta_data,
    )
except ImportError:
    pass

__all__ = [
    # Models - Dimensions
    "DimensionDefinition",
    "DimensionSet",
    # Models - Scoring
    "DimensionScore",
    "SingleRunScore",
    "EntityScore",
    "EntityScoreResult",
    # Models - Consolidation
    "EntityConsolidation",
    "ConsolidationResult",
    # Scorer
    "EntityScorer",
    "score_entities",
    # Visualizer
    "EntityVisualizer",
    # Consolidator
    "EntityConsolidator",
    "consolidate_entities",
    # Comparison
    "ParticipantComparison",
    "ComparisonResult",
    "GroupComparisonResult",
    "ComparisonVisualizer",
    "aitchison_distance",
    "wasserstein_distance_compositional",
    "compute_pairwise_distances",
    "compare_participants",
    "clr_transform",
    "ilr_transform",
    # Agreement
    "AgreementResult",
    "compute_agreement",
    "krippendorff_alpha",
    # Clustering
    "ClusteringResult",
    "PCAResult",
    "cluster_participants",
    "cluster_hierarchical",
    "cluster_kmeans",
    "pca_on_scores",
    # Detection
    "EntityDetector",
    "EntityDetectionResult",
]

# Conditionally add HDBSCAN if available
try:
    from qualitative_analysis.entity.clustering import cluster_hdbscan as _  # noqa: F401
    __all__.append("cluster_hdbscan")
    del _
except ImportError:
    pass

# Conditionally add Bayesian symbols if pymc is available
try:
    from qualitative_analysis.entity.bayesian import BayesianEntityModel as _  # noqa: F401
    __all__.extend([
        "BayesianEntityModel",
        "BayesianVisualizer",
        "BayesianModelResult",
        "check_pymc_available",
        "prepare_beta_data",
    ])
    del _
except ImportError:
    pass
