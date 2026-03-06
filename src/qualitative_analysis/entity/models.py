"""
Data models for entity consolidation, scoring, and visualization.

This module provides data structures for:
- Dimension definitions for multi-dimensional scoring
- Entity scores with uncertainty quantification
- Consolidation results for entity deduplication
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import logging
import statistics

from scipy.stats import t as t_dist

logger = logging.getLogger(__name__)


# =============================================================================
# DIMENSION DEFINITIONS
# =============================================================================

@dataclass
class DimensionDefinition:
    """
    Definition of a scoring dimension.

    Attributes:
        name: Short name for the dimension (e.g., "Social", "Ecological").
        description: Full description of what the dimension measures.
        min_anchor: Description of what a minimum score (0) represents.
        max_anchor: Description of what a maximum score (100) represents.
        scale_min: Minimum value on the scale (default 0).
        scale_max: Maximum value on the scale (default 100).

    Example:
        DimensionDefinition(
            name="Social",
            description="Degree to which the entity represents social/human factors",
            min_anchor="Purely technical or natural, no social dimension",
            max_anchor="Fundamentally about human behavior, institutions, or culture"
        )
    """
    name: str
    description: str
    min_anchor: str = ""
    max_anchor: str = ""
    scale_min: int = 0
    scale_max: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionDefinition":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            min_anchor=data.get("min_anchor", ""),
            max_anchor=data.get("max_anchor", ""),
            scale_min=data.get("scale_min", 0),
            scale_max=data.get("scale_max", 100),
        )

    def validate(self) -> bool:
        """Check if the dimension definition is valid."""
        return bool(self.name and self.description)


@dataclass
class DimensionSet:
    """
    A collection of dimension definitions for scoring.

    Attributes:
        name: Name of the dimension set (e.g., "SETS Framework").
        description: Description of the dimension set.
        dimensions: List of dimension definitions.
    """
    name: str
    description: str = ""
    dimensions: List[DimensionDefinition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DimensionSet":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            dimensions=[
                DimensionDefinition.from_dict(d) for d in data.get("dimensions", [])
            ],
        )

    @classmethod
    def sets_framework(cls) -> "DimensionSet":
        """Create the standard SETS (Social-Ecological-Technological) framework."""
        return cls(
            name="SETS Framework",
            description="Social-Ecological-Technological Systems framework for categorizing entities",
            dimensions=[
                DimensionDefinition(
                    name="Social",
                    description="Degree to which the entity represents social/human factors including behavior, institutions, culture, governance, and human relationships",
                    min_anchor="Purely technical or natural phenomenon with no social dimension",
                    max_anchor="Fundamentally about human behavior, social institutions, or cultural factors",
                ),
                DimensionDefinition(
                    name="Ecological",
                    description="Degree to which the entity relates to environmental or ecological systems including natural processes, ecosystems, and environmental conditions",
                    min_anchor="No environmental or ecological relevance",
                    max_anchor="Core environmental, ecological, or natural system concept",
                ),
                DimensionDefinition(
                    name="Technological",
                    description="Degree to which the entity represents technological factors including tools, infrastructure, engineering systems, and technical processes",
                    min_anchor="No technological or engineered component",
                    max_anchor="Fundamentally technological, engineered, or infrastructure-based",
                ),
            ],
        )


# =============================================================================
# SCALE CONFIGURATION
# =============================================================================

@dataclass
class ScaleConfig:
    """
    Score scale configuration for the entity scoring pipeline.

    Propagated from CLI args or auto-detected from score_metadata.json.
    Used by scoring, Bayesian modeling, and visualization components.
    """
    scale_min: int = 0
    scale_max: int = 100

    @property
    def scale_range(self) -> int:
        return self.scale_max - self.scale_min

    @property
    def midpoint(self) -> float:
        return (self.scale_min + self.scale_max) / 2

    def fraction(self, value: float) -> float:
        """Convert raw score to [0, 1] fraction."""
        return max(0, min((value - self.scale_min) / self.scale_range, 1.0))

    def from_fraction(self, frac: float) -> float:
        """Convert [0, 1] fraction back to raw scale."""
        return self.scale_min + frac * self.scale_range

    def rope_default(self) -> float:
        """Proportional ROPE: 5% of scale range."""
        return self.scale_range * 0.05

    def reference_levels(self) -> list:
        """Reference sizes for legends at 25%, 50%, 75% of scale."""
        r = self.scale_range
        return [
            (self.scale_min + 0.25 * r, f"{self.scale_min + 0.25 * r:.0f}"),
            (self.scale_min + 0.50 * r, f"{self.scale_min + 0.50 * r:.0f}"),
            (self.scale_min + 0.75 * r, f"{self.scale_min + 0.75 * r:.0f}"),
        ]

    @classmethod
    def from_metadata_file(cls, metadata_path) -> "ScaleConfig":
        """Load from score_metadata.json if scale fields present."""
        import json as _json
        from pathlib import Path
        with open(Path(metadata_path)) as f:
            meta = _json.load(f)
        return cls(
            scale_min=meta.get("scale_min", 0),
            scale_max=meta.get("scale_max", 100),
        )


# =============================================================================
# SCORING MODELS
# =============================================================================

@dataclass
class DimensionScore:
    """
    Score for a single dimension with uncertainty quantification.

    Attributes:
        dimension: Name of the dimension.
        mean: Mean score across runs.
        median: Median score across runs.
        mode: Most common score across runs.
        std_dev: Standard deviation of scores.
        confidence_interval_95: 95% confidence interval (low, high).
        coefficient_of_variation: CV = std_dev / mean (uncertainty metric).
        scores: Raw scores from each run.
        justification: LLM justification for the scores.
    """
    dimension: str
    mean: float
    median: float
    mode: float
    std_dev: float
    confidence_interval_95: Tuple[float, float]
    coefficient_of_variation: float
    scores: List[int]
    justification: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "std_dev": self.std_dev,
            "confidence_interval_95_low": self.confidence_interval_95[0],
            "confidence_interval_95_high": self.confidence_interval_95[1],
            "coefficient_of_variation": self.coefficient_of_variation,
            "scores": self.scores,
            "justification": self.justification,
        }

    @classmethod
    def from_scores(
        cls,
        dimension: str,
        scores: List[int],
        justification: str = "",
        scale_min: int = 0,
        scale_max: int = 100,
    ) -> "DimensionScore":
        """
        Create a DimensionScore from raw scores with computed statistics.

        Args:
            dimension: Name of the dimension.
            scores: List of scores from multiple runs.
            justification: Combined justification text.

        Returns:
            DimensionScore with computed statistics.
        """
        if not scores:
            return cls(
                dimension=dimension,
                mean=0.0,
                median=0.0,
                mode=0.0,
                std_dev=0.0,
                confidence_interval_95=(0.0, 0.0),
                coefficient_of_variation=0.0,
                scores=[],
                justification=justification,
            )

        n = len(scores)
        mean_val = statistics.mean(scores)
        median_val = statistics.median(scores)

        # Mode - handle multimodal cases by taking median of modes
        modes = statistics.multimode(scores)
        mode_val = float(statistics.median(modes))

        # Standard deviation
        std_dev = statistics.stdev(scores) if n > 1 else 0.0

        # 95% CI using t-distribution
        if n > 1:
            t_value = float(t_dist.ppf(0.975, df=n - 1))
            margin = t_value * std_dev / (n ** 0.5)
            ci_low = max(scale_min, mean_val - margin)
            ci_high = min(scale_max, mean_val + margin)
        else:
            ci_low = ci_high = mean_val

        # Coefficient of variation
        cv = std_dev / mean_val if mean_val > 0 else 0.0

        return cls(
            dimension=dimension,
            mean=round(mean_val, 2),
            median=round(median_val, 2),
            mode=round(mode_val, 2),
            std_dev=round(std_dev, 2),
            confidence_interval_95=(round(ci_low, 2), round(ci_high, 2)),
            coefficient_of_variation=round(cv, 4),
            scores=scores,
            justification=justification,
        )


@dataclass
class SingleRunScore:
    """
    Score from a single LLM scoring run.

    Attributes:
        run_number: Which run this is (1-indexed).
        dimension_scores: Map of dimension name to (score, justification).
        initial_observations: LLM's initial observations about the entity.
        raw_response: Raw LLM response for debugging.
        processing_time_ms: Time taken for this run in milliseconds.
    """
    run_number: int
    dimension_scores: Dict[str, Dict[str, Any]]  # {dim: {score: int, justification: str}}
    initial_observations: str = ""
    raw_response: str = ""
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityScore:
    """
    Complete scoring result for a single entity.

    Attributes:
        entity: The entity that was scored.
        text_id: Identifier for the source text.
        context: The context used for scoring.
        dimension_scores: Aggregated scores for each dimension.
        runs: Individual scoring runs (if include_raw_scores=True).
        num_runs: Number of scoring runs performed.
        processing_time_ms: Total processing time in milliseconds.
    """
    entity: str
    text_id: str
    context: str
    dimension_scores: Dict[str, DimensionScore]
    runs: List[SingleRunScore] = field(default_factory=list)
    num_runs: int = 1
    processing_time_ms: float = 0.0
    window_index: Optional[int] = None
    group: Optional[str] = None

    def to_dict(self, include_runs: bool = False) -> Dict[str, Any]:
        result = {
            "entity": self.entity,
            "text_id": self.text_id,
            "context": self.context,
            "window_index": self.window_index,
            "group": self.group,
            "num_runs": self.num_runs,
            "processing_time_ms": self.processing_time_ms,
        }

        # Flatten dimension scores for CSV compatibility
        for dim_name, score in self.dimension_scores.items():
            # Individual run scores (e.g., social_run1, social_run2, social_run3)
            for k, run_score in enumerate(score.scores, start=1):
                result[f"{dim_name}_run{k}"] = run_score
            result[f"{dim_name}_mean"] = score.mean
            result[f"{dim_name}_median"] = score.median
            result[f"{dim_name}_std"] = score.std_dev
            result[f"{dim_name}_ci_low"] = score.confidence_interval_95[0]
            result[f"{dim_name}_ci_high"] = score.confidence_interval_95[1]
            result[f"{dim_name}_cv"] = score.coefficient_of_variation
            result[f"{dim_name}_justification"] = score.justification

        if include_runs:
            result["runs"] = [r.to_dict() for r in self.runs]

        return result

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for CSV output."""
        return self.to_dict(include_runs=False)

    def get_primary_dimension(self) -> str:
        """Get the dimension with the highest mean score."""
        if not self.dimension_scores:
            return ""
        return max(self.dimension_scores.items(), key=lambda x: x[1].mean)[0]

    def get_scores_as_tuple(self, dimensions: List[str]) -> Tuple[float, ...]:
        """Get scores for specified dimensions as a tuple (for ternary plots)."""
        scores = []
        for dim in dimensions:
            if dim not in self.dimension_scores:
                logger.warning(
                    f"Dimension '{dim}' not found in scores for entity "
                    f"'{getattr(self, 'entity', '?')}' — defaulting to 0.0"
                )
                scores.append(0.0)
            else:
                scores.append(self.dimension_scores[dim].mean)
        return tuple(scores)


@dataclass
class EntityScoreResult:
    """
    Result from scoring multiple entities.

    Attributes:
        scores: List of entity scores.
        dimensions: Dimension definitions used for scoring.
        config: Configuration used for scoring.
        statistics: Aggregate statistics across all entities.
    """
    scores: List[EntityScore] = field(default_factory=list)
    dimensions: List[DimensionDefinition] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across all scored entities."""
        if not self.scores:
            return {}

        dim_names = list(self.scores[0].dimension_scores.keys()) if self.scores else []

        stats = {
            "total_entities": len(self.scores),
            "dimensions": {},
        }

        for dim in dim_names:
            dim_means = [s.dimension_scores[dim].mean for s in self.scores if dim in s.dimension_scores]
            if dim_means:
                stats["dimensions"][dim] = {
                    "mean": round(statistics.mean(dim_means), 2),
                    "median": round(statistics.median(dim_means), 2),
                    "std_dev": round(statistics.stdev(dim_means), 2) if len(dim_means) > 1 else 0.0,
                    "min": round(min(dim_means), 2),
                    "max": round(max(dim_means), 2),
                }

        self.statistics = stats
        return stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": [s.to_dict() for s in self.scores],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "config": self.config,
            "statistics": self.statistics or self.compute_statistics(),
        }


# =============================================================================
# CONSOLIDATION MODELS
# =============================================================================

@dataclass
class EntityConsolidation:
    """
    A single consolidation/merge of similar entities.

    Attributes:
        canonical: The canonical form selected for this group.
        variants: All entity variants that map to this canonical form.
        frequency: Total occurrences across all variants.
        avg_similarity: Average similarity within the group.
        confidence: Confidence in the consolidation.
    """
    canonical: str
    variants: List[str]
    frequency: int = 1
    avg_similarity: float = 1.0
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsolidationResult:
    """
    Result from entity consolidation.

    Attributes:
        consolidations: List of entity consolidations.
        mapping: Map from original entity to canonical form.
        config: Configuration used for consolidation.
        statistics: Summary statistics.
    """
    consolidations: List[EntityConsolidation] = field(default_factory=list)
    mapping: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for the consolidation."""
        original_count = len(self.mapping)
        canonical_count = len(set(self.mapping.values()))

        self.statistics = {
            "original_count": original_count,
            "consolidated_count": canonical_count,
            "reduction_count": original_count - canonical_count,
            "reduction_percentage": round(
                100 * (original_count - canonical_count) / original_count, 1
            ) if original_count > 0 else 0,
            "cluster_count": len(self.consolidations),
            "largest_cluster_size": max(
                (len(c.variants) for c in self.consolidations), default=0
            ),
        }
        return self.statistics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consolidations": [c.to_dict() for c in self.consolidations],
            "mapping": self.mapping,
            "config": self.config,
            "statistics": self.statistics or self.compute_statistics(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidationResult":
        return cls(
            consolidations=[
                EntityConsolidation(**c) for c in data.get("consolidations", [])
            ],
            mapping=data.get("mapping", {}),
            config=data.get("config", {}),
            statistics=data.get("statistics", {}),
        )

    def apply_to_entities(self, entities: List[str]) -> List[str]:
        """Apply the consolidation mapping to a list of entities."""
        return [self.mapping.get(e, e) for e in entities]
