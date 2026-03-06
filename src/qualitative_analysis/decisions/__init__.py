"""
Decision/factor extraction, normalization, and analysis module.

This module provides functionality for:
- Extracting decisions and their supporting/opposing factors from text
- Validating decisions using rule-based and LLM-based approaches
- Normalizing/consolidating decisions and factors via semantic clustering
- Aggregating statistics across multiple texts
- Visualizing decision/factor analysis results
"""

from .models import (
    Factor,
    Decision,
    WindowSummary,
    DecisionExtractionResult,
    DecisionConfig,
    AggregationResult,
)
from .extractor import DecisionExtractor
from .validator import DecisionValidator
from .normalizer import DecisionNormalizer, NormalizationResult
from .aggregator import DecisionAggregator, aggregate_decisions
from .visualizer import DecisionVisualizer

__all__ = [
    # Models
    "Factor",
    "Decision",
    "WindowSummary",
    "DecisionExtractionResult",
    "DecisionConfig",
    "AggregationResult",
    # Extraction
    "DecisionExtractor",
    # Validation
    "DecisionValidator",
    # Normalization
    "DecisionNormalizer",
    "NormalizationResult",
    # Aggregation
    "DecisionAggregator",
    "aggregate_decisions",
    # Visualization
    "DecisionVisualizer",
]
