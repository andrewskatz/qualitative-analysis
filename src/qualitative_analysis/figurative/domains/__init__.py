"""
Domain mapping and normalization for figurative language analysis.

This module provides:
- DomainExtractor: Extract source/target domains from figurative instances
- DomainNormalizer: Cluster and normalize domain labels  
- DomainGraph: Generate source→target relationship graphs
"""

from .models import (
    DomainMappedInstance,
    DomainCluster,
    NormalizationResult,
    DomainGraphNode,
    DomainGraphEdge,
    DomainGraphData,
    Checkpoint,
)
from .extractor import DomainExtractor
from .normalizer import DomainNormalizer
from .graph import DomainGraph

__all__ = [
    # Models
    "DomainMappedInstance",
    "DomainCluster",
    "NormalizationResult",
    "DomainGraphNode",
    "DomainGraphEdge",
    "DomainGraphData",
    "Checkpoint",
    # Classes
    "DomainExtractor",
    "DomainNormalizer",
    "DomainGraph",
]
