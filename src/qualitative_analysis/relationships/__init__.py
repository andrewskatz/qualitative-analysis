"""
Relationship extraction, normalization, and analysis module.

This module provides functionality for:
- Extracting entities and relationships from text
- Normalizing/consolidating entities and relationship types
- Verifying relationships with LLM second-pass
- Analyzing causal attributes of relationships
- Generating relationship network graphs
"""

from .detector import RelationshipDetector
from .models import (
    Relationship,
    RelationshipResult,
    EntityCluster,
    NormalizedEntity,
    NormalizedRelationship,
    NormalizationResult,
    VerifiedRelationship,
    CausalAttributes,
    CausalRelationship,
    CausalAnalysisResult,
    RelationshipGraphNode,
    RelationshipGraphEdge,
    RelationshipGraphData,
)
from .normalizer import RelationshipNormalizer, normalize_relationships
from .causal import CausalAnalyzer, analyze_causal
from .graph import RelationshipGraph, build_relationship_graph
from .verifier import RelationshipVerifier, VerificationResult, verify_relationships

__all__ = [
    # Detection
    "RelationshipDetector",
    "Relationship",
    "RelationshipResult",
    # Normalization
    "RelationshipNormalizer",
    "normalize_relationships",
    "EntityCluster",
    "NormalizedEntity",
    "NormalizedRelationship",
    "NormalizationResult",
    # Verification
    "RelationshipVerifier",
    "VerificationResult",
    "verify_relationships",
    "VerifiedRelationship",
    # Causal Analysis
    "CausalAnalyzer",
    "analyze_causal",
    "CausalAttributes",
    "CausalRelationship",
    "CausalAnalysisResult",
    # Graph
    "RelationshipGraph",
    "build_relationship_graph",
    "RelationshipGraphNode",
    "RelationshipGraphEdge",
    "RelationshipGraphData",
]
