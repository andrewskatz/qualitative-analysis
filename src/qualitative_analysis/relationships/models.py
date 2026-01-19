"""
Data models for relationship extraction, normalization, verification, causal analysis, and graph generation.

This module provides the data structures for the complete relationship analysis pipeline:
- Detection: Relationship, RelationshipResult
- Normalization: NormalizedEntity, NormalizedRelationship, NormalizationResult
- Verification: VerifiedRelationship
- Causal Analysis: CausalAttributes, CausalRelationship
- Graph: RelationshipGraphNode, RelationshipGraphEdge, RelationshipGraphData
- Utilities: EntityCluster, Checkpoint
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple


# =============================================================================
# DETECTION MODELS (Existing)
# =============================================================================

@dataclass
class Summary:
    """
    Represents a window-level summary.

    Attributes:
        text: The summary text content.
        window_index: Index of the window this summary describes.
    """
    text: str
    window_index: int = 0


@dataclass
class Relationship:
    """
    Represents a single relationship between two entities.

    Attributes:
        source: The source entity in the relationship.
        target: The target entity in the relationship.
        type: The type/nature of the relationship (e.g., "causes", "influences").
        description: Evidence or description of the relationship.
        window_index: Index of the window where this relationship was found.
        text_id: Optional identifier for the source text.
    """
    source: str
    target: str
    type: str
    description: str = ""
    window_index: int = 0
    text_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_tuple(self) -> Tuple[str, str, str]:
        """Return (source, type, target) tuple for deduplication."""
        return (self.source.lower().strip(), self.type.lower().strip(), self.target.lower().strip())


@dataclass
class RelationshipResult:
    """
    Final output of the relationship extraction process.

    Attributes:
        entities: List of unique entities found.
        relationships: List of extracted relationships.
        metadata: Additional metadata about the extraction.
    """
    entities: List[str] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": self.entities,
            "relationships": [r.to_dict() for r in self.relationships],
            "metadata": self.metadata,
        }


# =============================================================================
# NORMALIZATION MODELS
# =============================================================================

@dataclass
class EntityCluster:
    """
    A cluster of semantically similar entities.

    Attributes:
        canonical: The canonical/normalized label for this cluster.
        members: Original entity labels in this cluster.
        count: Total occurrences of entities in this cluster.
        avg_similarity: Average cosine similarity within cluster.
    """
    canonical: str
    members: List[str]
    count: int = 0
    avg_similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedEntity:
    """
    An entity after normalization/consolidation.

    Attributes:
        canonical: The canonical form of the entity.
        original: The original entity text before normalization.
        variants: All variants that map to this canonical form.
        frequency: Total occurrences across all variants.
        confidence: Confidence in the normalization.
    """
    canonical: str
    original: str = ""
    variants: List[str] = field(default_factory=list)
    frequency: int = 1
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedRelationship:
    """
    A relationship after entity and type normalization.

    Attributes:
        source: Normalized source entity.
        target: Normalized target entity.
        type: Normalized relationship type.
        original_source: Original source before normalization.
        original_target: Original target before normalization.
        original_type: Original type before normalization.
        description: Merged descriptions from all instances.
        descriptions: List of all individual descriptions.
        confidence: Confidence in the normalization.
        count: Number of instances merged into this relationship.
        window_indices: Windows where this relationship was found.
        text_ids: Source text identifiers.
    """
    source: str
    target: str
    type: str
    original_source: str = ""
    original_target: str = ""
    original_type: str = ""
    description: str = ""
    descriptions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    count: int = 1
    window_indices: List[int] = field(default_factory=list)
    text_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_tuple(self) -> Tuple[str, str, str]:
        """Return (source, type, target) tuple for deduplication."""
        return (self.source.lower().strip(), self.type.lower().strip(), self.target.lower().strip())


@dataclass
class NormalizationResult:
    """
    Result from relationship normalization/clustering.

    Attributes:
        entity_mapping: Map from original entity to canonical label.
        type_mapping: Map from original type to canonical label.
        entity_clusters: List of entity clusters.
        type_clusters: List of relationship type clusters.
        normalized_relationships: Relationships after normalization.
        config: Configuration used for normalization.
    """
    entity_mapping: Dict[str, str]
    type_mapping: Dict[str, str]
    entity_clusters: List[EntityCluster]
    type_clusters: List[EntityCluster]
    normalized_relationships: List[NormalizedRelationship] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "entity_mapping": self.entity_mapping,
            "type_mapping": self.type_mapping,
            "entity_clusters": [c.to_dict() for c in self.entity_clusters],
            "type_clusters": [c.to_dict() for c in self.type_clusters],
            "normalized_relationships": [r.to_dict() for r in self.normalized_relationships],
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationResult":
        """Deserialize from dictionary."""
        return cls(
            entity_mapping=data.get("entity_mapping", {}),
            type_mapping=data.get("type_mapping", {}),
            entity_clusters=[
                EntityCluster(**c) for c in data.get("entity_clusters", [])
            ],
            type_clusters=[
                EntityCluster(**c) for c in data.get("type_clusters", [])
            ],
            normalized_relationships=[
                NormalizedRelationship(**r) for r in data.get("normalized_relationships", [])
            ],
            config=data.get("config", {}),
        )


# =============================================================================
# VERIFICATION MODELS
# =============================================================================

@dataclass
class VerifiedRelationship:
    """
    A relationship that has been verified by a second LLM pass.

    Attributes:
        source: Source entity.
        target: Target entity.
        type: Relationship type.
        description: Relationship description/evidence.
        window_index: Window index where found.
        text_id: Source text identifier.
        verification_confidence: LLM confidence in the relationship (0.0-1.0).
        verification_note: Optional notes/corrections from verification.
        verified: Whether the relationship passed verification threshold.
        original_relationship: Reference to original relationship if available.
    """
    source: str
    target: str
    type: str
    description: str = ""
    window_index: int = 0
    text_id: str = ""
    verification_confidence: float = 1.0
    verification_note: str = ""
    verified: bool = True
    original_relationship: Optional[Relationship] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "description": self.description,
            "window_index": self.window_index,
            "text_id": self.text_id,
            "verification_confidence": self.verification_confidence,
            "verification_note": self.verification_note,
            "verified": self.verified,
        }
        return result


# =============================================================================
# CAUSAL ANALYSIS MODELS
# =============================================================================

@dataclass
class CausalAttributes:
    """
    Causal attributes for a relationship.

    Attributes:
        is_causal: Whether the relationship expresses causation.
        polarity: Direction of effect ("positive", "negative", "neutral").
        certainty: Confidence level ("certain", "likely", "possible").
        explicit_vs_implicit: Whether causation is stated ("explicit") or inferred ("implicit").
        reasoning: LLM reasoning for the classification.
    """
    is_causal: bool = False
    polarity: Optional[str] = None  # positive, negative, neutral (None if not causal)
    certainty: Optional[str] = None  # certain, likely, possible (None if not causal)
    explicit_vs_implicit: Optional[str] = None  # explicit, implicit (None if not causal)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_causal": self.is_causal,
            "polarity": self.polarity,
            "certainty": self.certainty,
            "explicit_vs_implicit": self.explicit_vs_implicit,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalAttributes":
        return cls(
            is_causal=data.get("is_causal", False),
            polarity=data.get("polarity"),
            certainty=data.get("certainty"),
            explicit_vs_implicit=data.get("explicit_vs_implicit"),
            reasoning=data.get("reasoning", data.get("explanation", "")),
        )


@dataclass
class CausalRelationship:
    """
    A relationship with causal attributes analyzed.

    Attributes:
        source: Source entity (normalized if applicable).
        target: Target entity (normalized if applicable).
        type: Relationship type (normalized if applicable).
        description: Primary description/evidence.
        causal: Causal attributes for this relationship.
        original_source: Original source before normalization.
        original_target: Original target before normalization.
        original_type: Original type before normalization.
        count: Number of instances merged (for normalized relationships).
        window_indices: Windows where this relationship was found.
        text_ids: Source text identifiers.
    """
    source: str
    target: str
    type: str
    description: str = ""
    causal: CausalAttributes = field(default_factory=CausalAttributes)
    original_source: str = ""
    original_target: str = ""
    original_type: str = ""
    count: int = 1
    window_indices: List[int] = field(default_factory=list)
    text_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "description": self.description,
            "is_causal": self.causal.is_causal,
            "polarity": self.causal.polarity,
            "certainty": self.causal.certainty,
            "explicit_vs_implicit": self.causal.explicit_vs_implicit,
            "reasoning": self.causal.reasoning,
            "original_source": self.original_source,
            "original_target": self.original_target,
            "original_type": self.original_type,
            "count": self.count,
            "window_indices": self.window_indices,
            "text_ids": self.text_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalRelationship":
        """Deserialize from dictionary."""
        causal = CausalAttributes(
            is_causal=data.get("is_causal", False),
            polarity=data.get("polarity"),
            certainty=data.get("certainty"),
            explicit_vs_implicit=data.get("explicit_vs_implicit"),
            reasoning=data.get("reasoning", data.get("causal_explanation", "")),
        )
        return cls(
            source=data.get("source", ""),
            target=data.get("target", ""),
            type=data.get("type", ""),
            description=data.get("description", ""),
            causal=causal,
            original_source=data.get("original_source", ""),
            original_target=data.get("original_target", ""),
            original_type=data.get("original_type", ""),
            count=data.get("count", 1),
            window_indices=data.get("window_indices", []),
            text_ids=data.get("text_ids", []),
        )

    @classmethod
    def from_relationship(
        cls,
        rel: Relationship,
        causal: Optional[CausalAttributes] = None
    ) -> "CausalRelationship":
        """Create from a base Relationship."""
        return cls(
            source=rel.source,
            target=rel.target,
            type=rel.type,
            description=rel.description,
            causal=causal or CausalAttributes(),
            window_indices=[rel.window_index],
            text_ids=[rel.text_id] if rel.text_id else [],
        )


@dataclass
class CausalAnalysisResult:
    """
    Result from causal analysis of relationships.

    Attributes:
        causal_relationships: List of relationships with causal attributes.
        stats: Summary statistics of causal analysis.
        config: Configuration used for analysis.
    """
    causal_relationships: List[CausalRelationship] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics from the analyzed relationships."""
        total = len(self.causal_relationships)
        if total == 0:
            return {"total": 0}

        causal_count = sum(1 for r in self.causal_relationships if r.causal.is_causal)

        # Polarity distribution
        polarity_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in self.causal_relationships:
            if r.causal.is_causal and r.causal.polarity:
                polarity_counts[r.causal.polarity] = polarity_counts.get(r.causal.polarity, 0) + 1

        # Certainty distribution
        certainty_counts = {"certain": 0, "likely": 0, "possible": 0}
        for r in self.causal_relationships:
            if r.causal.is_causal and r.causal.certainty:
                certainty_counts[r.causal.certainty] = certainty_counts.get(r.causal.certainty, 0) + 1

        # Explicit vs implicit
        explicit_counts = {"explicit": 0, "implicit": 0}
        for r in self.causal_relationships:
            if r.causal.is_causal and r.causal.explicit_vs_implicit:
                explicit_counts[r.causal.explicit_vs_implicit] = explicit_counts.get(
                    r.causal.explicit_vs_implicit, 0
                ) + 1

        self.stats = {
            "total": total,
            "causal_count": causal_count,
            "causal_percentage": round(100 * causal_count / total, 1) if total > 0 else 0,
            "non_causal_count": total - causal_count,
            "positive_count": polarity_counts.get("positive", 0),
            "negative_count": polarity_counts.get("negative", 0),
            "neutral_count": polarity_counts.get("neutral", 0),
            "certain_count": certainty_counts.get("certain", 0),
            "likely_count": certainty_counts.get("likely", 0),
            "possible_count": certainty_counts.get("possible", 0),
            "explicit_count": explicit_counts.get("explicit", 0),
            "implicit_count": explicit_counts.get("implicit", 0),
        }
        return self.stats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "causal_relationships": [r.to_dict() for r in self.causal_relationships],
            "stats": self.stats or self.compute_statistics(),
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalAnalysisResult":
        """Deserialize from dictionary."""
        # Support both 'causal_relationships' and 'relationships' keys
        rels_data = data.get("causal_relationships", data.get("relationships", []))
        relationships = [CausalRelationship.from_dict(r) for r in rels_data]
        return cls(
            causal_relationships=relationships,
            stats=data.get("stats", data.get("statistics", {})),
            config=data.get("config", {}),
        )


# =============================================================================
# GRAPH MODELS
# =============================================================================

@dataclass
class RelationshipGraphNode:
    """
    A node in the relationship graph representing an entity.

    Attributes:
        id: Unique node identifier (lowercase entity name).
        label: Display label (original case).
        frequency: Total occurrences of this entity.
        text_count: Number of unique texts mentioning this entity.
        source_text_ids: List of text IDs where this entity appears.
        snippets: Evidence snippets mentioning this entity.
        degree: Total connections (in + out).
        in_degree: Number of incoming edges.
        out_degree: Number of outgoing edges.
        betweenness: Betweenness centrality score.
        pagerank: PageRank score.
        cluster_id: Optional cluster assignment.
        raw_entities: Original entities mapped to this node (for normalized graphs).
    """
    id: str
    label: str
    frequency: int = 1
    text_count: int = 1
    source_text_ids: List[str] = field(default_factory=list)
    snippets: List[Dict[str, str]] = field(default_factory=list)
    degree: int = 0
    in_degree: int = 0
    out_degree: int = 0
    betweenness: float = 0.0
    pagerank: float = 0.0
    cluster_id: Optional[int] = None
    raw_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipGraphEdge:
    """
    An edge in the relationship graph.

    Attributes:
        id: Unique edge identifier.
        source: Source node ID.
        target: Target node ID.
        type: Relationship type.
        weight: Number of instances of this relationship.
        text_count: Number of unique texts with this relationship.
        evidence: List of evidence snippets/descriptions.
        source_text_ids: List of text IDs where this relationship appears.
        causal_attributes: Optional causal analysis results.
    """
    id: str
    source: str
    target: str
    type: str
    weight: int = 1
    text_count: int = 1
    evidence: List[str] = field(default_factory=list)
    source_text_ids: List[str] = field(default_factory=list)
    causal_attributes: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "type": self.type,
            "weight": self.weight,
            "text_count": self.text_count,
            "evidence": self.evidence,
            "source_text_ids": self.source_text_ids,
        }
        if self.causal_attributes:
            result["causal_attributes"] = self.causal_attributes
        return result


@dataclass
class RelationshipGraphData:
    """
    Complete relationship graph data structure.

    Attributes:
        nodes: List of entity nodes.
        edges: List of relationship edges.
        metrics_summary: Global graph metrics.
        is_normalized: Whether entities/types are normalized.
        has_causal: Whether causal attributes are included.
    """
    nodes: List[RelationshipGraphNode]
    edges: List[RelationshipGraphEdge]
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    is_normalized: bool = False
    has_causal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metrics_summary": self.metrics_summary,
            "is_normalized": self.is_normalized,
            "has_causal": self.has_causal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipGraphData":
        """Deserialize from dictionary."""
        nodes = [RelationshipGraphNode(**n) for n in data.get("nodes", [])]
        edges = [RelationshipGraphEdge(**e) for e in data.get("edges", [])]
        return cls(
            nodes=nodes,
            edges=edges,
            metrics_summary=data.get("metrics_summary", {}),
            is_normalized=data.get("is_normalized", False),
            has_causal=data.get("has_causal", False),
        )


# =============================================================================
# CHECKPOINT MODEL
# =============================================================================

@dataclass
class Checkpoint:
    """
    Checkpoint for resuming interrupted processing.

    Attributes:
        processed_indices: Indices of already-processed items.
        partial_results: Results for processed items.
        config: Configuration used.
        timestamp: When checkpoint was created.
        input_file: Path to input file.
        total_items: Total number of items to process.
        step: Current pipeline step (detect, normalize, verify, causal, graph).
    """
    processed_indices: List[int]
    partial_results: List[Dict[str, Any]]
    config: Dict[str, Any]
    timestamp: str
    input_file: str = ""
    total_items: int = 0
    step: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            processed_indices=data.get("processed_indices", []),
            partial_results=data.get("partial_results", []),
            config=data.get("config", {}),
            timestamp=data.get("timestamp", ""),
            input_file=data.get("input_file", ""),
            total_items=data.get("total_items", 0),
            step=data.get("step", ""),
        )
