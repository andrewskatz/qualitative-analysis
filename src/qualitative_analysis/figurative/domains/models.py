"""
Data models for domain mapping and normalization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DomainMappedInstance:
    """
    A figurative language instance with extracted source/target domains.
    
    Attributes:
        text: The figurative text/phrase
        type: Type of figurative language (metaphor, analogy, etc.)
        confidence: Original detection confidence
        explanation: Original explanation of why it's figurative
        window_index: Index of the text window containing this instance
        text_id: Optional identifier for the source text
        window_text: Optional context window text
        source_domain: Extracted source domain label
        target_domain: Extracted target domain label
        mapping_explanation: How source maps to target
        domain_confidence: Confidence in domain extraction
        source_domain_levels: Multi-level source domains (specific/moderate/abstract)
        target_domain_levels: Multi-level target domains (specific/moderate/abstract)
        raw_response: Raw LLM response for debugging
    """
    # Original instance fields
    text: str
    type: str
    confidence: float = 0.0
    explanation: str = ""
    window_index: int = 0
    text_id: str = ""
    window_text: str = ""
    
    # Domain fields
    source_domain: str = ""
    target_domain: str = ""
    mapping_explanation: str = ""
    domain_confidence: float = 0.0
    
    # Multi-level domains
    source_domain_levels: Dict[str, str] = field(default_factory=dict)
    target_domain_levels: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class DomainCluster:
    """
    A cluster of semantically similar domains.
    
    Attributes:
        canonical: The canonical/normalized label for this cluster
        members: Original domain labels in this cluster
        count: Total occurrences of domains in this cluster
        avg_similarity: Average cosine similarity within cluster
    """
    canonical: str
    members: List[str]
    count: int = 0
    avg_similarity: float = 0.0


@dataclass
class NormalizationResult:
    """
    Result from domain normalization/clustering.
    
    Attributes:
        source_mapping: Map from original source domain to canonical label
        target_mapping: Map from original target domain to canonical label
        source_clusters: List of source domain clusters
        target_clusters: List of target domain clusters
        config: Configuration used for normalization
    """
    source_mapping: Dict[str, str]
    target_mapping: Dict[str, str]
    source_clusters: List[DomainCluster]
    target_clusters: List[DomainCluster]
    config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "source_mapping": self.source_mapping,
            "target_mapping": self.target_mapping,
            "source_clusters": [
                {
                    "canonical": c.canonical,
                    "members": c.members,
                    "count": c.count,
                    "avg_similarity": c.avg_similarity,
                }
                for c in self.source_clusters
            ],
            "target_clusters": [
                {
                    "canonical": c.canonical,
                    "members": c.members,
                    "count": c.count,
                    "avg_similarity": c.avg_similarity,
                }
                for c in self.target_clusters
            ],
            "config": self.config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationResult":
        """Deserialize from dictionary."""
        return cls(
            source_mapping=data.get("source_mapping", {}),
            target_mapping=data.get("target_mapping", {}),
            source_clusters=[
                DomainCluster(
                    canonical=c["canonical"],
                    members=c["members"],
                    count=c.get("count", 0),
                    avg_similarity=c.get("avg_similarity", 0.0),
                )
                for c in data.get("source_clusters", [])
            ],
            target_clusters=[
                DomainCluster(
                    canonical=c["canonical"],
                    members=c["members"],
                    count=c.get("count", 0),
                    avg_similarity=c.get("avg_similarity", 0.0),
                )
                for c in data.get("target_clusters", [])
            ],
            config=data.get("config", {}),
        )


@dataclass
class DomainGraphNode:
    """
    A node in the domain graph.
    
    Attributes:
        id: Unique node identifier (lowercase domain name)
        label: Display label (title case)
        domain_type: "source", "target", or "both"
        frequency: Total occurrences
        as_source: Count as source domain
        as_target: Count as target domain
        raw_domains: Original domains mapped to this node (for normalized graphs)
    """
    id: str
    label: str
    domain_type: str  # "source", "target", or "both"
    frequency: int
    as_source: int
    as_target: int
    raw_domains: List[str] = field(default_factory=list)


@dataclass
class DomainGraphEdge:
    """
    An edge in the domain graph representing a source→target mapping.
    
    Attributes:
        id: Unique edge identifier
        source: Source node ID
        target: Target node ID
        weight: Number of instances with this mapping
        figurative_types: Types of figurative language using this mapping
        examples: Example figurative texts
        text_count: Number of unique texts using this mapping
    """
    id: str
    source: str
    target: str
    weight: int
    figurative_types: List[str]
    examples: List[str]
    text_count: int


@dataclass
class DomainGraphData:
    """
    Complete domain relationship graph.
    
    Attributes:
        nodes: List of domain nodes
        edges: List of source→target edges
        stats: Summary statistics
        is_normalized: Whether this graph uses normalized domains
    """
    nodes: List[DomainGraphNode]
    edges: List[DomainGraphEdge]
    stats: Dict[str, Any]
    is_normalized: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "domain_type": n.domain_type,
                    "frequency": n.frequency,
                    "as_source": n.as_source,
                    "as_target": n.as_target,
                    "raw_domains": n.raw_domains,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "weight": e.weight,
                    "figurative_types": e.figurative_types,
                    "examples": e.examples,
                    "text_count": e.text_count,
                }
                for e in self.edges
            ],
            "stats": self.stats,
            "is_normalized": self.is_normalized,
        }


@dataclass
class Checkpoint:
    """
    Checkpoint for resuming interrupted processing.
    
    Attributes:
        processed_indices: Indices of already-processed items
        partial_results: Results for processed items
        config: Configuration used
        timestamp: When checkpoint was created
        input_file: Path to input file
        total_items: Total number of items to process
    """
    processed_indices: List[int]
    partial_results: List[Dict[str, Any]]
    config: Dict[str, Any]
    timestamp: str
    input_file: str = ""
    total_items: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "processed_indices": self.processed_indices,
            "partial_results": self.partial_results,
            "config": self.config,
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "total_items": self.total_items,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Deserialize from dictionary."""
        return cls(
            processed_indices=data.get("processed_indices", []),
            partial_results=data.get("partial_results", []),
            config=data.get("config", {}),
            timestamp=data.get("timestamp", ""),
            input_file=data.get("input_file", ""),
            total_items=data.get("total_items", 0),
        )
