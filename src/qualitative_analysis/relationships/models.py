"""
Domain models for relationship extraction.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


@dataclass
class Summary:
    """
    Represents a window-level summary.
    """
    text: str
    window_index: int = 0


@dataclass
class Relationship:
    """
    Represents a single relationship between two entities.
    """
    source: str
    target: str
    type: str
    description: str = ""
    window_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipResult:
    """
    Final output of the relationship extraction process.
    """
    entities: List[str] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
