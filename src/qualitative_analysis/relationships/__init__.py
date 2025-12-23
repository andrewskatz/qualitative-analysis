"""
Relationship extraction module.
"""

from .detector import RelationshipDetector
from .models import Relationship, RelationshipResult

__all__ = ["RelationshipDetector", "Relationship", "RelationshipResult"]
