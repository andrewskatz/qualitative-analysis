"""
Domain models for figurative language detection.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from ..core.types import TextSpan

@dataclass
class Instance(TextSpan):
    """
    Represents a single figurative language instance.
    """
    type: str = "unknown"
    confidence: float = 0.0
    explanation: str = ""
    context_dependent: bool = False
    window_index: int = 0

@dataclass
class Summary:
    """
    Represents a temporary summary used in the two-step process.
    """
    text: str
    window_index: int

@dataclass
class DetectionResult:
    """
    Final output of the detection process.
    """
    contains_figurative: bool
    confidence: float
    instances: List[Instance] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
