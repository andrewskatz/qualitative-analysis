"""
Shared data types for the qualitative analysis suite.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

@dataclass
class TextSpan:
    """
    Represents a specific span of text within a larger document.
    """
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalysisResult:
    """
    Base class for any analysis result.
    """
    raw_text: str
    confidence: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
