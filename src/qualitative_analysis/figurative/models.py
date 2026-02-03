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


@dataclass
class DetectionCheckpoint:
    """
    Checkpoint for resuming interrupted figurative detection.
    
    Attributes:
        processed_text_ids: IDs of already-processed texts
        partial_summary_results: Serialized summary CSV rows
        partial_instance_results: Serialized instance CSV rows
        partial_window_results: Serialized window CSV rows
        config: CLI configuration used
        timestamp: When checkpoint was created
        input_file: Path to input CSV
        total_texts: Total number of texts to process
    """
    processed_text_ids: List[str]
    partial_summary_results: List[dict] = field(default_factory=list)
    partial_instance_results: List[dict] = field(default_factory=list)
    partial_window_results: List[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    timestamp: str = ""
    input_file: str = ""
    total_texts: int = 0
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "processed_text_ids": self.processed_text_ids,
            "partial_summary_results": self.partial_summary_results,
            "partial_instance_results": self.partial_instance_results,
            "partial_window_results": self.partial_window_results,
            "config": self.config,
            "timestamp": self.timestamp,
            "input_file": self.input_file,
            "total_texts": self.total_texts,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DetectionCheckpoint":
        """Deserialize from dictionary."""
        return cls(
            processed_text_ids=data.get("processed_text_ids", []),
            partial_summary_results=data.get("partial_summary_results", []),
            partial_instance_results=data.get("partial_instance_results", []),
            partial_window_results=data.get("partial_window_results", []),
            config=data.get("config", {}),
            timestamp=data.get("timestamp", ""),
            input_file=data.get("input_file", ""),
            total_texts=data.get("total_texts", 0),
        )

