"""
Data models for decision/factor extraction, normalization, and aggregation.

This module provides the data structures for the complete decision analysis pipeline:
- Extraction: Factor, Decision, WindowSummary, DecisionExtractionResult
- Configuration: DecisionConfig
- Aggregation: AggregationResult
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Factor:
    """
    A factor influencing a decision.

    Attributes:
        text: The factor text (verbatim or with pronoun resolution).
        decision_text: The decision this factor is linked to.
        polarity: Polarity of the factor: supporting, opposing, or neutral.
        window_index: Index of the window where this factor was found.
        text_id: Identifier for the source text.
        confidence: Confidence in the extraction (0.0-1.0).
    """

    text: str
    decision_text: str = ""
    polarity: str = "neutral"
    window_index: int = 0
    text_id: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Factor":
        return cls(
            text=data.get("text", data.get("factor", "")),
            decision_text=data.get("decision_text", data.get("decision", "")),
            polarity=data.get("polarity", "neutral"),
            window_index=data.get("window_index", 0),
            text_id=data.get("text_id", ""),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Decision:
    """
    A single decision extracted from text.

    Attributes:
        text: Decision action text (causal clauses stripped).
        original_text: Original text before normalization.
        factors: Factors linked to this decision.
        window_indices: Windows where this decision was found.
        text_ids: Source text identifiers.
        validation_passed: Whether the decision passed rule-based validation.
    """

    text: str
    original_text: str = ""
    factors: List[Factor] = field(default_factory=list)
    window_indices: List[int] = field(default_factory=list)
    text_ids: List[str] = field(default_factory=list)
    validation_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "original_text": self.original_text,
            "factors": [f.to_dict() for f in self.factors],
            "window_indices": self.window_indices,
            "text_ids": self.text_ids,
            "validation_passed": self.validation_passed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        factors = [
            Factor.from_dict(f) for f in data.get("factors", [])
        ]
        return cls(
            text=data.get("text", ""),
            original_text=data.get("original_text", ""),
            factors=factors,
            window_indices=data.get("window_indices", []),
            text_ids=data.get("text_ids", []),
            validation_passed=data.get("validation_passed", True),
        )


@dataclass
class WindowSummary:
    """
    Summary of a text window for semantic retrieval.

    Attributes:
        window_index: Index of the window this summary describes.
        text: Original window text.
        summary: 1-2 sentence summary.
        bullet_points: 3-5 key bullet points.
    """

    window_index: int
    text: str
    summary: str = ""
    bullet_points: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowSummary":
        return cls(
            window_index=data.get("window_index", 0),
            text=data.get("text", ""),
            summary=data.get("summary", ""),
            bullet_points=data.get("bullet_points", []),
        )


@dataclass
class DecisionExtractionResult:
    """
    Complete result of decision/factor extraction for one text.

    Attributes:
        text_id: Identifier for the source text.
        decisions: List of extracted decisions with linked factors.
        factors: All factors across all decisions (flat list).
        window_count: Number of windows processed.
        metadata: Additional metadata about the extraction.
    """

    text_id: str
    decisions: List[Decision] = field(default_factory=list)
    factors: List[Factor] = field(default_factory=list)
    window_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_id": self.text_id,
            "decisions": [d.to_dict() for d in self.decisions],
            "factors": [f.to_dict() for f in self.factors],
            "window_count": self.window_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionExtractionResult":
        decisions = [Decision.from_dict(d) for d in data.get("decisions", [])]
        factors = [Factor.from_dict(f) for f in data.get("factors", [])]
        return cls(
            text_id=data.get("text_id", ""),
            decisions=decisions,
            factors=factors,
            window_count=data.get("window_count", 0),
            metadata=data.get("metadata", {}),
        )

    def to_flat_decisions_rows(self) -> List[Dict[str, Any]]:
        """Flatten to one row per decision for CSV export."""
        rows = []
        for d in self.decisions:
            rows.append(
                {
                    "text_id": self.text_id,
                    "decision": d.text,
                    "original_decision": d.original_text,
                    "factor_count": len(d.factors),
                    "supporting_count": sum(
                        1 for f in d.factors if f.polarity == "supporting"
                    ),
                    "opposing_count": sum(
                        1 for f in d.factors if f.polarity == "opposing"
                    ),
                    "neutral_count": sum(
                        1 for f in d.factors if f.polarity == "neutral"
                    ),
                    "window_indices": ",".join(str(i) for i in d.window_indices),
                    "validation_passed": d.validation_passed,
                }
            )
        return rows

    def to_flat_factors_rows(self) -> List[Dict[str, Any]]:
        """Flatten to one row per factor for CSV export."""
        rows = []
        for f in self.factors:
            rows.append(
                {
                    "text_id": self.text_id,
                    "decision": f.decision_text,
                    "factor": f.text,
                    "polarity": f.polarity,
                    "window_index": f.window_index,
                    "confidence": f.confidence,
                }
            )
        return rows


@dataclass
class DecisionConfig:
    """
    Configuration for decision/factor extraction.

    Attributes:
        strategy: Extraction strategy (one_pass, semantic_two_pass, context_aware).
        temperature: LLM temperature for generation.
        validate_decisions: Enable rule-based decision validation.
        normalize_decisions: Strip causal clauses from decisions.
        dedup_method: Deduplication method (exact or semantic).
        decision_dedup_threshold: Cosine similarity threshold for decision dedup.
        factor_dedup_threshold: Cosine similarity threshold for factor dedup.
        factor_min_words: Minimum word count for valid factors.
        factor_max_decision_overlap: Maximum word overlap ratio with decision.
        top_k_windows: Number of top windows per decision (semantic two-pass).
        min_similarity: Minimum similarity for window relevance (semantic two-pass).
    """

    strategy: str = "semantic_two_pass"
    temperature: float = 0.3
    validate_decisions: bool = True
    normalize_decisions: bool = True
    dedup_method: str = "semantic"
    decision_dedup_threshold: float = 0.9
    factor_dedup_threshold: float = 0.8
    factor_min_words: int = 3
    factor_max_decision_overlap: float = 0.85
    top_k_windows: int = 3
    min_similarity: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AggregationResult:
    """
    Aggregated statistics across multiple decision extraction results.

    Attributes:
        total_decisions: Total decisions across all texts.
        total_factors: Total factors across all texts.
        unique_decisions: Number of unique decision texts.
        unique_factors: Number of unique factor texts.
        decision_frequency: Map of decision text to occurrence count.
        factor_frequency: Map of factor text to occurrence count.
        polarity_distribution: Counts per polarity (supporting/opposing/neutral).
        texts_per_decision: Map of decision text to count of texts containing it.
        texts_per_factor: Map of factor text to count of texts containing it.
        avg_decisions_per_text: Average decisions per text.
        avg_factors_per_decision: Average factors per decision.
    """

    total_decisions: int = 0
    total_factors: int = 0
    unique_decisions: int = 0
    unique_factors: int = 0
    decision_frequency: Dict[str, int] = field(default_factory=dict)
    factor_frequency: Dict[str, int] = field(default_factory=dict)
    polarity_distribution: Dict[str, int] = field(default_factory=dict)
    texts_per_decision: Dict[str, int] = field(default_factory=dict)
    texts_per_factor: Dict[str, int] = field(default_factory=dict)
    avg_decisions_per_text: float = 0.0
    avg_factors_per_decision: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AggregationResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
