"""
Causal Analyzer for classifying relationships as causal or non-causal.

Analyzes relationships to determine:
- is_causal: Whether the relationship represents cause-effect
- polarity: positive (increases), negative (decreases), or neutral
- certainty: certain, likely, or possible
- explicit_vs_implicit: whether causality is stated directly or inferred
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from .models import (
    Relationship,
    NormalizedRelationship,
    CausalAttributes,
    CausalRelationship,
    CausalAnalysisResult,
)

logger = logging.getLogger(__name__)

# Path to prompt template
PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_FILE = PROMPT_DIR / "causal_enrichment.txt"

# Valid values for causal attributes
VALID_POLARITIES = {"positive", "negative", "neutral"}
VALID_CERTAINTIES = {"certain", "likely", "possible"}
VALID_EXPLICIT_IMPLICIT = {"explicit", "implicit"}


class CausalAnalyzer:
    """
    Analyzes relationships for causal attributes using LLM.

    The analyzer determines whether each relationship represents a causal
    relationship (cause → effect) and classifies its attributes:
    - polarity: direction of influence (positive/negative/neutral)
    - certainty: confidence level (certain/likely/possible)
    - explicit_vs_implicit: whether causality is stated or inferred

    Example:
        >>> from qualitative_analysis.core.llm import LLMProvider
        >>> analyzer = CausalAnalyzer()
        >>> llm = LLMProvider(model="qwen3:8b", provider="ollama")
        >>> result = await analyzer.analyze(relationships, llm)
        >>> print(f"Found {result.stats['causal_count']} causal relationships")
    """

    def __init__(
        self,
        prompt_template: Optional[str] = None,
        temperature: float = 0.3,
    ):
        """
        Initialize the causal analyzer.

        Args:
            prompt_template: Custom prompt template string. If None, uses default.
            temperature: LLM temperature for generation.
        """
        self.temperature = temperature
        self._is_default_prompt = (prompt_template is None)

        if prompt_template:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = self._load_default_prompt()

    def _load_default_prompt(self) -> str:
        """Load the default prompt template from file."""
        if DEFAULT_PROMPT_FILE.exists():
            return DEFAULT_PROMPT_FILE.read_text(encoding="utf-8")
        else:
            # Fallback prompt if file not found
            logger.warning("Default causal prompt not found, using fallback")
            return """Analyze causality: {source} -> {relationship} -> {target}.
Evidence: {evidence}.
Return JSON with reasoning first: {{"reasoning": "...", "is_causal": bool, "polarity": "positive|negative|neutral", "certainty": "certain|likely|possible", "explicit_vs_implicit": "explicit|implicit"}}"""

    async def analyze(
        self,
        relationships: Union[List[Relationship], List[NormalizedRelationship]],
        llm_provider: Any,
        max_evidence: int = 3,
        batch_size: int = 10,
        on_progress: Optional[callable] = None,
    ) -> CausalAnalysisResult:
        """
        Analyze relationships for causal attributes.

        Args:
            relationships: List of Relationship or NormalizedRelationship objects.
            llm_provider: LLM provider instance with generate() method.
            max_evidence: Maximum evidence snippets to include per relationship.
            batch_size: Number of relationships to process between progress updates.
            on_progress: Optional callback(current, total) for progress tracking.

        Returns:
            CausalAnalysisResult with causal relationships and statistics.
        """
        if not relationships:
            return CausalAnalysisResult(
                causal_relationships=[],
                stats={
                    "total": 0,
                    "causal_count": 0,
                    "non_causal_count": 0,
                },
                config={},
            )

        logger.info(f"Analyzing {len(relationships)} relationships for causal attributes")

        # Statistics tracking
        stats = {
            "total": len(relationships),
            "causal_count": 0,
            "non_causal_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "certain_count": 0,
            "likely_count": 0,
            "possible_count": 0,
            "explicit_count": 0,
            "implicit_count": 0,
            "errors": 0,
        }

        causal_relationships = []

        for i, rel in enumerate(relationships):
            # Extract relationship data
            source = rel.source
            target = rel.target
            rel_type = rel.type

            # Get evidence/descriptions
            evidence = self._get_evidence(rel, max_evidence)

            try:
                # Analyze single relationship
                attrs = await self._analyze_single(
                    source=source,
                    target=target,
                    relationship=rel_type,
                    evidence=evidence,
                    llm_provider=llm_provider,
                )

                # Create CausalRelationship
                causal_rel = CausalRelationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    description=rel.description if hasattr(rel, "description") else "",
                    causal=attrs,
                )

                # Copy additional fields from NormalizedRelationship if present
                if isinstance(rel, NormalizedRelationship):
                    causal_rel.original_source = rel.original_source
                    causal_rel.original_target = rel.original_target
                    causal_rel.original_type = rel.original_type
                    causal_rel.count = rel.count
                    causal_rel.window_indices = rel.window_indices
                    causal_rel.text_ids = rel.text_ids
                elif isinstance(rel, Relationship):
                    causal_rel.window_indices = [rel.window_index]
                    causal_rel.text_ids = [rel.text_id] if rel.text_id else []

                causal_relationships.append(causal_rel)

                # Update statistics
                if attrs.is_causal:
                    stats["causal_count"] += 1
                    if attrs.polarity == "positive":
                        stats["positive_count"] += 1
                    elif attrs.polarity == "negative":
                        stats["negative_count"] += 1
                    else:
                        stats["neutral_count"] += 1

                    if attrs.certainty == "certain":
                        stats["certain_count"] += 1
                    elif attrs.certainty == "likely":
                        stats["likely_count"] += 1
                    elif attrs.certainty == "possible":
                        stats["possible_count"] += 1

                    if attrs.explicit_vs_implicit == "explicit":
                        stats["explicit_count"] += 1
                    elif attrs.explicit_vs_implicit == "implicit":
                        stats["implicit_count"] += 1
                else:
                    stats["non_causal_count"] += 1

            except Exception as e:
                logger.error(f"Error analyzing {source} -> {target}: {e}")
                stats["errors"] += 1

                # Create relationship with error status — NOT counted as non-causal
                causal_rel = CausalRelationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    description=rel.description if hasattr(rel, "description") else "",
                    causal=CausalAttributes(
                        is_causal=False,
                        polarity=None,
                        certainty=None,
                        explicit_vs_implicit=None,
                        reasoning=f"Error: {str(e)}",
                    ),
                )
                causal_relationships.append(causal_rel)

            # Progress callback
            if on_progress and (i + 1) % batch_size == 0:
                on_progress(i + 1, len(relationships))

        # Final progress callback
        if on_progress:
            on_progress(len(relationships), len(relationships))

        # Build config
        config = {
            "temperature": self.temperature,
            "max_evidence": max_evidence,
            "prompt_template": "default" if self._is_default_prompt else "custom",
        }

        logger.info(
            f"Causal analysis complete: {stats['causal_count']} causal, "
            f"{stats['non_causal_count']} non-causal out of {stats['total']}"
        )

        return CausalAnalysisResult(
            causal_relationships=causal_relationships,
            stats=stats,
            config=config,
        )

    def _get_evidence(
        self,
        rel: Union[Relationship, NormalizedRelationship],
        max_evidence: int,
    ) -> List[str]:
        """Extract evidence snippets from a relationship."""
        evidence = []

        # Try descriptions list first (from NormalizedRelationship)
        if hasattr(rel, "descriptions") and rel.descriptions:
            evidence = rel.descriptions[:max_evidence]
        # Fall back to single description
        elif rel.description:
            evidence = [rel.description]

        return evidence

    async def _analyze_single(
        self,
        source: str,
        target: str,
        relationship: str,
        evidence: List[str],
        llm_provider: Any,
    ) -> CausalAttributes:
        """
        Analyze a single relationship for causal attributes.

        Args:
            source: Source entity
            target: Target entity
            relationship: Relationship type
            evidence: List of evidence snippets
            llm_provider: LLM provider instance

        Returns:
            CausalAttributes with analysis results
        """
        # Format evidence
        if evidence:
            evidence_text = "\n".join(f"- {e}" for e in evidence)
        else:
            evidence_text = "No direct evidence available"

        # Format prompt
        prompt = self._prompt_template.format(
            source=source,
            target=target,
            relationship=relationship,
            evidence=evidence_text,
        )

        logger.debug(f"Analyzing: {source} -> {relationship} -> {target}")

        # Call LLM
        response = await llm_provider.generate(
            prompt=prompt,
            temperature=self.temperature,
        )

        # Parse response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> CausalAttributes:
        """Parse LLM response into CausalAttributes."""
        clean_response = response.strip()

        # Strip markdown code blocks
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()

        # Try to parse JSON
        try:
            result = json.loads(clean_response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*"is_causal"[^{}]*\}', clean_response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                # More aggressive extraction
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError(f"No valid JSON found in response: {clean_response[:200]}")

        # Extract and validate attributes
        is_causal = result.get("is_causal", False)
        reasoning = result.get("reasoning", result.get("explanation", ""))

        if is_causal:
            polarity = result.get("polarity")
            certainty = result.get("certainty")
            explicit_vs_implicit = result.get("explicit_vs_implicit")

            # Validate values — log when defaulting from invalid LLM output
            if polarity not in VALID_POLARITIES:
                logger.warning(f"Invalid polarity '{polarity}', defaulting to 'neutral'")
                polarity = "neutral"
            if certainty not in VALID_CERTAINTIES:
                logger.warning(f"Invalid certainty '{certainty}', defaulting to 'possible'")
                certainty = "possible"
            if explicit_vs_implicit not in VALID_EXPLICIT_IMPLICIT:
                logger.warning(f"Invalid explicit_vs_implicit '{explicit_vs_implicit}', defaulting to 'implicit'")
                explicit_vs_implicit = "implicit"
        else:
            polarity = None
            certainty = None
            explicit_vs_implicit = None

        return CausalAttributes(
            is_causal=is_causal,
            polarity=polarity,
            certainty=certainty,
            explicit_vs_implicit=explicit_vs_implicit,
            reasoning=reasoning,
        )

    def save(self, result: CausalAnalysisResult, path: Union[str, Path]) -> None:
        """Save causal analysis result to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved causal analysis result to {path}")

    def load(self, path: Union[str, Path]) -> CausalAnalysisResult:
        """Load causal analysis result from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return CausalAnalysisResult.from_dict(data)


async def analyze_causal(
    relationships: Union[List[Relationship], List[NormalizedRelationship]],
    llm_provider: Any,
    temperature: float = 0.3,
    max_evidence: int = 3,
) -> CausalAnalysisResult:
    """
    Convenience function to analyze relationships for causal attributes.

    Args:
        relationships: List of Relationship or NormalizedRelationship objects.
        llm_provider: LLM provider instance with generate() method.
        temperature: LLM temperature for generation.
        max_evidence: Maximum evidence snippets per relationship.

    Returns:
        CausalAnalysisResult with causal relationships and statistics.
    """
    analyzer = CausalAnalyzer(temperature=temperature)
    return await analyzer.analyze(
        relationships,
        llm_provider,
        max_evidence=max_evidence,
    )
