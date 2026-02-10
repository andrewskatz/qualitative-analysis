"""
Relationship Verifier for validating extracted relationships using a second LLM pass.

This module provides functionality for verifying relationships by re-prompting
the LLM with the original source text and the extracted relationships, asking
it to assess whether each relationship is actually supported by the text.

Example usage:
    from qualitative_analysis.relationships import RelationshipVerifier
    from qualitative_analysis.core.providers import OllamaProvider

    verifier = RelationshipVerifier()
    llm = OllamaProvider(model_name="qwen3:8b")

    result = await verifier.verify(
        relationships=relationships,
        source_texts={"text_1": "The rain caused flooding..."},
        llm_provider=llm,
        confidence_threshold=0.7
    )
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import (
    Relationship,
    NormalizedRelationship,
    VerifiedRelationship,
)

logger = logging.getLogger(__name__)

# Path to prompt template
PROMPT_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_FILE = PROMPT_DIR / "relationship_verification_v1.txt"


@dataclass
class VerificationResult:
    """
    Result from relationship verification.

    Attributes:
        verified_relationships: List of relationships that passed verification.
        rejected_relationships: List of relationships that failed verification.
        all_verifications: All relationships with verification status.
        statistics: Summary statistics.
        config: Configuration used for verification.
    """
    verified_relationships: List[VerifiedRelationship] = field(default_factory=list)
    rejected_relationships: List[VerifiedRelationship] = field(default_factory=list)
    all_verifications: List[VerifiedRelationship] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        total = len(self.all_verifications)
        verified = len(self.verified_relationships)
        rejected = len(self.rejected_relationships)

        # Confidence statistics
        if self.all_verifications:
            confidences = [v.verification_confidence for v in self.all_verifications]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
        else:
            avg_confidence = min_confidence = max_confidence = 0.0

        self.statistics = {
            "total": total,
            "verified_count": verified,
            "rejected_count": rejected,
            "verification_rate": round(100 * verified / total, 1) if total > 0 else 0,
            "avg_confidence": round(avg_confidence, 3),
            "min_confidence": round(min_confidence, 3),
            "max_confidence": round(max_confidence, 3),
        }
        return self.statistics

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "verified_relationships": [v.to_dict() for v in self.verified_relationships],
            "rejected_relationships": [v.to_dict() for v in self.rejected_relationships],
            "all_verifications": [v.to_dict() for v in self.all_verifications],
            "statistics": self.statistics or self.compute_statistics(),
            "config": self.config,
        }


class RelationshipVerifier:
    """
    Verifies relationships by re-prompting the LLM with source text.

    The verifier groups relationships by their source text (using text_id)
    and processes them in batches. For each batch, it asks the LLM to verify
    whether each relationship is actually supported by the text.

    This helps filter out hallucinated or unsupported relationships from
    the initial extraction pass.

    Attributes:
        temperature: LLM temperature for verification (low for consistency).
        confidence_threshold: Minimum confidence to accept a relationship.
        batch_size: Maximum relationships per LLM call.

    Example:
        >>> verifier = RelationshipVerifier(confidence_threshold=0.7)
        >>> result = await verifier.verify(
        ...     relationships=rels,
        ...     source_texts={"doc1": "The text content..."},
        ...     llm_provider=llm
        ... )
        >>> print(f"Verified: {result.statistics['verified_count']}/{result.statistics['total']}")
    """

    DEFAULT_CONFIDENCE_THRESHOLD = 0.7
    DEFAULT_BATCH_SIZE = 20

    def __init__(
        self,
        prompt_template: Optional[str] = None,
        temperature: float = 0.1,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the relationship verifier.

        Args:
            prompt_template: Custom prompt template. If None, uses default.
            temperature: LLM temperature (low for consistent verification).
            confidence_threshold: Minimum confidence to accept relationship.
            batch_size: Maximum relationships to verify in one LLM call.
        """
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        if prompt_template:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = self._load_default_prompt()

    def _load_default_prompt(self) -> str:
        """Load the default prompt template from file."""
        if DEFAULT_PROMPT_FILE.exists():
            return DEFAULT_PROMPT_FILE.read_text(encoding="utf-8")
        else:
            logger.warning("Default verification prompt not found, using fallback")
            return """Verify these relationships in the text:
Text: {text}
Relationships: {relationships_list}

Return JSON with "verifications": [{{"original_index": i, "is_supported": bool, "confidence": float, "correction": str|null, "evidence": str|null}}]"""

    async def verify(
        self,
        relationships: Union[List[Relationship], List[NormalizedRelationship]],
        source_texts: Optional[Dict[str, str]] = None,
        llm_provider: Any = None,
        confidence_threshold: Optional[float] = None,
        on_progress: Optional[callable] = None,
        window_texts: Optional[Dict[Tuple[str, int], str]] = None,
    ) -> VerificationResult:
        """
        Verify relationships against their source texts.

        Supports two modes:
        1. Window-level verification (preferred): Uses window_texts dict keyed by (text_id, window_index)
        2. Document-level verification: Falls back to source_texts dict keyed by text_id

        When window_texts is provided, relationships are verified against the specific
        text window they were extracted from, which is more accurate than using the
        full document.

        Args:
            relationships: List of relationships to verify.
            source_texts: Dict mapping text_id to full source text content (fallback).
            llm_provider: LLM provider instance with generate() method.
            confidence_threshold: Override the default confidence threshold.
            on_progress: Optional callback(current, total) for progress.
            window_texts: Dict mapping (text_id, window_index) to window text content (preferred).

        Returns:
            VerificationResult with verified and rejected relationships.
        """
        if not relationships:
            return VerificationResult(config={"confidence_threshold": self.confidence_threshold})

        threshold = confidence_threshold or self.confidence_threshold
        source_texts = source_texts or {}
        window_texts = window_texts or {}

        # Determine verification mode
        use_window_level = bool(window_texts)
        mode_str = "window-level" if use_window_level else "document-level"
        logger.info(f"Verifying {len(relationships)} relationships ({mode_str}, threshold={threshold})")

        all_verified = []
        verified_rels = []
        rejected_rels = []
        processed = 0

        if use_window_level:
            # Group by (text_id, window_index) for window-level verification
            by_window = self._group_by_window(relationships)

            for (text_id, window_index), rels_for_window in by_window.items():
                # Try window text first, then fall back to full document
                text = window_texts.get((text_id, window_index), "")
                if not text:
                    text = source_texts.get(text_id, "")

                if not text:
                    logger.warning(
                        f"No text found for text_id={text_id}, window={window_index}, "
                        f"skipping {len(rels_for_window)} relationships"
                    )
                    for rel in rels_for_window:
                        vrel = self._to_verified_relationship(
                            rel, verified=False, confidence=0.0, note="No source text available"
                        )
                        all_verified.append(vrel)
                        rejected_rels.append(vrel)
                    processed += len(rels_for_window)
                    if on_progress:
                        on_progress(processed, len(relationships))
                    continue

                # Process batch (usually small since grouped by window)
                try:
                    batch_results = await self._verify_batch(
                        rels_for_window, text, llm_provider
                    )

                    for vrel in batch_results:
                        all_verified.append(vrel)
                        if vrel.verified and vrel.verification_confidence >= threshold:
                            verified_rels.append(vrel)
                        else:
                            rejected_rels.append(vrel)

                except Exception as e:
                    logger.error(f"Error verifying batch for text_id={text_id}, window={window_index}: {e}")
                    for rel in rels_for_window:
                        vrel = self._to_verified_relationship(
                            rel, verified=False, confidence=0.0, note=f"Verification error: {str(e)}"
                        )
                        all_verified.append(vrel)
                        rejected_rels.append(vrel)

                processed += len(rels_for_window)
                if on_progress:
                    on_progress(processed, len(relationships))

        else:
            # Group by text_id for document-level verification
            by_text_id = self._group_by_text_id(relationships)

            for text_id, rels_for_text in by_text_id.items():
                source_text = source_texts.get(text_id, "")
                if not source_text:
                    logger.warning(
                        f"No source text found for text_id={text_id}, "
                        f"skipping {len(rels_for_text)} relationships"
                    )
                    for rel in rels_for_text:
                        vrel = self._to_verified_relationship(
                            rel, verified=False, confidence=0.0, note="No source text available"
                        )
                        all_verified.append(vrel)
                        rejected_rels.append(vrel)
                    processed += len(rels_for_text)
                    if on_progress:
                        on_progress(processed, len(relationships))
                    continue

                # Process in batches for longer documents
                for batch_start in range(0, len(rels_for_text), self.batch_size):
                    batch = rels_for_text[batch_start:batch_start + self.batch_size]

                    try:
                        batch_results = await self._verify_batch(
                            batch, source_text, llm_provider
                        )

                        for vrel in batch_results:
                            all_verified.append(vrel)
                            if vrel.verified and vrel.verification_confidence >= threshold:
                                verified_rels.append(vrel)
                            else:
                                rejected_rels.append(vrel)

                    except Exception as e:
                        logger.error(f"Error verifying batch for text_id={text_id}: {e}")
                        for rel in batch:
                            vrel = self._to_verified_relationship(
                                rel, verified=False, confidence=0.0, note=f"Verification error: {str(e)}"
                            )
                            all_verified.append(vrel)
                            rejected_rels.append(vrel)

                    processed += len(batch)
                    if on_progress:
                        on_progress(processed, len(relationships))

        # Build result
        result = VerificationResult(
            verified_relationships=verified_rels,
            rejected_relationships=rejected_rels,
            all_verifications=all_verified,
            config={
                "confidence_threshold": threshold,
                "temperature": self.temperature,
                "batch_size": self.batch_size,
                "verification_mode": mode_str,
            },
        )
        result.compute_statistics()

        logger.info(
            f"Verification complete: {result.statistics['verified_count']}/{result.statistics['total']} "
            f"verified ({result.statistics['verification_rate']}%)"
        )

        return result

    def _group_by_window(
        self,
        relationships: Union[List[Relationship], List[NormalizedRelationship]],
    ) -> Dict[Tuple[str, int], List]:
        """Group relationships by (text_id, window_index)."""
        by_window: Dict[Tuple[str, int], List] = defaultdict(list)

        for rel in relationships:
            if isinstance(rel, NormalizedRelationship):
                text_id = rel.text_ids[0] if rel.text_ids else ""
                window_index = rel.window_indices[0] if rel.window_indices else 0
            else:
                text_id = rel.text_id or ""
                window_index = rel.window_index

            by_window[(text_id, window_index)].append(rel)

        return dict(by_window)

    def _group_by_text_id(
        self,
        relationships: Union[List[Relationship], List[NormalizedRelationship]],
    ) -> Dict[str, List]:
        """Group relationships by their text_id."""
        by_text_id = defaultdict(list)

        for rel in relationships:
            # Handle different relationship types
            if isinstance(rel, NormalizedRelationship):
                # Normalized relationships may have multiple text_ids
                # Use the first one, or a combined key
                text_ids = rel.text_ids
                text_id = text_ids[0] if text_ids else ""
            else:
                text_id = rel.text_id or ""

            by_text_id[text_id].append(rel)

        return dict(by_text_id)

    async def _verify_batch(
        self,
        relationships: List,
        source_text: str,
        llm_provider: Any,
    ) -> List[VerifiedRelationship]:
        """Verify a batch of relationships against source text."""
        # Format relationships for prompt
        rels_formatted = ""
        for i, rel in enumerate(relationships):
            desc = rel.description if hasattr(rel, "description") and rel.description else ""
            rels_formatted += f"{i}. {rel.source} --[{rel.type}]--> {rel.target}"
            if desc:
                rels_formatted += f" ({desc})"
            rels_formatted += "\n"

        # Prepare prompt
        prompt = self._prompt_template.format(
            text=source_text,
            relationships_list=rels_formatted,
        )

        # Call LLM
        response = await llm_provider.generate(
            prompt=prompt,
            temperature=self.temperature,
        )

        # Parse response
        verifications = self._parse_response(response)

        # Map verifications back to relationships
        results = []
        verification_map = {v["original_index"]: v for v in verifications}

        if len(verification_map) < len(relationships):
            logger.warning(
                f"LLM returned verifications for {len(verification_map)}/{len(relationships)} "
                f"relationships — missing ones will be marked unverified"
            )

        for i, rel in enumerate(relationships):
            ver = verification_map.get(i, {})
            is_supported = ver.get("is_supported", False)
            confidence = ver.get("confidence", 0.0)
            correction = ver.get("correction")
            evidence = ver.get("evidence")

            note = ""
            if i not in verification_map:
                note = "Not evaluated — missing from LLM verification response"
            if correction:
                note = f"Correction: {correction}"
            if evidence:
                if note:
                    note += f" | Evidence: {evidence}"
                else:
                    note = f"Evidence: {evidence}"

            vrel = self._to_verified_relationship(
                rel,
                verified=is_supported,
                confidence=confidence,
                note=note,
            )
            results.append(vrel)

        return results

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into verification list."""
        clean_response = response.strip()

        # Strip markdown code blocks
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        if clean_response.startswith("```"):
            clean_response = clean_response[3:]
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]
        clean_response = clean_response.strip()

        try:
            result = json.loads(clean_response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            json_match = re.search(r'\{[^{}]*"verifications"[^{}]*\[.*?\]\s*\}', clean_response, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            else:
                # More aggressive extraction
                json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse verification response: {clean_response[:200]}")
                        return []
                else:
                    logger.error(f"No JSON found in verification response: {clean_response[:200]}")
                    return []

        verifications = result.get("verifications", [])

        # Validate and normalize
        normalized = []
        for v in verifications:
            if "original_index" not in v:
                continue

            normalized.append({
                "original_index": int(v["original_index"]),
                "is_supported": bool(v.get("is_supported", False)),
                "confidence": float(v.get("confidence", 0.5)),
                "correction": v.get("correction"),
                "evidence": v.get("evidence"),
            })

        return normalized

    def _to_verified_relationship(
        self,
        rel: Union[Relationship, NormalizedRelationship],
        verified: bool,
        confidence: float,
        note: str = "",
    ) -> VerifiedRelationship:
        """Convert a relationship to VerifiedRelationship."""
        # Get window_index
        if isinstance(rel, NormalizedRelationship):
            window_index = rel.window_indices[0] if rel.window_indices else 0
            text_id = rel.text_ids[0] if rel.text_ids else ""
        else:
            window_index = rel.window_index
            text_id = rel.text_id

        return VerifiedRelationship(
            source=rel.source,
            target=rel.target,
            type=rel.type,
            description=rel.description if hasattr(rel, "description") else "",
            window_index=window_index,
            text_id=text_id,
            verification_confidence=confidence,
            verification_note=note,
            verified=verified,
            original_relationship=rel if isinstance(rel, Relationship) else None,
        )

    def save(self, result: VerificationResult, path: Union[str, Path]) -> None:
        """Save verification result to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved verification result to {path}")

    def load(self, path: Union[str, Path]) -> VerificationResult:
        """Load verification result from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct VerifiedRelationship objects
        verified = [
            VerifiedRelationship(**v) for v in data.get("verified_relationships", [])
        ]
        rejected = [
            VerifiedRelationship(**v) for v in data.get("rejected_relationships", [])
        ]
        all_vers = [
            VerifiedRelationship(**v) for v in data.get("all_verifications", [])
        ]

        return VerificationResult(
            verified_relationships=verified,
            rejected_relationships=rejected,
            all_verifications=all_vers,
            statistics=data.get("statistics", {}),
            config=data.get("config", {}),
        )


async def verify_relationships(
    relationships: Union[List[Relationship], List[NormalizedRelationship]],
    llm_provider: Any,
    source_texts: Optional[Dict[str, str]] = None,
    window_texts: Optional[Dict[Tuple[str, int], str]] = None,
    confidence_threshold: float = 0.7,
    temperature: float = 0.1,
) -> VerificationResult:
    """
    Convenience function to verify relationships.

    Args:
        relationships: List of relationships to verify.
        llm_provider: LLM provider instance with generate() method.
        source_texts: Dict mapping text_id to full source text (fallback).
        window_texts: Dict mapping (text_id, window_index) to window text (preferred).
        confidence_threshold: Minimum confidence to accept relationship.
        temperature: LLM temperature for verification.

    Returns:
        VerificationResult with verified and rejected relationships.
    """
    verifier = RelationshipVerifier(
        temperature=temperature,
        confidence_threshold=confidence_threshold,
    )
    return await verifier.verify(
        relationships,
        source_texts=source_texts,
        llm_provider=llm_provider,
        confidence_threshold=confidence_threshold,
        window_texts=window_texts,
    )
