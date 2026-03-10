"""
Entity detection from raw text using LLM analysis.

This module provides standalone entity extraction, filling the UX gap between
raw text input and the entity scoring pipeline.

Example usage::

    from qualitative_analysis.entity.detector import EntityDetector
    from qualitative_analysis.core.providers import OllamaProvider

    llm = OllamaProvider(model_name="gpt-oss:120b")
    detector = EntityDetector(llm_provider=llm)

    result = await detector.detect(
        text="Abeesee faces a heating crisis...",
        text_id="interview_001",
    )
    print(result.entities)  # ["heating crisis", "renewable energy", ...]
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import logging

from qualitative_analysis.core.entity_extraction import (
    EntityResponse,
    extract_json_object_payload,
)
from qualitative_analysis.entity.prompts.loader import load_entity_prompt
from qualitative_analysis.core.text import SlidingWindowProcessor
from qualitative_analysis.core.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class EntityDetectionResult:
    """Result of entity detection for a single text."""

    text_id: str
    entities: List[str]
    entity_contexts: List[Dict[str, Any]]  # {entity, context, window_index}
    window_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text_id": self.text_id,
            "entities": self.entities,
            "entity_count": len(self.entities),
            "window_count": self.window_count,
            "entity_contexts": self.entity_contexts,
            "metadata": self.metadata,
        }


class EntityDetector:
    """
    Detect entities in text using LLM analysis.

    This class provides a standalone interface for entity extraction with
    its own prompts, independent of the relationships pipeline.

    Args:
        llm_provider: LLM provider instance for generating extractions.
        window_size: Number of units per window (default: 3).
        stride: Window stride in units (default: 2).
        prompt_version: Entity extraction prompt version (default: 1).
        chunk_unit: Windowing unit — 'sentences' or 'tokens' (default: 'sentences').
        tokenizer_name: Tokenizer for token-based chunking (default: 'cl100k_base').

    Example::

        detector = EntityDetector(llm_provider)
        result = await detector.detect(text, text_id="doc1")
        for entity in result.entities:
            print(entity)
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        window_size: int = 3,
        stride: int = 2,
        prompt_version: int = 1,
        chunk_unit: str = "sentences",
        tokenizer_name: str = "cl100k_base",
    ):
        """Initialize the entity detector."""
        self.llm_provider = llm_provider
        self.system_prompt = load_entity_prompt("system_prompt")
        self.prompt_template = load_entity_prompt("entity_extraction", version=prompt_version)
        self.text_processor = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride,
            chunk_unit=chunk_unit,
            tokenizer_name=tokenizer_name,
        )
        self.window_size = window_size
        self.stride = stride
        self.prompt_version = prompt_version
        self.chunk_unit = chunk_unit
        self.tokenizer_name = tokenizer_name

    async def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from a text fragment using the LLM.

        Args:
            text: Text to extract entities from.

        Returns:
            List of entity strings.
        """
        prompt = self.prompt_template.replace("{text}", text)

        response = await self.llm_provider.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
        )

        try:
            data = self._extract_json_payload(response)
            parsed = EntityResponse.model_validate(data)
        except Exception as exc:
            logger.warning(f"Entity extraction parse failed: {exc}")
            return []

        return [e.strip() for e in parsed.entities_and_concepts if e and e.strip()]

    @staticmethod
    def _extract_json_payload(response: str) -> dict:
        """Extract JSON object from an LLM response for entity extraction."""
        return extract_json_object_payload(response, error_context="entity extraction")

    async def detect(
        self,
        text: str,
        text_id: str,
        use_windowing: bool = True,
        max_context_length: int = 500,
    ) -> EntityDetectionResult:
        """
        Extract entities from text.

        Args:
            text: The text to analyze.
            text_id: Identifier for this text (e.g., participant ID).
            use_windowing: Whether to split text into overlapping windows.
                If False, processes entire text as a single unit.
            max_context_length: Maximum characters for context snippets.

        Returns:
            EntityDetectionResult containing extracted entities and contexts.
        """
        if not text or not text.strip():
            logger.warning(f"Empty text for {text_id}")
            return EntityDetectionResult(
                text_id=text_id,
                entities=[],
                entity_contexts=[],
                window_count=0,
            )

        # Split text into windows or process as single unit
        if use_windowing:
            windows = self.text_processor.process(text)
            if not windows:
                windows = [text]
        else:
            windows = [text]

        all_entities: List[str] = []
        entity_contexts: List[Dict[str, Any]] = []

        for idx, window in enumerate(windows):
            try:
                extracted = await self._extract_entities(window)
            except Exception as e:
                logger.error(f"Entity extraction failed for {text_id} window {idx}: {e}")
                extracted = []

            for entity in extracted:
                entity = entity.strip()
                if not entity:
                    continue

                # Track unique entities (case-sensitive)
                if entity not in all_entities:
                    all_entities.append(entity)

                # Store entity-context pair
                context = window[:max_context_length]
                if len(window) > max_context_length:
                    context += "..."

                entity_contexts.append({
                    "entity": entity,
                    "context": context,
                    "window_index": idx,
                })

        logger.info(
            f"Detected {len(all_entities)} unique entities from {len(windows)} windows for {text_id}"
        )

        return EntityDetectionResult(
            text_id=text_id,
            entities=all_entities,
            entity_contexts=entity_contexts,
            window_count=len(windows),
            metadata={
                "use_windowing": use_windowing,
                "window_size": self.window_size if use_windowing else None,
                "stride": self.stride if use_windowing else None,
                "chunk_unit": self.chunk_unit if use_windowing else None,
            },
        )

    async def detect_batch(
        self,
        texts: List[Dict[str, str]],
        text_col: str = "text",
        id_col: Optional[str] = None,
        use_windowing: bool = True,
        max_context_length: int = 500,
    ) -> List[EntityDetectionResult]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of dicts containing text and optional ID.
            text_col: Key for text content in each dict.
            id_col: Key for text ID in each dict (auto-generates if None).
            use_windowing: Whether to split texts into windows.
            max_context_length: Maximum context snippet length.

        Returns:
            List of EntityDetectionResult, one per input text.
        """
        results = []
        for i, row in enumerate(texts):
            text = row.get(text_col, "")
            text_id = row.get(id_col, f"text_{i}") if id_col else f"text_{i}"

            result = await self.detect(
                text=text,
                text_id=text_id,
                use_windowing=use_windowing,
                max_context_length=max_context_length,
            )
            results.append(result)

        return results
