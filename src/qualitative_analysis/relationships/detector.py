"""
Main relationship extraction detector.
"""

from typing import Optional, Dict, Any, List

import json
import logging
from ..core.llm import BaseLLMProvider
from ..core.text import SlidingWindowProcessor
from .components.context_buffer import EntityBuffer
from .components.entity_extractor import EntityExtractor
from .components.extractor import RelationshipExtractor
from .components.relationship_only_extractor import RelationshipOnlyExtractor
from .components.summarizer import Summarizer
from .components.summary_buffer import SummaryBuffer
from .models import RelationshipResult, Relationship
from .prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class RelationshipDetector:
    """
    High-level interface for extracting relationships from text.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        strategy: str = "two_pass",
        prompt_version: int = 2,
        entity_prompt_version: int = 1,
        relationship_prompt_version: int = 1,
        coref_prompt_version: int = 1,
        # Text processor config
        window_size: int = 3,
        stride: int = 2,
        chunk_unit: str = "sentences",
        tokenizer_name: str = "cl100k_base",
        summary_buffer_size: int = 5,
        summary_prompt_version: int = 1,
        enable_summaries: bool = True,
        include_summaries_in_prompt: bool = False,
        summary_min_windows: int = 2,
        # Return window-level results
        return_windows: bool = False,
        # Context buffer
        context_buffer_size: int = 0,
        # Coreference resolution
        coref_resolution: bool = False,
    ):
        self.llm = llm_provider

        if strategy not in {"two_pass", "one_pass"}:
            raise ValueError(f"Unsupported strategy: {strategy}")

        self.strategy = strategy
        self.text_processor = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride,
            chunk_unit=chunk_unit,
            tokenizer_name=tokenizer_name,
        )
        self.entity_extractor = EntityExtractor(self.llm, prompt_version=entity_prompt_version)
        self.relationship_extractor = RelationshipExtractor(self.llm, prompt_version=prompt_version)
        self.relationship_only_extractor = RelationshipOnlyExtractor(
            self.llm, prompt_version=relationship_prompt_version
        )
        self.summarizer = Summarizer(self.llm, prompt_version=summary_prompt_version)
        self.summary_buffer_size = summary_buffer_size
        self.enable_summaries = enable_summaries
        self.include_summaries_in_prompt = include_summaries_in_prompt
        self.summary_min_windows = max(1, summary_min_windows)
        self.return_windows = return_windows
        self.context_buffer = EntityBuffer(context_buffer_size)
        self.coref_resolution = coref_resolution
        self.coref_prompt = load_prompt("coref_resolution", version=coref_prompt_version)
        self.system_prompt = load_prompt("system_prompt", version=1)

    async def detect(
        self,
        text: str,
        current_entities: Optional[List[str]] = None,
        text_id: str = "",
    ) -> RelationshipResult:
        # Empty text guard
        if not text or not text.strip():
            logger.warning(f"Empty text for text_id={text_id}, returning empty result")
            return RelationshipResult(
                entities=[],
                relationships=[],
                metadata={"strategy": f"relationship_{self.strategy}", "window_count": 0,
                          "relationship_count": 0, "entity_count": 0, "windows": []},
            )

        # Reset context buffer per detect() call to prevent cross-text contamination
        self.context_buffer = EntityBuffer(self.context_buffer.max_size)

        if self.coref_resolution:
            text = await self._resolve_coreferences(text)

        windows = self.text_processor.process(text)
        if not windows:
            return RelationshipResult(
                entities=[],
                relationships=[],
                metadata={"strategy": f"relationship_{self.strategy}", "window_count": 0,
                          "relationship_count": 0, "entity_count": 0, "windows": []},
            )

        summaries_enabled = self.enable_summaries and len(windows) >= self.summary_min_windows
        summary_buffer = SummaryBuffer(self.summary_buffer_size) if summaries_enabled else None
        all_entities: List[str] = []
        all_relationships: List[Relationship] = []
        window_results: List[Dict[str, Any]] = []

        for i, window in enumerate(windows):
            try:
                summary = None
                summary_context = None
                if summary_buffer is not None:
                    summary = await self.summarizer.summarize(
                        window,
                        summary_buffer.get_context(),
                        window_index=i,
                    )
                    summary_buffer.add(summary)
                    if self.include_summaries_in_prompt:
                        summary_context = summary_buffer.get_formatted_context()
                        if summary_context == "No prior context.":
                            summary_context = None
                entities_seed = self._merge_entities(
                    current_entities or [],
                    self.context_buffer.get_entities(),
                )

                if self.strategy == "two_pass":
                    entities_from_llm = await self.entity_extractor.extract(window)
                    entities = self._merge_entities(entities_seed, entities_from_llm)
                    relationships = await self.relationship_only_extractor.extract(
                        window,
                        entities,
                        window_index=i,
                        summary_context=summary_context,
                    )
                    self.context_buffer.add(entities)
                else:
                    entities, relationships = await self.relationship_extractor.extract(
                        window,
                        entities_seed,
                        window_index=i,
                        summary_context=summary_context,
                    )
                    self.context_buffer.add(entities)

                # Propagate text_id to all extracted relationships
                if text_id:
                    for rel in relationships:
                        rel.text_id = text_id

                all_entities.extend(entities)
                all_relationships.extend(relationships)

                if self.return_windows:
                    window_results.append(
                        {
                            "window_index": i,
                            "window_text": window,
                            "summary": summary.text if summary else "",
                            "entities": entities,
                            "relationships": [rel.to_dict() for rel in relationships],
                            "relationship_count": len(relationships),
                        }
                    )
            except Exception as e:
                logger.error(f"Extraction failed for window {i} (text_id={text_id}): {e}")
                continue

        # Case-insensitive dedup preserving first-seen casing
        seen_lower: set = set()
        dedup_entities: List[str] = []
        for e in all_entities:
            if e and e.lower() not in seen_lower:
                dedup_entities.append(e)
                seen_lower.add(e.lower())
        dedup_entities.sort()
        return RelationshipResult(
            entities=dedup_entities,
            relationships=all_relationships,
            metadata={
                "strategy": f"relationship_{self.strategy}",
                "window_count": len(windows),
                "relationship_count": len(all_relationships),
                "entity_count": len(dedup_entities),
                "windows": window_results if self.return_windows else [],
            },
        )

    def _merge_entities(self, *lists: List[str]) -> List[str]:
        merged: List[str] = []
        seen: set = set()
        for items in lists:
            for item in items:
                value = item.strip() if isinstance(item, str) else ""
                if value and value.lower() not in seen:
                    merged.append(value)
                    seen.add(value.lower())
        return merged

    async def _resolve_coreferences(self, text: str) -> str:
        prompt = self.coref_prompt.replace("{text}", text)
        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,
            )
            data = self._extract_json_payload(response)
            resolved = data.get("resolved_text", "").strip()
            return resolved or text
        except Exception as exc:
            logger.warning(f"Coreference resolution failed: {exc}")
            return text

    def _extract_json_payload(self, response: str) -> dict:
        content = response.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]

        data = json.loads(content)
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object for coreference resolution.")
        return data
