"""
Figurative language scanner component.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional

from pydantic import BaseModel, ConfigDict, Field

from ...core.entity_extraction import extract_json_object_payload
from ...core.llm import BaseLLMProvider
from ..models import Instance
from ..prompts.loader import load_prompt
from ..prompts.types import generate_types_section, get_valid_types_hint, validate_types

logger = logging.getLogger(__name__)

class InstanceModel(BaseModel):
    """Pydantic model for LLM structured output."""
    model_config = ConfigDict(extra="ignore")

    text: str
    type: str
    confidence: float
    explanation: str
    context_dependent: bool


class DetectionResponse(BaseModel):
    """Structured response model for binary figurative detection."""
    model_config = ConfigDict(extra="ignore")

    has_figurative: bool = False
    confidence: float = 0.0
    reasoning: str = ""


class ExtractionEnvelope(BaseModel):
    """Top-level structured extraction envelope."""
    model_config = ConfigDict(extra="ignore")

    instances: List[Dict[str, Any]] = Field(default_factory=list)

class Scanner:
    def __init__(
        self,
        llm: BaseLLMProvider,
        prompt_version: int = 1,
        figurative_types: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.figurative_types = validate_types(figurative_types) if figurative_types else None
        self.system_prompt = load_prompt("system_prompt", version=1)
        
        # Use v2 prompts when type filtering is enabled, otherwise use specified version
        if self.figurative_types is not None:
            # v2 prompts support {figurative_types_section} and {valid_types_hint}
            self.detector_template = load_prompt("two_step_binary_detection", version=2)
            self.extractor_template = load_prompt("two_step_instance_extraction", version=2)
            self._use_type_filtering = True
        else:
            self.detector_template = load_prompt("two_step_binary_detection", version=prompt_version)
            self.extractor_template = load_prompt("two_step_instance_extraction", version=prompt_version)
            self._use_type_filtering = False

    def _format_prompt(self, template: str, text: str, prior_context: str) -> str:
        """Format a prompt template with type-aware substitution."""
        if self._use_type_filtering:
            return template.format(
                text=text,
                prior_summaries=prior_context,
                figurative_types_section=generate_types_section(self.figurative_types),
                valid_types_hint=get_valid_types_hint(self.figurative_types),
            )
        else:
            return template.format(text=text, prior_summaries=prior_context)

    @staticmethod
    def _clamp_confidence(confidence: float) -> float:
        """Clamp confidence scores into the public [0, 1] contract."""
        return max(0.0, min(1.0, float(confidence)))

    @staticmethod
    def _normalize_type(value: str) -> str:
        """Normalize a figurative type label for filtering."""
        return value.lower().strip().replace(" ", "_").replace("-", "_")

    def _build_instances(
        self,
        raw_instances: List[Dict[str, Any]],
        window_index: int,
    ) -> Tuple[List[Instance], int]:
        """Validate raw instances and convert them into public instance models."""
        instances: List[Instance] = []
        invalid_items_skipped = 0

        for item in raw_instances:
            try:
                model = InstanceModel.model_validate(item)
            except Exception:
                invalid_items_skipped += 1
                continue

            if self.figurative_types is not None:
                item_type = self._normalize_type(model.type)
                if item_type not in self.figurative_types:
                    logger.debug(
                        "Skipping instance with type '%s' - not in filter list",
                        model.type,
                    )
                    continue

            instances.append(
                Instance(
                    text=model.text,
                    type=model.type,
                    confidence=self._clamp_confidence(model.confidence),
                    explanation=model.explanation,
                    context_dependent=model.context_dependent,
                    window_index=window_index,
                )
            )

        return instances, invalid_items_skipped

    async def detect(self, text: str, prior_context: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Binary detection of figurative language.
        """
        prompt = self._format_prompt(self.detector_template, text, prior_context)
        metadata = {
            "structured_output_fallback": False,
            "parse_failed": False,
        }

        try:
            parsed = await self.llm.generate_json(
                prompt=prompt,
                schema=DetectionResponse,
                system_prompt=self.system_prompt,
                temperature=0.1,
            )
            return bool(parsed.has_figurative), self._clamp_confidence(parsed.confidence), metadata
        except Exception as exc:
            metadata["structured_output_fallback"] = True
            logger.warning(
                "Structured detection failed; retrying with raw JSON extraction: %s",
                exc,
            )

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.1,
            )
            data = extract_json_object_payload(response, error_context="figurative detection")
            parsed = DetectionResponse.model_validate(data)
            return bool(parsed.has_figurative), self._clamp_confidence(parsed.confidence), metadata
        except Exception as exc:
            logger.warning("Detection parse failed: %s", exc)
            metadata["parse_failed"] = True
            return False, 0.0, metadata

    async def extract(
        self, 
        text: str, 
        prior_context: str,
        window_index: int
    ) -> Tuple[List[Instance], Dict[str, Any]]:
        """
        Extract instances of figurative language.
        """
        prompt = self._format_prompt(self.extractor_template, text, prior_context)
        metadata = {
            "structured_output_fallback": False,
            "parse_failed": False,
            "invalid_items_skipped": 0,
        }

        try:
            parsed = await self.llm.generate_json(
                prompt=prompt,
                schema=ExtractionEnvelope,
                system_prompt=self.system_prompt,
                temperature=0.1,
            )
            raw_instances = parsed.instances
        except Exception as exc:
            metadata["structured_output_fallback"] = True
            logger.warning(
                "Structured extraction failed; retrying with raw JSON extraction: %s",
                exc,
            )

            try:
                response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    temperature=0.1,
                )
                data = extract_json_object_payload(response, error_context="figurative extraction")
                parsed = ExtractionEnvelope.model_validate(data)
                raw_instances = parsed.instances
            except Exception as fallback_exc:
                logger.warning("Extraction parse failed: %s", fallback_exc)
                metadata["parse_failed"] = True
                return [], metadata

        instances, invalid_items_skipped = self._build_instances(raw_instances, window_index)
        metadata["invalid_items_skipped"] = invalid_items_skipped
        return instances, metadata

