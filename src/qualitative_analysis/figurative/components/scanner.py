"""
Figurative language scanner component.
"""

import json
import logging
from typing import List, Tuple, Optional
from pydantic import BaseModel
from ...core.llm import BaseLLMProvider
from ..models import Instance
from ..prompts.loader import load_prompt
from ..prompts.types import generate_types_section, get_valid_types_hint, validate_types

logger = logging.getLogger(__name__)

class InstanceModel(BaseModel):
    """Pydantic model for LLM structured output."""
    text: str
    type: str
    confidence: float
    explanation: str
    context_dependent: bool

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

    async def detect(self, text: str, prior_context: str) -> Tuple[bool, float]:
        """
        Binary detection of figurative language.
        """
        prompt = self._format_prompt(self.detector_template, text, prior_context)

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1
        )

        try:
            data = json.loads(response.strip())
            has_figurative = bool(data.get("has_figurative", False))
            confidence = float(data.get("confidence", 0.0))
            if not 0.0 <= confidence <= 1.0:
                confidence = max(0.0, min(1.0, confidence))
            return has_figurative, confidence
        except json.JSONDecodeError:
            logger.warning("Detection returned non-JSON response; using heuristic fallback.")
        except Exception as exc:
            logger.warning(f"Detection parse failed; using heuristic fallback: {exc}")

        lower_resp = response.lower()
        has_figurative = "yes" in lower_resp or "true" in lower_resp
        return has_figurative, 1.0 if has_figurative else 0.0

    async def extract(
        self, 
        text: str, 
        prior_context: str,
        window_index: int
    ) -> List[Instance]:
        """
        Extract instances of figurative language.
        """
        prompt = self._format_prompt(self.extractor_template, text, prior_context)

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.1
            )

            data = json.loads(response.strip())
            raw_instances = data.get("instances", [])
            if not isinstance(raw_instances, list):
                logger.warning("Extraction response 'instances' is not a list.")
                return []

            instances: List[Instance] = []
            for item in raw_instances:
                try:
                    model = InstanceModel.model_validate(item)
                except Exception:
                    continue
                
                # If type filtering is enabled, skip instances that don't match
                if self.figurative_types is not None:
                    item_type = model.type.lower().strip().replace(" ", "_").replace("-", "_")
                    if item_type not in self.figurative_types:
                        logger.debug(f"Skipping instance with type '{model.type}' - not in filter list")
                        continue
                
                instances.append(Instance(
                    text=model.text,
                    type=model.type,
                    confidence=model.confidence,
                    explanation=model.explanation,
                    context_dependent=model.context_dependent,
                    window_index=window_index
                ))
            return instances

        except json.JSONDecodeError:
            logger.warning("Extraction returned non-JSON response.")
            return []
        except Exception as exc:
            logger.warning(f"Extraction failed: {exc}")
            return []

