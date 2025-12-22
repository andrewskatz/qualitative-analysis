"""
Figurative language scanner component.
"""

import json
import logging
from typing import List, Tuple
from pydantic import BaseModel
from ...core.llm import BaseLLMProvider
from ..models import Instance
from ..prompts.loader import load_prompt

logger = logging.getLogger(__name__)

class InstanceModel(BaseModel):
    """Pydantic model for LLM structured output."""
    text: str
    type: str
    confidence: float
    explanation: str
    context_dependent: bool

class Scanner:
    def __init__(self, llm: BaseLLMProvider, prompt_version: int = 1):
        self.llm = llm
        self.system_prompt = load_prompt("system_prompt", version=1)
        self.detector_template = load_prompt("two_step_binary_detection", version=prompt_version)
        self.extractor_template = load_prompt("two_step_instance_extraction", version=prompt_version)

    async def detect(self, text: str, prior_context: str) -> Tuple[bool, float]:
        """
        Binary detection of figurative language.
        """
        prompt = self.detector_template.format(text=text, prior_summaries=prior_context)

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
        prompt = self.extractor_template.format(text=text, prior_summaries=prior_context)

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
