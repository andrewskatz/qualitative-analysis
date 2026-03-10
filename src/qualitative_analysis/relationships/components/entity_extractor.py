"""
Entity extraction component for relationship pipeline.
"""

import logging
from typing import List

from ...core.entity_extraction import EntityResponse, extract_json_object_payload
from ...core.llm import BaseLLMProvider
from ..prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class EntityExtractor:
    def __init__(self, llm: BaseLLMProvider, prompt_version: int = 1):
        self.llm = llm
        self.system_prompt = load_prompt("system_prompt", version=1)
        self.prompt_template = load_prompt("entity_extraction", version=prompt_version)

    async def extract(self, text: str) -> List[str]:
        prompt = self._render_prompt(text)
        response = await self.llm.generate(
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

    def _render_prompt(self, text: str) -> str:
        prompt = self.prompt_template
        if "{text}" in prompt:
            prompt = prompt.replace("{text}", text)
        return prompt

    def _extract_json_payload(self, response: str) -> dict:
        return extract_json_object_payload(response, error_context="entity extraction")
