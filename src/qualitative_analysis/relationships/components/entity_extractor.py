"""
Entity extraction component for relationship pipeline.
"""

import json
import logging
from typing import List

from pydantic import BaseModel, Field, ConfigDict

from ...core.llm import BaseLLMProvider
from ..prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class EntityResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities_and_concepts: List[str] = Field(default_factory=list)


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
            raise ValueError("Expected JSON object for entity extraction.")
        return data
