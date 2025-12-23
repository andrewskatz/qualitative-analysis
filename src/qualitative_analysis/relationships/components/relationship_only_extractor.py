"""
Relationship-only extraction component for two-pass pipeline.
"""

import json
import logging
from typing import List

from pydantic import BaseModel, Field, ConfigDict

from ...core.llm import BaseLLMProvider
from ..models import Relationship
from ..prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class RelationshipModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    target: str
    type: str
    description: str = ""


class RelationshipOnlyResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relationships: List[RelationshipModel] = Field(default_factory=list)


class RelationshipOnlyExtractor:
    def __init__(self, llm: BaseLLMProvider, prompt_version: int = 1):
        self.llm = llm
        self.system_prompt = load_prompt("system_prompt", version=1)
        self.prompt_template = load_prompt("relationship_only", version=prompt_version)

    async def extract(
        self,
        text: str,
        current_entities: List[str],
        window_index: int,
        summary_context: str | None = None,
    ) -> List[Relationship]:
        prompt = self._render_prompt(text, current_entities, summary_context)
        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
        )

        try:
            data = self._extract_json_payload(response)
            parsed = RelationshipOnlyResponse.model_validate(data)
        except Exception as exc:
            logger.warning(f"Relationship-only parse failed: {exc}")
            return []

        relationships: List[Relationship] = []
        for rel in parsed.relationships:
            relationships.append(
                Relationship(
                    source=rel.source,
                    target=rel.target,
                    type=rel.type,
                    description=rel.description,
                    window_index=window_index,
                )
            )
        return relationships

    def _render_prompt(
        self,
        text: str,
        current_entities: List[str],
        summary_context: str | None,
    ) -> str:
        prompt = self.prompt_template
        if summary_context:
            prompt = f"{summary_context}\n\n{prompt}"
        if "{text}" in prompt:
            prompt = prompt.replace("{text}", text)
        if "{current_entities}" in prompt:
            prompt = prompt.replace("{current_entities}", json.dumps(current_entities, ensure_ascii=True))
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
            raise ValueError("Expected JSON object for relationship extraction.")
        return data
