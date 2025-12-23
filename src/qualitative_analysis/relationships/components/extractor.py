"""
Relationship extraction component.
"""

import json
import logging
from typing import List, Optional, Tuple

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


class ExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: List[str] = Field(default_factory=list)
    relationships: List[RelationshipModel] = Field(default_factory=list)


class RelationshipExtractor:
    def __init__(self, llm: BaseLLMProvider, prompt_version: int = 2):
        self.llm = llm
        self.system_prompt = load_prompt("system_prompt", version=1)
        self.prompt_template = load_prompt("relationship_extraction", version=prompt_version)

    async def extract(
        self,
        text: str,
        current_entities: Optional[List[str]],
        window_index: int,
        summary_context: Optional[str] = None,
    ) -> Tuple[List[str], List[Relationship]]:
        formatted_entities = self._format_entities(current_entities)
        prompt = self._render_prompt(text, formatted_entities, summary_context)

        response = await self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            temperature=0.1,
        )

        try:
            data = self._extract_json_payload(response)
            parsed = ExtractionResponse.model_validate(data)
        except Exception as exc:
            logger.warning(f"Relationship extraction parse failed: {exc}")
            return [], []

        entities = [e.strip() for e in parsed.entities if e and e.strip()]
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

        return entities, relationships

    def _format_entities(self, entities: Optional[List[str]]) -> str:
        if not entities:
            return "[]"
        return json.dumps([e for e in entities if e], ensure_ascii=True)

    def _render_prompt(self, text: str, current_entities: str, summary_context: Optional[str]) -> str:
        prompt = self.prompt_template
        if summary_context:
            prompt = f"{summary_context}\n\n{prompt}"
        if "{text}" in prompt:
            prompt = prompt.replace("{text}", text)
        if "{current_entities}" in prompt:
            prompt = prompt.replace("{current_entities}", current_entities)
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
        elif "\"entities\"" in content or "'entities'" in content:
            content = "{" + content + "}"

        data = json.loads(content)
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError("Expected JSON object for relationship extraction.")
        return data
