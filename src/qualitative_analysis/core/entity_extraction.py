"""Shared parsing helpers for entity extraction responses."""

import json
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class EntityResponse(BaseModel):
    """Structured entity-extraction response used across package pipelines."""

    model_config = ConfigDict(extra="ignore")

    entities_and_concepts: List[str] = Field(default_factory=list)


def extract_json_object_payload(response: str, *, error_context: str) -> dict:
    """Extract a JSON object from an LLM response, handling fences and wrappers."""
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
        raise ValueError(f"Expected JSON object for {error_context}.")
    return data