"""
Concrete LLM provider implementations.
"""

from typing import Optional, Type, Dict, Any
import json
import logging

import httpx
from pydantic import BaseModel

from .llm import BaseLLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    """
    Provider for local Ollama models.
    """
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = self.config.pop("base_url", "http://localhost:11434")
        self.timeout = self.config.pop("timeout", 60.0)
        self.log_prompts = bool(self.config.pop("log_prompts", False))
        self.log_responses = bool(self.config.pop("log_responses", self.log_prompts))

    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        **kwargs
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        # Merge defaults with kwargs
        options = self.config.copy()
        options.update(kwargs)

        # Ollama expects "format" at the top level, not in options.
        fmt = options.pop("format", None)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        if fmt:
            payload["format"] = fmt

        if self.log_prompts:
            print("\n[LLM SYSTEM PROMPT]", flush=True)
            print(system_prompt or "", flush=True)
            print("\n[LLM USER PROMPT]", flush=True)
            print(prompt, flush=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload
            )
            response.raise_for_status()
            data = response.json()

        content = data["message"]["content"]
        if self.log_responses:
            print("\n[LLM RESPONSE]", flush=True)
            print(content, flush=True)

        return content

    async def generate_json(
        self, 
        prompt: str, 
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None, 
        **kwargs
    ) -> BaseModel:
        # For structured output, we can use format='json' (if supported) 
        # or just prompt engineering + pydantic validation.
        # Ollama supports 'format="json"' in newer versions.
        
        system_prompt = system_prompt or ""
        system_prompt += f"\nRespond strictly in JSON matching this schema: {schema.model_json_schema()}"

        response_text = await self.generate(prompt, system_prompt, format="json", **kwargs)
        
        try:
            # Parse JSON and validate
            data = json.loads(response_text)
            return schema.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}. Response: {response_text}")
            raise
