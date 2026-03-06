"""
Concrete LLM provider implementations.
"""

from typing import Optional, Type, Dict, Any
import asyncio
import gc
import json
import logging
import re

import httpx
from pydantic import BaseModel

from .llm import BaseLLMProvider

logger = logging.getLogger(__name__)

class OllamaProvider(BaseLLMProvider):
    """
    Provider for local Ollama models.
    """

    _STOP_SEQUENCES = [
        "<|endoftext|>", "<|end|>", "<|im_end|>", "</s>", "<|end_of_text|>",
    ]

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = self.config.pop("base_url", "http://localhost:11434")
        self.timeout = self.config.pop("timeout", 60.0)
        self.enable_thinking = bool(self.config.pop("enable_thinking", False))
        self.log_prompts = bool(self.config.pop("log_prompts", False))
        self.log_responses = bool(self.config.pop("log_responses", self.log_prompts))
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client for connection reuse."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the persistent HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

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

        # Inject stop sequences so the model halts at EOS tokens
        # instead of continuing to generate garbage after valid output.
        options.setdefault("stop", self._STOP_SEQUENCES)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "options": options,
            "stream": False,
            "think": self.enable_thinking,
        }
        if fmt:
            payload["format"] = fmt

        if self.log_prompts:
            print("\n[LLM SYSTEM PROMPT]", flush=True)
            print(system_prompt or "", flush=True)
            print("\n[LLM USER PROMPT]", flush=True)
            print(prompt, flush=True)

        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/api/chat",
            json=payload
        )
        response.raise_for_status()
        data = response.json()

        content = data["message"]["content"]
        thinking = data["message"].get("thinking", "")

        if self.log_responses:
            if thinking:
                print("\n[LLM THINKING]", flush=True)
                print(thinking, flush=True)
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


class MLXProvider(BaseLLMProvider):
    """
    Provider for local MLX models using mlx-vlm on Apple Silicon.

    Supports Qwen3.5 MoE models and other mlx-vlm compatible models.
    The model is loaded once at initialization and reused across all
    generate() calls within the session.

    Requires: pip install 'mlx-vlm>=0.3.12' torchvision
    """

    _END_TOKENS = [
        "<|endoftext|>", "<|end|>", "<|im_end|>", "</s>", "<|end_of_text|>",
    ]

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        self.max_tokens = self.config.pop("max_tokens", 2048)
        self.enable_thinking = bool(self.config.pop("enable_thinking", False))
        self.log_prompts = bool(self.config.pop("log_prompts", False))
        self.log_responses = bool(self.config.pop("log_responses", self.log_prompts))

        # Import mlx_vlm with clear error message
        try:
            from mlx_vlm import load as _mlx_load, generate as _mlx_generate
            self._mlx_load = _mlx_load
            self._mlx_generate = _mlx_generate
        except ImportError as e:
            raise RuntimeError(
                "mlx-vlm is required for MLXProvider. "
                "Install with: pip install 'mlx-vlm>=0.3.12' torchvision"
            ) from e

        # Load model once at init
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        """Load model and processor. Called once at init."""
        logger.info(f"Loading MLX model: {self.model_name}")
        print(f"Loading MLX model: {self.model_name}")
        print("  (This may take 30-60 seconds on first load...)")

        self._model, self._processor = self._mlx_load(self.model_name)

        logger.info(f"MLX model loaded: {self.model_name}")
        print(f"  Model loaded successfully.")

    def _build_chat_prompt(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> str:
        """Format messages using the processor's chat template.

        Passes enable_thinking to the template for reasoning models
        (Qwen3.5 etc.) — False by default to suppress <think> blocks
        and get direct structured output.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking,
        )

        # Try processor.tokenizer (most common for mlx-vlm)
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Model doesn't support enable_thinking kwarg — retry without it
                template_kwargs.pop("enable_thinking", None)
                try:
                    return tokenizer.apply_chat_template(messages, **template_kwargs)
                except Exception as e:
                    logger.warning(f"Tokenizer chat template failed: {e}")
            except Exception as e:
                logger.warning(f"Tokenizer chat template failed: {e}")

        # Fallback: try processor directly
        if hasattr(self._processor, "apply_chat_template"):
            try:
                return self._processor.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                try:
                    return self._processor.apply_chat_template(messages, **template_kwargs)
                except Exception as e:
                    logger.warning(f"Processor chat template failed: {e}")
            except Exception as e:
                logger.warning(f"Processor chat template failed: {e}")

        # Last resort: simple concatenation
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"
        return prompt

    def _strip_eos(self, text: str) -> str:
        """Strip text after common EOS tokens."""
        for token in self._END_TOKENS:
            if token in text:
                text = text.split(token)[0]
                break
        return text.strip()

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        """Remove <think>...</think> reasoning blocks from response."""
        cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def _generate_sync(
        self, prompt: str, system_prompt: Optional[str], **kwargs
    ) -> str:
        """Synchronous generation — called via asyncio.to_thread()."""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        formatted_prompt = self._build_chat_prompt(prompt, system_prompt)

        if self.log_prompts:
            print("\n[MLX SYSTEM PROMPT]", flush=True)
            print(system_prompt or "", flush=True)
            print("\n[MLX USER PROMPT]", flush=True)
            print(prompt, flush=True)

        result = self._mlx_generate(
            model=self._model,
            processor=self._processor,
            prompt=formatted_prompt,
            image=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # GenerationResult has .text; handle plain string fallback
        raw_text = result.text if hasattr(result, "text") else str(result)
        clean_text = self._strip_eos(raw_text)
        clean_text = self._strip_think_blocks(clean_text)

        if self.log_responses:
            print("\n[MLX RESPONSE]", flush=True)
            print(clean_text, flush=True)

        return clean_text

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync, prompt, system_prompt, **kwargs
        )

    async def generate_json(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> BaseModel:
        system_prompt = system_prompt or ""
        system_prompt += (
            f"\nRespond strictly in JSON matching this schema: "
            f"{schema.model_json_schema()}"
        )

        response_text = await self.generate(prompt, system_prompt, **kwargs)

        try:
            data = json.loads(response_text)
            return schema.model_validate(data)
        except Exception:
            # Try brace-matching extraction before giving up
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return schema.model_validate(data)
            logger.error(f"Failed to parse JSON from MLX response: {response_text[:200]}")
            raise

    async def close(self) -> None:
        """Release the loaded model to free memory."""
        self._model = None
        self._processor = None
        gc.collect()
        logger.info("MLX model released")
