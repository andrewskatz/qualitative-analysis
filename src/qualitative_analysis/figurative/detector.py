"""
Main figurative language usage detector.
"""

from typing import Optional, Dict, Any, Union
from ..core.llm import BaseLLMProvider
from ..core.providers import OllamaProvider
from ..core.text import SlidingWindowProcessor
from .strategies.two_step import TwoStepWithSummariesStrategy
from .models import DetectionResult

class FigurativeDetector:
    """
    High-level interface for detecting figurative language.
    """
    
    def __init__(
        self,
        model_name: str = "mistral-small",
        provider: str = "ollama",
        provider_config: Optional[Dict[str, Any]] = None,
        strategy: str = "two_step",
        threshold: float = 0.5,
        summary_buffer_size: int = 5,
        prompt_version: int = 1,
        return_windows: bool = False,
        # Text processor config
        window_size: int = 3,
        stride: int = 2,
        # Dependency Injection
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        """
        Initialize the detector.
        """
        provider_config = provider_config or {}
        
        # 1. Setup LLM
        if llm_provider:
            self.llm = llm_provider
        elif provider == "ollama":
            self.llm = OllamaProvider(model_name, **provider_config)
        else:
            # Fallback or error
            raise ValueError(f"Unsupported provider: {provider}")

        # 2. Setup Text Processor
        self.text_processor = SlidingWindowProcessor(
            window_size=window_size,
            stride=stride
        )
        
        # 3. Setup Strategy
        if strategy == "two_step":
            self.strategy = TwoStepWithSummariesStrategy(
                llm=self.llm,
                text_processor=self.text_processor,
                threshold=threshold,
                summary_buffer_size=summary_buffer_size,
                prompt_version=prompt_version,
                return_windows=return_windows,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def detect(self, text: str) -> DetectionResult:
        """
        Analyze text for figurative language.
        """
        return await self.strategy.detect(text)
