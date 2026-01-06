"""
Main figurative language usage detector.
"""

from typing import Optional, Dict, Any, List
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
        figurative_types: Optional[List[str]] = None,
        # Text processor config
        window_size: int = 3,
        stride: int = 2,
        chunk_unit: str = "sentences",
        tokenizer_name: str = "cl100k_base",
        # Dependency Injection
        llm_provider: Optional[BaseLLMProvider] = None
    ):
        """
        Initialize the detector.
        
        Args:
            model_name: LLM model to use
            provider: LLM provider ("ollama", etc.)
            provider_config: Optional config dict for provider
            strategy: Detection strategy ("two_step")
            threshold: Confidence threshold for detection
            summary_buffer_size: Number of summaries to keep in context
            prompt_version: Version of prompts to use
            return_windows: Whether to return window-level results
            figurative_types: Optional list of figurative types to detect.
                If specified, only these types will be detected.
                Valid types: metaphor, simile, personification, hyperbole,
                idiom, irony, extended_metaphor, analogy, other
            window_size: Number of chunks per window
            stride: Number of chunks to slide
            chunk_unit: "sentences" or "tokens"
            tokenizer_name: Tokenizer for token-based chunking
            llm_provider: Optional pre-configured LLM provider (for testing)
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
            stride=stride,
            chunk_unit=chunk_unit,
            tokenizer_name=tokenizer_name,
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
                figurative_types=figurative_types,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def detect(self, text: str) -> DetectionResult:
        """
        Analyze text for figurative language.
        """
        return await self.strategy.detect(text)

