"""
Base class for figurative language detection strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ...core.llm import BaseLLMProvider
from ...core.text import TextProcessor

class BaseStrategy(ABC):
    """
    Abstract base class for strategies.
    """
    
    def __init__(
        self, 
        llm: BaseLLMProvider, 
        text_processor: TextProcessor,
        **kwargs
    ):
        self.llm = llm
        self.text_processor = text_processor
        self.config = kwargs
    
    @abstractmethod
    async def detect(self, text: str) -> Dict[str, Any]:
        """
        Run detection on text.
        """
        pass
