"""
Abstract LLM provider interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, Type
from pydantic import BaseModel

class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.config = kwargs

    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None, 
        **kwargs
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system instructions.
            **kwargs: Additional model parameters (temperature, etc.)
            
        Returns:
            Generated text string.
        """
        pass

    @abstractmethod
    async def generate_json(
        self, 
        prompt: str, 
        schema: Type[BaseModel],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        Generate structured data matching a Pydantic schema.
        
        Args:
            prompt: The user prompt.
            schema: Pydantic model class to validate against.
            system_prompt: Optional system instructions.
            **kwargs: Additional model parameters.
            
        Returns:
            Instance of the schema class.
        """
        pass
