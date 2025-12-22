"""
Test the pipeline with a mock LLM.
"""

import unittest
import asyncio
from typing import Optional, Type
from pydantic import BaseModel
from qualitative_analysis.core.llm import BaseLLMProvider
from qualitative_analysis.figurative.detector import FigurativeDetector
from qualitative_analysis.figurative.components.scanner import InstanceModel

class MockLLM(BaseLLMProvider):
    def __init__(self):
        super().__init__("mock-model")
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if "<text_to_summarize>" in prompt:
            return '{"summary_points": ["Summary point one", "Summary point two", "Summary point three"]}'
        if "<text_to_analyze>" in prompt and "has_figurative" in prompt:
            return '{"has_figurative": true, "confidence": 0.9, "reasoning": "Clear metaphor."}'
        if "<text_to_analyze>" in prompt and "instances" in prompt:
            return '{"instances": [{"text": "time is a thief", "type": "metaphor", "confidence": 0.9, "explanation": "Time steals moments", "context_dependent": false}]}'
        return '{"summary_points": ["Fallback summary point"]}'

    async def generate_json(self, prompt: str, schema: Type[BaseModel], **kwargs) -> BaseModel:
        # Construct the class instance expected by the scanner
        # schema here is ExtractionResponse class
        return schema(instances=[
            InstanceModel(
                text="time is a thief",
                type="metaphor",
                confidence=0.9,
                explanation="Time steals moments",
                context_dependent=False
            )
        ])

class TestFigurativePipeline(unittest.TestCase):
    def test_pipeline(self):
        async def run_test():
            mock_llm = MockLLM()
            detector = FigurativeDetector(llm_provider=mock_llm, window_size=5, stride=5)
            
            text = "Time is a thief. " * 10
            
            result = await detector.detect(text)
            
            self.assertTrue(result.contains_figurative)
            self.assertTrue(len(result.instances) > 0)
            self.assertEqual(result.instances[0].text, "time is a thief")
            self.assertEqual(result.instances[0].type, "metaphor")

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()
