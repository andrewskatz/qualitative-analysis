import csv
import tempfile
import unittest
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from qualitative_analysis.core.llm import BaseLLMProvider
from qualitative_analysis.figurative.domains import DomainExtractor


class DomainLLM(BaseLLMProvider):
    def __init__(self):
        super().__init__("mock-model")

    async def generate(self, prompt: str, system_prompt=None, **kwargs) -> str:
        return '{"source_domain": "theft", "target_domain": "time", "mapping_explanation": "Time is framed as stealing", "confidence": 0.88}'

    async def generate_json(self, prompt: str, schema: Type[BaseModel], **kwargs) -> BaseModel:
        raise NotImplementedError("Domain extraction uses raw JSON parsing in this test")


class TestFigurativeDetectToMapContract(unittest.IsolatedAsyncioTestCase):
    async def test_extract_from_csv_does_not_require_offsets(self):
        extractor = DomainExtractor(llm_provider=DomainLLM(), multi_level=False, prompt_version="v1")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "instances.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "text_id",
                        "window_index",
                        "window_text",
                        "instance_text",
                        "type",
                        "confidence",
                        "explanation",
                        "start_char",
                        "end_char",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "text_id": "row-1",
                        "window_index": 3,
                        "window_text": "Before time is a thief after.",
                        "instance_text": "time is a thief",
                        "type": "metaphor",
                        "confidence": 0.91,
                        "explanation": "Time is described as stealing.",
                        "start_char": "",
                        "end_char": "",
                    }
                )

            results = await extractor.extract_from_csv(csv_path)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.text, "time is a thief")
        self.assertEqual(result.type, "metaphor")
        self.assertEqual(result.text_id, "row-1")
        self.assertEqual(result.window_index, 3)
        self.assertEqual(result.window_text, "Before time is a thief after.")
        self.assertEqual(result.source_domain, "theft")
        self.assertEqual(result.target_domain, "time")


if __name__ == "__main__":
    unittest.main()