"""
Live integration test for the figurative pipeline using Ollama.

Set RUN_OLLAMA_LIVE_TESTS=1 to enable.
"""

import os
import unittest
import asyncio

from qualitative_analysis.figurative.detector import FigurativeDetector


RUN_LIVE = os.getenv("RUN_OLLAMA_LIVE_TESTS") == "1"


class TestLiveOllamaPipeline(unittest.TestCase):
    @unittest.skipUnless(RUN_LIVE, "Set RUN_OLLAMA_LIVE_TESTS=1 to run live Ollama tests.")
    def test_live_ollama_pipeline(self):
        async def run_test():
            detector = FigurativeDetector(
                model_name="qwen3:30b-a3b-instruct-2507-q4_K_M",
                provider="ollama",
                strategy="two_step",
                prompt_version=1,
                window_size=3,
                stride=2
            )

            samples = [
                {
                    "label": "short_figurative",
                    "expected_contains": True,
                    "text": "Her voice was velvet, smoothing the rough edges of the room."
                },
                {
                    "label": "short_literal",
                    "expected_contains": False,
                    "text": "The meeting starts at 3 PM and will last 30 minutes."
                },
                {
                    "label": "long_figurative",
                    "expected_contains": True,
                    "text": (
                        "The startup planted a seed in a crowded market. "
                        "Over the next year, it grew roots through patient partnerships, "
                        "and by spring it was bearing fruit in new regions. "
                        "Investors said the company had finally found its sunlight."
                    )
                },
                {
                    "label": "long_literal",
                    "expected_contains": False,
                    "text": (
                        "The report summarizes quarterly revenue, operating expenses, and headcount. "
                        "It notes that research costs increased by 8% due to new laboratory equipment. "
                        "The next release is scheduled for September after additional QA testing."
                    )
                },
            ]

            matches = 0
            for sample in samples:
                result = await detector.detect(sample["text"])

                print(f"\n[{sample['label']}]")
                print(f"contains_figurative: {result.contains_figurative}")
                print(f"confidence: {result.confidence}")
                print(f"instances: {len(result.instances)}")
                print(f"metadata: {result.metadata}")

                self.assertIsInstance(result.contains_figurative, bool)
                self.assertIsInstance(result.confidence, float)
                self.assertIsInstance(result.instances, list)
                self.assertIsInstance(result.metadata, dict)

                if result.metadata:
                    self.assertIn("window_count", result.metadata)
                    self.assertIn("instance_count", result.metadata)

                if result.contains_figurative == sample["expected_contains"]:
                    matches += 1

            self.assertGreaterEqual(
                matches,
                1,
                f"Model classification matched 0/{len(samples)} expected labels."
            )

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
