import unittest

from qualitative_analysis.core.llm import BaseLLMProvider
from qualitative_analysis.figurative.components.scanner import Scanner
from qualitative_analysis.figurative.models import Instance, Summary
from qualitative_analysis.figurative.strategies.two_step import TwoStepWithSummariesStrategy


class ControlledLLM(BaseLLMProvider):
    def __init__(
        self,
        *,
        fail_generate_json_for=None,
        raw_detect_response=None,
        raw_extract_response=None,
        structured_detect_response=None,
        structured_extract_instances=None,
    ):
        super().__init__("mock-model")
        self.fail_generate_json_for = set(fail_generate_json_for or [])
        self.raw_detect_response = raw_detect_response or '{"has_figurative": true, "confidence": 0.8}'
        self.raw_extract_response = raw_extract_response or '{"instances": []}'
        self.structured_detect_response = structured_detect_response or {
            "has_figurative": True,
            "confidence": 0.8,
            "reasoning": "clear figurative language",
        }
        self.structured_extract_instances = structured_extract_instances or []
        self.generate_calls = []
        self.generate_json_calls = []

    async def generate(self, prompt: str, system_prompt=None, **kwargs) -> str:
        mode = "detect" if "has_figurative" in prompt else "extract"
        self.generate_calls.append(mode)
        return self.raw_detect_response if mode == "detect" else self.raw_extract_response

    async def generate_json(self, prompt: str, schema, system_prompt=None, **kwargs):
        mode = "detect" if "has_figurative" in schema.model_fields else "extract"
        self.generate_json_calls.append(mode)
        if mode in self.fail_generate_json_for:
            raise ValueError(f"forced {mode} failure")
        if mode == "detect":
            return schema(**self.structured_detect_response)
        return schema(instances=self.structured_extract_instances)


class StubTextProcessor:
    def __init__(self, windows, window_size=1, stride=1):
        self.windows = windows
        self.window_size = window_size
        self.stride = stride

    def process(self, text):
        return list(self.windows)


class TestScanner(unittest.IsolatedAsyncioTestCase):
    async def test_detect_prefers_provider_structured_output(self):
        llm = ControlledLLM(structured_detect_response={"has_figurative": True, "confidence": 1.4})
        scanner = Scanner(llm)

        has_figurative, confidence, metadata = await scanner.detect("Time is a thief.", "No prior context.")

        self.assertTrue(has_figurative)
        self.assertEqual(confidence, 1.0)
        self.assertFalse(metadata["structured_output_fallback"])
        self.assertFalse(metadata["parse_failed"])
        self.assertEqual(llm.generate_json_calls, ["detect"])
        self.assertEqual(llm.generate_calls, [])

    async def test_detect_parse_failure_is_conservative(self):
        llm = ControlledLLM(
            fail_generate_json_for={"detect"},
            raw_detect_response="yes definitely figurative",
        )
        scanner = Scanner(llm)

        has_figurative, confidence, metadata = await scanner.detect("Time is a thief.", "No prior context.")

        self.assertFalse(has_figurative)
        self.assertEqual(confidence, 0.0)
        self.assertTrue(metadata["structured_output_fallback"])
        self.assertTrue(metadata["parse_failed"])
        self.assertEqual(llm.generate_json_calls, ["detect"])
        self.assertEqual(llm.generate_calls, ["detect"])

    async def test_extract_skips_invalid_items_and_reports_count(self):
        llm = ControlledLLM(
            fail_generate_json_for={"extract"},
            raw_extract_response='{"instances": [{"text": "time is a thief", "type": "metaphor", "confidence": 0.9, "explanation": "Time steals moments", "context_dependent": false}, {"text": "broken item"}]}'
        )
        scanner = Scanner(llm)

        instances, metadata = await scanner.extract("Time is a thief.", "No prior context.", window_index=2)

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].window_index, 2)
        self.assertEqual(instances[0].type, "metaphor")
        self.assertTrue(metadata["structured_output_fallback"])
        self.assertFalse(metadata["parse_failed"])
        self.assertEqual(metadata["invalid_items_skipped"], 1)

    async def test_extract_respects_type_filtering(self):
        llm = ControlledLLM(
            structured_extract_instances=[
                {"text": "time is a thief", "type": "metaphor", "confidence": 0.9, "explanation": "Time steals moments", "context_dependent": False},
                {"text": "as quiet as snow", "type": "simile", "confidence": 0.7, "explanation": "Explicit comparison", "context_dependent": False},
            ]
        )
        scanner = Scanner(llm, figurative_types=["metaphor"])

        instances, metadata = await scanner.extract("Time is a thief.", "No prior context.", window_index=0)

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].type, "metaphor")
        self.assertEqual(metadata["invalid_items_skipped"], 0)


class TestTwoStepWithSummariesStrategy(unittest.IsolatedAsyncioTestCase):
    async def test_detection_context_excludes_current_window_summary(self):
        strategy = TwoStepWithSummariesStrategy(
            llm=ControlledLLM(),
            text_processor=StubTextProcessor(["first window", "second window"]),
        )
        seen_contexts = []

        async def fake_summarize(text, prior_summaries, window_index):
            return Summary(text=f"summary-{window_index}", window_index=window_index)

        async def fake_detect(text, prior_context):
            seen_contexts.append(prior_context)
            return False, 0.0, {"structured_output_fallback": False, "parse_failed": False}

        async def fake_extract(text, prior_context, window_index):
            return [], {"structured_output_fallback": False, "parse_failed": False, "invalid_items_skipped": 0}

        strategy.summarizer.summarize = fake_summarize
        strategy.scanner.detect = fake_detect
        strategy.scanner.extract = fake_extract

        await strategy.detect("ignored")

        self.assertEqual(seen_contexts[0], "No prior context.")
        self.assertIn("summary-0", seen_contexts[1])
        self.assertNotIn("summary-1", seen_contexts[1])

    async def test_strategy_aggregates_scanner_metadata(self):
        strategy = TwoStepWithSummariesStrategy(
            llm=ControlledLLM(),
            text_processor=StubTextProcessor(["first window", "second window"]),
        )
        detect_calls = 0

        async def fake_summarize(text, prior_summaries, window_index):
            return Summary(text=f"summary-{window_index}", window_index=window_index)

        async def fake_detect(text, prior_context):
            nonlocal detect_calls
            detect_calls += 1
            if detect_calls == 1:
                return False, 0.0, {"structured_output_fallback": True, "parse_failed": True}
            return True, 0.9, {"structured_output_fallback": False, "parse_failed": False}

        async def fake_extract(text, prior_context, window_index):
            return [], {"structured_output_fallback": True, "parse_failed": True, "invalid_items_skipped": 2}

        strategy.summarizer.summarize = fake_summarize
        strategy.scanner.detect = fake_detect
        strategy.scanner.extract = fake_extract

        result = await strategy.detect("ignored")
        stats = result.metadata["scanner_stats"]

        self.assertEqual(stats["detection_structured_output_fallbacks"], 1)
        self.assertEqual(stats["detection_parse_failures"], 1)
        self.assertEqual(stats["extraction_structured_output_fallbacks"], 1)
        self.assertEqual(stats["extraction_parse_failures"], 1)
        self.assertEqual(stats["invalid_extraction_items_skipped"], 2)

    async def test_strategy_deduplicates_overlapping_window_instances(self):
        windows = [
            "Before time is a thief.",
            "time is a thief. After.",
        ]
        strategy = TwoStepWithSummariesStrategy(
            llm=ControlledLLM(),
            text_processor=StubTextProcessor(windows, window_size=2, stride=1),
        )

        async def fake_summarize(text, prior_summaries, window_index):
            return Summary(text=f"summary-{window_index}", window_index=window_index)

        async def fake_detect(text, prior_context):
            return True, 0.9, {"structured_output_fallback": False, "parse_failed": False}

        async def fake_extract(text, prior_context, window_index):
            return [
                Instance(
                    text="time is a thief",
                    type="metaphor",
                    confidence=0.7 if window_index == 0 else 0.9,
                    explanation="shared overlap",
                    context_dependent=False,
                    window_index=window_index,
                )
            ], {"structured_output_fallback": False, "parse_failed": False, "invalid_items_skipped": 0}

        strategy.summarizer.summarize = fake_summarize
        strategy.scanner.detect = fake_detect
        strategy.scanner.extract = fake_extract

        result = await strategy.detect("ignored")

        self.assertEqual(len(result.instances), 1)
        self.assertEqual(result.metadata["raw_instance_count"], 2)
        self.assertEqual(result.metadata["instance_count"], 1)
        self.assertEqual(result.metadata["deduplicated_instance_count"], 1)
        instance = result.instances[0]
        self.assertEqual(instance.window_index, 1)
        self.assertEqual(instance.support_count, 2)
        self.assertEqual(instance.supporting_window_indices, [0, 1])
        self.assertEqual(instance.alignment_status, "exact")
        self.assertEqual(instance.start_char, 0)
        self.assertEqual(instance.end_char, len("time is a thief"))

    async def test_strategy_uses_normalized_exact_alignment(self):
        window = 'He called it “time’s a thief” again.'
        strategy = TwoStepWithSummariesStrategy(
            llm=ControlledLLM(),
            text_processor=StubTextProcessor([window], window_size=1, stride=1),
        )

        async def fake_summarize(text, prior_summaries, window_index):
            return Summary(text=f"summary-{window_index}", window_index=window_index)

        async def fake_detect(text, prior_context):
            return True, 0.9, {"structured_output_fallback": False, "parse_failed": False}

        async def fake_extract(text, prior_context, window_index):
            return [
                Instance(
                    text="time's a thief",
                    type="metaphor",
                    confidence=0.9,
                    explanation="normalized apostrophe",
                    context_dependent=False,
                    window_index=window_index,
                )
            ], {"structured_output_fallback": False, "parse_failed": False, "invalid_items_skipped": 0}

        strategy.summarizer.summarize = fake_summarize
        strategy.scanner.detect = fake_detect
        strategy.scanner.extract = fake_extract

        result = await strategy.detect("ignored")

        instance = result.instances[0]
        self.assertEqual(instance.alignment_status, "normalized_exact")
        self.assertEqual(instance.text, "time's a thief")
        self.assertEqual(instance.start_char, window.index("time’s a thief"))
        self.assertEqual(instance.end_char, window.index("time’s a thief") + len("time’s a thief"))

    async def test_strategy_marks_ambiguous_alignment_without_offsets(self):
        window = "time is a thief; time is a thief."
        strategy = TwoStepWithSummariesStrategy(
            llm=ControlledLLM(),
            text_processor=StubTextProcessor([window], window_size=1, stride=1),
        )

        async def fake_summarize(text, prior_summaries, window_index):
            return Summary(text=f"summary-{window_index}", window_index=window_index)

        async def fake_detect(text, prior_context):
            return True, 0.9, {"structured_output_fallback": False, "parse_failed": False}

        async def fake_extract(text, prior_context, window_index):
            return [
                Instance(
                    text="time is a thief",
                    type="metaphor",
                    confidence=0.9,
                    explanation="repeated phrase",
                    context_dependent=False,
                    window_index=window_index,
                )
            ], {"structured_output_fallback": False, "parse_failed": False, "invalid_items_skipped": 0}

        strategy.summarizer.summarize = fake_summarize
        strategy.scanner.detect = fake_detect
        strategy.scanner.extract = fake_extract

        result = await strategy.detect("ignored")

        instance = result.instances[0]
        self.assertEqual(instance.alignment_status, "ambiguous")
        self.assertIsNone(instance.start_char)
        self.assertIsNone(instance.end_char)

    async def test_strategy_preserves_instance_when_alignment_is_only_approximate(self):
        window = "Time is a thief."
        strategy = TwoStepWithSummariesStrategy(
            llm=ControlledLLM(),
            text_processor=StubTextProcessor([window], window_size=1, stride=1),
        )

        async def fake_summarize(text, prior_summaries, window_index):
            return Summary(text=f"summary-{window_index}", window_index=window_index)

        async def fake_detect(text, prior_context):
            return True, 0.9, {"structured_output_fallback": False, "parse_failed": False}

        async def fake_extract(text, prior_context, window_index):
            return [
                Instance(
                    text="Time is a thieff",
                    type="metaphor",
                    confidence=0.9,
                    explanation="minor drift",
                    context_dependent=False,
                    window_index=window_index,
                )
            ], {"structured_output_fallback": False, "parse_failed": False, "invalid_items_skipped": 0}

        strategy.summarizer.summarize = fake_summarize
        strategy.scanner.detect = fake_detect
        strategy.scanner.extract = fake_extract

        result = await strategy.detect("ignored")

        self.assertEqual(len(result.instances), 1)
        self.assertEqual(result.metadata["instance_count"], 1)
        instance = result.instances[0]
        self.assertEqual(instance.alignment_status, "approximate")
        self.assertIsNone(instance.start_char)
        self.assertIsNone(instance.end_char)


if __name__ == "__main__":
    unittest.main()