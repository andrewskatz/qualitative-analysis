"""
Tests for entity detection module.

Tests cover:
- EntityDetector initialization and configuration
- Single text entity detection
- Batch entity detection
- Windowing behavior
- Empty/edge case handling
- Output structure validation
"""

import asyncio
import json
import pytest
from typing import List

from qualitative_analysis.core.text import SlidingWindowProcessor
from qualitative_analysis.entity.detector import (
    EntityDetector,
    EntityDetectionResult,
)


def run_async(coro):
    """Helper to run async functions in sync tests."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: List[str] = None):
        """
        Initialize with canned responses.

        Args:
            responses: List of JSON responses to return in sequence.
                If None, returns a default entity list.
        """
        self.responses = responses or []
        self.call_count = 0
        self.last_system_prompt = None
        self.last_prompt = None
        self.last_kwargs = None

    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        **kwargs,
    ) -> str:
        """Return mock response."""
        self.last_system_prompt = system_prompt
        self.last_prompt = prompt
        self.last_kwargs = kwargs

        if self.responses:
            response = self.responses[self.call_count % len(self.responses)]
        else:
            # Default response with some entities
            response = '{"entities_and_concepts": ["renewable energy", "heating costs", "community support"]}'

        self.call_count += 1
        return response


class MockTokenizer:
    """Minimal tokenizer stub for offline token-windowing tests."""

    def encode(self, text: str) -> List[int]:
        return [ord(ch) for ch in text]

    def decode(self, tokens: List[int]) -> str:
        return "".join(chr(token) for token in tokens)


class TestEntityDetectionResult:
    """Tests for EntityDetectionResult dataclass."""

    def test_basic_creation(self):
        """EntityDetectionResult should store all fields correctly."""
        result = EntityDetectionResult(
            text_id="test_001",
            entities=["entity1", "entity2"],
            entity_contexts=[
                {"entity": "entity1", "context": "some context", "window_index": 0},
                {"entity": "entity2", "context": "more context", "window_index": 1},
            ],
            window_count=2,
        )

        assert result.text_id == "test_001"
        assert len(result.entities) == 2
        assert len(result.entity_contexts) == 2
        assert result.window_count == 2

    def test_to_dict_serialization(self):
        """to_dict() should produce JSON-serializable dict."""
        result = EntityDetectionResult(
            text_id="test_001",
            entities=["entity1", "entity2"],
            entity_contexts=[
                {"entity": "entity1", "context": "ctx", "window_index": 0},
            ],
            window_count=1,
            metadata={"key": "value"},
        )

        d = result.to_dict()

        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

        # Check expected keys
        assert "text_id" in d
        assert "entities" in d
        assert "entity_count" in d
        assert d["entity_count"] == 2


class TestEntityDetector:
    """Tests for EntityDetector class."""

    @pytest.fixture
    def mock_tokenizer(self, monkeypatch):
        monkeypatch.setattr(
            SlidingWindowProcessor,
            "_get_tokenizer",
            lambda self: MockTokenizer(),
        )

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM provider."""
        return MockLLMProvider()

    @pytest.fixture
    def detector(self, mock_llm):
        """Create detector with mock LLM."""
        return EntityDetector(
            llm_provider=mock_llm,
            window_size=3,
            stride=2,
        )

    def test_detect_single_text(self, detector):
        """Detection should extract entities from text."""
        result = run_async(detector.detect(
            text="Abeesee faces a heating crisis. Renewable energy could help.",
            text_id="test_001",
        ))

        assert isinstance(result, EntityDetectionResult)
        assert result.text_id == "test_001"
        assert len(result.entities) > 0
        assert result.window_count >= 1

    def test_detect_with_windowing(self):
        """Windowing should split text into multiple windows."""
        # Create text with many sentences
        sentences = [f"This is sentence number {i}." for i in range(10)]
        long_text = " ".join(sentences)

        # Response includes window-specific entity
        responses = [
            '{"entities_and_concepts": ["entity_from_window_0"]}',
            '{"entities_and_concepts": ["entity_from_window_1"]}',
            '{"entities_and_concepts": ["entity_from_window_2"]}',
            '{"entities_and_concepts": ["entity_from_window_3"]}',
            '{"entities_and_concepts": ["entity_from_window_4"]}',
        ]
        mock_llm = MockLLMProvider(responses)
        detector = EntityDetector(mock_llm, window_size=3, stride=2)

        result = run_async(detector.detect(long_text, text_id="test"))

        # Should have processed multiple windows
        assert result.window_count > 1

    def test_detect_no_windowing(self, detector):
        """No-windowing mode should process entire text as one unit."""
        long_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

        result = run_async(detector.detect(
            text=long_text,
            text_id="test_001",
            use_windowing=False,
        ))

        # Should be exactly 1 window
        assert result.window_count == 1

    def test_detect_empty_text(self, detector):
        """Empty text should return empty result."""
        result = run_async(detector.detect(text="", text_id="empty"))

        assert result.text_id == "empty"
        assert result.entities == []
        assert result.entity_contexts == []
        assert result.window_count == 0

    def test_detect_whitespace_only(self, detector):
        """Whitespace-only text should return empty result."""
        result = run_async(detector.detect(text="   \n\t  ", text_id="whitespace"))

        assert result.entities == []
        assert result.window_count == 0

    def test_detect_requests_json_response_mode_by_default(self):
        """Detection should request provider JSON mode when configured."""
        mock_llm = MockLLMProvider()
        detector = EntityDetector(mock_llm, response_format="json")

        result = run_async(detector.detect(text="Some text.", text_id="test"))

        assert isinstance(result, EntityDetectionResult)
        assert mock_llm.last_kwargs["format"] == "json"
        assert result.metadata["response_format"] == "json"

    def test_detect_can_use_prompt_only_response_mode(self):
        """Prompt-only mode should not force provider JSON mode."""
        mock_llm = MockLLMProvider()
        detector = EntityDetector(mock_llm, response_format="prompt")

        result = run_async(detector.detect(text="Some text.", text_id="test"))

        assert isinstance(result, EntityDetectionResult)
        assert "format" not in mock_llm.last_kwargs
        assert result.metadata["response_format"] == "prompt"

    def test_entity_deduplication(self):
        """Duplicate entities within same text should be tracked uniquely."""
        # Same entity returned in multiple windows
        responses = [
            '{"entities_and_concepts": ["renewable energy", "heating"]}',
            '{"entities_and_concepts": ["renewable energy", "community"]}',
        ]
        mock_llm = MockLLMProvider(responses)
        detector = EntityDetector(mock_llm, window_size=2, stride=1)

        text = "First. Second. Third. Fourth."
        result = run_async(detector.detect(text, text_id="test"))

        # "renewable energy" appears in both windows but should only be in entities once
        assert result.entities.count("renewable energy") == 1
        assert "renewable energy" in result.entities

    def test_extract_entities_accepts_markdown_fenced_json(self):
        mock_llm = MockLLMProvider([
            '```json\n{"entities_and_concepts": ["renewable energy", "heating costs"]}\n```'
        ])
        detector = EntityDetector(mock_llm, window_size=3, stride=2)

        entities = run_async(detector._extract_entities("Some text here."))

        assert entities == ["renewable energy", "heating costs"]

    def test_extract_entities_accepts_embedded_json(self):
        mock_llm = MockLLMProvider([
            'Here are the entities: {"entities_and_concepts": ["community support", "solar panels"]} Thanks.'
        ])
        detector = EntityDetector(mock_llm, window_size=3, stride=2)

        entities = run_async(detector._extract_entities("Some text here."))

        assert entities == ["community support", "solar panels"]

    def test_entity_contexts_include_window_info(self, detector):
        """Entity contexts should include window index."""
        result = run_async(detector.detect(
            text="Some text here.",
            text_id="test",
        ))

        for ctx in result.entity_contexts:
            assert "entity" in ctx
            assert "context" in ctx
            assert "window_index" in ctx
            assert isinstance(ctx["window_index"], int)

    def test_context_truncation(self, mock_llm):
        """Long contexts should be truncated."""
        detector = EntityDetector(mock_llm, window_size=3, stride=2)

        # Create a very long sentence
        long_text = "Word " * 200 + "end."

        result = run_async(detector.detect(
            text=long_text,
            text_id="test",
            max_context_length=100,
        ))

        for ctx in result.entity_contexts:
            # Context should be truncated
            assert len(ctx["context"]) <= 103  # 100 + "..."

    def test_detect_batch(self, detector):
        """Batch detection should process multiple texts."""
        texts = [
            {"text": "First document about energy.", "id": "doc1"},
            {"text": "Second document about climate.", "id": "doc2"},
            {"text": "Third document about policy.", "id": "doc3"},
        ]

        results = run_async(detector.detect_batch(
            texts=texts,
            text_col="text",
            id_col="id",
        ))

        assert len(results) == 3
        assert results[0].text_id == "doc1"
        assert results[1].text_id == "doc2"
        assert results[2].text_id == "doc3"

    def test_detect_batch_auto_id(self, detector):
        """Batch detection should auto-generate IDs when not provided."""
        texts = [
            {"text": "First document."},
            {"text": "Second document."},
        ]

        results = run_async(detector.detect_batch(
            texts=texts,
            text_col="text",
            id_col=None,
        ))

        assert results[0].text_id == "text_0"
        assert results[1].text_id == "text_1"

    def test_metadata_stored(self, detector):
        """Detection result should include metadata about processing."""
        result = run_async(detector.detect(
            text="Some text here.",
            text_id="test",
            use_windowing=True,
        ))

        assert "use_windowing" in result.metadata
        assert result.metadata["use_windowing"] is True
        assert "window_size" in result.metadata

    def test_llm_parse_error_handled(self):
        """LLM returning invalid JSON should be handled gracefully."""
        mock_llm = MockLLMProvider(["not valid json at all"])
        detector = EntityDetector(mock_llm)

        # Should not raise - should return empty entities
        result = run_async(detector.detect(text="Some text.", text_id="test"))

        # May have empty entities due to parse failure
        assert isinstance(result, EntityDetectionResult)

    def test_token_based_windowing(self, mock_tokenizer):
        """Token-based windowing should split text by token count."""
        mock_llm = MockLLMProvider()
        detector = EntityDetector(
            mock_llm,
            window_size=10,
            stride=5,
            chunk_unit="tokens",
        )

        # Create text long enough to produce multiple token windows
        text = "The community discussed renewable energy options for heating. " * 5

        result = run_async(detector.detect(text, text_id="token_test"))

        assert isinstance(result, EntityDetectionResult)
        assert result.window_count >= 1
        assert len(result.entities) > 0
        assert result.metadata["chunk_unit"] == "tokens"

    def test_chunk_unit_in_metadata(self, mock_tokenizer):
        """Detection metadata should include chunk_unit."""
        mock_llm = MockLLMProvider()

        # Sentence-based
        detector_sent = EntityDetector(mock_llm, chunk_unit="sentences")
        result_sent = run_async(detector_sent.detect(text="Some text.", text_id="test"))
        assert result_sent.metadata["chunk_unit"] == "sentences"

        # Token-based
        detector_tok = EntityDetector(mock_llm, chunk_unit="tokens")
        result_tok = run_async(detector_tok.detect(text="Some text.", text_id="test"))
        assert result_tok.metadata["chunk_unit"] == "tokens"

    def test_uses_entity_system_prompt(self):
        """Detector should use its own system prompt, not the relationships one."""
        mock_llm = MockLLMProvider()
        detector = EntityDetector(mock_llm)

        run_async(detector.detect(text="Some text here.", text_id="test"))

        # Should use entity-specific system prompt
        assert mock_llm.last_system_prompt is not None
        assert "qualitative research analyst" in mock_llm.last_system_prompt
        assert "relationship analyst" not in mock_llm.last_system_prompt

    def test_user_prompt_contains_text(self):
        """User prompt should contain the input text."""
        mock_llm = MockLLMProvider()
        detector = EntityDetector(mock_llm)

        run_async(detector.detect(text="Abeesee faces a heating crisis.", text_id="test"))

        assert mock_llm.last_prompt is not None
        assert "Abeesee faces a heating crisis." in mock_llm.last_prompt
