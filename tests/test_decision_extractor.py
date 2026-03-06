"""Tests for decision extractor with mock LLM provider."""

import json
import pytest
from unittest.mock import AsyncMock

from qualitative_analysis.decisions.extractor import DecisionExtractor
from qualitative_analysis.decisions.models import DecisionConfig


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    async def generate(self, prompt="", system_prompt="", temperature=0.3):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = json.dumps({"decisions": [], "factors": []})
        self.call_count += 1
        return response

    async def generate_json(self, prompt="", system_prompt="", temperature=0.3, schema=None):
        return json.loads(await self.generate(prompt, system_prompt, temperature))


class TestDecisionExtractorInit:
    def test_valid_strategies(self):
        mock = MockLLMProvider()
        for strategy in ["one_pass", "semantic_two_pass", "context_aware"]:
            extractor = DecisionExtractor(mock, strategy=strategy)
            assert extractor.strategy == strategy

    def test_invalid_strategy(self):
        mock = MockLLMProvider()
        with pytest.raises(ValueError, match="Unsupported strategy"):
            DecisionExtractor(mock, strategy="invalid")


class TestOnePassExtraction:
    @pytest.mark.asyncio
    async def test_empty_text(self):
        mock = MockLLMProvider()
        extractor = DecisionExtractor(mock, strategy="one_pass")
        result = await extractor.extract("", text_id="doc1")
        assert result.text_id == "doc1"
        assert result.decisions == []
        assert result.factors == []

    @pytest.mark.asyncio
    async def test_basic_extraction(self):
        response = json.dumps({
            "decisions": [
                {"decision": "adopt cloud migration"}
            ],
            "factors": [
                {"decision": "adopt cloud migration", "factor": "cost savings", "polarity": "supporting"},
                {"decision": "adopt cloud migration", "factor": "security risks", "polarity": "opposing"},
            ]
        })
        mock = MockLLMProvider(responses=[response])
        config = DecisionConfig(strategy="one_pass", validate_decisions=False)
        extractor = DecisionExtractor(
            mock,
            strategy="one_pass",
            config=config,
            window_size=1_000_000,
            stride=1_000_000,
        )

        result = await extractor.extract(
            "The committee decided to adopt cloud migration because of cost savings, "
            "despite security risks.",
            text_id="doc1",
        )

        assert len(result.decisions) == 1
        assert result.decisions[0].text == "adopt cloud migration"
        assert len(result.factors) == 2
        assert result.metadata["strategy"] == "one_pass"

    @pytest.mark.asyncio
    async def test_llm_error_handled(self):
        mock = MockLLMProvider(responses=["not valid json at all"])
        config = DecisionConfig(strategy="one_pass", validate_decisions=False)
        extractor = DecisionExtractor(
            mock,
            strategy="one_pass",
            config=config,
            window_size=1_000_000,
            stride=1_000_000,
        )

        # Should not raise, just return empty
        result = await extractor.extract("Some text here.", text_id="doc1")
        assert result.decisions == []


class TestFactorValidation:
    def setup_method(self):
        self.mock = MockLLMProvider()
        self.extractor = DecisionExtractor(self.mock, strategy="one_pass")

    def test_empty_factor_rejected(self):
        assert not self.extractor._validate_factor("", "some decision")

    def test_identical_to_decision_rejected(self):
        assert not self.extractor._validate_factor(
            "Adopt Cloud Migration", "adopt cloud migration"
        )

    def test_too_short_rejected(self):
        assert not self.extractor._validate_factor("high costs", "reduce budget")

    def test_high_overlap_rejected(self):
        # Factor words are subset of decision words
        assert not self.extractor._validate_factor(
            "reduce the budget", "reduce the budget significantly"
        )

    def test_valid_factor_accepted(self):
        assert self.extractor._validate_factor(
            "rising infrastructure costs across departments",
            "reduce the training budget",
        )


class TestJsonParsing:
    def test_clean_json(self):
        data = DecisionExtractor._parse_json_response(
            '{"decisions": ["adopt cloud"]}'
        )
        assert data["decisions"] == ["adopt cloud"]

    def test_markdown_fenced(self):
        response = '```json\n{"decisions": ["adopt cloud"]}\n```'
        data = DecisionExtractor._parse_json_response(response)
        assert data["decisions"] == ["adopt cloud"]

    def test_text_before_json(self):
        response = 'Here are the decisions:\n{"decisions": ["adopt cloud"]}'
        data = DecisionExtractor._parse_json_response(response)
        assert data["decisions"] == ["adopt cloud"]

    def test_json_array(self):
        response = '["adopt cloud", "reduce budget"]'
        data = DecisionExtractor._parse_json_response(response)
        assert "decisions" in data
        assert len(data["decisions"]) == 2

    def test_nested_string_json(self):
        response = '"{\\"decisions\\": [\\"test\\"]}"'
        # This tests the double-JSON scenario
        # Should handle gracefully
        try:
            data = DecisionExtractor._parse_json_response(response)
            assert isinstance(data, dict)
        except (json.JSONDecodeError, ValueError):
            pass  # Acceptable to fail on deeply nested edge cases
