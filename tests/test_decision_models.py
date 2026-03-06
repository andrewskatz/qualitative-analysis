"""Tests for decision/factor data models."""

import pytest
from qualitative_analysis.decisions.models import (
    Factor,
    Decision,
    WindowSummary,
    DecisionExtractionResult,
    DecisionConfig,
    AggregationResult,
)


class TestFactor:
    def test_basic_creation(self):
        f = Factor(text="rising costs", decision_text="reduce budget", polarity="supporting")
        assert f.text == "rising costs"
        assert f.decision_text == "reduce budget"
        assert f.polarity == "supporting"

    def test_defaults(self):
        f = Factor(text="some factor")
        assert f.decision_text == ""
        assert f.polarity == "neutral"
        assert f.window_index == 0
        assert f.text_id == ""
        assert f.confidence == 1.0

    def test_to_dict_roundtrip(self):
        f = Factor(
            text="rising costs",
            decision_text="reduce budget",
            polarity="opposing",
            window_index=3,
            text_id="doc1",
            confidence=0.95,
        )
        d = f.to_dict()
        f2 = Factor.from_dict(d)
        assert f2.text == f.text
        assert f2.decision_text == f.decision_text
        assert f2.polarity == f.polarity
        assert f2.window_index == f.window_index
        assert f2.text_id == f.text_id
        assert f2.confidence == f.confidence

    def test_from_dict_alt_keys(self):
        """Test from_dict with 'factor' and 'decision' keys (web app format)."""
        data = {"factor": "high demand", "decision": "expand operations", "polarity": "supporting"}
        f = Factor.from_dict(data)
        assert f.text == "high demand"
        assert f.decision_text == "expand operations"


class TestDecision:
    def test_basic_creation(self):
        d = Decision(text="reduce training budget")
        assert d.text == "reduce training budget"
        assert d.factors == []
        assert d.window_indices == []

    def test_with_factors(self):
        factors = [
            Factor(text="cost overruns", polarity="supporting"),
            Factor(text="employee morale", polarity="opposing"),
        ]
        d = Decision(text="reduce budget", factors=factors)
        assert len(d.factors) == 2

    def test_to_dict_roundtrip(self):
        factors = [Factor(text="cost overruns", polarity="supporting")]
        d = Decision(
            text="reduce budget",
            original_text="reduce budget because costs",
            factors=factors,
            window_indices=[0, 2],
            text_ids=["doc1"],
            validation_passed=True,
        )
        data = d.to_dict()
        d2 = Decision.from_dict(data)
        assert d2.text == d.text
        assert d2.original_text == d.original_text
        assert len(d2.factors) == 1
        assert d2.factors[0].text == "cost overruns"
        assert d2.window_indices == [0, 2]


class TestWindowSummary:
    def test_creation(self):
        ws = WindowSummary(
            window_index=0,
            text="The committee discussed...",
            summary="Discussion of budget issues.",
            bullet_points=["Budget reviewed", "Cuts proposed"],
        )
        assert ws.window_index == 0
        assert len(ws.bullet_points) == 2

    def test_roundtrip(self):
        ws = WindowSummary(window_index=1, text="text", summary="summary", bullet_points=["a", "b"])
        data = ws.to_dict()
        ws2 = WindowSummary.from_dict(data)
        assert ws2.window_index == ws.window_index
        assert ws2.summary == ws.summary


class TestDecisionExtractionResult:
    def test_empty_result(self):
        r = DecisionExtractionResult(text_id="doc1")
        assert r.decisions == []
        assert r.factors == []
        assert r.window_count == 0

    def test_to_dict_roundtrip(self):
        factors = [Factor(text="cost", decision_text="cut budget", polarity="supporting")]
        decisions = [Decision(text="cut budget", factors=factors)]
        r = DecisionExtractionResult(
            text_id="doc1",
            decisions=decisions,
            factors=factors,
            window_count=5,
            metadata={"strategy": "one_pass"},
        )
        data = r.to_dict()
        r2 = DecisionExtractionResult.from_dict(data)
        assert r2.text_id == "doc1"
        assert len(r2.decisions) == 1
        assert len(r2.factors) == 1
        assert r2.window_count == 5

    def test_to_flat_decisions_rows(self):
        factors = [
            Factor(text="cost", decision_text="cut budget", polarity="supporting"),
            Factor(text="morale", decision_text="cut budget", polarity="opposing"),
        ]
        decisions = [Decision(text="cut budget", factors=factors, window_indices=[0, 1])]
        r = DecisionExtractionResult(text_id="doc1", decisions=decisions, factors=factors)

        rows = r.to_flat_decisions_rows()
        assert len(rows) == 1
        assert rows[0]["decision"] == "cut budget"
        assert rows[0]["supporting_count"] == 1
        assert rows[0]["opposing_count"] == 1
        assert rows[0]["text_id"] == "doc1"

    def test_to_flat_factors_rows(self):
        factors = [
            Factor(text="cost", decision_text="cut budget", polarity="supporting", window_index=0),
            Factor(text="morale", decision_text="cut budget", polarity="opposing", window_index=1),
        ]
        r = DecisionExtractionResult(text_id="doc1", factors=factors)

        rows = r.to_flat_factors_rows()
        assert len(rows) == 2
        assert rows[0]["factor"] == "cost"
        assert rows[0]["polarity"] == "supporting"
        assert rows[1]["factor"] == "morale"


class TestDecisionConfig:
    def test_defaults(self):
        c = DecisionConfig()
        assert c.strategy == "semantic_two_pass"
        assert c.validate_decisions is True
        assert c.normalize_decisions is True
        assert c.decision_dedup_threshold == 0.9

    def test_roundtrip(self):
        c = DecisionConfig(strategy="one_pass", temperature=0.5)
        data = c.to_dict()
        c2 = DecisionConfig.from_dict(data)
        assert c2.strategy == "one_pass"
        assert c2.temperature == 0.5


class TestAggregationResult:
    def test_defaults(self):
        a = AggregationResult()
        assert a.total_decisions == 0
        assert a.polarity_distribution == {}

    def test_roundtrip(self):
        a = AggregationResult(
            total_decisions=10,
            total_factors=25,
            unique_decisions=5,
            decision_frequency={"reduce budget": 3, "expand team": 2},
            polarity_distribution={"supporting": 15, "opposing": 8, "neutral": 2},
        )
        data = a.to_dict()
        a2 = AggregationResult.from_dict(data)
        assert a2.total_decisions == 10
        assert a2.unique_decisions == 5
        assert a2.polarity_distribution["supporting"] == 15
