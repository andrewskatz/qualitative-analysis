"""Tests for decision/factor aggregation."""

import pytest
from qualitative_analysis.decisions.models import (
    Decision,
    DecisionExtractionResult,
    Factor,
)
from qualitative_analysis.decisions.aggregator import (
    DecisionAggregator,
    aggregate_decisions,
)


def _make_result(
    text_id: str,
    decisions_data: list,
) -> DecisionExtractionResult:
    """Helper to build a DecisionExtractionResult from simplified data."""
    decisions = []
    all_factors = []
    for d_text, factors_list in decisions_data:
        factors = []
        for f_text, polarity in factors_list:
            f = Factor(text=f_text, decision_text=d_text, polarity=polarity, text_id=text_id)
            factors.append(f)
            all_factors.append(f)
        decisions.append(Decision(text=d_text, factors=factors, text_ids=[text_id]))
    return DecisionExtractionResult(
        text_id=text_id, decisions=decisions, factors=all_factors
    )


class TestDecisionAggregator:
    def test_empty_results(self):
        agg = DecisionAggregator()
        result = agg.aggregate([])
        assert result.total_decisions == 0
        assert result.total_factors == 0

    def test_single_text(self):
        r = _make_result("doc1", [
            ("reduce budget", [
                ("cost overruns", "supporting"),
                ("employee morale", "opposing"),
            ]),
            ("expand team", [
                ("growing demand", "supporting"),
            ]),
        ])
        result = aggregate_decisions([r])

        assert result.total_decisions == 2
        assert result.total_factors == 3
        assert result.unique_decisions == 2
        assert result.unique_factors == 3
        assert result.decision_frequency["reduce budget"] == 1
        assert result.decision_frequency["expand team"] == 1
        assert result.polarity_distribution["supporting"] == 2
        assert result.polarity_distribution["opposing"] == 1
        assert result.texts_per_decision["reduce budget"] == 1

    def test_multiple_texts(self):
        r1 = _make_result("doc1", [
            ("reduce budget", [("cost overruns", "supporting")]),
        ])
        r2 = _make_result("doc2", [
            ("reduce budget", [("quarterly losses", "supporting")]),
            ("expand team", [("talent shortage", "opposing")]),
        ])
        result = aggregate_decisions([r1, r2])

        assert result.total_decisions == 3
        assert result.unique_decisions == 2
        assert result.decision_frequency["reduce budget"] == 2
        assert result.decision_frequency["expand team"] == 1
        assert result.texts_per_decision["reduce budget"] == 2
        assert result.texts_per_decision["expand team"] == 1

    def test_averages(self):
        r1 = _make_result("doc1", [
            ("d1", [("f1", "supporting"), ("f2", "opposing")]),
        ])
        r2 = _make_result("doc2", [
            ("d2", [("f3", "neutral")]),
            ("d3", []),
        ])
        result = aggregate_decisions([r1, r2])

        assert result.avg_decisions_per_text == 1.5  # 3 decisions / 2 texts
        assert result.avg_factors_per_decision == 1.0  # 3 factors / 3 decisions

    def test_polarity_distribution(self):
        r = _make_result("doc1", [
            ("d1", [
                ("f1", "supporting"),
                ("f2", "supporting"),
                ("f3", "opposing"),
                ("f4", "neutral"),
            ]),
        ])
        result = aggregate_decisions([r])

        assert result.polarity_distribution["supporting"] == 2
        assert result.polarity_distribution["opposing"] == 1
        assert result.polarity_distribution["neutral"] == 1
