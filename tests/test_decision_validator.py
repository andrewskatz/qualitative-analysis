"""Tests for decision validator."""

import pytest
from qualitative_analysis.decisions.validator import DecisionValidator


class TestDecisionValidator:
    def setup_method(self):
        self.validator = DecisionValidator()

    # --- Valid decisions ---

    def test_valid_decision_verb(self):
        assert self.validator.is_valid_decision("We decided to adopt cloud migration")

    def test_valid_chose(self):
        assert self.validator.is_valid_decision("The team chose PostgreSQL")

    def test_valid_will(self):
        assert self.validator.is_valid_decision("The company will expand operations")

    def test_valid_plan_to(self):
        assert self.validator.is_valid_decision("They plan to reduce costs")

    def test_valid_imperative(self):
        assert self.validator.is_valid_decision("Standardize on PostgreSQL for all services")

    def test_valid_imperative_reduce(self):
        assert self.validator.is_valid_decision("Reduce the training budget by 20%")

    def test_valid_imperative_deploy(self):
        assert self.validator.is_valid_decision("Deploy the new system next quarter")

    def test_valid_approved(self):
        assert self.validator.is_valid_decision("Management approved the new hiring plan")

    def test_valid_rejected(self):
        assert self.validator.is_valid_decision("The board rejected the merger proposal")

    def test_valid_agreed(self):
        assert self.validator.is_valid_decision("They agreed to postpone the launch")

    # --- Invalid decisions ---

    def test_invalid_empty(self):
        assert not self.validator.is_valid_decision("")

    def test_invalid_whitespace(self):
        assert not self.validator.is_valid_decision("   ")

    def test_invalid_none(self):
        assert not self.validator.is_valid_decision(None)

    def test_invalid_too_short(self):
        assert not self.validator.is_valid_decision("yes")

    def test_invalid_two_words(self):
        assert not self.validator.is_valid_decision("reduce costs")

    def test_invalid_discussion(self):
        assert not self.validator.is_valid_decision("They discussed the budget issue at length")

    def test_invalid_considering(self):
        assert not self.validator.is_valid_decision("The team is considering several options")

    def test_invalid_might(self):
        assert not self.validator.is_valid_decision("We might look at alternatives")

    def test_invalid_suggestion(self):
        assert not self.validator.is_valid_decision("She suggested a new approach to the problem")

    def test_invalid_recommendation(self):
        assert not self.validator.is_valid_decision("The report recommended switching providers")

    def test_invalid_no_indicators(self):
        assert not self.validator.is_valid_decision("The market has shown strong growth this year")

    def test_invalid_observation(self):
        assert not self.validator.is_valid_decision("Costs have increased significantly")

    # --- Batch validation ---

    def test_validate_decisions_filters(self):
        decisions = [
            "We decided to adopt cloud migration",
            "They discussed alternatives",
            "Reduce the budget by 15%",
            "yes",
        ]
        valid = self.validator.validate_decisions(decisions)
        assert len(valid) == 2
        assert "We decided to adopt cloud migration" in valid
        assert "Reduce the budget by 15%" in valid

    def test_validate_decisions_empty_list(self):
        assert self.validator.validate_decisions([]) == []

    # --- Min words ---

    def test_custom_min_words(self):
        validator = DecisionValidator(min_words=5)
        assert not validator.is_valid_decision("Adopt cloud migration now")
        assert validator.is_valid_decision("The team decided to adopt cloud migration")


class TestStripCausalClauses:
    def test_because(self):
        result = DecisionValidator.strip_causal_clauses(
            "reduce training budget because cost overruns"
        )
        assert result == "reduce training budget"

    def test_due_to(self):
        result = DecisionValidator.strip_causal_clauses(
            "expand the team due to increased demand"
        )
        assert result == "expand the team"

    def test_given(self):
        result = DecisionValidator.strip_causal_clauses(
            "pause hiring given market conditions"
        )
        assert result == "pause hiring"

    def test_since(self):
        result = DecisionValidator.strip_causal_clauses(
            "adopt cloud infrastructure since costs are lower"
        )
        assert result == "adopt cloud infrastructure"

    def test_owing_to(self):
        result = DecisionValidator.strip_causal_clauses(
            "delay launch owing to supply chain issues"
        )
        assert result == "delay launch"

    def test_on_account_of(self):
        result = DecisionValidator.strip_causal_clauses(
            "close the facility on account of declining revenue"
        )
        assert result == "close the facility"

    def test_no_causal_clause(self):
        result = DecisionValidator.strip_causal_clauses("adopt cloud migration")
        assert result == "adopt cloud migration"

    def test_case_insensitive(self):
        result = DecisionValidator.strip_causal_clauses(
            "reduce budget Because of high costs"
        )
        assert result == "reduce budget"

    def test_preserves_whitespace(self):
        result = DecisionValidator.strip_causal_clauses("  adopt cloud migration  ")
        assert result == "adopt cloud migration"
