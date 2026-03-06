"""
Decision Validator

Validates that extracted text represents an actual decision rather than
a statement, fact, observation, or discussion. Also provides normalization
to strip causal clauses from decision text.
"""

import re
from typing import List


class DecisionValidator:
    """Validates that text represents an actual decision."""

    # Verbs that typically indicate decisions (strong indicators)
    DECISION_VERBS = [
        "decide", "decided", "decides",
        "choose", "chose", "chosen",
        "adopt", "adopted", "adopts",
        "implement", "implemented", "implements",
        "launch", "launched", "launches",
        "approve", "approved", "approves",
        "reject", "rejected", "rejects",
        "select", "selected", "selects",
        "opt", "opted", "opts",
        "commit", "committed", "commits",
        "agree", "agreed", "agrees",
        "determine", "determined", "determines",
        "resolve", "resolved", "resolves",
        "plan", "planned", "plans",
        "will",
        "going to",
        "intend", "intended", "intends",
        "pursue", "pursued", "pursues",
        "proceed", "proceeded", "proceeds",
        "standardize", "standardized", "standardizes",
        "deploy", "deployed", "deploys",
        "pause", "paused", "pauses",
        "expand", "expanded", "expands",
        "phase out", "phased out", "phases out",
    ]

    # Patterns that typically indicate decisions
    DECISION_PATTERNS = [
        "to ",
        "will ",
        "should ",
        "must ",
        "going to",
        "plan to",
        "intend to",
        "decided to",
        "chose to",
        "agreed to",
    ]

    # Words that indicate non-decisions (discussions, observations, facts)
    NON_DECISION_INDICATORS = [
        "discuss", "discussed", "discussing",
        "consider", "considered", "considering",
        "might", "maybe", "perhaps",
        "could", "would",
        "suggest", "suggested", "suggesting",
        "recommend", "recommended", "recommending",
        "think", "thought", "thinking",
        "believe", "believed", "believing",
        "have increased", "has increased",
        "have decreased", "has decreased",
        "have risen", "has risen",
        "have fallen", "has fallen",
        "directive to",
        "requirement to",
    ]

    # Action verbs that can start imperative decisions (commands)
    IMPERATIVE_VERBS = [
        "adopt", "implement", "launch", "approve", "reject",
        "select", "standardize", "deploy", "pause", "expand",
        "phase out", "increase", "reduce", "cut", "eliminate",
        "establish", "create", "develop", "build", "maintain",
    ]

    # Regex patterns for stripping causal clauses
    _CAUSAL_PATTERNS = [
        r"\s+because\s+.*$",
        r"\s+due\s+to\s+.*$",
        r"\s+given\s+.*$",
        r"\s+since\s+.*$",
        r"\s+as\s+.*$",
        r"\s+owing\s+to\s+.*$",
        r"\s+on\s+account\s+of\s+.*$",
    ]

    def __init__(self, min_words: int = 3, verbose: bool = False):
        """
        Initialize validator.

        Args:
            min_words: Minimum number of words required for a valid decision.
            verbose: Whether to print validation details.
        """
        self.min_words = min_words
        self.verbose = verbose

    def is_valid_decision(self, decision_text: str) -> bool:
        """
        Validate that text represents an actual decision.

        Args:
            decision_text: The text to validate.

        Returns:
            True if text represents a decision, False otherwise.
        """
        if not decision_text or not decision_text.strip():
            return False

        text = decision_text.strip()
        lower = text.lower()

        # Check minimum length
        word_count = len(text.split())
        if word_count < self.min_words:
            return False

        # Check for non-decision indicators (discussions, suggestions)
        has_non_decision = any(
            indicator in lower for indicator in self.NON_DECISION_INDICATORS
        )
        if has_non_decision:
            return False

        # Check for decision verbs
        has_decision_verb = any(verb in lower for verb in self.DECISION_VERBS)

        # Check for decision patterns
        has_decision_pattern = any(
            pattern in lower for pattern in self.DECISION_PATTERNS
        )

        # Check for imperative form (starts with action verb)
        first_word = text.split()[0].lower() if text.split() else ""
        is_imperative = any(
            first_word == verb.split()[0] for verb in self.IMPERATIVE_VERBS
        )

        return has_decision_verb or has_decision_pattern or is_imperative

    def validate_decisions(self, decisions: List[str]) -> List[str]:
        """
        Validate a list of decisions and return only valid ones.

        Args:
            decisions: List of decision texts to validate.

        Returns:
            List of valid decisions.
        """
        valid_decisions = []

        for decision in decisions:
            if self.is_valid_decision(decision):
                valid_decisions.append(decision)

        if self.verbose and len(decisions) != len(valid_decisions):
            filtered_count = len(decisions) - len(valid_decisions)
            print(f"Filtered out {filtered_count} non-decision(s)")

        return valid_decisions

    @staticmethod
    def strip_causal_clauses(text: str) -> str:
        """
        Normalize decision text by stripping causal clauses.

        Removes phrases like "because X", "due to X", "given X",
        "since X", "as X", "owing to X", "on account of X".

        Args:
            text: Decision text to normalize.

        Returns:
            Decision text with causal clauses removed.

        Example:
            >>> DecisionValidator.strip_causal_clauses(
            ...     "reduce training budget because cost overruns"
            ... )
            'reduce training budget'
        """
        normalized = text.strip()
        for pattern in DecisionValidator._CAUSAL_PATTERNS:
            normalized = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
        return normalized.strip()
