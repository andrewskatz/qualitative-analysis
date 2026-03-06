"""
Decision/factor aggregation across multiple extraction results.

Computes frequency counts, polarity distribution, and cross-text coverage.
"""

from collections import Counter
from typing import Dict, List, Set

from .models import AggregationResult, DecisionExtractionResult


class DecisionAggregator:
    """Compute aggregate statistics across multiple extraction results."""

    def aggregate(
        self,
        results: List[DecisionExtractionResult],
    ) -> AggregationResult:
        """
        Aggregate statistics from multiple extraction results.

        Args:
            results: List of DecisionExtractionResult from multiple texts.

        Returns:
            AggregationResult with frequency counts and distributions.
        """
        all_decision_texts: List[str] = []
        all_factor_texts: List[str] = []
        polarity_counts: Counter = Counter()
        decision_to_texts: Dict[str, Set[str]] = {}
        factor_to_texts: Dict[str, Set[str]] = {}

        for result in results:
            text_id = result.text_id

            # Collect decisions
            for d in result.decisions:
                all_decision_texts.append(d.text)
                decision_to_texts.setdefault(d.text, set()).add(text_id)

            # Collect factors
            for f in result.factors:
                all_factor_texts.append(f.text)
                polarity_counts[f.polarity] += 1
                factor_to_texts.setdefault(f.text, set()).add(text_id)

        decision_frequency = dict(Counter(all_decision_texts))
        factor_frequency = dict(Counter(all_factor_texts))

        texts_per_decision = {
            d: len(text_ids) for d, text_ids in decision_to_texts.items()
        }
        texts_per_factor = {
            f: len(text_ids) for f, text_ids in factor_to_texts.items()
        }

        num_texts = len(results) if results else 1
        total_decisions = len(all_decision_texts)
        total_factors = len(all_factor_texts)

        avg_decisions_per_text = total_decisions / num_texts if num_texts else 0
        avg_factors_per_decision = (
            total_factors / total_decisions if total_decisions else 0
        )

        return AggregationResult(
            total_decisions=total_decisions,
            total_factors=total_factors,
            unique_decisions=len(decision_frequency),
            unique_factors=len(factor_frequency),
            decision_frequency=decision_frequency,
            factor_frequency=factor_frequency,
            polarity_distribution=dict(polarity_counts),
            texts_per_decision=texts_per_decision,
            texts_per_factor=texts_per_factor,
            avg_decisions_per_text=round(avg_decisions_per_text, 2),
            avg_factors_per_decision=round(avg_factors_per_decision, 2),
        )


def aggregate_decisions(
    results: List[DecisionExtractionResult],
) -> AggregationResult:
    """Module-level convenience function for aggregation."""
    return DecisionAggregator().aggregate(results)
