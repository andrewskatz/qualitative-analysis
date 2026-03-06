"""
Decision/factor normalization via semantic clustering.

Clusters semantically similar decisions and factors across multiple
extraction results, assigning canonical labels to each cluster.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .models import DecisionExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class DecisionCluster:
    """
    A cluster of semantically similar decisions.

    Attributes:
        canonical: The canonical/representative decision text.
        members: Original decision texts in this cluster.
        count: Total occurrences across all texts.
        avg_similarity: Average cosine similarity within cluster.
    """

    canonical: str
    members: List[str]
    count: int = 0
    avg_similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizationResult:
    """
    Result of decision/factor normalization.

    Attributes:
        decision_mapping: Map from original decision text to canonical.
        factor_mapping: Map from original factor text to canonical.
        decision_clusters: List of decision clusters.
        factor_clusters: List of factor clusters.
        config: Configuration used.
    """

    decision_mapping: Dict[str, str] = field(default_factory=dict)
    factor_mapping: Dict[str, str] = field(default_factory=dict)
    decision_clusters: List[DecisionCluster] = field(default_factory=list)
    factor_clusters: List[DecisionCluster] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_mapping": self.decision_mapping,
            "factor_mapping": self.factor_mapping,
            "decision_clusters": [c.to_dict() for c in self.decision_clusters],
            "factor_clusters": [c.to_dict() for c in self.factor_clusters],
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizationResult":
        return cls(
            decision_mapping=data.get("decision_mapping", {}),
            factor_mapping=data.get("factor_mapping", {}),
            decision_clusters=[
                DecisionCluster(**c) for c in data.get("decision_clusters", [])
            ],
            factor_clusters=[
                DecisionCluster(**c) for c in data.get("factor_clusters", [])
            ],
            config=data.get("config", {}),
        )


class DecisionNormalizer:
    """
    Normalize decisions and factors using semantic clustering.

    Uses embedding-based similarity to cluster semantically similar
    decisions (and factors) across multiple extraction results, then
    assigns canonical labels based on the most frequent member.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        decision_threshold: float = 0.85,
        factor_threshold: float = 0.80,
    ):
        """
        Args:
            embedding_model: Name of the sentence-transformers model.
            decision_threshold: Cosine similarity threshold for clustering decisions.
            factor_threshold: Cosine similarity threshold for clustering factors.
        """
        self.embedding_model = embedding_model
        self.decision_threshold = decision_threshold
        self.factor_threshold = factor_threshold
        self._embedding_service = None

    def normalize(
        self,
        results: List[DecisionExtractionResult],
    ) -> NormalizationResult:
        """
        Normalize decisions and factors across multiple extraction results.

        Args:
            results: List of extraction results from multiple texts.

        Returns:
            NormalizationResult with mappings and clusters.
        """
        # Collect all unique decisions and factors
        all_decisions: List[str] = []
        decision_counts: Dict[str, int] = {}
        all_factors: List[str] = []
        factor_counts: Dict[str, int] = {}

        for r in results:
            for d in r.decisions:
                all_decisions.append(d.text)
                decision_counts[d.text] = decision_counts.get(d.text, 0) + 1
            for f in r.factors:
                all_factors.append(f.text)
                factor_counts[f.text] = factor_counts.get(f.text, 0) + 1

        # Get unique texts
        unique_decisions = list(dict.fromkeys(all_decisions))
        unique_factors = list(dict.fromkeys(all_factors))

        # Cluster decisions
        decision_clusters = self._cluster_texts(
            unique_decisions, decision_counts, self.decision_threshold
        )

        # Cluster factors
        factor_clusters = self._cluster_texts(
            unique_factors, factor_counts, self.factor_threshold
        )

        # Build mappings
        decision_mapping = {}
        for cluster in decision_clusters:
            for member in cluster.members:
                decision_mapping[member] = cluster.canonical

        factor_mapping = {}
        for cluster in factor_clusters:
            for member in cluster.members:
                factor_mapping[member] = cluster.canonical

        return NormalizationResult(
            decision_mapping=decision_mapping,
            factor_mapping=factor_mapping,
            decision_clusters=decision_clusters,
            factor_clusters=factor_clusters,
            config={
                "embedding_model": self.embedding_model,
                "decision_threshold": self.decision_threshold,
                "factor_threshold": self.factor_threshold,
            },
        )

    def _cluster_texts(
        self,
        texts: List[str],
        counts: Dict[str, int],
        threshold: float,
    ) -> List[DecisionCluster]:
        """
        Cluster texts using embedding similarity.

        Uses greedy agglomerative approach: for each text, find the most
        similar existing cluster and merge if above threshold.
        """
        if not texts:
            return []

        if len(texts) == 1:
            return [
                DecisionCluster(
                    canonical=texts[0],
                    members=texts[:],
                    count=counts.get(texts[0], 1),
                    avg_similarity=1.0,
                )
            ]

        try:
            embedding_service = self._get_embedding_service()
            embeddings = embedding_service.embed(texts)

            from sklearn.metrics.pairwise import cosine_similarity

            sims = cosine_similarity(embeddings)
        except Exception as e:
            logger.warning(
                f"Embedding failed, falling back to exact match: {e}"
            )
            # Fallback: each text is its own cluster
            return [
                DecisionCluster(
                    canonical=t,
                    members=[t],
                    count=counts.get(t, 1),
                    avg_similarity=1.0,
                )
                for t in texts
            ]

        # Greedy clustering
        used = set()
        clusters: List[DecisionCluster] = []

        for i in range(len(texts)):
            if i in used:
                continue

            members = [texts[i]]
            member_indices = [i]
            used.add(i)

            for j in range(i + 1, len(texts)):
                if j in used:
                    continue
                if sims[i][j] >= threshold:
                    members.append(texts[j])
                    member_indices.append(j)
                    used.add(j)

            # Canonical = most frequent member
            canonical = max(members, key=lambda m: counts.get(m, 0))
            total_count = sum(counts.get(m, 0) for m in members)

            # Average similarity
            if len(member_indices) > 1:
                pair_sims = []
                for a in range(len(member_indices)):
                    for b in range(a + 1, len(member_indices)):
                        pair_sims.append(
                            sims[member_indices[a]][member_indices[b]]
                        )
                avg_sim = float(sum(pair_sims) / len(pair_sims))
            else:
                avg_sim = 1.0

            clusters.append(
                DecisionCluster(
                    canonical=canonical,
                    members=members,
                    count=total_count,
                    avg_similarity=round(avg_sim, 4),
                )
            )

        return clusters

    def _get_embedding_service(self):
        """Lazy-load embedding service."""
        if self._embedding_service is None:
            from ..core.embeddings import EmbeddingService

            self._embedding_service = EmbeddingService(
                model_name=self.embedding_model,
            )
        return self._embedding_service
