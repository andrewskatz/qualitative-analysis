"""
Entity consolidation module for semantic deduplication.

This module provides functionality for consolidating (deduplicating) entities
based on semantic similarity using embeddings and clustering.

Example usage:
    from qualitative_analysis.entity import EntityConsolidator

    consolidator = EntityConsolidator()

    # Consolidate a list of entities
    result = consolidator.consolidate(
        entities=["climate change", "Climate Change", "global climate change", "flooding"],
        threshold=0.85,
        canonical_method="shortest"
    )

    # Apply mapping to new data
    normalized = result.apply_to_entities(["climate change", "Climate Change"])
    # Returns: ["climate change", "climate change"]
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np

from qualitative_analysis.core.embeddings import EmbeddingService, get_embedding_service
from qualitative_analysis.entity.models import EntityConsolidation, ConsolidationResult

logger = logging.getLogger(__name__)


# Canonical selection methods
CanonicalMethod = Literal["shortest", "frequent", "representative", "first"]


class EntityConsolidator:
    """
    Consolidates semantically similar entities into canonical forms.

    Uses embedding-based clustering to identify groups of similar entities
    and selects a canonical (representative) form for each group.

    Attributes:
        embedding_service: Service for computing embeddings and clustering.
        default_threshold: Default similarity threshold for clustering.

    Example:
        >>> consolidator = EntityConsolidator()
        >>> result = consolidator.consolidate(
        ...     entities=["renewable energy", "Renewable Energy", "clean energy"],
        ...     threshold=0.85
        ... )
        >>> print(result.mapping)
        {'renewable energy': 'renewable energy',
         'Renewable Energy': 'renewable energy',
         'clean energy': 'clean energy'}
    """

    # Threshold presets for convenience
    THRESHOLD_STRICT = 0.90      # Very similar only
    THRESHOLD_MODERATE = 0.85   # Default - good balance
    THRESHOLD_LOOSE = 0.75      # More aggressive merging

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize the entity consolidator.

        Args:
            embedding_model: Name of the sentence transformer model for embeddings.
            embedding_service: Optional pre-configured embedding service.
                              If not provided, creates one with the specified model.
        """
        self.embedding_model = embedding_model

        if embedding_service is not None:
            self.embedding_service = embedding_service
        else:
            self.embedding_service = get_embedding_service(model_name=embedding_model)

    def consolidate(
        self,
        entities: List[str],
        threshold: float = THRESHOLD_MODERATE,
        canonical_method: CanonicalMethod = "shortest",
        frequencies: Optional[Dict[str, int]] = None,
        linkage: str = "complete",
    ) -> ConsolidationResult:
        """
        Consolidate a list of entities by semantic similarity.

        Args:
            entities: List of entity strings to consolidate.
            threshold: Similarity threshold (0.0 to 1.0). Higher = stricter.
                      Use presets: THRESHOLD_STRICT, THRESHOLD_MODERATE, THRESHOLD_LOOSE.
            canonical_method: Method for selecting canonical form:
                - "shortest": Use shortest string (default)
                - "frequent": Use most frequent entity
                - "representative": Use entity closest to cluster centroid
                - "first": Use first occurrence
            frequencies: Optional dict mapping entity -> occurrence count.
                        Required for "frequent" canonical_method.
            linkage: Clustering linkage type ("complete", "average", "single").

        Returns:
            ConsolidationResult with:
                - consolidations: List of EntityConsolidation objects
                - mapping: Dict from original entity to canonical form
                - statistics: Summary statistics
        """
        if not entities:
            return ConsolidationResult(
                config={"threshold": threshold, "canonical_method": canonical_method}
            )

        # Deduplicate while preserving first occurrence order
        seen = set()
        unique_entities = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique_entities.append(e)

        logger.info(f"Consolidating {len(unique_entities)} unique entities (threshold={threshold})")

        # Check if embedding service is available
        if not self.embedding_service.is_available:
            logger.warning("Embedding service not available. Returning identity mapping.")
            return self._identity_result(unique_entities, threshold, canonical_method)

        # Cluster entities
        cluster_info = self.embedding_service.cluster_with_info(
            texts=unique_entities,
            threshold=threshold,
            linkage=linkage,
        )

        # Build consolidations
        consolidations = []
        mapping = {}

        for cluster in cluster_info:
            items = cluster["items"]
            indices = cluster["indices"]
            avg_similarity = cluster["avg_similarity"]

            # Select canonical form
            canonical = self._select_canonical(
                items=items,
                indices=indices,
                method=canonical_method,
                frequencies=frequencies,
                unique_entities=unique_entities,
            )

            # Create consolidation record
            consolidation = EntityConsolidation(
                canonical=canonical,
                variants=items,
                frequency=sum(frequencies.get(item, 1) for item in items) if frequencies else len(items),
                avg_similarity=round(avg_similarity, 4),
                confidence=round(avg_similarity, 4),  # Use avg_similarity as confidence
            )
            consolidations.append(consolidation)

            # Add to mapping
            for item in items:
                mapping[item] = canonical

        # Create result
        result = ConsolidationResult(
            consolidations=consolidations,
            mapping=mapping,
            config={
                "threshold": threshold,
                "canonical_method": canonical_method,
                "embedding_model": self.embedding_model,
                "linkage": linkage,
                "original_count": len(unique_entities),
            },
        )
        result.compute_statistics()

        logger.info(
            f"Consolidation complete: {result.statistics.get('original_count', 0)} -> "
            f"{result.statistics.get('consolidated_count', 0)} entities "
            f"({result.statistics.get('reduction_percentage', 0):.1f}% reduction)"
        )

        return result

    def consolidate_from_csv(
        self,
        csv_path: Union[str, Path],
        entity_col: str = "entity",
        frequency_col: Optional[str] = None,
        threshold: float = THRESHOLD_MODERATE,
        canonical_method: CanonicalMethod = "shortest",
    ) -> ConsolidationResult:
        """
        Consolidate entities from a CSV file.

        Args:
            csv_path: Path to CSV file containing entities.
            entity_col: Column name for entity strings.
            frequency_col: Optional column name for frequency counts.
            threshold: Similarity threshold for clustering.
            canonical_method: Method for selecting canonical form.

        Returns:
            ConsolidationResult with consolidation mapping.
        """
        csv_path = Path(csv_path)
        entities = []
        frequencies = {} if frequency_col else None

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entity = row.get(entity_col, "").strip()
                if entity:
                    entities.append(entity)
                    if frequencies is not None and frequency_col in row:
                        try:
                            freq = int(row[frequency_col])
                            frequencies[entity] = frequencies.get(entity, 0) + freq
                        except ValueError:
                            pass

        return self.consolidate(
            entities=entities,
            threshold=threshold,
            canonical_method=canonical_method,
            frequencies=frequencies,
        )

    def _select_canonical(
        self,
        items: List[str],
        indices: List[int],
        method: CanonicalMethod,
        frequencies: Optional[Dict[str, int]],
        unique_entities: List[str],
    ) -> str:
        """Select the canonical form for a cluster based on the specified method."""
        if not items:
            return ""

        if len(items) == 1:
            return items[0]

        if method == "shortest":
            # Shortest string (often the root/canonical form)
            return min(items, key=len)

        elif method == "frequent":
            # Most frequent entity
            if frequencies:
                return max(items, key=lambda x: frequencies.get(x, 0))
            # Fall back to shortest if no frequencies
            return min(items, key=len)

        elif method == "representative":
            # Entity closest to cluster centroid (most central)
            try:
                embeddings = self.embedding_service.embed(items, normalize=True)
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)  # Normalize centroid

                # Find item with highest similarity to centroid
                similarities = np.dot(embeddings, centroid)
                best_idx = int(np.argmax(similarities))
                return items[best_idx]
            except Exception as e:
                logger.warning(f"Error computing representative, falling back to shortest: {e}")
                return min(items, key=len)

        elif method == "first":
            # First occurrence in original list
            min_idx = min(indices)
            return unique_entities[min_idx]

        else:
            # Default to shortest
            return min(items, key=len)

    def _identity_result(
        self,
        entities: List[str],
        threshold: float,
        canonical_method: str,
    ) -> ConsolidationResult:
        """Create an identity mapping (no consolidation) as fallback."""
        consolidations = [
            EntityConsolidation(
                canonical=e,
                variants=[e],
                frequency=1,
                avg_similarity=1.0,
                confidence=1.0,
            )
            for e in entities
        ]

        mapping = {e: e for e in entities}

        return ConsolidationResult(
            consolidations=consolidations,
            mapping=mapping,
            config={
                "threshold": threshold,
                "canonical_method": canonical_method,
                "embedding_model": self.embedding_model,
                "note": "Identity mapping - embedding service unavailable",
            },
            statistics={
                "original_count": len(entities),
                "consolidated_count": len(entities),
                "reduction_count": 0,
                "reduction_percentage": 0.0,
                "cluster_count": len(entities),
                "largest_cluster_size": 1,
            },
        )

    def save(self, result: ConsolidationResult, path: Union[str, Path]) -> None:
        """
        Save consolidation result to JSON file.

        Args:
            result: ConsolidationResult to save.
            path: Output file path.
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved consolidation result to {path}")

    def load(self, path: Union[str, Path]) -> ConsolidationResult:
        """
        Load consolidation result from JSON file.

        Args:
            path: Input file path.

        Returns:
            ConsolidationResult loaded from file.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ConsolidationResult.from_dict(data)


# Convenience function for simple use cases
def consolidate_entities(
    entities: List[str],
    threshold: float = 0.85,
    canonical_method: CanonicalMethod = "shortest",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> ConsolidationResult:
    """
    Convenience function to consolidate entities.

    Args:
        entities: List of entity strings.
        threshold: Similarity threshold (0.0 to 1.0).
        canonical_method: Method for selecting canonical form.
        embedding_model: Sentence transformer model name.

    Returns:
        ConsolidationResult with consolidation mapping.

    Example:
        >>> result = consolidate_entities(
        ...     ["climate change", "Climate Change", "global warming"],
        ...     threshold=0.85
        ... )
        >>> print(result.mapping)
    """
    consolidator = EntityConsolidator(embedding_model=embedding_model)
    return consolidator.consolidate(
        entities=entities,
        threshold=threshold,
        canonical_method=canonical_method,
    )
