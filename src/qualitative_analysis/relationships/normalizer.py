"""
Relationship normalizer for consolidating entities and relationship types.

Uses the shared embedding service for semantic clustering of:
- Entity names (e.g., "climate change" and "Climate Change" and "global climate change")
- Relationship types (e.g., "causes" and "leads to" and "results in")

After normalization, relationships with matching (source, type, target) tuples
are merged, aggregating their evidence and metadata.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from ..core.embeddings import EmbeddingService, get_embedding_service
from .models import (
    Relationship,
    EntityCluster,
    NormalizedRelationship,
    NormalizationResult,
)

logger = logging.getLogger(__name__)

# Conservativeness presets for clustering
THRESHOLD_PRESETS = {
    "conservative": 0.85,  # Only very similar items clustered
    "moderate": 0.75,      # Default balance
    "aggressive": 0.60,    # More aggressive merging
}

# Canonical selection methods
CANONICAL_METHODS = ["shortest", "frequent", "representative", "llm"]


class RelationshipNormalizer:
    """
    Normalizes relationships by clustering similar entities and relationship types.

    The normalization pipeline:
    1. Extract unique entities (sources + targets) and relationship types
    2. Cluster entities semantically using embeddings
    3. Cluster relationship types semantically
    4. Generate canonical labels for each cluster
    5. Apply mappings to relationships
    6. Merge relationships with identical normalized (source, type, target) tuples

    Example:
        >>> normalizer = RelationshipNormalizer()
        >>> result = normalizer.normalize(relationships, entity_threshold=0.85)
        >>> print(f"Reduced {len(relationships)} to {len(result.normalized_relationships)}")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the normalizer.

        Args:
            embedding_model: SentenceTransformer model name for embeddings.
            device: Device for computation ("cpu", "cuda", "mps", or None for auto).
        """
        self.embedding_model_name = embedding_model
        self.device = device
        self._embedding_service: Optional[EmbeddingService] = None

    def _get_embedding_service(self) -> EmbeddingService:
        """Lazy load the embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                model_name=self.embedding_model_name,
                device=self.device,
            )
        return self._embedding_service

    def normalize(
        self,
        relationships: List[Relationship],
        entity_threshold: Union[float, str] = "moderate",
        type_threshold: Union[float, str] = "moderate",
        canonical_method: str = "shortest",
        llm_provider: Optional[Any] = None,
        normalize_entities: bool = True,
        normalize_types: bool = True,
    ) -> NormalizationResult:
        """
        Normalize relationships by clustering entities and types.

        Args:
            relationships: List of Relationship objects to normalize.
            entity_threshold: Similarity threshold for entity clustering.
                            Can be float (0.0-1.0) or preset name.
            type_threshold: Similarity threshold for type clustering.
                           Can be float (0.0-1.0) or preset name.
            canonical_method: How to select canonical labels:
                            - "shortest": Use shortest string
                            - "frequent": Use most frequent variant
                            - "representative": Use most central embedding
                            - "llm": Use LLM to generate abstract label
            llm_provider: LLM provider for canonical_method="llm".
            normalize_entities: Whether to normalize entity names.
            normalize_types: Whether to normalize relationship types.

        Returns:
            NormalizationResult with mappings, clusters, and normalized relationships.
        """
        if not relationships:
            return NormalizationResult(
                entity_mapping={},
                type_mapping={},
                entity_clusters=[],
                type_clusters=[],
                normalized_relationships=[],
                config={"error": "No relationships provided"},
            )

        # Resolve thresholds
        entity_thresh = self._resolve_threshold(entity_threshold)
        type_thresh = self._resolve_threshold(type_threshold)

        logger.info(
            f"Normalizing {len(relationships)} relationships: "
            f"entity_threshold={entity_thresh}, type_threshold={type_thresh}, "
            f"canonical_method={canonical_method}"
        )

        # Extract unique entities and types with counts
        entity_counts, type_counts = self._count_entities_and_types(relationships)

        entities = list(entity_counts.keys())
        types = list(type_counts.keys())

        logger.info(f"Found {len(entities)} unique entities, {len(types)} unique types")

        # Get embedding service
        embedding_service = self._get_embedding_service()

        # Cluster entities
        if normalize_entities and len(entities) > 1:
            entity_clusters = self._cluster_items(
                entities, entity_thresh, embedding_service
            )
            entity_clusters = self._assign_canonical_labels(
                entity_clusters,
                entity_counts,
                canonical_method,
                embedding_service,
                llm_provider,
            )
        else:
            # No clustering - each entity is its own cluster
            entity_clusters = [
                EntityCluster(canonical=e, members=[e], count=entity_counts[e])
                for e in entities
            ]

        # Cluster relationship types
        if normalize_types and len(types) > 1:
            type_clusters = self._cluster_items(
                types, type_thresh, embedding_service
            )
            type_clusters = self._assign_canonical_labels(
                type_clusters,
                type_counts,
                canonical_method,
                embedding_service,
                llm_provider,
            )
        else:
            type_clusters = [
                EntityCluster(canonical=t, members=[t], count=type_counts[t])
                for t in types
            ]

        # Build mappings
        entity_mapping = self._build_mapping(entity_clusters)
        type_mapping = self._build_mapping(type_clusters)

        # Apply normalization and merge relationships
        normalized_relationships = self._apply_normalization_and_merge(
            relationships, entity_mapping, type_mapping
        )

        # Build config
        config = {
            "entity_threshold": entity_thresh,
            "type_threshold": type_thresh,
            "canonical_method": canonical_method,
            "normalize_entities": normalize_entities,
            "normalize_types": normalize_types,
            "embedding_model": self.embedding_model_name,
            "original_count": len(relationships),
            "normalized_count": len(normalized_relationships),
        }

        logger.info(
            f"Normalization complete: {len(entity_clusters)} entity clusters, "
            f"{len(type_clusters)} type clusters, "
            f"{len(relationships)} → {len(normalized_relationships)} relationships"
        )

        return NormalizationResult(
            entity_mapping=entity_mapping,
            type_mapping=type_mapping,
            entity_clusters=entity_clusters,
            type_clusters=type_clusters,
            normalized_relationships=normalized_relationships,
            config=config,
        )

    def _resolve_threshold(self, threshold: Union[float, str]) -> float:
        """Convert threshold preset name to float value."""
        if isinstance(threshold, float):
            return threshold
        if isinstance(threshold, str):
            if threshold in THRESHOLD_PRESETS:
                return THRESHOLD_PRESETS[threshold]
            try:
                return float(threshold)
            except ValueError:
                pass
        logger.warning(f"Unknown threshold '{threshold}', using moderate (0.75)")
        return 0.75

    def _count_entities_and_types(
        self,
        relationships: List[Relationship],
    ) -> tuple[Dict[str, int], Dict[str, int]]:
        """Count occurrences of entities and relationship types."""
        entity_counts: Dict[str, int] = defaultdict(int)
        type_counts: Dict[str, int] = defaultdict(int)

        for rel in relationships:
            entity_counts[rel.source] += 1
            entity_counts[rel.target] += 1
            type_counts[rel.type] += 1

        return dict(entity_counts), dict(type_counts)

    def _cluster_items(
        self,
        items: List[str],
        threshold: float,
        embedding_service: EmbeddingService,
    ) -> List[EntityCluster]:
        """Cluster items using the embedding service."""
        if not items:
            return []

        if len(items) == 1:
            return [EntityCluster(canonical="", members=items, avg_similarity=1.0)]

        # Use the embedding service's cluster_with_info method
        cluster_info = embedding_service.cluster_with_info(
            items, threshold=threshold, linkage="complete"
        )

        # Convert to EntityCluster objects
        clusters = []
        for info in cluster_info:
            clusters.append(EntityCluster(
                canonical="",  # Will be set later
                members=sorted(info["items"]),
                count=info["count"],
                avg_similarity=round(info["avg_similarity"], 3),
            ))

        return clusters

    def _assign_canonical_labels(
        self,
        clusters: List[EntityCluster],
        counts: Dict[str, int],
        method: str,
        embedding_service: EmbeddingService,
        llm_provider: Optional[Any] = None,
    ) -> List[EntityCluster]:
        """Assign canonical labels to clusters using the specified method."""
        for cluster in clusters:
            members = cluster.members

            if len(members) == 1:
                cluster.canonical = members[0]
                cluster.count = counts.get(members[0], 1)
                continue

            # Update count to sum of member counts
            cluster.count = sum(counts.get(m, 1) for m in members)

            if method == "shortest":
                # Use shortest string
                cluster.canonical = min(members, key=len)

            elif method == "frequent":
                # Use most frequent variant
                cluster.canonical = max(members, key=lambda m: counts.get(m, 0))

            elif method == "representative":
                # Use most central member (highest avg similarity to others)
                cluster.canonical = self._find_representative(members, embedding_service)

            elif method == "llm" and llm_provider is not None:
                # Use LLM to generate canonical label
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                canonical = loop.run_until_complete(
                    self._generate_canonical_llm(members, llm_provider)
                )
                cluster.canonical = canonical or min(members, key=len)

            else:
                # Default to shortest
                cluster.canonical = min(members, key=len)

        return clusters

    def _find_representative(
        self,
        members: List[str],
        embedding_service: EmbeddingService,
    ) -> str:
        """Find the most representative member (highest avg similarity to others)."""
        if len(members) <= 1:
            return members[0] if members else ""

        # Compute pairwise similarity matrix
        sim_matrix = embedding_service.compute_pairwise_similarity(members)

        # Find member with highest average similarity to others
        best_idx = 0
        best_avg = -1.0

        for i in range(len(members)):
            # Average similarity to all other members
            avg_sim = sum(sim_matrix[i, j] for j in range(len(members)) if i != j)
            avg_sim /= (len(members) - 1)

            if avg_sim > best_avg:
                best_avg = avg_sim
                best_idx = i

        return members[best_idx]

    async def _generate_canonical_llm(
        self,
        members: List[str],
        llm_provider: Any,
    ) -> Optional[str]:
        """Generate a canonical label using an LLM."""
        prompt = f"""Given these semantically similar labels that all refer to the same concept:
{chr(10).join(f'- {m}' for m in members)}

Generate a single canonical label that best represents this group.

Respond with a JSON object:
{{"reasoning": "...", "canonical_label": "..."}}

The canonical_label should be:
- 1-4 words
- Clear and unambiguous
- The most standard/common form of the concept

Respond with ONLY the JSON object."""

        try:
            response = await llm_provider.generate(prompt=prompt, temperature=0.3)

            # Parse JSON response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            data = json.loads(cleaned)
            return data.get("canonical_label", "").strip()

        except Exception as e:
            logger.warning(f"LLM canonical generation failed: {e}")
            return None

    def _build_mapping(self, clusters: List[EntityCluster]) -> Dict[str, str]:
        """Build mapping from original items to canonical labels."""
        mapping = {}
        for cluster in clusters:
            for member in cluster.members:
                mapping[member] = cluster.canonical
        return mapping

    def _apply_normalization_and_merge(
        self,
        relationships: List[Relationship],
        entity_mapping: Dict[str, str],
        type_mapping: Dict[str, str],
    ) -> List[NormalizedRelationship]:
        """Apply normalization mappings and merge duplicate relationships."""
        # Group relationships by normalized (source, type, target) tuple
        grouped: Dict[tuple, List[Relationship]] = defaultdict(list)

        for rel in relationships:
            norm_source = entity_mapping.get(rel.source, rel.source)
            norm_target = entity_mapping.get(rel.target, rel.target)
            norm_type = type_mapping.get(rel.type, rel.type)

            key = (norm_source.lower(), norm_type.lower(), norm_target.lower())
            grouped[key].append(rel)

        # Merge grouped relationships
        normalized = []
        for (norm_source_lower, norm_type_lower, norm_target_lower), rels in grouped.items():
            # Use the first relationship's normalized forms for display
            first = rels[0]
            norm_source = entity_mapping.get(first.source, first.source)
            norm_target = entity_mapping.get(first.target, first.target)
            norm_type = type_mapping.get(first.type, first.type)

            # Collect all descriptions, window indices, and text IDs
            descriptions = []
            window_indices = []
            text_ids = []
            original_sources = set()
            original_targets = set()
            original_types = set()

            for rel in rels:
                if rel.description and rel.description not in descriptions:
                    descriptions.append(rel.description)
                if rel.window_index not in window_indices:
                    window_indices.append(rel.window_index)
                if rel.text_id and rel.text_id not in text_ids:
                    text_ids.append(rel.text_id)
                original_sources.add(rel.source)
                original_targets.add(rel.target)
                original_types.add(rel.type)

            # Merge description (first one as primary, or combine)
            merged_description = descriptions[0] if descriptions else ""

            normalized.append(NormalizedRelationship(
                source=norm_source,
                target=norm_target,
                type=norm_type,
                original_source="; ".join(sorted(original_sources)),
                original_target="; ".join(sorted(original_targets)),
                original_type="; ".join(sorted(original_types)),
                description=merged_description,
                descriptions=descriptions,
                confidence=1.0,  # Could compute based on cluster similarity
                count=len(rels),
                window_indices=sorted(window_indices),
                text_ids=sorted(text_ids),
            ))

        # Sort by count (most frequent first)
        normalized.sort(key=lambda r: r.count, reverse=True)

        return normalized

    def save(self, result: NormalizationResult, path: Union[str, Path]) -> None:
        """Save normalization result to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved normalization result to {path}")

    def load(self, path: Union[str, Path]) -> NormalizationResult:
        """Load normalization result from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return NormalizationResult.from_dict(data)


def normalize_relationships(
    relationships: List[Relationship],
    entity_threshold: Union[float, str] = "moderate",
    type_threshold: Union[float, str] = "moderate",
    canonical_method: str = "shortest",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> NormalizationResult:
    """
    Convenience function to normalize relationships without instantiating the class.

    Args:
        relationships: List of Relationship objects.
        entity_threshold: Threshold for entity clustering (float or preset name).
        type_threshold: Threshold for type clustering (float or preset name).
        canonical_method: Method for selecting canonical labels.
        embedding_model: SentenceTransformer model name.

    Returns:
        NormalizationResult with normalized relationships.
    """
    normalizer = RelationshipNormalizer(embedding_model=embedding_model)
    return normalizer.normalize(
        relationships,
        entity_threshold=entity_threshold,
        type_threshold=type_threshold,
        canonical_method=canonical_method,
    )
