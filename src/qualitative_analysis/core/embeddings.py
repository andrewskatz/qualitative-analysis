"""
Shared embedding service for semantic operations across qualitative analysis modules.

This module provides a unified interface for generating embeddings, computing
semantic similarity, and clustering text based on meaning. It's used by:
- Entity consolidation (deduplicating similar entities)
- Relationship normalization (clustering relationship types)
- Semantic graph layouts (positioning nodes by meaning)
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from sklearn.cluster import AgglomerativeClustering
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Shared embedding service for semantic operations.

    Provides embedding generation, similarity computation, and clustering
    functionality used across entity and relationship analysis modules.

    Attributes:
        model_name: Name of the sentence transformer model
        model: The loaded SentenceTransformer model
        device: Device to run computations on (cpu, cuda, mps)

    Example:
        >>> service = EmbeddingService()
        >>> similarity = service.compute_similarity("climate change", "global warming")
        >>> print(f"Similarity: {similarity:.2f}")
        Similarity: 0.89
    """

    # Default models for different use cases
    DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality
    HIGH_QUALITY_MODEL = "all-mpnet-base-v2"  # Slower, better quality
    MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Multi-language

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        trust_remote_code: bool = False
    ):
        """
        Initialize the embedding service.

        Args:
            model_name: Name of the sentence transformer model to use.
                       Defaults to 'all-MiniLM-L6-v2' for good speed/quality balance.
            device: Device for computation ('cpu', 'cuda', 'mps', or None for auto).
            trust_remote_code: Whether to trust remote code for custom models.
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.model: Optional[SentenceTransformer] = None
        self._embedding_dim: Optional[int] = None

        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return

        try:
            logger.info(f"Loading embedding model: {model_name} on {self.device}")
            self.model = SentenceTransformer(
                model_name,
                device=self.device,
                trust_remote_code=trust_remote_code
            )
            # Cache embedding dimension
            test_embedding = self.model.encode(["test"])
            self._embedding_dim = test_embedding.shape[1]
            logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            self.model = None

    def _detect_device(self) -> str:
        """Auto-detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS can be unstable, default to CPU for reliability
                return "cpu"
        except ImportError:
            pass
        return "cpu"

    @property
    def is_available(self) -> bool:
        """Check if the embedding service is ready to use."""
        return self.model is not None

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim or 0

    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            normalize: Whether to L2-normalize embeddings (for cosine similarity).

        Returns:
            NumPy array of shape (len(texts), embedding_dim).

        Raises:
            RuntimeError: If the model is not available.
        """
        if not self.model:
            raise RuntimeError("Embedding model not available")

        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100
        )
        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Cosine similarity score between 0.0 and 1.0.
        """
        if not self.model or not text1 or not text2:
            return 0.0

        try:
            embeddings = self.embed([text1, text2], normalize=True)
            # Cosine similarity of normalized vectors is just dot product
            similarity = float(np.dot(embeddings[0], embeddings[1]))
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0

    def compute_batch_similarity(
        self,
        query: str,
        candidates: List[str]
    ) -> List[float]:
        """
        Compute similarity between a query and multiple candidates efficiently.

        Args:
            query: Query text string.
            candidates: List of candidate text strings.

        Returns:
            List of similarity scores, one per candidate.
        """
        if not self.model or not query or not candidates:
            return [0.0] * len(candidates)

        try:
            # Embed query and candidates together for efficiency
            all_texts = [query] + candidates
            embeddings = self.embed(all_texts, normalize=True)

            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]

            # Dot product with normalized vectors = cosine similarity
            similarities = np.dot(candidate_embeddings, query_embedding)
            return similarities.tolist()
        except Exception as e:
            logger.error(f"Error computing batch similarity: {e}")
            return [0.0] * len(candidates)

    def compute_pairwise_similarity(self, texts: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            Symmetric similarity matrix of shape (len(texts), len(texts)).
        """
        if not self.model or not texts:
            return np.array([])

        embeddings = self.embed(texts, normalize=True)
        # For normalized vectors, similarity matrix is just matrix multiplication
        similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix

    def cluster(
        self,
        texts: List[str],
        threshold: float = 0.75,
        method: str = "agglomerative",
        linkage: str = "complete"
    ) -> List[List[int]]:
        """
        Cluster texts by semantic similarity.

        Args:
            texts: List of text strings to cluster.
            threshold: Similarity threshold for clustering (0.0 to 1.0).
                      Higher = stricter (fewer, tighter clusters).
            method: Clustering method ('agglomerative').
            linkage: Linkage type for agglomerative ('complete', 'average', 'single').

        Returns:
            List of clusters, where each cluster is a list of indices into texts.
        """
        if not self.model or not texts or not HAS_SKLEARN:
            return [[i] for i in range(len(texts))]  # Each text in its own cluster

        if len(texts) == 1:
            return [[0]]

        try:
            embeddings = self.embed(texts, normalize=True)

            # Convert similarity threshold to distance threshold
            # Cosine distance = 1 - cosine similarity
            distance_threshold = max(1.0 - threshold, 1e-6)

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage=linkage
            )

            labels = clustering.fit_predict(embeddings)

            # Group indices by cluster label
            clusters: Dict[int, List[int]] = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

            return list(clusters.values())
        except Exception as e:
            logger.error(f"Error clustering texts: {e}")
            return [[i] for i in range(len(texts))]

    def cluster_with_info(
        self,
        texts: List[str],
        threshold: float = 0.75,
        linkage: str = "complete"
    ) -> List[Dict[str, Any]]:
        """
        Cluster texts and return detailed cluster information.

        Args:
            texts: List of text strings to cluster.
            threshold: Similarity threshold for clustering (0.0 to 1.0).
            linkage: Linkage type ('complete', 'average', 'single').

        Returns:
            List of cluster info dicts with:
            - indices: List of indices in this cluster
            - items: List of text items in this cluster
            - canonical: Suggested canonical form (shortest item)
            - avg_similarity: Average pairwise similarity within cluster
        """
        if not texts:
            return []

        cluster_indices = self.cluster(texts, threshold=threshold, linkage=linkage)

        # Compute pairwise similarity for avg_similarity calculation
        if self.model and len(texts) > 1:
            sim_matrix = self.compute_pairwise_similarity(texts)
        else:
            sim_matrix = None

        results = []
        for indices in cluster_indices:
            items = [texts[i] for i in indices]

            # Compute average similarity within cluster
            avg_sim = 1.0
            if sim_matrix is not None and len(indices) > 1:
                cluster_sims = []
                for i, idx1 in enumerate(indices):
                    for idx2 in indices[i+1:]:
                        cluster_sims.append(sim_matrix[idx1, idx2])
                if cluster_sims:
                    avg_sim = float(np.mean(cluster_sims))

            # Select canonical form (shortest string heuristic)
            canonical = min(items, key=len)

            results.append({
                "indices": indices,
                "items": items,
                "canonical": canonical,
                "avg_similarity": avg_sim,
                "count": len(indices)
            })

        # Sort by cluster size (largest first)
        results.sort(key=lambda x: x["count"], reverse=True)
        return results

    def find_similar(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.7,
        top_k: Optional[int] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Find candidates similar to a query.

        Args:
            query: Query text string.
            candidates: List of candidate text strings.
            threshold: Minimum similarity threshold.
            top_k: Maximum number of results (None for all above threshold).

        Returns:
            List of (index, text, similarity) tuples, sorted by similarity descending.
        """
        if not candidates:
            return []

        similarities = self.compute_batch_similarity(query, candidates)

        # Filter and sort
        results = [
            (i, candidates[i], sim)
            for i, sim in enumerate(similarities)
            if sim >= threshold
        ]
        results.sort(key=lambda x: x[2], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results


# Module-level convenience functions for simple use cases
_default_service: Optional[EmbeddingService] = None


def get_embedding_service(
    model_name: str = EmbeddingService.DEFAULT_MODEL,
    **kwargs
) -> EmbeddingService:
    """
    Get or create a shared embedding service instance.

    For simple use cases, this provides a singleton-like pattern.
    For different models or configurations, create instances directly.

    Args:
        model_name: Model name (uses cached instance if same model).
        **kwargs: Additional arguments for EmbeddingService.

    Returns:
        EmbeddingService instance.
    """
    global _default_service

    if _default_service is None or _default_service.model_name != model_name:
        _default_service = EmbeddingService(model_name=model_name, **kwargs)

    return _default_service
