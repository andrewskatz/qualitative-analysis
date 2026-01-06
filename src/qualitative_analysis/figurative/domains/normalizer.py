"""
Domain normalizer for clustering and canonicalizing domain labels.

Uses sentence embeddings and agglomerative clustering to group
semantically similar domain labels.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from .models import (
    DomainMappedInstance,
    DomainCluster,
    NormalizationResult,
)

logger = logging.getLogger(__name__)

# Conservativeness presets
THRESHOLD_MAP = {
    "conservative": 0.85,
    "moderate": 0.75,
    "aggressive": 0.60,
}


def _detect_device() -> str:
    """Auto-detect the best available device for embeddings."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class DomainNormalizer:
    """
    Normalizes domain labels by clustering semantically similar ones.
    
    Uses SentenceTransformer embeddings and AgglomerativeClustering.
    """
    
    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
    ):
        """
        Initialize the normalizer.
        
        Args:
            embedding_model: SentenceTransformer model name/path
            device: Device to use ("cpu", "cuda", "mps", or None for auto)
        """
        self.embedding_model_name = embedding_model
        self.device = device or _detect_device()
        self._embedding_model = None  # Lazy load
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name} on {self.device}")
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device,
            )
        return self._embedding_model
    
    def normalize(
        self,
        instances: List[DomainMappedInstance],
        conservativeness: str = "moderate",
        similarity_threshold: Optional[float] = None,
        cluster_mode: str = "separate",
        canonical_method: str = "representative",
        abstraction_level: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> NormalizationResult:
        """
        Normalize domains by clustering semantically similar labels.
        
        Args:
            instances: List of DomainMappedInstance objects
            conservativeness: "conservative", "moderate", "aggressive", or "custom"
            similarity_threshold: Custom threshold (only used if conservativeness="custom")
            cluster_mode: "separate" (cluster source/target independently) or 
                         "together" (cluster all domains together)
            canonical_method: "representative" (pick most central member) or 
                             "llm" (use LLM to generate abstract label)
            abstraction_level: For multi-level instances, which level to normalize:
                              "specific", "moderate", or "abstract"
            llm_model: LLM model for canonical_method="llm"
            llm_provider: LLM provider for canonical_method="llm"
            llm_config: Additional LLM config for canonical_method="llm"
            
        Returns:
            NormalizationResult with mappings and clusters
        """
        # Determine threshold
        if conservativeness == "custom":
            if similarity_threshold is None:
                raise ValueError("similarity_threshold required when conservativeness='custom'")
            threshold = similarity_threshold
        else:
            threshold = THRESHOLD_MAP.get(conservativeness, 0.75)
        
        logger.info(f"Starting normalization: threshold={threshold}, mode={cluster_mode}, method={canonical_method}")
        
        # Extract unique domains with counts
        source_counts: Dict[str, int] = defaultdict(int)
        target_counts: Dict[str, int] = defaultdict(int)
        
        for inst in instances:
            # Get domain at selected abstraction level
            source_domain = self._get_domain_at_level(
                inst, "source", abstraction_level
            )
            target_domain = self._get_domain_at_level(
                inst, "target", abstraction_level
            )
            
            if source_domain:
                source_counts[source_domain] += 1
            if target_domain:
                target_counts[target_domain] += 1
        
        source_domains = list(source_counts.keys())
        target_domains = list(target_counts.keys())
        
        logger.info(f"Found {len(source_domains)} unique source domains, {len(target_domains)} unique target domains")
        
        # Get embedding model
        embedding_model = self._get_embedding_model()
        
        # Cluster domains
        if cluster_mode == "together":
            all_domains = list(set(source_domains + target_domains))
            clusters = self._cluster_domains(all_domains, threshold, embedding_model)
            source_clusters = clusters
            target_clusters = clusters
        else:
            source_clusters = self._cluster_domains(source_domains, threshold, embedding_model)
            target_clusters = self._cluster_domains(target_domains, threshold, embedding_model)
        
        # Generate canonical labels
        if canonical_method == "llm":
            if not llm_model:
                raise ValueError("llm_model required when canonical_method='llm'")
            import asyncio
            source_clusters = asyncio.get_event_loop().run_until_complete(
                self._generate_canonical_labels_llm(
                    source_clusters, llm_model, llm_provider or "ollama", llm_config or {}
                )
            )
            target_clusters = asyncio.get_event_loop().run_until_complete(
                self._generate_canonical_labels_llm(
                    target_clusters, llm_model, llm_provider or "ollama", llm_config or {}
                )
            )
        else:
            source_clusters = self._generate_canonical_labels_representative(
                source_clusters, embedding_model
            )
            target_clusters = self._generate_canonical_labels_representative(
                target_clusters, embedding_model
            )
        
        # Add counts to clusters
        for cluster in source_clusters:
            cluster.count = sum(source_counts.get(m, 0) for m in cluster.members)
        for cluster in target_clusters:
            cluster.count = sum(target_counts.get(m, 0) for m in cluster.members)
        
        # Build mappings
        source_mapping = {}
        for cluster in source_clusters:
            for member in cluster.members:
                source_mapping[member] = cluster.canonical
        
        target_mapping = {}
        for cluster in target_clusters:
            for member in cluster.members:
                target_mapping[member] = cluster.canonical
        
        # Build config
        config = {
            "conservativeness": conservativeness,
            "similarity_threshold": threshold,
            "cluster_mode": cluster_mode,
            "canonical_method": canonical_method,
            "abstraction_level": abstraction_level,
            "embedding_model": self.embedding_model_name,
        }
        
        logger.info(f"Normalization complete: {len(source_clusters)} source clusters, {len(target_clusters)} target clusters")
        
        return NormalizationResult(
            source_mapping=source_mapping,
            target_mapping=target_mapping,
            source_clusters=source_clusters,
            target_clusters=target_clusters,
            config=config,
        )
    
    def _get_domain_at_level(
        self,
        instance: DomainMappedInstance,
        domain_type: str,
        abstraction_level: Optional[str],
    ) -> str:
        """Get domain at specified abstraction level."""
        if domain_type == "source":
            primary = instance.source_domain
            levels = instance.source_domain_levels
        else:
            primary = instance.target_domain
            levels = instance.target_domain_levels
        
        if abstraction_level and levels:
            return levels.get(abstraction_level) or primary
        return primary
    
    def _cluster_domains(
        self,
        domains: List[str],
        threshold: float,
        embedding_model,
    ) -> List[DomainCluster]:
        """Cluster domains using agglomerative clustering."""
        from sklearn.cluster import AgglomerativeClustering
        
        if len(domains) == 0:
            return []
        
        if len(domains) == 1:
            return [DomainCluster(
                canonical="",  # Will be set later
                members=domains,
                avg_similarity=1.0,
            )]
        
        # Generate embeddings
        embeddings = embedding_model.encode(domains)
        
        # Agglomerative clustering with cosine distance
        distance_threshold = max(1 - threshold, 1e-6)
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="complete",
        )
        
        labels = clustering.fit_predict(embeddings)
        
        # Group by cluster
        clusters_dict: Dict[int, List[str]] = defaultdict(list)
        for domain, label in zip(domains, labels):
            clusters_dict[label].append(domain)
        
        # Calculate average similarity within each cluster
        result = []
        for label, members in clusters_dict.items():
            if len(members) == 1:
                avg_sim = 1.0
            else:
                # Get indices for this cluster
                indices = [i for i, d in enumerate(domains) if d in members]
                cluster_embeddings = embeddings[indices]
                
                # Compute pairwise similarities
                sims = []
                for i in range(len(cluster_embeddings)):
                    for j in range(i + 1, len(cluster_embeddings)):
                        vec1 = cluster_embeddings[i]
                        vec2 = cluster_embeddings[j]
                        sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                        sims.append(sim)
                avg_sim = sum(sims) / len(sims) if sims else 1.0
            
            result.append(DomainCluster(
                canonical="",  # Will be set later
                members=sorted(members),
                avg_similarity=round(avg_sim, 3),
            ))
        
        return result
    
    def _generate_canonical_labels_representative(
        self,
        clusters: List[DomainCluster],
        embedding_model,
    ) -> List[DomainCluster]:
        """Generate canonical labels by selecting the most representative member."""
        for cluster in clusters:
            members = cluster.members
            
            if len(members) == 1:
                cluster.canonical = members[0].upper()
                continue
            
            # Get embeddings for cluster members
            embeddings = embedding_model.encode(members)
            
            # Find member with highest average similarity to others
            best_idx = 0
            best_avg_sim = -1
            
            for i in range(len(members)):
                sims = []
                for j in range(len(members)):
                    if i != j:
                        vec1 = embeddings[i]
                        vec2 = embeddings[j]
                        sim = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                        sims.append(sim)
                avg_sim = sum(sims) / len(sims) if sims else 0
                
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_idx = i
            
            cluster.canonical = members[best_idx].upper()
        
        return clusters
    
    async def _generate_canonical_labels_llm(
        self,
        clusters: List[DomainCluster],
        model: str,
        provider: str,
        config: Dict[str, Any],
    ) -> List[DomainCluster]:
        """Generate canonical labels using an LLM."""
        from ...core.providers import OllamaProvider
        
        llm = OllamaProvider(model, **config)
        
        prompt_template = """Given these semantically similar domain labels:
{members}

Generate a single canonical label that best represents this group of concepts.
The label should be:
- 1-3 words
- More abstract/general than the specific members
- Written in UPPERCASE

Respond with only the canonical label, nothing else."""
        
        for cluster in clusters:
            if len(cluster.members) == 1:
                cluster.canonical = cluster.members[0].upper()
                continue
            
            members_str = "\n".join(f"- {m}" for m in cluster.members)
            prompt = prompt_template.format(members=members_str)
            
            try:
                response = await llm.generate(prompt=prompt, temperature=0.3)
                cluster.canonical = response.strip().upper()
            except Exception as e:
                logger.warning(f"LLM canonical generation failed: {e}")
                # Fall back to representative
                cluster.canonical = cluster.members[0].upper()
        
        return clusters
    
    def save(self, result: NormalizationResult, path: Path) -> None:
        """Save normalization result to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved normalization to {path}")
    
    def load(self, path: Path) -> NormalizationResult:
        """Load normalization result from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return NormalizationResult.from_dict(data)
