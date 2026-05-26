"""
Domain normalizer for clustering and canonicalizing domain labels.

Uses sentence embeddings and agglomerative clustering to group
semantically similar domain labels.
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from .models import (
    DomainMappedInstance,
    DomainCluster,
    NormalizationResult,
    NormalizeCheckpoint,
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
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval: int = 10,
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
            checkpoint_path: Optional path to save checkpoints for LLM method
            checkpoint_interval: How often to save checkpoints (every N clusters)
            
        Returns:
            NormalizationResult with mappings and clusters
        """
        if canonical_method == "llm":
            if not llm_model:
                raise ValueError("llm_model required when canonical_method='llm'")
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(
                    self.normalize_async(
                        instances,
                        conservativeness=conservativeness,
                        similarity_threshold=similarity_threshold,
                        cluster_mode=cluster_mode,
                        canonical_method=canonical_method,
                        abstraction_level=abstraction_level,
                        llm_model=llm_model,
                        llm_provider=llm_provider,
                        llm_config=llm_config,
                        checkpoint_path=checkpoint_path,
                        checkpoint_interval=checkpoint_interval,
                    )
                )
            raise RuntimeError(
                "DomainNormalizer.normalize(..., canonical_method='llm') cannot be called "
                "from an active event loop; use 'await DomainNormalizer.normalize_async(...)' instead."
            )

        threshold, source_counts, target_counts, embedding_model, source_clusters, target_clusters = (
            self._prepare_normalization(
                instances,
                conservativeness,
                similarity_threshold,
                cluster_mode,
                abstraction_level,
                canonical_method,
            )
        )
        source_clusters, target_clusters = self._generate_representative_canonical_labels(
            source_clusters,
            target_clusters,
            embedding_model,
        )
        return self._build_normalization_result(
            source_counts,
            target_counts,
            source_clusters,
            target_clusters,
            conservativeness,
            threshold,
            cluster_mode,
            canonical_method,
            abstraction_level,
        )

    async def normalize_async(
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
        checkpoint_path: Optional[Path] = None,
        checkpoint_interval: int = 10,
    ) -> NormalizationResult:
        """Async normalization entry point for event-loop-safe LLM canonicalization."""
        threshold, source_counts, target_counts, embedding_model, source_clusters, target_clusters = (
            self._prepare_normalization(
                instances,
                conservativeness,
                similarity_threshold,
                cluster_mode,
                abstraction_level,
                canonical_method,
            )
        )

        if canonical_method == "llm":
            if not llm_model:
                raise ValueError("llm_model required when canonical_method='llm'")
            source_clusters, target_clusters = await self._generate_canonical_labels_llm_with_checkpoint(
                source_clusters,
                target_clusters,
                llm_model,
                llm_provider or "ollama",
                llm_config or {},
                checkpoint_path,
                checkpoint_interval,
            )
        else:
            source_clusters, target_clusters = self._generate_representative_canonical_labels(
                source_clusters,
                target_clusters,
                embedding_model,
            )

        return self._build_normalization_result(
            source_counts,
            target_counts,
            source_clusters,
            target_clusters,
            conservativeness,
            threshold,
            cluster_mode,
            canonical_method,
            abstraction_level,
        )

    def _prepare_normalization(
        self,
        instances: List[DomainMappedInstance],
        conservativeness: str,
        similarity_threshold: Optional[float],
        cluster_mode: str,
        abstraction_level: Optional[str],
        canonical_method: str,
    ) -> tuple:
        """Prepare counts, embeddings, and clusters for normalization."""
        threshold = self._resolve_threshold(conservativeness, similarity_threshold)
        logger.info(
            f"Starting normalization: threshold={threshold}, mode={cluster_mode}, method={canonical_method}"
        )

        source_counts: Dict[str, int] = defaultdict(int)
        target_counts: Dict[str, int] = defaultdict(int)
        for inst in instances:
            source_domain = self._get_domain_at_level(inst, "source", abstraction_level)
            target_domain = self._get_domain_at_level(inst, "target", abstraction_level)
            if source_domain:
                source_counts[source_domain] += 1
            if target_domain:
                target_counts[target_domain] += 1

        source_domains = list(source_counts.keys())
        target_domains = list(target_counts.keys())
        logger.info(
            f"Found {len(source_domains)} unique source domains, {len(target_domains)} unique target domains"
        )

        embedding_model = self._get_embedding_model()
        if cluster_mode == "together":
            all_domains = list(set(source_domains + target_domains))
            clusters = self._cluster_domains(all_domains, threshold, embedding_model)
            source_clusters = clusters
            target_clusters = clusters
        else:
            source_clusters = self._cluster_domains(source_domains, threshold, embedding_model)
            target_clusters = self._cluster_domains(target_domains, threshold, embedding_model)

        return (
            threshold,
            source_counts,
            target_counts,
            embedding_model,
            source_clusters,
            target_clusters,
        )

    def _resolve_threshold(
        self,
        conservativeness: str,
        similarity_threshold: Optional[float],
    ) -> float:
        """Resolve a threshold value from preset or custom settings."""
        if conservativeness == "custom":
            if similarity_threshold is None:
                raise ValueError("similarity_threshold required when conservativeness='custom'")
            return similarity_threshold
        return THRESHOLD_MAP.get(conservativeness, 0.75)

    def _generate_representative_canonical_labels(
        self,
        source_clusters: List[DomainCluster],
        target_clusters: List[DomainCluster],
        embedding_model,
    ) -> tuple[List[DomainCluster], List[DomainCluster]]:
        """Generate representative canonical labels for source and target clusters."""
        source_clusters = self._generate_canonical_labels_representative(
            source_clusters, embedding_model
        )
        target_clusters = self._generate_canonical_labels_representative(
            target_clusters, embedding_model
        )
        return source_clusters, target_clusters

    def _build_normalization_result(
        self,
        source_counts: Dict[str, int],
        target_counts: Dict[str, int],
        source_clusters: List[DomainCluster],
        target_clusters: List[DomainCluster],
        conservativeness: str,
        threshold: float,
        cluster_mode: str,
        canonical_method: str,
        abstraction_level: Optional[str],
    ) -> NormalizationResult:
        """Finalize normalization result after canonical labels are assigned."""
        for cluster in source_clusters:
            cluster.count = sum(source_counts.get(member, 0) for member in cluster.members)
        for cluster in target_clusters:
            cluster.count = sum(target_counts.get(member, 0) for member in cluster.members)

        source_mapping = {}
        for cluster in source_clusters:
            for member in cluster.members:
                source_mapping[member] = cluster.canonical

        target_mapping = {}
        for cluster in target_clusters:
            for member in cluster.members:
                target_mapping[member] = cluster.canonical

        config = {
            "conservativeness": conservativeness,
            "similarity_threshold": threshold,
            "cluster_mode": cluster_mode,
            "canonical_method": canonical_method,
            "abstraction_level": abstraction_level,
            "embedding_model": self.embedding_model_name,
        }

        logger.info(
            f"Normalization complete: {len(source_clusters)} source clusters, {len(target_clusters)} target clusters"
        )
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

Respond with a JSON object in this exact format:
{{
  "reasoning": "Explanation of why this label was chosen...",
  "canonical_label": "LABEL"
}}

The canonical_label should be:
- 1-3 words
- More abstract/general than the specific members
- Written in UPPERCASE

Respond with ONLY the JSON object, nothing else."""
        
        for cluster in clusters:
            if len(cluster.members) == 1:
                cluster.canonical = cluster.members[0].upper()
                continue
            
            members_str = "\n".join(f"- {m}" for m in cluster.members)
            prompt = prompt_template.format(members=members_str)
            
            try:
                response = await llm.generate(prompt=prompt, temperature=0.3)
                
                # Parse JSON response
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()
                
                data = json.loads(cleaned)
                cluster.canonical = data.get("canonical_label", "").strip().upper()
                
                # Log reasoning if available
                if "reasoning" in data:
                    logger.debug(f"Canonical label reasoning for {cluster.canonical}: {data['reasoning']}")
                    
            except Exception as e:
                logger.warning(f"LLM canonical generation failed: {e}")
                # Fall back to representative
                cluster.canonical = cluster.members[0].upper()
        
        return clusters
    
    async def _generate_canonical_labels_llm_with_checkpoint(
        self,
        source_clusters: List[DomainCluster],
        target_clusters: List[DomainCluster],
        model: str,
        provider: str,
        config: Dict[str, Any],
        checkpoint_path: Optional[Path],
        checkpoint_interval: int,
    ) -> tuple:
        """
        Generate canonical labels using an LLM with checkpoint support.
        
        Processes source clusters first, then target clusters, saving checkpoints
        periodically to allow resumption if interrupted.
        """
        from ...core.providers import OllamaProvider
        
        llm = OllamaProvider(model, **config)
        
        # Load checkpoint if exists
        checkpoint = None
        processed_source = set()
        processed_target = set()
        
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = self._load_normalize_checkpoint(checkpoint_path)
            if checkpoint:
                processed_source = set(checkpoint.processed_source_indices)
                processed_target = set(checkpoint.processed_target_indices)
                
                # Restore canonical labels from checkpoint
                for i, cluster_data in enumerate(checkpoint.source_clusters):
                    if i < len(source_clusters):
                        source_clusters[i].canonical = cluster_data.get("canonical", "")
                for i, cluster_data in enumerate(checkpoint.target_clusters):
                    if i < len(target_clusters):
                        target_clusters[i].canonical = cluster_data.get("canonical", "")
                
                logger.info(
                    f"Resuming from checkpoint: {len(processed_source)}/{len(source_clusters)} source, "
                    f"{len(processed_target)}/{len(target_clusters)} target clusters processed"
                )
        
        # Process source clusters
        total_source = len(source_clusters)
        print(f"\nGenerating canonical labels for {total_source} source clusters...")
        for i, cluster in enumerate(source_clusters):
            if i in processed_source:
                continue
            
            await self._process_single_cluster_llm(cluster, llm)
            processed_source.add(i)
            
            # Print progress
            print(f"\rSource clusters: {len(processed_source)}/{total_source} ({100*len(processed_source)/total_source:.1f}%)", end="", flush=True)
            
            # Save checkpoint periodically
            if checkpoint_path and (len(processed_source) % checkpoint_interval == 0):
                self._save_normalize_checkpoint(
                    checkpoint_path,
                    "source",
                    source_clusters,
                    target_clusters,
                    list(processed_source),
                    list(processed_target),
                    total_source,
                    len(target_clusters),
                )
                logger.info(f"Checkpoint saved: {len(processed_source)}/{total_source} source clusters")
        
        print()  # Newline after source progress
        
        # Process target clusters
        total_target = len(target_clusters)
        print(f"Generating canonical labels for {total_target} target clusters...")
        for i, cluster in enumerate(target_clusters):
            if i in processed_target:
                continue
            
            await self._process_single_cluster_llm(cluster, llm)
            processed_target.add(i)
            
            # Print progress
            print(f"\rTarget clusters: {len(processed_target)}/{total_target} ({100*len(processed_target)/total_target:.1f}%)", end="", flush=True)
            
            # Save checkpoint periodically
            if checkpoint_path and (len(processed_target) % checkpoint_interval == 0):
                self._save_normalize_checkpoint(
                    checkpoint_path,
                    "target",
                    source_clusters,
                    target_clusters,
                    list(processed_source),
                    list(processed_target),
                    total_source,
                    total_target,
                )
                logger.info(f"Checkpoint saved: {len(processed_target)}/{total_target} target clusters")
        
        print()  # Newline after target progress
        
        # Delete checkpoint on successful completion
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Removed checkpoint file: {checkpoint_path}")
        
        return source_clusters, target_clusters
    
    async def _process_single_cluster_llm(self, cluster: DomainCluster, llm) -> None:
        """Process a single cluster to generate its canonical label via LLM."""
        if len(cluster.members) == 1:
            cluster.canonical = cluster.members[0].upper()
            return
        
        prompt_template = """Given these semantically similar domain labels:
{members}

Generate a single canonical label that best represents this group of concepts.

Respond with a JSON object in this exact format:
{{
  "reasoning": "Explanation of why this label was chosen...",
  "canonical_label": "LABEL"
}}

The canonical_label should be:
- 1-3 words
- More abstract/general than the specific members
- Written in UPPERCASE

Respond with ONLY the JSON object, nothing else."""
        
        members_str = "\n".join(f"- {m}" for m in cluster.members)
        prompt = prompt_template.format(members=members_str)
        
        try:
            response = await llm.generate(prompt=prompt, temperature=0.3)
            
            # Parse JSON response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            cluster.canonical = data.get("canonical_label", "").strip().upper()
            
            if "reasoning" in data:
                logger.debug(f"Canonical label reasoning for {cluster.canonical}: {data['reasoning']}")
                
        except Exception as e:
            logger.warning(f"LLM canonical generation failed: {e}")
            cluster.canonical = cluster.members[0].upper()
    
    def _save_normalize_checkpoint(
        self,
        path: Path,
        phase: str,
        source_clusters: List[DomainCluster],
        target_clusters: List[DomainCluster],
        processed_source_indices: List[int],
        processed_target_indices: List[int],
        total_source: int,
        total_target: int,
    ) -> None:
        """Save normalization checkpoint to file."""
        checkpoint = NormalizeCheckpoint(
            phase=phase,
            processed_source_indices=processed_source_indices,
            processed_target_indices=processed_target_indices,
            source_clusters=[
                {"canonical": c.canonical, "members": c.members, "avg_similarity": c.avg_similarity}
                for c in source_clusters
            ],
            target_clusters=[
                {"canonical": c.canonical, "members": c.members, "avg_similarity": c.avg_similarity}
                for c in target_clusters
            ],
            config={"embedding_model": self.embedding_model_name},
            timestamp=datetime.now().isoformat(),
            total_source_clusters=total_source,
            total_target_clusters=total_target,
        )
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
    
    def _load_normalize_checkpoint(self, path: Path) -> Optional[NormalizeCheckpoint]:
        """Load normalization checkpoint from file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return NormalizeCheckpoint.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load normalize checkpoint: {e}")
            return None
    
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
