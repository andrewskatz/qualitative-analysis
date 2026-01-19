"""
Domain graph generator for visualizing source→target relationships.

Generates nodes and edges from domain-mapped figurative instances.
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

from .models import (
    DomainMappedInstance,
    DomainGraphNode,
    DomainGraphEdge,
    DomainGraphData,
    NormalizationResult,
)

logger = logging.getLogger(__name__)


class DomainGraph:
    """
    Generates domain relationship graphs from figurative instances.
    
    Supports both raw and normalized domain graphs.
    """
    
    def __init__(
        self,
        instances: List[DomainMappedInstance],
        normalization: Optional[NormalizationResult] = None,
    ):
        """
        Initialize the graph generator.
        
        Args:
            instances: List of domain-mapped instances
            normalization: Optional normalization result for normalized graphs
        """
        self.instances = instances
        self.normalization = normalization
        self._graph_data: Optional[DomainGraphData] = None
    
    def generate(self, abstraction_level: Optional[str] = None) -> DomainGraphData:
        """
        Generate the domain graph.
        
        Args:
            abstraction_level: For multi-level instances, which level to use
            
        Returns:
            DomainGraphData with nodes and edges
        """
        is_normalized = self.normalization is not None
        
        # Track node data
        nodes_data: Dict[str, Dict[str, Any]] = {}  # id -> {label, as_source, as_target, raw_domains}
        
        # Track edge data
        edges_data: Dict[str, Dict[str, Any]] = {}  # "source->target" -> {weight, types, examples, texts}
        
        for inst in self.instances:
            # Get domains (normalized if available)
            source = self._get_domain(inst, "source", abstraction_level)
            target = self._get_domain(inst, "target", abstraction_level)
            
            if not source or not target:
                continue
            
            # Normalize for node IDs
            source_id = source.lower().strip()
            target_id = target.lower().strip()
            
            if source_id == target_id:
                continue  # Skip self-loops
            
            # Get original raw domains for tracking
            raw_source = self._get_raw_domain(inst, "source", abstraction_level)
            raw_target = self._get_raw_domain(inst, "target", abstraction_level)
            
            # Update source node
            if source_id not in nodes_data:
                nodes_data[source_id] = {
                    "label": source.title(),
                    "as_source": 0,
                    "as_target": 0,
                    "raw_domains": set(),
                }
            nodes_data[source_id]["as_source"] += 1
            if raw_source:
                nodes_data[source_id]["raw_domains"].add(raw_source)
            
            # Update target node
            if target_id not in nodes_data:
                nodes_data[target_id] = {
                    "label": target.title(),
                    "as_source": 0,
                    "as_target": 0,
                    "raw_domains": set(),
                }
            nodes_data[target_id]["as_target"] += 1
            if raw_target:
                nodes_data[target_id]["raw_domains"].add(raw_target)
            
            # Update edge
            edge_key = f"{source_id}->{target_id}"
            if edge_key not in edges_data:
                edges_data[edge_key] = {
                    "source": source_id,
                    "target": target_id,
                    "weight": 0,
                    "types": set(),
                    "examples": [],
                    "text_ids": set(),
                }
            edges_data[edge_key]["weight"] += 1
            edges_data[edge_key]["types"].add(inst.type)
            if len(edges_data[edge_key]["examples"]) < 5:  # Keep up to 5 examples
                edges_data[edge_key]["examples"].append(inst.text)
            if inst.text_id:
                edges_data[edge_key]["text_ids"].add(inst.text_id)
        
        # Build nodes
        nodes = []
        for node_id, data in nodes_data.items():
            domain_type = "both"
            if data["as_source"] > 0 and data["as_target"] == 0:
                domain_type = "source"
            elif data["as_target"] > 0 and data["as_source"] == 0:
                domain_type = "target"
            
            nodes.append(DomainGraphNode(
                id=node_id,
                label=data["label"],
                domain_type=domain_type,
                frequency=data["as_source"] + data["as_target"],
                as_source=data["as_source"],
                as_target=data["as_target"],
                raw_domains=sorted(data["raw_domains"]) if is_normalized else [],
            ))
        
        # Build edges
        edges = []
        for i, (edge_key, data) in enumerate(edges_data.items()):
            edges.append(DomainGraphEdge(
                id=f"e{i}",
                source=data["source"],
                target=data["target"],
                weight=data["weight"],
                figurative_types=sorted(data["types"]),
                examples=data["examples"],
                text_count=len(data["text_ids"]),
            ))
        
        # Sort by weight descending
        edges.sort(key=lambda e: e.weight, reverse=True)
        nodes.sort(key=lambda n: n.frequency, reverse=True)
        
        # Compute stats
        stats = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "instance_count": len(self.instances),
            "is_normalized": is_normalized,
            "source_only_nodes": sum(1 for n in nodes if n.domain_type == "source"),
            "target_only_nodes": sum(1 for n in nodes if n.domain_type == "target"),
            "both_nodes": sum(1 for n in nodes if n.domain_type == "both"),
            "max_edge_weight": max(e.weight for e in edges) if edges else 0,
        }
        
        self._graph_data = DomainGraphData(
            nodes=nodes,
            edges=edges,
            stats=stats,
            is_normalized=is_normalized,
        )
        
        logger.info(f"Generated graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return self._graph_data
    
    def _get_domain(
        self,
        instance: DomainMappedInstance,
        domain_type: str,
        abstraction_level: Optional[str],
    ) -> str:
        """Get domain, applying normalization if available."""
        # First get the raw domain
        raw = self._get_raw_domain(instance, domain_type, abstraction_level)
        
        if not raw:
            return ""
        
        # Apply normalization if available
        if self.normalization:
            if domain_type == "source":
                return self.normalization.source_mapping.get(raw, raw)
            else:
                return self.normalization.target_mapping.get(raw, raw)
        
        return raw
    
    def _get_raw_domain(
        self,
        instance: DomainMappedInstance,
        domain_type: str,
        abstraction_level: Optional[str],
    ) -> str:
        """Get raw domain at specified abstraction level."""
        if domain_type == "source":
            primary = instance.source_domain
            levels = instance.source_domain_levels
        else:
            primary = instance.target_domain
            levels = instance.target_domain_levels
        
        if abstraction_level and levels:
            return levels.get(abstraction_level) or primary
        return primary
    
    def _compute_semantic_positions(
        self,
        domain_labels: List[str],
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
    ) -> Dict[str, tuple]:
        """
        Compute 2D positions for domains using UMAP projection of embeddings.
        
        Args:
            domain_labels: List of domain label strings
            embedding_model: SentenceTransformer model name
            device: Device for embeddings ("cpu", "cuda", "mps", or None for auto)
            
        Returns:
            Dict mapping domain label (lowercase) -> (x, y) tuple
        """
        try:
            import umap
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.warning(f"Missing dependency for semantic layout: {e}")
            return {}
        
        if len(domain_labels) < 2:
            # Can't project with <2 points
            return {label.lower(): (0.5, 0.5) for label in domain_labels}
        
        # Auto-detect device
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        
        # Generate embeddings
        logger.info(f"[SemanticLayout] Generating embeddings for {len(domain_labels)} domains")
        model = SentenceTransformer(embedding_model, device=device)
        embeddings = model.encode(domain_labels)
        
        # PCA preprocessing: reduce to intermediate space retaining ~90% variance
        # This improves UMAP stability and reduces curse of dimensionality effects
        from sklearn.decomposition import PCA
        
        embedding_dim = embeddings.shape[1]
        n_samples = embeddings.shape[0]
        
        # Only apply PCA if we have more dimensions than samples and high-dimensional data
        if embedding_dim > 50 and n_samples >= 3:
            # Use 90% variance retention or cap at 50 components
            max_components = min(n_samples - 1, 50)
            pca = PCA(n_components=min(max_components, embedding_dim))
            embeddings_reduced = pca.fit_transform(embeddings)
            
            # Calculate explained variance
            explained_var = sum(pca.explained_variance_ratio_) * 100
            logger.info(f"[SemanticLayout] PCA reduced {embedding_dim}D → {embeddings_reduced.shape[1]}D (retaining {explained_var:.1f}% variance)")
            embeddings = embeddings_reduced
        
        # Project to 2D with UMAP
        n_neighbors = min(15, len(domain_labels) - 1)
        logger.info(f"[SemanticLayout] Running UMAP projection with n_neighbors={n_neighbors}")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42  # Reproducible
        )
        coords_2d = reducer.fit_transform(embeddings)
        
        # Normalize to [0, 1] range
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()
        
        positions = {}
        for i, label in enumerate(domain_labels):
            x_norm = (coords_2d[i, 0] - x_min) / (x_max - x_min + 1e-6)
            y_norm = (coords_2d[i, 1] - y_min) / (y_max - y_min + 1e-6)
            positions[label.lower()] = (float(x_norm), float(y_norm))
        
        logger.info(f"[SemanticLayout] Computed positions for {len(positions)} domains")
        return positions

    def _cluster_nodes_2d(
        self,
        positions: Dict[str, tuple],
        min_cluster_size: int = 3,
        method: str = "hdbscan",
    ) -> Dict[str, int]:
        """
        Cluster nodes based on their 2D positions.
        
        Args:
            positions: Dict mapping node_id -> (x, y)
            min_cluster_size: Minimum nodes per cluster
            method: Clustering method (hdbscan or agglomerative)
            
        Returns:
            Dict mapping node_id -> cluster_id (-1 for noise/outliers)
        """
        import numpy as np
        
        if len(positions) < min_cluster_size:
            return {node_id: 0 for node_id in positions}
        
        # Convert to array
        node_ids = list(positions.keys())
        coords = np.array([positions[nid] for nid in node_ids])
        
        if method == "hdbscan":
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=2,
                    metric='euclidean',
                )
                labels = clusterer.fit_predict(coords)
                logger.info(f"[Clustering] HDBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
            except ImportError:
                logger.warning("hdbscan not installed, falling back to agglomerative")
                method = "agglomerative"
        
        if method == "agglomerative":
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = max(2, len(positions) // min_cluster_size)
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(coords)
            logger.info(f"[Clustering] Agglomerative created {n_clusters} clusters")
        
        return {node_id: int(labels[i]) for i, node_id in enumerate(node_ids)}

    async def _generate_cluster_labels_llm(
        self,
        clusters: Dict[int, List[str]],
        model: str,
        provider_config: Dict[str, Any],
    ) -> Dict[int, str]:
        """
        Generate descriptive labels for clusters using LLM.
        
        Args:
            clusters: Dict mapping cluster_id -> list of domain labels
            model: LLM model name
            provider_config: LLM provider configuration
            
        Returns:
            Dict mapping cluster_id -> generated label
        """
        from ...core.providers import OllamaProvider
        
        llm = OllamaProvider(model, **provider_config)
        labels = {}
        
        prompt_template = """Given these domain labels from a semantic cluster:
{members}

Generate a short 2-4 word descriptive label that captures the common theme.

Respond with JSON:
{{
  "reasoning": "Brief explanation",
  "label": "CLUSTER LABEL"
}}

Respond with ONLY the JSON, nothing else."""

        for cluster_id, members in clusters.items():
            if cluster_id == -1:
                labels[cluster_id] = "Other"
                continue
            
            if len(members) == 1:
                labels[cluster_id] = members[0].title()
                continue
            
            members_str = "\n".join(f"- {m}" for m in members[:10])  # Limit to 10
            prompt = prompt_template.format(members=members_str)
            
            try:
                import json as json_module
                response = await llm.generate(prompt=prompt, temperature=0.3)
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                data = json_module.loads(cleaned.strip())
                labels[cluster_id] = data.get("label", f"Cluster {cluster_id}")
                logger.debug(f"Cluster {cluster_id} label: {labels[cluster_id]} (reason: {data.get('reasoning', 'N/A')})")
            except Exception as e:
                logger.warning(f"LLM cluster labeling failed for cluster {cluster_id}: {e}")
                labels[cluster_id] = f"Cluster {cluster_id}"
        
        logger.info(f"[Clustering] Generated {len(labels)} cluster labels")
        return labels

    def _get_cluster_hull_points(
        self,
        positions: Dict[str, tuple],
        cluster_assignments: Dict[str, int],
        cluster_id: int,
    ) -> List[tuple]:
        """Get points for drawing cluster boundary."""
        points = [
            positions[nid] for nid, cid in cluster_assignments.items()
            if cid == cluster_id and nid in positions
        ]
        return points


    def to_json(self) -> Dict[str, Any]:
        """Export graph as JSON-serializable dict."""
        if not self._graph_data:
            self.generate()
        return self._graph_data.to_dict()
    
    def to_csv_nodes(self, path: Path) -> None:
        """Export nodes to CSV."""
        if not self._graph_data:
            self.generate()
        
        fieldnames = [
            "id", "label", "domain_type", "frequency", 
            "as_source", "as_target", "raw_domains"
        ]
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for node in self._graph_data.nodes:
                writer.writerow({
                    "id": node.id,
                    "label": node.label,
                    "domain_type": node.domain_type,
                    "frequency": node.frequency,
                    "as_source": node.as_source,
                    "as_target": node.as_target,
                    "raw_domains": "|".join(node.raw_domains),
                })
        
        logger.info(f"Saved nodes CSV to {path}")
    
    def to_csv_edges(self, path: Path) -> None:
        """Export edges to CSV."""
        if not self._graph_data:
            self.generate()
        
        fieldnames = [
            "id", "source", "target", "weight",
            "figurative_types", "examples", "text_count"
        ]
        
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for edge in self._graph_data.edges:
                writer.writerow({
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "figurative_types": "|".join(edge.figurative_types),
                    "examples": "|||".join(edge.examples),
                    "text_count": edge.text_count,
                })
        
        logger.info(f"Saved edges CSV to {path}")
    
    def to_png(
        self,
        path: Path,
        figsize: tuple = (16, 12),
        dpi: int = 150,
        layout: str = "spring",
    ) -> None:
        """
        Export graph as PNG visualization.
        
        Args:
            path: Output path for PNG file
            figsize: Figure size (width, height) in inches
            dpi: Resolution
            layout: Layout algorithm ("spring", "circular", "kamada_kawai", "semantic")
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("networkx or matplotlib not installed. Skipping PNG generation.")
            return
        
        if not self._graph_data:
            self.generate()
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in self._graph_data.nodes:
            G.add_node(
                node.id,
                label=node.label,
                domain_type=node.domain_type,
                frequency=node.frequency,
            )
        
        # Add edges with weights
        for edge in self._graph_data.edges:
            G.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
            )
        
        if len(G.nodes()) == 0:
            logger.warning("No nodes in graph. Skipping PNG generation.")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Select layout
        if layout == "semantic":
            # Use embedding-based semantic positioning
            domain_labels = [G.nodes[node].get("label", node) for node in G.nodes()]
            semantic_positions = self._compute_semantic_positions(domain_labels)
            if semantic_positions:
                # Map positions back to node IDs
                pos = {}
                for node in G.nodes():
                    label = G.nodes[node].get("label", node).lower()
                    if label in semantic_positions:
                        pos[node] = semantic_positions[label]
                    else:
                        pos[node] = (0.5, 0.5)
                logger.info(f"Using semantic layout with {len(pos)} positioned nodes")
            else:
                logger.warning("Semantic layout failed, falling back to spring layout")
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Color nodes by type
        node_colors = []
        for node in G.nodes():
            domain_type = G.nodes[node].get("domain_type", "both")
            if domain_type == "source":
                node_colors.append("#4CAF50")  # Green for source-only
            elif domain_type == "target":
                node_colors.append("#2196F3")  # Blue for target-only
            else:
                node_colors.append("#9C27B0")  # Purple for both
        
        # Size nodes by frequency
        node_sizes = [
            max(300, min(2000, G.nodes[node].get("frequency", 1) * 200))
            for node in G.nodes()
        ]
        
        # Draw edges with width based on weight
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [1 + (w / max_weight) * 4 for w in edge_weights]
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color="#888888",
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
        )
        
        # Draw labels
        labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_weight="bold",
            ax=ax,
        )
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#4CAF50", label="Source Domain"),
            Patch(facecolor="#2196F3", label="Target Domain"),
            Patch(facecolor="#9C27B0", label="Both"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)
        
        # Title and styling
        stats = self._graph_data.stats
        is_norm = "Normalized" if stats.get("is_normalized") else "Raw"
        ax.set_title(
            f"Domain Relationship Graph ({is_norm})\n"
            f"{stats['node_count']} nodes, {stats['edge_count']} edges",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")
        
        # Save
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        
        logger.info(f"Saved graph PNG to {path}")
    
    def to_png_clustered(
        self,
        path: Path,
        figsize: tuple = (20, 16),
        dpi: int = 150,
        min_cluster_size: int = 3,
        noise_handling: str = "label",
        model: Optional[str] = None,
        provider_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Export graph as PNG with cluster-based region labeling.
        
        Args:
            path: Output path for PNG file
            figsize: Figure size (width, height) in inches
            dpi: Resolution
            min_cluster_size: Minimum nodes per cluster
            noise_handling: How to handle outliers: "label", "hide", or "other"
            model: LLM model for cluster labeling (if None, uses generic labels)
            provider_config: LLM provider configuration
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib.patches import Polygon
            from scipy.spatial import ConvexHull
        except ImportError as e:
            logger.warning(f"Missing dependency for clustered PNG: {e}")
            return
        
        if not self._graph_data:
            self.generate()
        
        # Create networkx graph
        G = nx.DiGraph()
        for node in self._graph_data.nodes:
            G.add_node(
                node.id,
                label=node.label,
                domain_type=node.domain_type,
                frequency=node.frequency,
            )
        for edge in self._graph_data.edges:
            G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        if len(G.nodes()) == 0:
            logger.warning("No nodes in graph. Skipping PNG generation.")
            return
        
        # Get semantic positions
        domain_labels = [G.nodes[node].get("label", node) for node in G.nodes()]
        semantic_positions = self._compute_semantic_positions(domain_labels)
        
        if not semantic_positions:
            logger.warning("Could not compute semantic positions")
            return
        
        # Map positions to node IDs
        pos = {}
        for node in G.nodes():
            label = G.nodes[node].get("label", node).lower()
            if label in semantic_positions:
                pos[node] = semantic_positions[label]
            else:
                pos[node] = (0.5, 0.5)
        
        # Cluster nodes in 2D space
        cluster_assignments = self._cluster_nodes_2d(pos, min_cluster_size)
        
        # Group nodes by cluster
        clusters: Dict[int, List[str]] = {}
        for node_id, cluster_id in cluster_assignments.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            label = G.nodes[node_id].get("label", node_id) if node_id in G.nodes() else node_id
            clusters[cluster_id].append(label)
        
        # Generate cluster labels
        if model and provider_config:
            import asyncio
            cluster_labels = asyncio.get_event_loop().run_until_complete(
                self._generate_cluster_labels_llm(clusters, model, provider_config)
            )
        else:
            cluster_labels = {cid: f"Cluster {cid}" if cid >= 0 else "Other" for cid in clusters}
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Color palette for clusters - use distinct colors per cluster
        import matplotlib.cm as cm
        n_clusters = len([c for c in clusters if c >= 0])
        cluster_colors = cm.tab20(np.linspace(0, 1, max(n_clusters, 20)))
        
        # Build node -> cluster color mapping
        node_cluster_colors = {}
        for node_id, cluster_id in cluster_assignments.items():
            if cluster_id >= 0:
                node_cluster_colors[node_id] = cluster_colors[cluster_id % len(cluster_colors)]
            else:
                node_cluster_colors[node_id] = (0.7, 0.7, 0.7, 0.8)  # Gray for noise
        
        # Draw cluster regions (convex hulls) - skip outliers (cluster_id = -1)
        for cluster_id, members in clusters.items():
            if cluster_id == -1:
                continue  # Never draw hull for outliers - they span the whole graph
            
            points = self._get_cluster_hull_points(pos, cluster_assignments, cluster_id)
            if len(points) >= 3:
                try:
                    coords = np.array(points)
                    hull = ConvexHull(coords)
                    hull_points = coords[hull.vertices]
                    
                    # Draw filled polygon
                    color = cluster_colors[cluster_id % len(cluster_colors)] if cluster_id >= 0 else (0.8, 0.8, 0.8, 0.3)
                    polygon = Polygon(
                        hull_points,
                        alpha=0.15,
                        facecolor=color,
                        edgecolor=color[:3] if len(color) >= 3 else color,
                        linewidth=1.5,
                    )
                    ax.add_patch(polygon)
                except Exception as e:
                    logger.debug(f"Could not draw hull for cluster {cluster_id}: {e}")
        
        # Draw edges
        edge_weights = [G[u][v].get("weight", 1) for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [0.3 + (w / max_weight) * 1.5 for w in edge_weights]
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color="#CCCCCC",
            width=edge_widths,
            alpha=0.2,
            arrows=True,
            arrowsize=8,
            ax=ax,
        )
        
        # Draw nodes: COLOR = cluster, SHAPE = source/target/both
        for node in G.nodes():
            domain_type = G.nodes[node].get("domain_type", "both")
            x, y = pos[node]
            frequency = G.nodes[node].get("frequency", 1)
            size = max(30, min(200, frequency * 20))
            
            # Shape based on domain type
            if domain_type == "source":
                marker = "s"  # Square for source
            elif domain_type == "target":
                marker = "^"  # Triangle for target
            else:
                marker = "o"  # Circle for both
            
            # Color based on cluster
            color = node_cluster_colors.get(node, (0.5, 0.5, 0.5, 1))
            
            ax.scatter([x], [y], s=size, c=[color], marker=marker, alpha=0.85, zorder=5, edgecolors='white', linewidths=0.5)
        
        # Collect cluster label annotations for adjustText
        texts = []
        for cluster_id, members in clusters.items():
            if cluster_id == -1 and noise_handling != "label":
                continue
            
            points = self._get_cluster_hull_points(pos, cluster_assignments, cluster_id)
            if len(points) >= 1:
                coords = np.array(points)
                centroid = coords.mean(axis=0)
                label_text = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
                
                text = ax.annotate(
                    label_text,
                    centroid,
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor='gray', linewidth=0.5),
                    zorder=10,
                )
                texts.append(text)
        
        # Apply label repulsion using adjustText if available
        try:
            from adjustText import adjust_text
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))
            logger.info(f"[Clustering] Applied label repulsion to {len(texts)} cluster labels")
        except ImportError:
            logger.warning("adjustText not installed. Labels may overlap. Install with: pip install adjustText")
        
        # Add legend for shapes only (colors are per-cluster, too many to list)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="gray", markersize=10, label="Source Domain"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=10, label="Target Domain"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10, label="Both"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9, title="Domain Type (shape)")
        
        # Title
        stats = self._graph_data.stats
        ax.set_title(
            f"Domain Graph with Semantic Clusters\n"
            f"{stats['node_count']} nodes, {len([c for c in clusters if c >= 0])} clusters",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")
        
        # Save
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        
        logger.info(f"Saved clustered graph PNG to {path}")

    
    def save(
        self,
        output_dir: Path,
        format: str = "json",
        prefix: str = "domain_graph",
        abstraction_level: Optional[str] = None,
        visualize: bool = False,
        layout: str = "spring",
    ) -> List[Path]:
        """
        Save graph to file(s).
        
        Args:
            output_dir: Directory to save files
            format: "json", "csv", or "both"
            prefix: Filename prefix
            abstraction_level: For multi-level instances, which level to use
            visualize: Whether to also generate PNG visualization
            layout: Layout algorithm for PNG ("spring", "circular", "kamada_kawai")
            
        Returns:
            List of paths to saved files
        """
        if not self._graph_data:
            self.generate(abstraction_level=abstraction_level)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        if format in ("json", "both"):
            json_path = output_dir / f"{prefix}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, indent=2)
            saved_paths.append(json_path)
            logger.info(f"Saved graph JSON to {json_path}")
        
        if format in ("csv", "both"):
            nodes_path = output_dir / f"{prefix}_nodes.csv"
            edges_path = output_dir / f"{prefix}_edges.csv"
            self.to_csv_nodes(nodes_path)
            self.to_csv_edges(edges_path)
            saved_paths.extend([nodes_path, edges_path])
        
        if visualize:
            png_path = output_dir / f"{prefix}.png"
            self.to_png(png_path, layout=layout)
            saved_paths.append(png_path)
        
        return saved_paths
