"""
Relationship Graph Generator for building and visualizing entity relationship networks.

Generates graph data from relationships with:
- Node metrics (degree, betweenness, PageRank)
- Edge weights and evidence aggregation
- Optional causal attribute integration
- Multiple layout algorithms including semantic positioning
- Export to JSON, CSV, PNG, and GEXF formats
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import networkx as nx
import numpy as np

from .models import (
    Relationship,
    NormalizedRelationship,
    CausalRelationship,
    RelationshipGraphNode,
    RelationshipGraphEdge,
    RelationshipGraphData,
)

logger = logging.getLogger(__name__)

# Causal edge colors (from web app conventions)
POLARITY_COLORS = {
    "positive": "#22c55e",  # Green
    "negative": "#ef4444",  # Red
    "neutral": "#94a3b8",   # Slate gray
    None: "#94a3b8",        # Default gray for non-causal
}

# Certainty line styles for matplotlib
CERTAINTY_STYLES = {
    "certain": "solid",
    "likely": (0, (5, 3)),      # Dashed
    "possible": (0, (2, 2)),    # Dotted
    None: "solid",              # Default solid
}


class RelationshipGraph:
    """
    Build and analyze relationship networks.

    Supports building graphs from raw, normalized, or causal relationships,
    computing network metrics, and exporting to multiple formats.

    Example:
        >>> from qualitative_analysis.relationships.graph import RelationshipGraph
        >>> graph = RelationshipGraph()
        >>> graph_data = graph.build(relationships)
        >>> graph.to_png("output/graph.png", layout="semantic")
    """

    def __init__(self):
        """Initialize the relationship graph generator."""
        self._graph_data: Optional[RelationshipGraphData] = None
        self._nx_graph: Optional[nx.DiGraph] = None
        self._positions: Optional[Dict[str, Tuple[float, float]]] = None

    def build(
        self,
        relationships: Union[
            List[Relationship],
            List[NormalizedRelationship],
            List[CausalRelationship],
        ],
        include_causal: bool = True,
        min_edge_weight: int = 1,
    ) -> RelationshipGraphData:
        """
        Build graph from a list of relationships.

        Args:
            relationships: List of Relationship, NormalizedRelationship, or CausalRelationship objects.
            include_causal: Whether to include causal attributes in edges (if available).
            min_edge_weight: Minimum edge weight to include in graph.

        Returns:
            RelationshipGraphData with nodes, edges, and metrics.
        """
        if not relationships:
            return RelationshipGraphData(
                nodes=[],
                edges=[],
                metrics_summary={"node_count": 0, "edge_count": 0},
            )

        # Detect relationship type
        is_normalized = isinstance(relationships[0], NormalizedRelationship)
        is_causal = isinstance(relationships[0], CausalRelationship)

        logger.info(f"Building graph from {len(relationships)} relationships (normalized={is_normalized}, causal={is_causal})")

        # Track node data: id -> {label, frequency, text_ids, snippets, raw_entities}
        nodes_data: Dict[str, Dict[str, Any]] = {}

        # Track edge data: "source->target->type" -> {weight, evidence, text_ids, causal_attrs}
        edges_data: Dict[str, Dict[str, Any]] = {}

        for rel in relationships:
            source = rel.source.strip()
            target = rel.target.strip()
            rel_type = rel.type.strip()

            if not source or not target:
                continue

            # Normalize IDs
            source_id = source.lower()
            target_id = target.lower()

            # Skip self-loops
            if source_id == target_id:
                continue

            # Get evidence/description
            description = getattr(rel, "description", "")
            descriptions = getattr(rel, "descriptions", [description] if description else [])

            # Get text identifiers
            if hasattr(rel, "text_ids") and rel.text_ids:
                text_ids = rel.text_ids
            elif hasattr(rel, "text_id") and rel.text_id:
                text_ids = [rel.text_id]
            else:
                text_ids = []

            # Get count for normalized relationships
            count = getattr(rel, "count", 1)

            # Get raw/original entities for normalized relationships
            original_source = getattr(rel, "original_source", source)
            original_target = getattr(rel, "original_target", target)

            # Update source node
            if source_id not in nodes_data:
                nodes_data[source_id] = {
                    "label": source,
                    "frequency": 0,
                    "text_ids": set(),
                    "snippets": [],
                    "raw_entities": set(),
                }
            nodes_data[source_id]["frequency"] += count
            nodes_data[source_id]["text_ids"].update(text_ids)
            if original_source and original_source != source:
                nodes_data[source_id]["raw_entities"].add(original_source)
            if descriptions and len(nodes_data[source_id]["snippets"]) < 5:
                for d in descriptions[:2]:
                    if d not in nodes_data[source_id]["snippets"]:
                        nodes_data[source_id]["snippets"].append(d)

            # Update target node
            if target_id not in nodes_data:
                nodes_data[target_id] = {
                    "label": target,
                    "frequency": 0,
                    "text_ids": set(),
                    "snippets": [],
                    "raw_entities": set(),
                }
            nodes_data[target_id]["frequency"] += count
            nodes_data[target_id]["text_ids"].update(text_ids)
            if original_target and original_target != target:
                nodes_data[target_id]["raw_entities"].add(original_target)
            if descriptions and len(nodes_data[target_id]["snippets"]) < 5:
                for d in descriptions[:2]:
                    if d not in nodes_data[target_id]["snippets"]:
                        nodes_data[target_id]["snippets"].append(d)

            # Update edge
            edge_key = f"{source_id}->{target_id}->{rel_type.lower()}"
            if edge_key not in edges_data:
                edges_data[edge_key] = {
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                    "weight": 0,
                    "evidence": [],
                    "text_ids": set(),
                    "causal_attributes": None,
                }
            edges_data[edge_key]["weight"] += count
            edges_data[edge_key]["text_ids"].update(text_ids)
            if descriptions:
                for d in descriptions[:3]:
                    if d and d not in edges_data[edge_key]["evidence"]:
                        edges_data[edge_key]["evidence"].append(d)
                        if len(edges_data[edge_key]["evidence"]) >= 5:
                            break

            # Add causal attributes if available
            if is_causal and include_causal:
                causal = getattr(rel, "causal", None)
                if causal and causal.is_causal:
                    edges_data[edge_key]["causal_attributes"] = {
                        "is_causal": causal.is_causal,
                        "polarity": causal.polarity,
                        "certainty": causal.certainty,
                        "explicit_vs_implicit": causal.explicit_vs_implicit,
                    }

        # Build NetworkX graph for metrics
        G = nx.DiGraph()
        for node_id, data in nodes_data.items():
            G.add_node(node_id, label=data["label"], frequency=data["frequency"])

        for edge_key, data in edges_data.items():
            if data["weight"] >= min_edge_weight:
                G.add_edge(data["source"], data["target"], weight=data["weight"], type=data["type"])

        self._nx_graph = G

        # Compute metrics
        if len(G.nodes) > 0:
            betweenness = nx.betweenness_centrality(G)
            pagerank = nx.pagerank(G, weight="weight")
            in_degrees = dict(G.in_degree())
            out_degrees = dict(G.out_degree())
        else:
            betweenness = {}
            pagerank = {}
            in_degrees = {}
            out_degrees = {}

        # Build node objects
        nodes = []
        for node_id, data in nodes_data.items():
            if node_id not in G.nodes:
                continue  # Skip nodes without edges meeting min_edge_weight

            nodes.append(RelationshipGraphNode(
                id=node_id,
                label=data["label"],
                frequency=data["frequency"],
                text_count=len(data["text_ids"]),
                source_text_ids=sorted(data["text_ids"]),
                snippets=[{"text": s} for s in data["snippets"][:5]],
                degree=in_degrees.get(node_id, 0) + out_degrees.get(node_id, 0),
                in_degree=in_degrees.get(node_id, 0),
                out_degree=out_degrees.get(node_id, 0),
                betweenness=round(betweenness.get(node_id, 0), 6),
                pagerank=round(pagerank.get(node_id, 0), 6),
                raw_entities=sorted(data["raw_entities"]) if is_normalized else [],
            ))

        # Build edge objects
        edges = []
        edge_idx = 0
        for edge_key, data in edges_data.items():
            if data["weight"] < min_edge_weight:
                continue

            edges.append(RelationshipGraphEdge(
                id=f"e{edge_idx}",
                source=data["source"],
                target=data["target"],
                type=data["type"],
                weight=data["weight"],
                text_count=len(data["text_ids"]),
                evidence=data["evidence"][:5],
                source_text_ids=sorted(data["text_ids"]),
                causal_attributes=data["causal_attributes"],
            ))
            edge_idx += 1

        # Sort by weight/frequency
        nodes.sort(key=lambda n: n.frequency, reverse=True)
        edges.sort(key=lambda e: e.weight, reverse=True)

        # Compute summary metrics
        metrics_summary = self._compute_metrics_summary(G, nodes, edges)

        self._graph_data = RelationshipGraphData(
            nodes=nodes,
            edges=edges,
            metrics_summary=metrics_summary,
            is_normalized=is_normalized,
            has_causal=is_causal and include_causal,
        )

        logger.info(f"Built graph with {len(nodes)} nodes and {len(edges)} edges")

        return self._graph_data

    def _compute_metrics_summary(
        self,
        G: nx.DiGraph,
        nodes: List[RelationshipGraphNode],
        edges: List[RelationshipGraphEdge],
    ) -> Dict[str, Any]:
        """Compute global graph metrics."""
        if len(G.nodes) == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0,
                "average_degree": 0,
            }

        metrics = {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "density": round(nx.density(G), 6),
            "average_degree": round(sum(n.degree for n in nodes) / len(nodes), 2) if nodes else 0,
        }

        # Connected components (treating as undirected for this metric)
        if len(G.nodes) > 0:
            undirected = G.to_undirected()
            metrics["connected_components"] = nx.number_connected_components(undirected)
            metrics["is_connected"] = nx.is_connected(undirected)

        # Top nodes by centrality
        if nodes:
            metrics["top_nodes_by_degree"] = [n.label for n in sorted(nodes, key=lambda x: x.degree, reverse=True)[:5]]
            metrics["top_nodes_by_pagerank"] = [n.label for n in sorted(nodes, key=lambda x: x.pagerank, reverse=True)[:5]]
            metrics["top_nodes_by_betweenness"] = [n.label for n in sorted(nodes, key=lambda x: x.betweenness, reverse=True)[:5]]

        # Edge statistics
        if edges:
            metrics["max_edge_weight"] = max(e.weight for e in edges)
            metrics["causal_edge_count"] = sum(1 for e in edges if e.causal_attributes)

        return metrics

    def filter(
        self,
        min_edge_weight: int = 1,
        relationship_types: Optional[List[str]] = None,
        include_isolated: bool = False,
    ) -> RelationshipGraphData:
        """
        Filter the graph by criteria.

        Args:
            min_edge_weight: Minimum edge weight to include.
            relationship_types: Only include these relationship types (None = all).
            include_isolated: Include nodes with no edges after filtering.

        Returns:
            Filtered RelationshipGraphData.
        """
        if not self._graph_data:
            raise ValueError("No graph data. Call build() first.")

        # Filter edges
        filtered_edges = []
        for edge in self._graph_data.edges:
            if edge.weight < min_edge_weight:
                continue
            if relationship_types and edge.type.lower() not in [t.lower() for t in relationship_types]:
                continue
            filtered_edges.append(edge)

        # Get connected nodes
        connected_nodes = set()
        for edge in filtered_edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)

        # Filter nodes
        filtered_nodes = []
        for node in self._graph_data.nodes:
            if include_isolated or node.id in connected_nodes:
                filtered_nodes.append(node)

        return RelationshipGraphData(
            nodes=filtered_nodes,
            edges=filtered_edges,
            metrics_summary={
                "node_count": len(filtered_nodes),
                "edge_count": len(filtered_edges),
                "filtered": True,
            },
            is_normalized=self._graph_data.is_normalized,
            has_causal=self._graph_data.has_causal,
        )

    def _filter_labels(self, labels: Dict[str, str], mode: str) -> Dict[str, str]:
        """
        Filter which node labels to display based on mode.

        Args:
            labels: Dict mapping node_id to label string.
            mode: Filter mode - "all", "none", "top:N", or "pagerank:N".

        Returns:
            Filtered labels dict.
        """
        if mode == "none":
            return {}
        if mode == "all":
            return labels
        if mode.startswith("top:"):
            try:
                n = int(mode.split(":")[1])
                top_nodes = sorted(self._graph_data.nodes, key=lambda x: x.degree, reverse=True)[:n]
                return {node.id: labels[node.id] for node in top_nodes if node.id in labels}
            except (ValueError, IndexError):
                return labels
        if mode.startswith("pagerank:"):
            try:
                n = int(mode.split(":")[1])
                top_nodes = sorted(self._graph_data.nodes, key=lambda x: x.pagerank, reverse=True)[:n]
                return {node.id: labels[node.id] for node in top_nodes if node.id in labels}
            except (ValueError, IndexError):
                return labels
        return labels

    def _apply_repulsion(
        self,
        positions: Dict[str, Tuple[float, float]],
        min_distance: float = 0.05,
        iterations: int = 50,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Apply force-based repulsion to prevent node overlap.

        Args:
            positions: Dict mapping node_id to (x, y) position.
            min_distance: Minimum distance between nodes.
            iterations: Number of repulsion iterations.

        Returns:
            Updated positions with repulsion applied.
        """
        if len(positions) < 2:
            return positions

        nodes = list(positions.keys())
        coords = np.array([positions[n] for n in nodes], dtype=float)

        for _ in range(iterations):
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    diff = coords[i] - coords[j]
                    dist = np.linalg.norm(diff)
                    if dist < min_distance and dist > 1e-6:
                        force = (min_distance - dist) / 2
                        direction = diff / dist
                        coords[i] += direction * force
                        coords[j] -= direction * force

        # Normalize back to [0, 1] range
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_range = x_max - x_min if x_max > x_min else 1
        y_range = y_max - y_min if y_max > y_min else 1

        return {
            n: (
                float((coords[i, 0] - x_min) / x_range),
                float((coords[i, 1] - y_min) / y_range),
            )
            for i, n in enumerate(nodes)
        }

    def _get_edge_colors_and_styles(
        self,
        G: "nx.DiGraph",
        color_by: str = "polarity",
    ) -> Tuple[List[str], List[Any]]:
        """
        Get edge colors and line styles based on causal attributes.

        Args:
            G: NetworkX graph.
            color_by: How to color edges - "polarity", "type", or "none".

        Returns:
            Tuple of (colors list, styles list) for each edge.
        """
        edge_list = list(G.edges())
        colors = []
        styles = []

        # Build lookup for causal attributes
        causal_lookup = {}
        if self._graph_data and self._graph_data.edges:
            for edge in self._graph_data.edges:
                key = (edge.source, edge.target)
                causal_lookup[key] = edge.causal_attributes

        for u, v in edge_list:
            causal_attrs = causal_lookup.get((u, v), {}) or {}

            # Determine color
            if color_by == "polarity" and causal_attrs:
                polarity = causal_attrs.get("polarity")
                colors.append(POLARITY_COLORS.get(polarity, POLARITY_COLORS[None]))
            elif color_by == "type":
                # Could add type-based coloring here
                colors.append("#94a3b8")
            else:
                colors.append("#94a3b8")

            # Determine line style
            if causal_attrs:
                certainty = causal_attrs.get("certainty")
                styles.append(CERTAINTY_STYLES.get(certainty, CERTAINTY_STYLES[None]))
            else:
                styles.append("solid")

        return colors, styles

    def _draw_legend(
        self,
        ax,
        color_by: str,
        has_causal: bool,
        show_certainty: bool = True,
        show_node_roles: bool = False,
    ) -> None:
        """
        Draw legend for edge colors, styles, and node shapes.

        Args:
            ax: Matplotlib axis.
            color_by: Color mode used.
            has_causal: Whether graph has causal attributes.
            show_certainty: Whether to show certainty line styles.
            show_node_roles: Whether to show node role shapes.
        """
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines

        handles = []

        # Node role shapes
        if show_node_roles:
            role_legend = [
                ("^", "Source (cause)", "#6366f1"),
                ("s", "Target (effect)", "#6366f1"),
                ("o", "Both (mediator)", "#6366f1"),
            ]
            for marker, label, color in role_legend:
                handles.append(mlines.Line2D(
                    [], [],
                    marker=marker,
                    color="white",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=10,
                    linestyle="None",
                    label=label,
                ))

        # Polarity colors
        if color_by == "polarity" and has_causal:
            for polarity in ["positive", "negative", "neutral"]:
                color = POLARITY_COLORS[polarity]
                handles.append(mpatches.Patch(
                    color=color,
                    label=f"{polarity.title()} causality"
                ))

        # Certainty line styles
        if show_certainty and has_causal:
            for certainty, style in [("certain", "solid"), ("possible", "dotted")]:
                handles.append(mlines.Line2D(
                    [], [],
                    color="gray",
                    linestyle=style,
                    label=f"{certainty.title()}"
                ))

        # Explicitness arrow styles
        if has_causal:
            # Explicit: large filled arrow
            handles.append(mlines.Line2D(
                [], [],
                color="gray",
                marker=">",
                markersize=12,
                linestyle="-",
                label="Explicit"
            ))
            # Implicit: smaller simple arrow
            handles.append(mlines.Line2D(
                [], [],
                color="gray",
                marker=">",
                markersize=8,
                linestyle="-",
                label="Implicit"
            ))

        if handles:
            ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)

    def _cluster_nodes_2d(
        self,
        positions: Dict[str, Tuple[float, float]],
        min_cluster_size: int = 3,
        method: str = "hdbscan",
    ) -> Dict[str, int]:
        """
        Cluster nodes based on their 2D positions.

        Args:
            positions: Dict mapping node_id -> (x, y).
            min_cluster_size: Minimum nodes per cluster.
            method: Clustering method ("hdbscan" or "agglomerative").

        Returns:
            Dict mapping node_id -> cluster_id (-1 for noise/outliers).
        """
        if len(positions) < min_cluster_size:
            return {node_id: 0 for node_id in positions}

        node_ids = list(positions.keys())
        coords = np.array([positions[nid] for nid in node_ids])

        labels = None

        if method == "hdbscan":
            try:
                import hdbscan
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=2,
                    metric='euclidean',
                )
                labels = clusterer.fit_predict(coords)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                logger.info(f"[Clustering] HDBSCAN found {n_clusters} clusters")
            except ImportError:
                logger.warning("hdbscan not installed, falling back to agglomerative")
                method = "agglomerative"

        if method == "agglomerative" or labels is None:
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = max(2, len(positions) // min_cluster_size)
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(coords)
            logger.info(f"[Clustering] Agglomerative created {n_clusters} clusters")

        return {node_id: int(labels[i]) for i, node_id in enumerate(node_ids)}

    def _get_cluster_hull_points(
        self,
        positions: Dict[str, Tuple[float, float]],
        cluster_assignments: Dict[str, int],
        cluster_id: int,
    ) -> List[Tuple[float, float]]:
        """Get points for drawing cluster boundary (convex hull)."""
        points = [
            positions[nid] for nid, cid in cluster_assignments.items()
            if cid == cluster_id and nid in positions
        ]
        return points

    async def _generate_cluster_labels_llm(
        self,
        clusters: Dict[int, List[str]],
        model: str,
        base_url: str = "http://localhost:11434",
    ) -> Dict[int, str]:
        """
        Generate descriptive labels for clusters using LLM.

        Args:
            clusters: Dict mapping cluster_id -> list of node labels in that cluster.
            model: LLM model name (e.g., "qwen3:8b").
            base_url: Ollama base URL.

        Returns:
            Dict mapping cluster_id -> generated descriptive label.
        """
        from qualitative_analysis.core.providers import OllamaProvider

        llm = OllamaProvider(model, base_url=base_url)
        labels = {}

        prompt_template = """Given these entity labels from a semantic cluster in a relationship graph:
{members}

Generate a short 2-4 word descriptive label that captures the common theme or category.

Respond with JSON:
{{
  "reasoning": "Brief explanation of the common theme",
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

            # Limit to 15 members for prompt
            members_str = "\n".join(f"- {m}" for m in members[:15])
            prompt = prompt_template.format(members=members_str)

            try:
                response = await llm.generate(prompt=prompt, temperature=0.3)
                cleaned = response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                data = json.loads(cleaned.strip())
                labels[cluster_id] = data.get("label", f"Cluster {cluster_id}")
                logger.debug(f"Cluster {cluster_id} label: {labels[cluster_id]}")
            except Exception as e:
                logger.warning(f"LLM cluster labeling failed for cluster {cluster_id}: {e}")
                labels[cluster_id] = f"Cluster {cluster_id}"

        logger.info(f"[Clustering] Generated {len(labels)} cluster labels via LLM")
        return labels

    def _get_node_roles(self) -> Dict[str, str]:
        """
        Determine the role of each node: source-only, target-only, or both.

        Returns:
            Dict mapping node_id -> role ("source", "target", or "both").
        """
        if not self._graph_data:
            return {}

        sources = set()
        targets = set()

        for edge in self._graph_data.edges:
            sources.add(edge.source)
            targets.add(edge.target)

        roles = {}
        all_nodes = sources | targets

        for node_id in all_nodes:
            is_source = node_id in sources
            is_target = node_id in targets

            if is_source and is_target:
                roles[node_id] = "both"
            elif is_source:
                roles[node_id] = "source"
            else:
                roles[node_id] = "target"

        return roles

    def _compute_semantic_positions(
        self,
        node_labels: List[str],
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        umap_min_dist: float = 0.3,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute 2D positions using UMAP projection of node label embeddings.

        Args:
            node_labels: List of node label strings.
            embedding_model: SentenceTransformer model name.
            device: Device for embeddings (None for auto-detect).
            umap_min_dist: UMAP min_dist parameter. Higher values spread nodes more.

        Returns:
            Dict mapping node label (lowercase) -> (x, y) position.
        """
        try:
            import umap
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            logger.warning(f"Missing dependency for semantic layout: {e}")
            return {}

        if len(node_labels) < 2:
            return {label.lower(): (0.5, 0.5) for label in node_labels}

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
        logger.info(f"[SemanticLayout] Generating embeddings for {len(node_labels)} nodes")
        model = SentenceTransformer(embedding_model, device=device)
        embeddings = model.encode(node_labels)

        # PCA preprocessing for stability
        from sklearn.decomposition import PCA

        embedding_dim = embeddings.shape[1]
        n_samples = embeddings.shape[0]

        if embedding_dim > 50 and n_samples >= 3:
            max_components = min(n_samples - 1, 50)
            pca = PCA(n_components=min(max_components, embedding_dim))
            embeddings = pca.fit_transform(embeddings)
            logger.info(f"[SemanticLayout] PCA reduced to {embeddings.shape[1]}D")

        # UMAP projection
        n_neighbors = min(15, len(node_labels) - 1)
        logger.info(f"[SemanticLayout] Running UMAP with n_neighbors={n_neighbors}, min_dist={umap_min_dist}")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
            metric='cosine',
            random_state=42,
        )
        coords_2d = reducer.fit_transform(embeddings)

        # Normalize to [0, 1]
        x_min, x_max = coords_2d[:, 0].min(), coords_2d[:, 0].max()
        y_min, y_max = coords_2d[:, 1].min(), coords_2d[:, 1].max()

        positions = {}
        for i, label in enumerate(node_labels):
            x_norm = (coords_2d[i, 0] - x_min) / (x_max - x_min + 1e-6)
            y_norm = (coords_2d[i, 1] - y_min) / (y_max - y_min + 1e-6)
            positions[label.lower()] = (float(x_norm), float(y_norm))

        logger.info(f"[SemanticLayout] Computed positions for {len(positions)} nodes")
        return positions

    def to_png(
        self,
        path: Union[str, Path],
        figsize: Tuple[int, int] = (16, 12),
        dpi: int = 150,
        layout: str = "spring",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        umap_min_dist: float = 0.3,
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0,
        show_edge_labels: bool = False,
        title: Optional[str] = None,
        color_by: str = "polarity",
        labels_mode: str = "all",
        show_legend: bool = True,
        layout_spacing: float = 1.0,
        cluster_nodes: bool = False,
        min_cluster_size: int = 3,
        show_cluster_hulls: bool = True,
        cluster_label_min_size: int = 5,
        cluster_label_model: Optional[str] = None,
        cluster_label_base_url: str = "http://localhost:11434",
        show_node_roles: bool = False,
    ) -> str:
        """
        Export graph as PNG visualization.

        Args:
            path: Output path for PNG file.
            figsize: Figure size (width, height) in inches.
            dpi: Resolution.
            layout: Layout algorithm ("spring", "circular", "kamada_kawai", "semantic", "hierarchical", "community").
            embedding_model: Model for semantic layout.
            umap_min_dist: UMAP min_dist parameter. Higher values spread nodes more. When cluster_nodes=True, this is overridden to 0.1 to preserve density structure for HDBSCAN.
            node_size_scale: Multiplier for node sizes.
            edge_width_scale: Multiplier for edge widths.
            show_edge_labels: Whether to show relationship types on edges.
            title: Optional title for the graph.
            color_by: Edge coloring mode ("polarity", "type", "none").
            labels_mode: Label display mode ("all", "none", "top:N", "pagerank:N").
            show_legend: Whether to show legend for colors/styles.
            layout_spacing: Multiplier for node spacing in layouts.
            cluster_nodes: Whether to cluster nodes and color by cluster (requires semantic layout).
            min_cluster_size: Minimum nodes per cluster when clustering.
            show_cluster_hulls: Whether to draw convex hulls around clusters.
            cluster_label_min_size: Minimum cluster size to show label (reduces clutter). Default 5.
            cluster_label_model: LLM model for generating cluster labels (e.g., "qwen3:8b"). If None, uses generic labels.
            cluster_label_base_url: Ollama base URL for cluster labeling.
            show_node_roles: Whether to use different shapes for source-only, target-only, and both nodes.

        Returns:
            Path to saved PNG file.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            logger.warning("matplotlib not installed. Skipping PNG generation.")
            return ""

        if not self._graph_data or not self._nx_graph:
            raise ValueError("No graph data. Call build() first.")

        path = Path(path)
        G = self._nx_graph

        if len(G.nodes) == 0:
            logger.warning("No nodes in graph. Skipping PNG generation.")
            return ""

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Select layout
        # Adjust spring constant based on node count and spacing
        k_value = 2.0 * layout_spacing * (1 + len(G.nodes()) / 100)

        if layout == "semantic":
            node_labels = [G.nodes[n].get("label", n) for n in G.nodes()]
            # When clustering is enabled, use lower min_dist (0.1) to preserve density
            # structure for HDBSCAN. Higher min_dist spreads nodes uniformly which
            # destroys the density signal clustering relies on.
            effective_min_dist = umap_min_dist
            if cluster_nodes:
                effective_min_dist = 0.1
                if umap_min_dist != 0.1:
                    logger.info(
                        f"[Clustering] Overriding umap_min_dist from {umap_min_dist} to 0.1 "
                        "for better cluster detection"
                    )
            semantic_positions = self._compute_semantic_positions(
                node_labels, embedding_model, umap_min_dist=effective_min_dist
            )
            if semantic_positions:
                pos = {}
                for node in G.nodes():
                    label = G.nodes[node].get("label", node).lower()
                    pos[node] = semantic_positions.get(label, (0.5, 0.5))
                # Apply repulsion to prevent overlap
                pos = self._apply_repulsion(pos, min_distance=0.03 * layout_spacing, iterations=100)
            else:
                logger.warning("Semantic layout failed, using spring layout")
                pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)
        elif layout == "hierarchical":
            # Try graphviz dot layout for directed graphs (using pydot)
            try:
                from networkx.drawing.nx_pydot import graphviz_layout
                pos = graphviz_layout(G, prog="dot")
                # Normalize positions to [0, 1] range
                xs = [p[0] for p in pos.values()]
                ys = [p[1] for p in pos.values()]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                x_range = x_max - x_min if x_max > x_min else 1
                y_range = y_max - y_min if y_max > y_min else 1
                pos = {n: ((p[0] - x_min) / x_range, (p[1] - y_min) / y_range) for n, p in pos.items()}
                logger.info("[Layout] Using hierarchical (graphviz dot) layout")
            except ImportError:
                logger.warning("pydot/graphviz not available, using spring layout. Install with: pip install pydot && brew install graphviz")
                pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "community":
            # Detect communities using Louvain and position by group
            try:
                # Convert to undirected for community detection
                G_undirected = G.to_undirected()
                communities = list(nx.community.louvain_communities(G_undirected, seed=42))
                n_communities = len(communities)
                logger.info(f"[Layout] Detected {n_communities} communities using Louvain")

                # Arrange communities in a grid
                import math
                grid_size = math.ceil(math.sqrt(n_communities))

                pos = {}
                for i, community in enumerate(communities):
                    # Grid position for this community
                    row = i // grid_size
                    col = i % grid_size
                    center_x = (col + 0.5) / grid_size
                    center_y = (row + 0.5) / grid_size

                    # Create subgraph for this community and use spring layout
                    subgraph = G.subgraph(community)
                    if len(community) == 1:
                        node = list(community)[0]
                        pos[node] = (center_x, center_y)
                    else:
                        sub_pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
                        # Scale and translate to community position
                        scale = 0.8 / grid_size  # Leave some margin
                        for node, (x, y) in sub_pos.items():
                            # Normalize sub_pos to [0, 1] then scale and translate
                            pos[node] = (
                                center_x + (x - 0.5) * scale,
                                center_y + (y - 0.5) * scale
                            )
            except Exception as e:
                logger.warning(f"Community layout failed ({e}), using spring layout")
                pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)
        else:  # spring layout
            pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)

        self._positions = pos

        # Node sizes based on frequency/degree
        node_sizes = []
        for node in G.nodes():
            freq = G.nodes[node].get("frequency", 1)
            size = max(300, min(2000, freq * 100)) * node_size_scale
            node_sizes.append(size)

        # Clustering (if enabled and using semantic layout)
        cluster_assignments = None
        cluster_colors_map = None
        cluster_text_objects = []
        n_clusters = 0

        if cluster_nodes and layout == "semantic" and len(pos) >= min_cluster_size:
            cluster_assignments = self._cluster_nodes_2d(pos, min_cluster_size=min_cluster_size)

            # Get unique clusters (excluding noise = -1)
            unique_clusters = sorted(set(c for c in cluster_assignments.values() if c >= 0))
            n_clusters = len(unique_clusters)

            # Generate cluster color palette
            import matplotlib.cm as cm
            cluster_palette = cm.tab20(np.linspace(0, 1, max(n_clusters, 20)))

            # Build node -> color mapping
            cluster_colors_map = {}
            for node_id, cluster_id in cluster_assignments.items():
                if cluster_id >= 0:
                    cluster_colors_map[node_id] = cluster_palette[cluster_id % len(cluster_palette)]
                else:
                    cluster_colors_map[node_id] = (0.7, 0.7, 0.7, 0.8)  # Gray for noise

            # Draw cluster hulls (before edges/nodes so they're in background)
            if show_cluster_hulls and n_clusters > 0:
                try:
                    from scipy.spatial import ConvexHull
                    from matplotlib.patches import Polygon

                    for cluster_id in unique_clusters:
                        points = self._get_cluster_hull_points(pos, cluster_assignments, cluster_id)
                        if len(points) >= 3:
                            coords = np.array(points)
                            hull = ConvexHull(coords)
                            hull_points = coords[hull.vertices]

                            color = cluster_palette[cluster_id % len(cluster_palette)]
                            polygon = Polygon(
                                hull_points,
                                alpha=0.12,
                                facecolor=color,
                                edgecolor=color[:3] if len(color) >= 3 else color,
                                linewidth=1.5,
                            )
                            ax.add_patch(polygon)
                except ImportError:
                    logger.warning("scipy not available for convex hull drawing")
                except Exception as e:
                    logger.debug(f"Could not draw cluster hulls: {e}")

        # Node colors - cluster-based or single color
        if cluster_colors_map:
            node_color = [cluster_colors_map.get(node, (0.5, 0.5, 0.5, 1)) for node in G.nodes()]
        else:
            node_color = "#6366f1"  # Indigo

        # Edge widths and alphas based on weight
        edge_list = list(G.edges())
        if edge_list:
            edge_weights = [G[u][v].get("weight", 1) for u, v in edge_list]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [max(0.5, (w / max_weight) * 3) * edge_width_scale for w in edge_weights]
            # Alpha proportional to weight: range from 0.2 (min) to 0.8 (max)
            edge_alphas = [0.2 + 0.6 * (w / max_weight) for w in edge_weights]
        else:
            edge_widths = []
            edge_alphas = []

        # Get edge colors and styles based on causal attributes
        edge_colors, edge_styles = self._get_edge_colors_and_styles(G, color_by=color_by)

        # Convert edge colors to RGBA with per-edge alpha
        import matplotlib.colors as mcolors
        edge_colors_rgba = []
        for i, color in enumerate(edge_colors):
            try:
                rgb = mcolors.to_rgb(color)
                alpha = edge_alphas[i] if i < len(edge_alphas) else 0.6
                edge_colors_rgba.append((*rgb, alpha))
            except (ValueError, KeyError):
                edge_colors_rgba.append((0.58, 0.64, 0.72, 0.6))  # Default gray with 0.6 alpha

        # Draw edges - group by line style AND explicitness for different visual encoding
        if edge_list:
            # Group edges by (style, explicitness)
            style_groups = {}
            for i, (u, v) in enumerate(edge_list):
                style = edge_styles[i] if i < len(edge_styles) else "solid"
                # Get explicitness from edge data
                causal_attrs = G[u][v].get("causal_attributes", {})
                explicitness = causal_attrs.get("explicit_vs_implicit", "") if causal_attrs else ""
                # Group key combines style and explicitness
                group_key = (str(style), explicitness)
                if group_key not in style_groups:
                    style_groups[group_key] = {
                        "edges": [],
                        "colors": [],
                        "widths": [],
                        "style": style,
                        "explicitness": explicitness,
                    }
                style_groups[group_key]["edges"].append((u, v))
                style_groups[group_key]["colors"].append(edge_colors_rgba[i] if i < len(edge_colors_rgba) else (0.58, 0.64, 0.72, 0.6))
                style_groups[group_key]["widths"].append(edge_widths[i] if i < len(edge_widths) else 1.0)

            # Draw each style group with appropriate arrow style
            for group_key, group in style_groups.items():
                # Arrow style based on explicitness:
                # - explicit: filled arrow (default)
                # - implicit: smaller, simpler arrow
                # - unknown/empty: medium arrow
                explicitness = group["explicitness"]
                if explicitness == "explicit":
                    arrowsize = 15
                    arrowstyle = "-|>"
                elif explicitness == "implicit":
                    arrowsize = 10
                    arrowstyle = "->"
                else:
                    arrowsize = 12
                    arrowstyle = "-|>"

                nx.draw_networkx_edges(
                    G, pos, ax=ax,
                    edgelist=group["edges"],
                    edge_color=group["colors"],
                    width=group["widths"],
                    arrows=True,
                    arrowsize=arrowsize,
                    arrowstyle=arrowstyle,
                    connectionstyle="arc3,rad=0.1",
                    style=group["style"],
                )

        # Draw edge labels if requested
        if show_edge_labels and edge_list:
            edge_labels = {(u, v): G[u][v].get("type", "") for u, v in edge_list}
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, ax=ax,
                font_size=6,
                font_color="#64748b",
            )

        # Draw nodes - with optional role-based shapes
        if show_node_roles:
            # Get node roles (source-only, target-only, both)
            node_roles = self._get_node_roles()

            # Define markers for each role
            role_markers = {
                "source": "^",   # Triangle pointing up (causes)
                "target": "s",   # Square (effects)
                "both": "o",     # Circle (mediators)
            }

            # Group nodes by role
            role_groups = {"source": [], "target": [], "both": []}
            for i, node in enumerate(G.nodes()):
                role = node_roles.get(node, "both")
                role_groups[role].append((node, i))

            # Draw each role group with its marker
            for role, nodes_with_idx in role_groups.items():
                if not nodes_with_idx:
                    continue

                nodelist = [n for n, _ in nodes_with_idx]
                if cluster_colors_map:
                    colors = [cluster_colors_map.get(n, (0.5, 0.5, 0.5, 1)) for n in nodelist]
                else:
                    colors = "#6366f1"
                sizes = [node_sizes[idx] for _, idx in nodes_with_idx]

                nx.draw_networkx_nodes(
                    G, pos, ax=ax,
                    nodelist=nodelist,
                    node_color=colors,
                    node_size=sizes,
                    node_shape=role_markers[role],
                    alpha=0.9,
                    edgecolors="white",
                    linewidths=1.5,
                )
        else:
            # Standard circular nodes
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_color=node_color,
                node_size=node_sizes,
                alpha=0.9,
                edgecolors="white",
                linewidths=1.5,
            )

        # Draw cluster labels at centroids (if clustering enabled)
        if cluster_nodes and cluster_assignments and n_clusters > 0:
            # Group nodes by cluster
            clusters_dict: Dict[int, List[str]] = {}
            for node_id, cluster_id in cluster_assignments.items():
                if cluster_id not in clusters_dict:
                    clusters_dict[cluster_id] = []
                label = G.nodes[node_id].get("label", node_id) if node_id in G.nodes() else node_id
                clusters_dict[cluster_id].append(label)

            # Generate cluster labels
            if cluster_label_model:
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If already in async context, create task
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self._generate_cluster_labels_llm(clusters_dict, cluster_label_model, cluster_label_base_url)
                            )
                            cluster_labels = future.result()
                    else:
                        cluster_labels = loop.run_until_complete(
                            self._generate_cluster_labels_llm(clusters_dict, cluster_label_model, cluster_label_base_url)
                        )
                except RuntimeError:
                    cluster_labels = asyncio.run(
                        self._generate_cluster_labels_llm(clusters_dict, cluster_label_model, cluster_label_base_url)
                    )
            else:
                cluster_labels = {cid: f"Cluster {cid}" if cid >= 0 else "Other" for cid in clusters_dict}

            # Draw cluster labels at centroids (only for clusters >= cluster_label_min_size)
            # Collect text objects for adjustText
            cluster_text_objects = []
            labeled_count = 0
            for cluster_id in unique_clusters:
                points = self._get_cluster_hull_points(pos, cluster_assignments, cluster_id)
                if len(points) < cluster_label_min_size:
                    continue  # Skip small clusters to reduce clutter
                if len(points) >= 1:
                    coords = np.array(points)
                    centroid = coords.mean(axis=0)
                    label_text = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")

                    text_obj = ax.annotate(
                        label_text,
                        centroid,
                        fontsize=9,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        color="#1e293b",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="#cbd5e1",
                            alpha=0.85,
                        ),
                        zorder=10,
                    )
                    cluster_text_objects.append(text_obj)
                    labeled_count += 1

            if labeled_count < n_clusters:
                logger.info(f"[Clustering] Showing labels for {labeled_count}/{n_clusters} clusters (min size: {cluster_label_min_size})")

        # Draw labels (with filtering) - collect text objects for adjustText
        all_labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
        filtered_labels = self._filter_labels(all_labels, labels_mode)
        node_text_objects = []
        if filtered_labels:
            for node, label in filtered_labels.items():
                x, y = pos[node]
                text_obj = ax.text(
                    x, y, label,
                    fontsize=8,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=5,
                )
                node_text_objects.append(text_obj)

        # Apply adjustText to reduce label overlap
        all_text_objects = node_text_objects
        if cluster_nodes and cluster_assignments:
            all_text_objects = cluster_text_objects + node_text_objects

        if all_text_objects:
            try:
                from adjustText import adjust_text
                adjust_text(
                    all_text_objects,
                    ax=ax,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3, lw=0.5),
                    expand_points=(1.2, 1.4),
                    force_points=(0.5, 0.8),
                )
                logger.info(f"[Labels] Applied adjustText to {len(all_text_objects)} labels")
            except ImportError:
                logger.warning("adjustText not installed. Labels may overlap. Install with: pip install adjustText")

        # Draw legend
        if show_legend:
            has_causal = self._graph_data.has_causal if self._graph_data else False
            self._draw_legend(ax, color_by=color_by, has_causal=has_causal, show_node_roles=show_node_roles)

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            stats = self._graph_data.metrics_summary
            if cluster_nodes and n_clusters > 0:
                title_text = f"Relationship Graph (Semantic Clusters)\n{stats.get('node_count', 0)} nodes, {stats.get('edge_count', 0)} edges, {n_clusters} clusters"
            else:
                title_text = f"Relationship Graph\n{stats.get('node_count', 0)} nodes, {stats.get('edge_count', 0)} edges"
            ax.set_title(title_text, fontsize=14, fontweight="bold")

        ax.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()

        logger.info(f"Saved graph PNG to {path}")
        return str(path)

    def to_json(self, path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Export graph as JSON.

        Args:
            path: Optional path to save JSON file.

        Returns:
            Graph data as dictionary.
        """
        if not self._graph_data:
            raise ValueError("No graph data. Call build() first.")

        data = self._graph_data.to_dict()

        if path:
            path = Path(path)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved graph JSON to {path}")

        return data

    def to_csv(
        self,
        output_dir: Union[str, Path],
        prefix: str = "graph",
    ) -> Tuple[str, str]:
        """
        Export graph as CSV files (nodes and edges).

        Args:
            output_dir: Output directory.
            prefix: Filename prefix.

        Returns:
            Tuple of (nodes_path, edges_path).
        """
        if not self._graph_data:
            raise ValueError("No graph data. Call build() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write nodes
        nodes_path = output_dir / f"{prefix}_nodes.csv"
        node_fields = [
            "id", "label", "frequency", "text_count", "degree",
            "in_degree", "out_degree", "betweenness", "pagerank",
            "source_text_ids", "raw_entities",
        ]

        with open(nodes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=node_fields)
            writer.writeheader()
            for node in self._graph_data.nodes:
                writer.writerow({
                    "id": node.id,
                    "label": node.label,
                    "frequency": node.frequency,
                    "text_count": node.text_count,
                    "degree": node.degree,
                    "in_degree": node.in_degree,
                    "out_degree": node.out_degree,
                    "betweenness": node.betweenness,
                    "pagerank": node.pagerank,
                    "source_text_ids": json.dumps(node.source_text_ids),
                    "raw_entities": "|".join(node.raw_entities),
                })

        logger.info(f"Saved nodes CSV to {nodes_path}")

        # Write edges
        edges_path = output_dir / f"{prefix}_edges.csv"
        edge_fields = [
            "id", "source", "target", "type", "weight", "text_count",
            "evidence", "source_text_ids",
            "is_causal", "polarity", "certainty", "explicit_vs_implicit",
        ]

        with open(edges_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=edge_fields)
            writer.writeheader()
            for edge in self._graph_data.edges:
                causal = edge.causal_attributes or {}
                writer.writerow({
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "weight": edge.weight,
                    "text_count": edge.text_count,
                    "evidence": "|||".join(edge.evidence[:3]),
                    "source_text_ids": json.dumps(edge.source_text_ids),
                    "is_causal": causal.get("is_causal", ""),
                    "polarity": causal.get("polarity", ""),
                    "certainty": causal.get("certainty", ""),
                    "explicit_vs_implicit": causal.get("explicit_vs_implicit", ""),
                })

        logger.info(f"Saved edges CSV to {edges_path}")

        return str(nodes_path), str(edges_path)

    def to_gexf(self, path: Union[str, Path]) -> str:
        """
        Export graph as GEXF for Gephi.

        Args:
            path: Output path for GEXF file.

        Returns:
            Path to saved GEXF file.
        """
        if not self._nx_graph:
            raise ValueError("No graph data. Call build() first.")

        path = Path(path)

        # Add attributes for GEXF export
        G = self._nx_graph.copy()

        # Add node attributes
        for node in self._graph_data.nodes:
            if node.id in G.nodes:
                G.nodes[node.id]["frequency"] = node.frequency
                G.nodes[node.id]["degree"] = node.degree
                G.nodes[node.id]["betweenness"] = node.betweenness
                G.nodes[node.id]["pagerank"] = node.pagerank

        # Add edge attributes
        for edge in self._graph_data.edges:
            if G.has_edge(edge.source, edge.target):
                G[edge.source][edge.target]["type"] = edge.type
                G[edge.source][edge.target]["weight"] = edge.weight
                if edge.causal_attributes:
                    G[edge.source][edge.target]["is_causal"] = edge.causal_attributes.get("is_causal", False)
                    G[edge.source][edge.target]["polarity"] = edge.causal_attributes.get("polarity", "")

        nx.write_gexf(G, path)
        logger.info(f"Saved graph GEXF to {path}")

        return str(path)

    def save(
        self,
        output_dir: Union[str, Path],
        prefix: str = "relationship_graph",
        formats: List[str] = None,
        layout: str = "spring",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        umap_min_dist: float = 0.3,
        color_by: str = "polarity",
        labels_mode: str = "all",
        show_legend: bool = True,
        layout_spacing: float = 1.0,
        cluster_nodes: bool = False,
        min_cluster_size: int = 3,
        show_cluster_hulls: bool = True,
        cluster_label_min_size: int = 5,
        cluster_label_model: Optional[str] = None,
        cluster_label_base_url: str = "http://localhost:11434",
        show_node_roles: bool = False,
    ) -> List[str]:
        """
        Save graph to multiple formats.

        Args:
            output_dir: Output directory.
            prefix: Filename prefix.
            formats: List of formats ("json", "csv", "png", "gexf"). Default: ["json", "csv"].
            layout: Layout for PNG visualization.
            embedding_model: Model for semantic layout.
            umap_min_dist: UMAP min_dist parameter for semantic layout.
            color_by: Edge coloring mode ("polarity", "type", "none").
            labels_mode: Label display mode ("all", "none", "top:N", "pagerank:N").
            show_legend: Whether to show legend.
            layout_spacing: Multiplier for node spacing.
            cluster_nodes: Whether to cluster nodes and color by cluster.
            min_cluster_size: Minimum nodes per cluster.
            show_cluster_hulls: Whether to draw convex hulls around clusters.
            cluster_label_min_size: Minimum cluster size to show label (reduces clutter). Default 5.
            cluster_label_model: LLM model for generating cluster labels. If None, uses generic labels.
            cluster_label_base_url: Ollama base URL for cluster labeling.
            show_node_roles: Whether to use different shapes for source-only, target-only, and both nodes.

        Returns:
            List of paths to saved files.
        """
        if formats is None:
            formats = ["json", "csv"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        if "json" in formats:
            json_path = output_dir / f"{prefix}.json"
            self.to_json(json_path)
            saved_paths.append(str(json_path))

        if "csv" in formats:
            nodes_path, edges_path = self.to_csv(output_dir, prefix)
            saved_paths.extend([nodes_path, edges_path])

        if "png" in formats:
            png_path = output_dir / f"{prefix}.png"
            self.to_png(
                png_path,
                layout=layout,
                embedding_model=embedding_model,
                umap_min_dist=umap_min_dist,
                color_by=color_by,
                labels_mode=labels_mode,
                show_legend=show_legend,
                layout_spacing=layout_spacing,
                cluster_nodes=cluster_nodes,
                min_cluster_size=min_cluster_size,
                show_cluster_hulls=show_cluster_hulls,
                cluster_label_min_size=cluster_label_min_size,
                cluster_label_model=cluster_label_model,
                cluster_label_base_url=cluster_label_base_url,
                show_node_roles=show_node_roles,
            )
            saved_paths.append(str(png_path))

        if "gexf" in formats:
            gexf_path = output_dir / f"{prefix}.gexf"
            self.to_gexf(gexf_path)
            saved_paths.append(str(gexf_path))

        return saved_paths


    def build_individual_graphs(
        self,
        relationships: Union[
            List[Relationship],
            List[NormalizedRelationship],
            List[CausalRelationship],
        ],
        output_dir: Union[str, Path],
        formats: List[str] = None,
        layout: str = "spring",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        umap_min_dist: float = 0.3,
        color_by: str = "polarity",
        labels_mode: str = "all",
        show_legend: bool = True,
        layout_spacing: float = 1.0,
        min_edge_weight: int = 1,
        cluster_nodes: bool = False,
        min_cluster_size: int = 3,
        show_cluster_hulls: bool = True,
        cluster_label_min_size: int = 5,
        cluster_label_model: Optional[str] = None,
        cluster_label_base_url: str = "http://localhost:11434",
        show_node_roles: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Build and save one graph per text_id.

        Args:
            relationships: List of relationship objects.
            output_dir: Output directory for all graphs.
            formats: Output formats (default: ["png"]).
            layout: Layout algorithm.
            embedding_model: Model for semantic layout.
            umap_min_dist: UMAP min_dist parameter for semantic layout.
            color_by: Edge coloring mode.
            labels_mode: Label display mode.
            show_legend: Whether to show legend.
            layout_spacing: Node spacing multiplier.
            min_edge_weight: Minimum edge weight.
            cluster_nodes: Whether to cluster nodes and color by cluster.
            min_cluster_size: Minimum nodes per cluster.
            show_cluster_hulls: Whether to draw convex hulls around clusters.
            cluster_label_min_size: Minimum cluster size to show label (reduces clutter). Default 5.
            cluster_label_model: LLM model for generating cluster labels. If None, uses generic labels.
            cluster_label_base_url: Ollama base URL for cluster labeling.
            show_node_roles: Whether to use different shapes for source-only, target-only, and both nodes.

        Returns:
            Dict mapping text_id to list of saved file paths.
        """
        if formats is None:
            formats = ["png"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Group relationships by text_id
        by_text: Dict[str, List] = defaultdict(list)
        for rel in relationships:
            if hasattr(rel, "text_ids") and rel.text_ids:
                for tid in rel.text_ids:
                    by_text[tid].append(rel)
            elif hasattr(rel, "text_id") and rel.text_id:
                by_text[rel.text_id].append(rel)
            else:
                by_text["unknown"].append(rel)

        logger.info(f"Building individual graphs for {len(by_text)} text_ids")

        results = {}
        for text_id, rels in by_text.items():
            if len(rels) == 0:
                continue

            # Build graph for this text_id
            self.build(rels, min_edge_weight=min_edge_weight)

            # Create safe filename
            safe_text_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in text_id)
            prefix = f"graph_{safe_text_id}"

            # Save
            saved_paths = self.save(
                output_dir,
                prefix=prefix,
                formats=formats,
                layout=layout,
                embedding_model=embedding_model,
                umap_min_dist=umap_min_dist,
                color_by=color_by,
                labels_mode=labels_mode,
                show_legend=show_legend,
                layout_spacing=layout_spacing,
                cluster_nodes=cluster_nodes,
                min_cluster_size=min_cluster_size,
                show_cluster_hulls=show_cluster_hulls,
                cluster_label_min_size=cluster_label_min_size,
                cluster_label_model=cluster_label_model,
                cluster_label_base_url=cluster_label_base_url,
                show_node_roles=show_node_roles,
            )
            results[text_id] = saved_paths

            logger.info(f"Built graph for {text_id}: {len(rels)} relationships")

        return results


def build_relationship_graph(
    relationships: Union[
        List[Relationship],
        List[NormalizedRelationship],
        List[CausalRelationship],
    ],
    include_causal: bool = True,
    min_edge_weight: int = 1,
) -> RelationshipGraphData:
    """
    Convenience function to build a relationship graph.

    Args:
        relationships: List of relationship objects.
        include_causal: Include causal attributes if available.
        min_edge_weight: Minimum edge weight to include.

    Returns:
        RelationshipGraphData with nodes, edges, and metrics.
    """
    graph = RelationshipGraph()
    return graph.build(relationships, include_causal, min_edge_weight)
