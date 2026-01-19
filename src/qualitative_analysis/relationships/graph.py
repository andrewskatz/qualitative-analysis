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

from .models import (
    Relationship,
    NormalizedRelationship,
    CausalRelationship,
    RelationshipGraphNode,
    RelationshipGraphEdge,
    RelationshipGraphData,
)

logger = logging.getLogger(__name__)


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

    def _compute_semantic_positions(
        self,
        node_labels: List[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute 2D positions using UMAP projection of node label embeddings.

        Args:
            node_labels: List of node label strings.
            embedding_model: SentenceTransformer model name.
            device: Device for embeddings (None for auto-detect).

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
        logger.info(f"[SemanticLayout] Running UMAP with n_neighbors={n_neighbors}")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
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
        embedding_model: str = "all-MiniLM-L6-v2",
        node_size_scale: float = 1.0,
        edge_width_scale: float = 1.0,
        show_edge_labels: bool = False,
        title: Optional[str] = None,
    ) -> str:
        """
        Export graph as PNG visualization.

        Args:
            path: Output path for PNG file.
            figsize: Figure size (width, height) in inches.
            dpi: Resolution.
            layout: Layout algorithm ("spring", "circular", "kamada_kawai", "semantic").
            embedding_model: Model for semantic layout.
            node_size_scale: Multiplier for node sizes.
            edge_width_scale: Multiplier for edge widths.
            show_edge_labels: Whether to show relationship types on edges.
            title: Optional title for the graph.

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
        if layout == "semantic":
            node_labels = [G.nodes[n].get("label", n) for n in G.nodes()]
            semantic_positions = self._compute_semantic_positions(node_labels, embedding_model)
            if semantic_positions:
                pos = {}
                for node in G.nodes():
                    label = G.nodes[node].get("label", node).lower()
                    pos[node] = semantic_positions.get(label, (0.5, 0.5))
            else:
                logger.warning("Semantic layout failed, using spring layout")
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        self._positions = pos

        # Node sizes based on frequency/degree
        node_sizes = []
        for node in G.nodes():
            freq = G.nodes[node].get("frequency", 1)
            size = max(300, min(2000, freq * 100)) * node_size_scale
            node_sizes.append(size)

        # Node colors - single color for simplicity
        node_color = "#6366f1"  # Indigo

        # Edge widths based on weight
        edge_list = list(G.edges())
        if edge_list:
            edge_weights = [G[u][v].get("weight", 1) for u, v in edge_list]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [max(0.5, (w / max_weight) * 3) * edge_width_scale for w in edge_weights]
        else:
            edge_widths = []

        # Draw edges
        if edge_list:
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                edge_color="#94a3b8",
                width=edge_widths,
                alpha=0.5,
                arrows=True,
                arrowsize=15,
                connectionstyle="arc3,rad=0.1",
            )

        # Draw edge labels if requested
        if show_edge_labels and edge_list:
            edge_labels = {(u, v): G[u][v].get("type", "") for u, v in edge_list}
            nx.draw_networkx_edge_labels(
                G, pos, edge_labels=edge_labels, ax=ax,
                font_size=6,
                font_color="#64748b",
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_color,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors="white",
            linewidths=1.5,
        )

        # Draw labels
        labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels=labels, ax=ax,
            font_size=8,
            font_weight="bold",
        )

        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")
        else:
            stats = self._graph_data.metrics_summary
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
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> List[str]:
        """
        Save graph to multiple formats.

        Args:
            output_dir: Output directory.
            prefix: Filename prefix.
            formats: List of formats ("json", "csv", "png", "gexf"). Default: ["json", "csv"].
            layout: Layout for PNG visualization.
            embedding_model: Model for semantic layout.

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
            self.to_png(png_path, layout=layout, embedding_model=embedding_model)
            saved_paths.append(str(png_path))

        if "gexf" in formats:
            gexf_path = output_dir / f"{prefix}.gexf"
            self.to_gexf(gexf_path)
            saved_paths.append(str(gexf_path))

        return saved_paths


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
