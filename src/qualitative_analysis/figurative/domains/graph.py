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
    
    def save(
        self,
        output_dir: Path,
        format: str = "json",
        prefix: str = "domain_graph",
        abstraction_level: Optional[str] = None,
    ) -> List[Path]:
        """
        Save graph to file(s).
        
        Args:
            output_dir: Directory to save files
            format: "json", "csv", or "both"
            prefix: Filename prefix
            abstraction_level: For multi-level instances, which level to use
            
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
        
        return saved_paths
