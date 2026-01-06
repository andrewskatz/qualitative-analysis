"""
CLI for domain mapping, normalization, and graph generation.

Entry point: qualitative-domains
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .domains import (
    DomainExtractor,
    DomainNormalizer,
    DomainGraph,
    DomainMappedInstance,
    NormalizationResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="qualitative-domains",
        description="Domain mapping and normalization for figurative language analysis.",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # ==================== EXTRACT SUBCOMMAND ====================
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract source/target domains from figurative instances.",
    )
    extract_parser.add_argument(
        "input_csv",
        help="Path to CSV file containing figurative instances.",
    )
    extract_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: <input>_domains.csv)",
    )
    extract_parser.add_argument(
        "--text-col",
        default="text",
        help="Column name for figurative text (default: text)",
    )
    extract_parser.add_argument(
        "--type-col",
        default="type",
        help="Column name for figurative type (default: type)",
    )
    extract_parser.add_argument(
        "--window-col",
        default=None,
        help="Column name for window context (optional)",
    )
    extract_parser.add_argument(
        "--id-col",
        default=None,
        help="Column name for text ID (optional)",
    )
    extract_parser.add_argument(
        "--multi-level",
        action="store_true",
        help="Extract domains at multiple abstraction levels",
    )
    extract_parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model name",
    )
    extract_parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    extract_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file path for resumable processing",
    )
    extract_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N items",
    )
    extract_parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Print LLM prompts and responses to the terminal",
    )
    
    # ==================== NORMALIZE SUBCOMMAND ====================
    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize/cluster domain labels.",
    )
    normalize_parser.add_argument(
        "input_csv",
        help="Path to CSV file with domain-mapped instances.",
    )
    normalize_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: <input>_normalized.csv)",
    )
    normalize_parser.add_argument(
        "--conservativeness",
        default="moderate",
        choices=["conservative", "moderate", "aggressive", "custom"],
        help="Clustering conservativeness (default: moderate)",
    )
    normalize_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom similarity threshold (required if conservativeness=custom)",
    )
    normalize_parser.add_argument(
        "--cluster-mode",
        default="separate",
        choices=["separate", "together"],
        help="Cluster source/target separately or together",
    )
    normalize_parser.add_argument(
        "--canonical-method",
        default="representative",
        choices=["representative", "llm"],
        help="How to generate canonical labels",
    )
    normalize_parser.add_argument(
        "--abstraction-level",
        default=None,
        choices=["specific", "moderate", "abstract"],
        help="Which abstraction level to normalize",
    )
    normalize_parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Sentence embedding model",
    )
    normalize_parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device for embeddings (default: auto)",
    )
    normalize_parser.add_argument(
        "--save-config",
        default=None,
        help="Save normalization config to JSON file",
    )
    normalize_parser.add_argument(
        "--load-config",
        default=None,
        help="Load and apply normalization from JSON file",
    )
    
    # ==================== GRAPH SUBCOMMAND ====================
    graph_parser = subparsers.add_parser(
        "graph",
        help="Generate domain relationship graph.",
    )
    graph_parser.add_argument(
        "input_csv",
        help="Path to CSV file with domain-mapped instances.",
    )
    graph_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path/prefix (default: <input>_graph)",
    )
    graph_parser.add_argument(
        "--format",
        default="json",
        choices=["json", "csv", "both"],
        help="Output format (default: json)",
    )
    graph_parser.add_argument(
        "--load-normalization",
        default=None,
        help="Apply normalization from JSON file",
    )
    graph_parser.add_argument(
        "--abstraction-level",
        default=None,
        choices=["specific", "moderate", "abstract"],
        help="Which abstraction level to use",
    )
    
    # ==================== PIPELINE SUBCOMMAND ====================
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full pipeline: extract → normalize → graph",
    )
    pipeline_parser.add_argument(
        "input_csv",
        help="Path to CSV file containing figurative instances.",
    )
    pipeline_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside input)",
    )
    pipeline_parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip normalization step",
    )
    pipeline_parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph generation step",
    )
    pipeline_parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model name",
    )
    pipeline_parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    pipeline_parser.add_argument(
        "--conservativeness",
        default="moderate",
        choices=["conservative", "moderate", "aggressive"],
        help="Clustering conservativeness",
    )
    pipeline_parser.add_argument(
        "--graph-format",
        default="both",
        choices=["json", "csv", "both"],
        help="Graph output format",
    )
    pipeline_parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Print LLM prompts and responses to the terminal",
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "extract":
            return asyncio.run(_run_extract(args))
        elif args.command == "normalize":
            return _run_normalize(args)
        elif args.command == "graph":
            return _run_graph(args)
        elif args.command == "pipeline":
            return asyncio.run(_run_pipeline(args))
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


async def _run_extract(args) -> int:
    """Run domain extraction."""
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path = Path(args.output) if args.output else input_path.with_suffix(".domains.csv")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    
    print(f"Extracting domains from: {input_path}")
    print(f"Output: {output_path}")
    
    extractor = DomainExtractor(
        model_name=args.model,
        provider="ollama",
        provider_config={
            "base_url": args.base_url,
            "log_prompts": args.log_llm,
            "log_responses": args.log_llm,
        },
        multi_level=args.multi_level,
    )
    
    def on_progress(current, total):
        print(f"\rProgress: {current}/{total} ({100*current/total:.1f}%)", end="", flush=True)
    
    results = await extractor.extract_from_csv(
        input_path,
        text_col=args.text_col,
        type_col=args.type_col,
        window_col=args.window_col,
        text_id_col=args.id_col,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        on_progress=on_progress,
    )
    
    print()  # Newline after progress
    
    # Save results
    _save_instances_csv(results, output_path, multi_level=args.multi_level)
    
    print(f"\nExtracted {len(results)} domain mappings")
    print(f"Saved to: {output_path}")
    
    return 0


def _run_normalize(args) -> int:
    """Run domain normalization."""
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_path = Path(args.output) if args.output else input_path.with_suffix(".normalized.csv")
    
    print(f"Normalizing domains from: {input_path}")
    
    # Load instances
    instances = _load_instances_csv(input_path)
    print(f"Loaded {len(instances)} instances")
    
    # Check for loading existing normalization
    if args.load_config:
        normalizer = DomainNormalizer(
            embedding_model=args.embedding_model,
            device=args.device,
        )
        normalization = normalizer.load(Path(args.load_config))
        print(f"Loaded normalization from: {args.load_config}")
    else:
        # Run normalization
        normalizer = DomainNormalizer(
            embedding_model=args.embedding_model,
            device=args.device,
        )
        
        print(f"Clustering with {args.conservativeness} conservativeness...")
        normalization = normalizer.normalize(
            instances,
            conservativeness=args.conservativeness,
            similarity_threshold=args.threshold,
            cluster_mode=args.cluster_mode,
            canonical_method=args.canonical_method,
            abstraction_level=args.abstraction_level,
        )
        
        print(f"Created {len(normalization.source_clusters)} source clusters, "
              f"{len(normalization.target_clusters)} target clusters")
        
        if args.save_config:
            normalizer.save(normalization, Path(args.save_config))
            print(f"Saved normalization config to: {args.save_config}")
    
    # Apply normalization and save
    _save_normalized_csv(instances, normalization, output_path, args.abstraction_level)
    print(f"Saved normalized instances to: {output_path}")
    
    return 0


def _run_graph(args) -> int:
    """Run graph generation."""
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_prefix = args.output or str(input_path.with_suffix(""))
    output_dir = Path(output_prefix).parent
    prefix = Path(output_prefix).name
    
    print(f"Generating graph from: {input_path}")
    
    # Load instances
    instances = _load_instances_csv(input_path)
    print(f"Loaded {len(instances)} instances")
    
    # Load normalization if specified
    normalization = None
    if args.load_normalization:
        normalizer = DomainNormalizer()
        normalization = normalizer.load(Path(args.load_normalization))
        print(f"Loaded normalization from: {args.load_normalization}")
    
    # Generate graph
    graph = DomainGraph(instances, normalization)
    saved_paths = graph.save(
        output_dir,
        format=args.format,
        prefix=prefix,
        abstraction_level=args.abstraction_level,
    )
    
    stats = graph._graph_data.stats
    print(f"\nGraph statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Edges: {stats['edge_count']}")
    print(f"  Normalized: {stats['is_normalized']}")
    
    print(f"\nSaved to:")
    for p in saved_paths:
        print(f"  {p}")
    
    return 0


async def _run_pipeline(args) -> int:
    """Run full pipeline: extract → normalize → graph."""
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = output_dir / f"domains_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running pipeline on: {input_path}")
    print(f"Output directory: {run_dir}")
    
    # Step 1: Extract
    print("\n=== Step 1: Domain Extraction ===")
    extractor = DomainExtractor(
        model_name=args.model,
        provider="ollama",
        provider_config={
            "base_url": args.base_url,
            "log_prompts": getattr(args, 'log_llm', False),
            "log_responses": getattr(args, 'log_llm', False),
        },
        multi_level=True,
    )
    
    def on_progress(current, total):
        print(f"\rProgress: {current}/{total} ({100*current/total:.1f}%)", end="", flush=True)
    
    instances = await extractor.extract_from_csv(
        input_path,
        on_progress=on_progress,
    )
    print()
    
    domains_path = run_dir / "domains.csv"
    _save_instances_csv(instances, domains_path, multi_level=True)
    print(f"Saved domains to: {domains_path}")
    
    # Step 2: Normalize (optional)
    normalization = None
    if not args.skip_normalize:
        print("\n=== Step 2: Domain Normalization ===")
        normalizer = DomainNormalizer()
        normalization = normalizer.normalize(
            instances,
            conservativeness=args.conservativeness,
        )
        
        norm_path = run_dir / "normalization.json"
        normalizer.save(normalization, norm_path)
        print(f"Saved normalization to: {norm_path}")
        
        normalized_path = run_dir / "normalized.csv"
        _save_normalized_csv(instances, normalization, normalized_path)
        print(f"Saved normalized instances to: {normalized_path}")
    
    # Step 3: Graph (optional)
    if not args.skip_graph:
        print("\n=== Step 3: Graph Generation ===")
        graph = DomainGraph(instances, normalization)
        saved_paths = graph.save(
            run_dir,
            format=args.graph_format,
            prefix="graph",
        )
        print(f"Saved graph to: {saved_paths}")
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Results in: {run_dir}")
    
    return 0


# ==================== UTILITY FUNCTIONS ====================

def _save_instances_csv(
    instances: List[DomainMappedInstance],
    path: Path,
    multi_level: bool = False,
) -> None:
    """Save domain-mapped instances to CSV."""
    fieldnames = [
        "text_id", "text", "type", "confidence", "explanation",
        "window_index", "window_text",
        "source_domain", "target_domain", "mapping_explanation", "domain_confidence",
    ]
    
    if multi_level:
        fieldnames.extend([
            "source_specific", "source_moderate", "source_abstract",
            "target_specific", "target_moderate", "target_abstract",
        ])
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for inst in instances:
            row = {
                "text_id": inst.text_id,
                "text": inst.text,
                "type": inst.type,
                "confidence": inst.confidence,
                "explanation": inst.explanation,
                "window_index": inst.window_index,
                "window_text": inst.window_text,
                "source_domain": inst.source_domain,
                "target_domain": inst.target_domain,
                "mapping_explanation": inst.mapping_explanation,
                "domain_confidence": inst.domain_confidence,
            }
            
            if multi_level:
                row["source_specific"] = inst.source_domain_levels.get("specific", "")
                row["source_moderate"] = inst.source_domain_levels.get("moderate", "")
                row["source_abstract"] = inst.source_domain_levels.get("abstract", "")
                row["target_specific"] = inst.target_domain_levels.get("specific", "")
                row["target_moderate"] = inst.target_domain_levels.get("moderate", "")
                row["target_abstract"] = inst.target_domain_levels.get("abstract", "")
            
            writer.writerow(row)


def _load_instances_csv(path: Path) -> List[DomainMappedInstance]:
    """Load domain-mapped instances from CSV."""
    instances = []
    
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            source_levels = {}
            target_levels = {}
            
            # Check for multi-level columns
            if "source_specific" in row:
                source_levels = {
                    "specific": row.get("source_specific", ""),
                    "moderate": row.get("source_moderate", ""),
                    "abstract": row.get("source_abstract", ""),
                }
            if "target_specific" in row:
                target_levels = {
                    "specific": row.get("target_specific", ""),
                    "moderate": row.get("target_moderate", ""),
                    "abstract": row.get("target_abstract", ""),
                }
            
            instances.append(DomainMappedInstance(
                text_id=row.get("text_id", ""),
                text=row.get("text", ""),
                type=row.get("type", "unknown"),
                confidence=float(row.get("confidence", 0) or 0),
                explanation=row.get("explanation", ""),
                window_index=int(row.get("window_index", 0) or 0),
                window_text=row.get("window_text", ""),
                source_domain=row.get("source_domain", ""),
                target_domain=row.get("target_domain", ""),
                mapping_explanation=row.get("mapping_explanation", ""),
                domain_confidence=float(row.get("domain_confidence", 0) or 0),
                source_domain_levels=source_levels,
                target_domain_levels=target_levels,
            ))
    
    return instances


def _save_normalized_csv(
    instances: List[DomainMappedInstance],
    normalization: NormalizationResult,
    path: Path,
    abstraction_level: Optional[str] = None,
) -> None:
    """Save instances with normalized domain columns."""
    fieldnames = [
        "text_id", "text", "type",
        "source_domain", "source_domain_normalized",
        "target_domain", "target_domain_normalized",
        "mapping_explanation",
    ]
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for inst in instances:
            # Get domain at abstraction level
            source = inst.source_domain
            target = inst.target_domain
            if abstraction_level:
                source = inst.source_domain_levels.get(abstraction_level) or source
                target = inst.target_domain_levels.get(abstraction_level) or target
            
            writer.writerow({
                "text_id": inst.text_id,
                "text": inst.text,
                "type": inst.type,
                "source_domain": source,
                "source_domain_normalized": normalization.source_mapping.get(source, source),
                "target_domain": target,
                "target_domain_normalized": normalization.target_mapping.get(target, target),
                "mapping_explanation": inst.mapping_explanation,
            })


if __name__ == "__main__":
    sys.exit(main())
