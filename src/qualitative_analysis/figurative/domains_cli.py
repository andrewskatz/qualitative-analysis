"""
CLI for domain mapping, normalization, and graph generation.

This module can be used standalone via `qualitative-domains` command (deprecated)
or through the unified CLI via `qa figurative map|normalize|graph|pipeline`.
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

from qualitative_analysis.core.cli_utils import (
    emit_deprecation_warning,
    resolve_nested_output_dir,
)
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


# ==================== ARGUMENT FUNCTIONS ====================
# These are used by both the standalone CLI and the unified CLI

def add_map_args(parser: argparse.ArgumentParser) -> None:
    """Add domain mapping arguments to a parser."""
    parser.add_argument(
        "input_csv",
        help="Path to CSV file containing figurative instances (for example, detector instances CSV).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: <input>_domains.csv)",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Column name for figurative text (default: text; auto-detects instance_text)",
    )
    parser.add_argument(
        "--type-col",
        default="type",
        help="Column name for figurative type (default: type)",
    )
    parser.add_argument(
        "--window-col",
        default=None,
        help="Column name for window context (optional; auto-detects window_text from detector CSV)",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Column name for text ID (optional; auto-detects text_id from detector CSV)",
    )
    parser.add_argument(
        "--multi-level",
        action="store_true",
        help="Map domains at multiple abstraction levels",
    )
    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model name",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file path for resumable processing",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Save checkpoint every N items",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print LLM prompts and responses to the terminal",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="(Alias for --verbose) Print LLM prompts and responses to the terminal",
    )


def add_normalize_args(parser: argparse.ArgumentParser) -> None:
    """Add domain normalization arguments to a parser."""
    parser.add_argument(
        "input_csv",
        help="Path to CSV file with domain-mapped instances.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: <input>_normalized.csv)",
    )
    parser.add_argument(
        "--merge-threshold",
        default="normal",
        help="Similarity threshold for merging domains. Use 'strict' (0.85), 'normal' (0.75), 'loose' (0.60), or a number 0.0-1.0 (default: normal)",
    )
    parser.add_argument(
        "--cluster-mode",
        default="separate",
        choices=["separate", "together"],
        help="Cluster source/target separately or together",
    )
    parser.add_argument(
        "--canonical-method",
        default="representative",
        choices=["representative", "llm"],
        help="How to generate canonical labels",
    )
    parser.add_argument(
        "--abstraction-level",
        default=None,
        choices=["specific", "moderate", "abstract"],
        help="Which abstraction level to normalize",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Sentence embedding model",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device for embeddings (default: auto)",
    )
    parser.add_argument(
        "--save-config",
        default=None,
        help="Save normalization config to JSON file",
    )
    parser.add_argument(
        "--load-config",
        default=None,
        help="Load and apply normalization from JSON file",
    )
    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="LLM model for canonical-method=llm",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print LLM prompts and responses to the terminal",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="(Alias for --verbose) Print LLM prompts and responses to the terminal",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file path for resumable LLM processing",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N clusters (default: 10)",
    )


def add_graph_args(parser: argparse.ArgumentParser) -> None:
    """Add graph generation arguments to a parser."""
    parser.add_argument(
        "input_csv",
        help="Path to CSV file with domain-mapped instances.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path/prefix (default: <input>_graph)",
    )
    parser.add_argument(
        "--format",
        default="json",
        choices=["json", "csv", "both"],
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--load-normalization",
        default=None,
        help="Apply normalization from JSON file",
    )
    parser.add_argument(
        "--abstraction-level",
        default=None,
        choices=["specific", "moderate", "abstract"],
        help="Which abstraction level to use",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate PNG visualization of the graph",
    )
    parser.add_argument(
        "--layout",
        default="spring",
        choices=["spring", "circular", "kamada_kawai", "semantic"],
        help="Layout algorithm for PNG visualization. 'semantic' positions nodes by embedding similarity. (default: spring)",
    )
    parser.add_argument(
        "--cluster-labels",
        action="store_true",
        help="Enable cluster-based region labeling (reduces visual clutter)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum nodes per cluster (default: 3)",
    )
    parser.add_argument(
        "--noise-handling",
        default="label",
        choices=["label", "hide", "other"],
        help="How to handle outlier nodes: label (small text), hide, or group as 'other' (default: label)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model for generating cluster labels (optional)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print LLM prompts and responses to the terminal",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="(Alias for --verbose) Print LLM prompts and responses to the terminal",
    )
    parser.add_argument(
        "--min-label-size",
        type=int,
        default=None,
        help="Only label clusters with at least N members (reduces clutter)",
    )
    parser.add_argument(
        "--max-cluster-labels",
        type=int,
        default=None,
        help="Only show labels for the N largest clusters",
    )
    parser.add_argument(
        "--hide-node-labels",
        action="store_true",
        help="Hide individual node labels, show only cluster region labels",
    )


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Add full pipeline arguments to a parser."""
    parser.add_argument(
        "input_csv",
        help="Path to CSV file containing figurative instances.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside input)",
    )
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip normalization step",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph generation step",
    )
    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="Ollama model name",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "--conservativeness",
        default="moderate",
        choices=["conservative", "moderate", "aggressive"],
        help="Clustering conservativeness",
    )
    parser.add_argument(
        "--graph-format",
        default="both",
        choices=["json", "csv", "both"],
        help="Graph output format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print LLM prompts and responses to the terminal",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="(Alias for --verbose) Print LLM prompts and responses to the terminal",
    )


# ==================== MAIN ENTRY POINT ====================


def main() -> int:
    """
    Legacy entry point for standalone CLI.
    
    DEPRECATED: Use `qa figurative map|normalize|graph|pipeline` instead.
    """
    emit_deprecation_warning("qualitative-domains", "qa figurative <command>")
    
    parser = argparse.ArgumentParser(
        prog="qualitative-domains",
        description="Domain mapping and normalization for figurative language analysis.",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Add subparsers using the extracted functions
    map_parser = subparsers.add_parser("map", help="Map source/target domains from figurative instances.")
    add_map_args(map_parser)
    
    normalize_parser = subparsers.add_parser("normalize", help="Normalize/cluster domain labels.")
    add_normalize_args(normalize_parser)
    
    graph_parser = subparsers.add_parser("graph", help="Generate domain relationship graph.")
    add_graph_args(graph_parser)
    
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline: extract → normalize → graph")
    add_pipeline_args(pipeline_parser)
    
    args = parser.parse_args()
    
    try:
        if args.command == "map":
            return asyncio.run(run_map(args))
        elif args.command == "normalize":
            return run_normalize(args)
        elif args.command == "graph":
            return run_graph(args)
        elif args.command == "pipeline":
            return asyncio.run(run_pipeline(args))
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


# ==================== HANDLER FUNCTIONS ====================
# These are the actual implementation functions, callable from unified CLI

async def run_map(args) -> int:
    """Run domain mapping."""
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Resolve output directory with nested structure
    output_dir, timestamp = resolve_nested_output_dir(
        input_path, 
        "map",
        output_dir=getattr(args, 'output_dir', None),
    )
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"domain_mappings_{timestamp}.csv"
    
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    
    start_time = datetime.now()
    print(f"Mapping domains from: {input_path}")
    print(f"Output: {output_path}")

    # Auto-detect text column if using default "text"
    text_col = args.text_col
    window_col = args.window_col
    id_col = args.id_col
    
    # Read header once to check for columns
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            
            # Auto-detect text column
            if text_col == "text":
                if "instance_text" in header and "text" not in header:
                    text_col = "instance_text"
                    print(f"Auto-detected text column: '{text_col}'")
            
            # Auto-detect window column if not specified
            if window_col is None:
                if "window_text" in header:
                    window_col = "window_text"
                    print(f"Auto-detected window column: '{window_col}'")

            # Auto-detect ID column if not specified
            if id_col is None:
                if "text_id" in header:
                    id_col = "text_id"
                    print(f"Auto-detected ID column: '{id_col}'")
                elif "id" in header:
                    id_col = "id"
                    print(f"Auto-detected ID column: '{id_col}'")
                    
        except StopIteration:
            pass

    extractor = DomainExtractor(
        model_name=args.model,
        provider="ollama",
        provider_config={
            "base_url": args.base_url,
            "log_prompts": getattr(args, 'verbose', False) or getattr(args, 'log_llm', False),
            "log_responses": getattr(args, 'verbose', False) or getattr(args, 'log_llm', False),
        },
        multi_level=args.multi_level,
    )
    
    def on_progress(current, total):
        print(f"\rProgress: {current}/{total} ({100*current/total:.1f}%)", end="", flush=True)
    
    results = await extractor.extract_from_csv(
        input_path,
        text_col=text_col,
        type_col=args.type_col,
        window_col=window_col,
        text_id_col=id_col,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=args.checkpoint_interval,
        on_progress=on_progress,
    )
    
    print()  # Newline after progress
    
    # Save results
    _save_instances_csv(results, output_path, multi_level=args.multi_level)
    
    end_time = datetime.now()
    
    # Write metadata JSON
    metadata = {
        "step": "domain_mapping",
        "run_id": f"mapping_{timestamp}",
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "cli_args": {
            "input_csv": str(input_path),
            "output": str(output_path),
            "text_col": args.text_col,
            "type_col": args.type_col,
            "window_col": args.window_col,
            "id_col": args.id_col,
            "multi_level": args.multi_level,
            "model": args.model,
            "base_url": args.base_url,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "checkpoint_interval": args.checkpoint_interval,
        },
        "stats": {
            "instances_processed": len(results),
            "domains_mapped": sum(1 for r in results if r.source_domain or r.target_domain),
        },
        "outputs": {
            "domain_mappings_csv": str(output_path),
        },
        "package_version": "0.1.0",
    }
    
    metadata_path = output_path.parent / f"mapping_metadata_{timestamp}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMapped {len(results)} domain mappings")
    print(f"Saved to: {output_path}")
    print(f"Wrote metadata: {metadata_path}")
    
    return 0


def run_normalize(args) -> int:
    """Run domain normalization."""
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Resolve output directory with nested structure
    output_dir, timestamp = resolve_nested_output_dir(
        input_path,
        "normalize",
        output_dir=getattr(args, 'output_dir', None),
    )
    start_time = datetime.now()
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = output_dir / f"normalized_{timestamp}.csv"
    
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
        
        # Parse merge threshold
        merge_threshold = args.merge_threshold
        threshold_presets = {"strict": 0.85, "normal": 0.75, "loose": 0.60}
        
        if merge_threshold in threshold_presets:
            threshold_value = threshold_presets[merge_threshold]
            print(f"Clustering with {merge_threshold} merge threshold ({threshold_value})...")
        else:
            try:
                threshold_value = float(merge_threshold)
                if not 0.0 <= threshold_value <= 1.0:
                    raise ValueError("Threshold must be between 0.0 and 1.0")
                print(f"Clustering with custom merge threshold ({threshold_value})...")
            except ValueError:
                print(f"Invalid merge-threshold: {merge_threshold}. Use 'strict', 'normal', 'loose', or a number 0.0-1.0")
                return 1
        
        # Build LLM config if using LLM canonical method
        llm_config = None
        if args.canonical_method == "llm":
            _log_llm = getattr(args, 'verbose', False) or getattr(args, 'log_llm', False)
            llm_config = {
                "base_url": args.base_url,
                "log_prompts": _log_llm,
                "log_responses": _log_llm,
            }
        
        normalization = normalizer.normalize(
            instances,
            conservativeness="custom",
            similarity_threshold=threshold_value,
            cluster_mode=args.cluster_mode,
            canonical_method=args.canonical_method,
            abstraction_level=args.abstraction_level,
            llm_model=args.model if args.canonical_method == "llm" else None,
            llm_provider="ollama" if args.canonical_method == "llm" else None,
            llm_config=llm_config,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            checkpoint_interval=args.checkpoint_interval,
        )
        
        print(f"Created {len(normalization.source_clusters)} source clusters, "
              f"{len(normalization.target_clusters)} target clusters")
        
        # Always save normalization config (auto-save)
        config_path = Path(args.save_config) if args.save_config else output_path.parent / f"normalization_config_{timestamp}.json"
        normalizer.save(normalization, config_path)
        print(f"Saved normalization config to: {config_path}")
    
    # Apply normalization and save
    _save_normalized_csv(instances, normalization, output_path, args.abstraction_level)
    print(f"Saved normalized instances to: {output_path}")
    
    end_time = datetime.now()
    
    # Write metadata JSON
    metadata = {
        "step": "domain_normalization",
        "run_id": f"normalize_{timestamp}",
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "cli_args": {
            "input_csv": str(input_path),
            "output": str(output_path),
            "merge_threshold": args.merge_threshold,
            "cluster_mode": args.cluster_mode,
            "canonical_method": args.canonical_method,
            "canonical_model": args.model if args.canonical_method == "llm" else None,
            "abstraction_level": args.abstraction_level or "moderate",
            "embedding_model": args.embedding_model,
            "device": args.device,
            "load_config": args.load_config,
            "save_config": args.save_config,
        },
        "stats": {
            "instances_processed": len(instances),
            "source_clusters": len(normalization.source_clusters),
            "target_clusters": len(normalization.target_clusters),
        },
        "outputs": {
            "normalized_csv": str(output_path),
            "normalization_config": str(config_path) if not args.load_config else None,
        },
        "package_version": "0.1.0",
    }
    
    metadata_path = output_path.parent / f"normalize_metadata_{timestamp}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {metadata_path}")
    
    return 0


def run_graph(args) -> int:
    """Run graph generation."""
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Resolve output directory with nested structure
    output_dir, timestamp = resolve_nested_output_dir(
        input_path,
        "graph",
        output_dir=getattr(args, 'output_dir', None),
    )
    start_time = datetime.now()
    
    # Use output_dir for graph files
    if args.output:
        output_prefix = args.output
        graph_output_dir = Path(output_prefix).parent
        prefix = Path(output_prefix).name
    else:
        graph_output_dir = output_dir
        prefix = f"graph_{timestamp}"
        output_prefix = str(graph_output_dir / prefix)
    
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
        graph_output_dir,
        format=args.format,
        prefix=prefix,
        abstraction_level=args.abstraction_level,
        visualize=args.visualize and not getattr(args, 'cluster_labels', False),
        layout=args.layout,
    )
    
    # Generate clustered visualization if requested
    if getattr(args, 'cluster_labels', False) and args.visualize:
        clustered_path = graph_output_dir / f"{prefix}_clustered.png"
        provider_config = {
            "base_url": getattr(args, 'base_url', 'http://localhost:11434'),
            "log_prompts": getattr(args, 'log_llm', False),
            "log_responses": getattr(args, 'log_llm', False),
        }
        graph.to_png_clustered(
            clustered_path,
            min_cluster_size=getattr(args, 'min_cluster_size', 3),
            noise_handling=getattr(args, 'noise_handling', 'label'),
            model=getattr(args, 'model', None),
            provider_config=provider_config if getattr(args, 'model', None) else None,
            min_label_size=getattr(args, 'min_label_size', None),
            max_cluster_labels=getattr(args, 'max_cluster_labels', None),
            hide_node_labels=getattr(args, 'hide_node_labels', False),
        )
        saved_paths.append(clustered_path)
        print(f"Generated clustered visualization: {clustered_path}")
    
    end_time = datetime.now()
    
    stats = graph._graph_data.stats
    print(f"\nGraph statistics:")
    print(f"  Nodes: {stats['node_count']}")
    print(f"  Edges: {stats['edge_count']}")
    print(f"  Normalized: {stats['is_normalized']}")
    
    # Write metadata JSON
    metadata = {
        "step": "graph_generation",
        "run_id": f"graph_{timestamp}",
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "cli_args": {
            "input_csv": str(input_path),
            "output_prefix": output_prefix,
            "format": args.format,
            "load_normalization": args.load_normalization,
            "abstraction_level": args.abstraction_level,
        },
        "stats": stats,
        "outputs": {
            "files": [str(p) for p in saved_paths],
        },
        "package_version": "0.1.0",
    }
    
    metadata_path = output_dir / f"graph_metadata_{timestamp}.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved to:")
    for p in saved_paths:
        print(f"  {p}")
    print(f"Wrote metadata: {metadata_path}")
    
    return 0


async def run_pipeline(args) -> int:
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
