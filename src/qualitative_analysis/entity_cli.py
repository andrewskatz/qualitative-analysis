"""
CLI commands for entity analysis.

Provides commands for:
- Entity scoring (`qa entity score`) - Multi-dimensional LLM-based scoring
- Entity consolidation (`qa entity consolidate`) - Semantic deduplication
- Entity visualization (`qa entity viz`) - Ternary plots and radar charts
"""

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from qualitative_analysis.core.cli_utils import PACKAGE_VERSION

logger = logging.getLogger(__name__)


# =============================================================================
# ENTITY SCORING COMMAND
# =============================================================================

def add_entity_score_args(parser: argparse.ArgumentParser) -> None:
    """
    Add entity scoring arguments to a parser.

    This is used by the unified CLI via `qa entity score`.
    """
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with entities to score.",
    )
    parser.add_argument(
        "--entity-col",
        default="entity",
        help="Column name for entities (default: entity).",
    )
    parser.add_argument(
        "--context-col",
        default="context",
        help="Column name for context (default: context).",
    )
    parser.add_argument(
        "--text-id-col",
        default="text_id",
        help="Column name for text IDs (default: text_id).",
    )
    parser.add_argument(
        "--dimensions",
        default="sets",
        help="Dimension set: 'sets' for SETS framework, or path to JSON file with custom dimensions (default: sets).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of scoring runs per entity for uncertainty estimation (default: 3).",
    )
    parser.add_argument(
        "--model",
        default="gpt-oss:120b",
        help="LLM model name (default: gpt-oss:120b).",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "openai", "anthropic"],
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL for Ollama provider (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature for generation (default: 0.3).",
    )
    parser.add_argument(
        "--research-context",
        default=None,
        help="Path to JSON file with research context (data_type, data_collection_context, research_question).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates scored_* subdirectory).",
    )
    parser.add_argument(
        "--output-format",
        default="csv,json",
        help="Output formats, comma-separated: csv, json (default: csv,json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entities to score (for testing).",
    )


async def run_entity_score(args: argparse.Namespace) -> int:
    """
    Run entity scoring with the given arguments.

    This scores entities along multiple dimensions using LLM analysis
    with uncertainty quantification through multiple runs.

    Args:
        args: Parsed arguments namespace with scoring configuration

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.entity.scorer import EntityScorer
    from qualitative_analysis.entity.models import DimensionSet, DimensionDefinition
    from qualitative_analysis.core.providers import OllamaProvider

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"scored_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dimensions
    if args.dimensions.lower() == "sets":
        dimension_set = DimensionSet.sets_framework()
        dimensions = dimension_set.dimensions
        print(f"Using SETS framework ({len(dimensions)} dimensions)")
    else:
        # Load from JSON file
        dim_path = Path(args.dimensions)
        if not dim_path.exists():
            raise SystemExit(f"Dimensions file not found: {dim_path}")

        with open(dim_path, "r", encoding="utf-8") as f:
            dim_data = json.load(f)

        if "dimensions" in dim_data:
            dimensions = [DimensionDefinition.from_dict(d) for d in dim_data["dimensions"]]
        else:
            dimensions = [DimensionDefinition.from_dict(d) for d in dim_data]

        print(f"Loaded {len(dimensions)} custom dimensions from {dim_path}")

    # Load research context if provided
    research_context = None
    if args.research_context:
        rc_path = Path(args.research_context)
        if rc_path.exists():
            with open(rc_path, "r", encoding="utf-8") as f:
                research_context = json.load(f)
            print(f"Loaded research context from {rc_path}")

    # Read entities from CSV
    print(f"Reading entities from: {input_path}")
    entities = _read_entities_csv(
        input_path,
        entity_col=args.entity_col,
        context_col=args.context_col,
        text_id_col=args.text_id_col,
        limit=args.limit,
    )

    if not entities:
        print("No entities found in input CSV.")
        return 1

    print(f"Loaded {len(entities)} entities")

    # Initialize LLM provider
    if args.provider == "ollama":
        llm_provider = OllamaProvider(
            model_name=args.model,
            base_url=args.base_url,
        )
    else:
        raise SystemExit(f"Provider '{args.provider}' not yet supported. Use 'ollama'.")

    # Initialize scorer
    scorer = EntityScorer(temperature=args.temperature)

    # Score entities
    print(f"\nScoring {len(entities)} entities with {args.num_runs} runs each...")
    print(f"Model: {args.model} ({args.provider})")
    print(f"Temperature: {args.temperature}")
    print()

    def progress_callback(current: int, total: int):
        print(f"  Progress: {current}/{total} entities scored", end="\r")

    result = await scorer.score_entities(
        entities=entities,
        dimensions=dimensions,
        llm_provider=llm_provider,
        num_runs=args.num_runs,
        research_context=research_context,
        on_progress=progress_callback,
    )

    print()  # Clear progress line

    # Print summary
    stats = result.statistics
    print(f"\nScoring Summary:")
    print(f"  Entities scored: {stats.get('total_entities', 0)}")

    if "dimensions" in stats:
        print(f"\n  Dimension Statistics:")
        for dim_name, dim_stats in stats["dimensions"].items():
            print(f"    {dim_name.upper()}:")
            print(f"      Mean: {dim_stats.get('mean', 0):.1f}")
            print(f"      Median: {dim_stats.get('median', 0):.1f}")
            print(f"      Std Dev: {dim_stats.get('std_dev', 0):.1f}")
            print(f"      Range: {dim_stats.get('min', 0):.1f} - {dim_stats.get('max', 0):.1f}")

    # Parse output formats
    formats = [f.strip().lower() for f in args.output_format.split(",")]

    # Export
    output_prefix = input_path.stem

    if "json" in formats:
        json_path = output_dir / f"{output_prefix}_scores.json"
        scorer.save(result, json_path)
        print(f"\nWrote scores JSON: {json_path}")

    if "csv" in formats:
        csv_path = output_dir / f"{output_prefix}_scores.csv"
        _write_scores_csv(result, csv_path)
        print(f"Wrote scores CSV: {csv_path}")

    return 0


def _read_entities_csv(
    path: Path,
    entity_col: str = "entity",
    context_col: str = "context",
    text_id_col: str = "text_id",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Read entities from CSV file."""
    entities = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        # Check required columns
        if entity_col not in reader.fieldnames:
            raise SystemExit(f"Entity column '{entity_col}' not found. Available: {reader.fieldnames}")

        has_context = context_col in reader.fieldnames
        has_text_id = text_id_col in reader.fieldnames

        if not has_context:
            print(f"Warning: Context column '{context_col}' not found. Using empty context.")

        for row in reader:
            entity = row.get(entity_col, "").strip()
            if not entity:
                continue

            entities.append({
                "entity": entity,
                "context": row.get(context_col, "") if has_context else "",
                "text_id": row.get(text_id_col, "") if has_text_id else "",
            })

            if limit and len(entities) >= limit:
                break

    return entities


def _write_scores_csv(result: Any, path: Path) -> None:
    """Write entity scores to CSV file."""
    if not result.scores:
        return

    # Get all field names from first score
    first_score = result.scores[0].to_flat_dict()
    fieldnames = list(first_score.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for score in result.scores:
            writer.writerow(score.to_flat_dict())


# =============================================================================
# ENTITY CONSOLIDATION COMMAND
# =============================================================================

def add_entity_consolidate_args(parser: argparse.ArgumentParser) -> None:
    """Add entity consolidation arguments."""
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with entities to consolidate.",
    )
    parser.add_argument(
        "--entity-col",
        default="entity",
        help="Column name for entities (default: entity).",
    )
    parser.add_argument(
        "--frequency-col",
        default=None,
        help="Optional column name for frequency counts (for 'frequent' canonical method).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold 0.0-1.0 (default: 0.85). Higher = stricter matching.",
    )
    parser.add_argument(
        "--canonical-method",
        choices=["shortest", "frequent", "representative", "first"],
        default="shortest",
        help="Method for selecting canonical form (default: shortest).",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates consolidated_* subdirectory).",
    )
    parser.add_argument(
        "--output-format",
        default="csv,json",
        help="Output formats, comma-separated: csv, json (default: csv,json).",
    )


async def run_entity_consolidate(args: argparse.Namespace) -> int:
    """
    Run entity consolidation with the given arguments.

    This consolidates (deduplicates) semantically similar entities
    using embedding-based clustering.

    Args:
        args: Parsed arguments namespace with consolidation configuration

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.entity.consolidator import EntityConsolidator

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"consolidated_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read entities from CSV
    print(f"Reading entities from: {input_path}")
    entities, frequencies = _read_entities_for_consolidation(
        input_path,
        entity_col=args.entity_col,
        frequency_col=args.frequency_col,
    )

    if not entities:
        print("No entities found in input CSV.")
        return 1

    print(f"Loaded {len(entities)} entities")

    # Initialize consolidator
    print(f"\nInitializing embedding model: {args.embedding_model}")
    consolidator = EntityConsolidator(embedding_model=args.embedding_model)

    # Run consolidation
    print(f"Consolidating with threshold={args.threshold}, method={args.canonical_method}...")
    result = consolidator.consolidate(
        entities=entities,
        threshold=args.threshold,
        canonical_method=args.canonical_method,
        frequencies=frequencies,
    )

    # Print summary
    stats = result.statistics
    print(f"\nConsolidation Summary:")
    print(f"  Original entities: {stats.get('original_count', 0)}")
    print(f"  Consolidated to: {stats.get('consolidated_count', 0)}")
    print(f"  Reduction: {stats.get('reduction_count', 0)} ({stats.get('reduction_percentage', 0):.1f}%)")
    print(f"  Clusters: {stats.get('cluster_count', 0)}")
    print(f"  Largest cluster: {stats.get('largest_cluster_size', 0)} entities")

    # Show top clusters (those with more than 1 item)
    multi_clusters = [c for c in result.consolidations if len(c.variants) > 1]
    if multi_clusters:
        print(f"\nTop Merged Clusters:")
        for cluster in multi_clusters[:10]:
            print(f"  '{cluster.canonical}' <- {cluster.variants}")

    # Parse output formats
    formats = [f.strip().lower() for f in args.output_format.split(",")]

    # Export
    output_prefix = input_path.stem

    if "json" in formats:
        json_path = output_dir / f"{output_prefix}_consolidation.json"
        consolidator.save(result, json_path)
        print(f"\nWrote consolidation JSON: {json_path}")

    if "csv" in formats:
        # Write mapping CSV
        mapping_path = output_dir / f"{output_prefix}_mapping.csv"
        _write_consolidation_mapping_csv(result, mapping_path)
        print(f"Wrote mapping CSV: {mapping_path}")

        # Write clusters CSV
        clusters_path = output_dir / f"{output_prefix}_clusters.csv"
        _write_consolidation_clusters_csv(result, clusters_path)
        print(f"Wrote clusters CSV: {clusters_path}")

    return 0


def _read_entities_for_consolidation(
    path: Path,
    entity_col: str = "entity",
    frequency_col: Optional[str] = None,
) -> tuple:
    """Read entities from CSV file for consolidation."""
    entities = []
    frequencies: Dict[str, int] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], {}

        if entity_col not in reader.fieldnames:
            raise SystemExit(f"Entity column '{entity_col}' not found. Available: {reader.fieldnames}")

        has_frequency = frequency_col and frequency_col in reader.fieldnames

        for row in reader:
            entity = row.get(entity_col, "").strip()
            if not entity:
                continue

            entities.append(entity)

            if has_frequency:
                try:
                    freq = int(row[frequency_col])
                    frequencies[entity] = frequencies.get(entity, 0) + freq
                except ValueError:
                    pass

    return entities, frequencies if frequencies else None


def _write_consolidation_mapping_csv(result: Any, path: Path) -> None:
    """Write entity mapping to CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_entity", "canonical_entity"])

        for original, canonical in sorted(result.mapping.items()):
            writer.writerow([original, canonical])


def _write_consolidation_clusters_csv(result: Any, path: Path) -> None:
    """Write cluster information to CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical", "variants", "variant_count", "avg_similarity", "confidence"])

        for cluster in result.consolidations:
            writer.writerow([
                cluster.canonical,
                "|".join(cluster.variants),
                len(cluster.variants),
                cluster.avg_similarity,
                cluster.confidence,
            ])


def add_entity_viz_args(parser: argparse.ArgumentParser) -> None:
    """Add entity visualization arguments."""
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with scored entities (from 'qa entity score').",
    )
    parser.add_argument(
        "--type",
        choices=["ternary", "radar"],
        default="ternary",
        help="Visualization type: 'ternary' for 3-dimension plots, 'radar' for multi-dimension (default: ternary).",
    )
    parser.add_argument(
        "--dimensions",
        default="social,ecological,technological",
        help="Dimensions to visualize, comma-separated (default: social,ecological,technological).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for visualization PNG (default: auto-generated in same directory).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title for the visualization.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide entity labels (useful for many entities).",
    )
    parser.add_argument(
        "--direct-labels",
        action="store_true",
        help="Use direct labels instead of numbered labels with legend.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=10,
        help="Maximum entities to show on radar chart (default: 10).",
    )
    parser.add_argument(
        "--figsize",
        default="14,10",
        help="Figure size as 'width,height' in inches (default: 14,10).",
    )


async def run_entity_viz(args: argparse.Namespace) -> int:
    """
    Run entity visualization with the given arguments.

    This generates visualizations (ternary plots, radar charts) from
    scored entity data.

    Args:
        args: Parsed arguments namespace with visualization configuration

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.entity.visualizer import EntityVisualizer

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Parse dimensions
    dimension_names = [d.strip() for d in args.dimensions.split(",")]
    print(f"Visualizing dimensions: {dimension_names}")

    # Validate for ternary
    if args.type == "ternary" and len(dimension_names) != 3:
        raise SystemExit(f"Ternary plots require exactly 3 dimensions, got {len(dimension_names)}")

    # Read scored entities from CSV
    print(f"Reading scored entities from: {input_path}")
    entity_scores = _read_scored_entities_csv(input_path, dimension_names)

    if not entity_scores:
        print("No entities found in input CSV.")
        return 1

    print(f"Loaded {len(entity_scores)} scored entities")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_{args.type}.png"

    # Parse figsize
    try:
        figsize = tuple(int(x) for x in args.figsize.split(","))
    except ValueError:
        figsize = (14, 10)

    # Initialize visualizer
    visualizer = EntityVisualizer()

    # Generate visualization
    if args.type == "ternary":
        title = args.title or f"Entity Classifications ({', '.join(d.capitalize() for d in dimension_names)})"
        visualizer.generate_ternary_plot(
            entity_scores=entity_scores,
            dimension_names=dimension_names,
            output_path=output_path,
            title=title,
            figsize=figsize,
            show_labels=not args.no_labels,
            use_numbered_labels=not args.direct_labels,
        )
    elif args.type == "radar":
        title = args.title or f"Entity Scores (Radar)"
        visualizer.generate_radar_chart(
            entity_scores=entity_scores,
            dimension_names=dimension_names,
            output_path=output_path,
            title=title,
            figsize=figsize,
            max_entities=args.max_entities,
        )

    print(f"\nVisualization saved to: {output_path}")
    return 0


def _read_scored_entities_csv(
    path: Path,
    dimension_names: List[str],
) -> List[Dict[str, Any]]:
    """Read scored entities from CSV file (output from 'qa entity score')."""
    entities = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        for row in reader:
            entity = row.get("entity", "").strip()
            if not entity:
                continue

            entity_data = {
                "entity": entity,
                "text_id": row.get("text_id", ""),
                "dimensions": {},
            }

            # Extract dimension scores
            for dim in dimension_names:
                dim_key = dim.lower()

                # Try various column name formats
                mean_key = f"{dim_key}_mean"
                score_key = dim_key

                if mean_key in row:
                    try:
                        entity_data["dimensions"][dim_key] = {
                            "mean": float(row[mean_key]),
                            "median": float(row.get(f"{dim_key}_median", row[mean_key])),
                            "std": float(row.get(f"{dim_key}_std", 0)),
                            "cv": float(row.get(f"{dim_key}_cv", 0)),
                        }
                    except ValueError:
                        entity_data["dimensions"][dim_key] = {"mean": 50}
                elif score_key in row:
                    try:
                        entity_data["dimensions"][dim_key] = {"mean": float(row[score_key])}
                    except ValueError:
                        entity_data["dimensions"][dim_key] = {"mean": 50}
                else:
                    entity_data["dimensions"][dim_key] = {"mean": 50}

            entities.append(entity_data)

    return entities
