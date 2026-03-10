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

import numpy as np

from qualitative_analysis.core.cli_utils import PACKAGE_VERSION

logger = logging.getLogger(__name__)


# =============================================================================
# DOWNSTREAM SCALE RESOLUTION HELPERS
# =============================================================================

def _resolve_scale(args: argparse.Namespace, input_csv_dir: Path) -> "ScaleConfig":
    """
    Resolve a score scale for commands that read previously scored outputs.

    This helper is intentionally metadata-aware for downstream consumers like
    visualization and comparison commands. The scoring command derives scale
    from the chosen dimensions and only applies explicit CLI overrides before
    writing fresh score_metadata.json for later consumers.

    Resolution priority:
    1. Explicit CLI args (--scale-min, --scale-max)
    2. score_metadata.json in same directory as input
    3. Defaults (0, 100)
    """
    from qualitative_analysis.entity.models import ScaleConfig

    scale_min, scale_max = 0, 100

    # Try metadata file
    meta_path = input_csv_dir / "score_metadata.json"
    if meta_path.exists():
        try:
            sc = ScaleConfig.from_metadata_file(meta_path)
            scale_min, scale_max = sc.scale_min, sc.scale_max
        except Exception:
            pass  # Fall back to defaults

    # CLI overrides take priority
    if getattr(args, 'scale_min', None) is not None:
        scale_min = args.scale_min
    if getattr(args, 'scale_max', None) is not None:
        scale_max = args.scale_max

    return ScaleConfig(scale_min=scale_min, scale_max=scale_max)


def _apply_scoring_scale_overrides(
    dimensions: List["DimensionDefinition"],
    args: argparse.Namespace,
) -> List["DimensionDefinition"]:
    """
    Apply explicit scoring CLI scale overrides to the loaded dimensions.

    Scoring intentionally does not auto-detect scale from score_metadata.json;
    it uses the scale defined on the selected dimensions unless the caller
    passes --scale-min/--scale-max.
    """
    if getattr(args, "scale_min", None) is None and getattr(args, "scale_max", None) is None:
        return dimensions

    for dim in dimensions:
        if args.scale_min is not None:
            dim.scale_min = args.scale_min
        if args.scale_max is not None:
            dim.scale_max = args.scale_max

    return dimensions


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
        "--scale-min",
        type=int,
        default=None,
        help="Minimum score value. Overrides scale_min on all dimensions (default: per-dimension, typically 0).",
    )
    parser.add_argument(
        "--scale-max",
        type=int,
        default=None,
        help="Maximum score value. Overrides scale_max on all dimensions (default: per-dimension, typically 100).",
    )
    parser.add_argument(
        "--prompt-version",
        default="v2",
        choices=["v2", "v3", "v4"],
        help="Prompt template version: v2 (score then justify), v3 (justify then score), v4 (scores only, no CoT) (default: v2).",
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
        choices=["ollama", "mlx", "openai", "anthropic"],
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
        "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable reasoning/thinking mode for models that support it (e.g., Qwen3.5). Default: disabled.",
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
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print LLM prompts and responses to terminal for monitoring.",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="(Alias for --verbose) Print LLM prompts and responses to terminal.",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        default=False,
        help="Disable checkpointing. By default, progress is saved after each entity and resumed on restart.",
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

    # Scoring uses the dimension source plus explicit CLI overrides only.
    dimensions = _apply_scoring_scale_overrides(dimensions, args)
    if getattr(args, 'scale_min', None) is not None or getattr(args, 'scale_max', None) is not None:
        print(f"Scale override: {dimensions[0].scale_min}-{dimensions[0].scale_max}")

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

    total_entities = len(entities)
    print(f"Loaded {total_entities} entities")

    # Checkpoint setup
    from qualitative_analysis.entity.models import EntityScoreResult
    use_checkpoint = not getattr(args, 'no_checkpoint', False)
    output_prefix = input_path.stem
    checkpoint_path = output_dir / f"{output_prefix}_checkpoint.jsonl"
    checkpoint_scores = []

    if use_checkpoint and checkpoint_path.exists():
        checkpoint_scores, scored_keys = _load_checkpoint(checkpoint_path, dimensions)
        if scored_keys:
            print(f"\nResuming from checkpoint: {len(scored_keys)} entities already scored")
            entities = [e for e in entities if _entity_checkpoint_key(e) not in scored_keys]
            skipped = total_entities - len(entities)
            print(f"Skipping {skipped} already-scored entities, {len(entities)} remaining")

    if not entities and checkpoint_scores:
        # All entities already scored — just finalize output
        print("\nAll entities already scored in checkpoint. Finalizing output...")
        result = EntityScoreResult(
            scores=checkpoint_scores,
            dimensions=dimensions,
            config={"num_runs": args.num_runs, "temperature": args.temperature, "errors": 0},
        )
        result.compute_statistics()
    elif not entities:
        print("No entities to score.")
        return 1
    else:
        # Initialize LLM provider
        log_llm = getattr(args, 'verbose', False) or getattr(args, 'log_llm', False)
        if args.provider == "ollama":
            llm_provider = OllamaProvider(
                model_name=args.model,
                base_url=args.base_url,
            )
        elif args.provider == "mlx":
            from qualitative_analysis.core.providers import MLXProvider
            llm_provider = MLXProvider(
                model_name=args.model,
                enable_thinking=getattr(args, 'enable_thinking', False),
                log_prompts=log_llm,
                log_responses=log_llm,
            )
        else:
            raise SystemExit(f"Provider '{args.provider}' not yet supported. Use 'ollama' or 'mlx'.")

        # Initialize scorer
        prompt_version = getattr(args, 'prompt_version', 'v2')
        scorer = EntityScorer(
            prompt_version=prompt_version,
            temperature=args.temperature,
            verbose=log_llm,
        )
        print(f"Prompt version: {prompt_version}")

        # Score entities
        n_already = len(checkpoint_scores)
        print(f"\nScoring {len(entities)} entities with {args.num_runs} runs each...")
        if n_already:
            print(f"  ({n_already} previously scored, {len(entities)} remaining)")
        print(f"Model: {args.model} ({args.provider})")
        print(f"Temperature: {args.temperature}")
        if use_checkpoint:
            print(f"Checkpoint: {checkpoint_path}")
        print()

        num_runs = args.num_runs

        def progress_callback(current: int, total: int, entity_name: str = ""):
            overall = n_already + current
            label = (entity_name[:40] + "...") if len(entity_name) > 40 else entity_name
            print(f"  [{overall}/{total_entities}] Scored:  {label:<43}", end="\r")

        def run_progress_callback(entity_idx, entity_total, entity_name, run_num, total_runs):
            overall = n_already + entity_idx
            label = (entity_name[:40] + "...") if len(entity_name) > 40 else entity_name
            print(f"  [{overall}/{total_entities}] Run {run_num}/{total_runs}: {label:<38}", end="\r")

        def checkpoint_callback(score):
            _append_to_checkpoint(checkpoint_path, score)

        result = await scorer.score_entities(
            entities=entities,
            dimensions=dimensions,
            llm_provider=llm_provider,
            num_runs=num_runs,
            research_context=research_context,
            on_progress=progress_callback,
            on_entity_scored=checkpoint_callback if use_checkpoint else None,
            on_run_progress=run_progress_callback,
        )

        # Close LLM provider connection
        if hasattr(llm_provider, 'close'):
            await llm_provider.close()

        # Merge checkpoint scores with newly scored entities
        if checkpoint_scores:
            result.scores = checkpoint_scores + result.scores
            result.compute_statistics()

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

    if "json" in formats:
        json_path = output_dir / f"{output_prefix}_scores.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nWrote scores JSON: {json_path}")

    if "csv" in formats:
        csv_path = output_dir / f"{output_prefix}_scores.csv"
        _write_scores_csv(result, csv_path)
        print(f"Wrote scores CSV: {csv_path}")

    # Write metadata
    dim_names = [d.name.lower() for d in dimensions]
    metadata = {
        "input_file": str(input_path),
        "model": args.model,
        "provider": args.provider,
        "temperature": args.temperature,
        "num_runs": args.num_runs,
        "dimensions": args.dimensions,
        "dimension_names": dim_names,
        "scale_min": dimensions[0].scale_min,
        "scale_max": dimensions[0].scale_max,
        "prompt_version": getattr(args, 'prompt_version', 'v2'),
        "total_entities_scored": len(result.scores),
        "errors": result.config.get("errors", 0),
        "timestamp": datetime.now().isoformat(),
        "package_version": PACKAGE_VERSION,
    }
    metadata_path = output_dir / "score_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata: {metadata_path}")

    # Clean up checkpoint file on successful completion
    if use_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint file removed (results saved to output files)")

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
        has_window_index = "window_index" in reader.fieldnames
        has_group = "group" in reader.fieldnames

        if not has_context:
            print(f"Warning: Context column '{context_col}' not found. Using empty context.")

        for row in reader:
            entity = row.get(entity_col, "").strip()
            if not entity:
                continue

            ent_dict = {
                "entity": entity,
                "context": row.get(context_col, "") if has_context else "",
                "text_id": row.get(text_id_col, "") if has_text_id else "",
            }

            if has_window_index:
                try:
                    ent_dict["window_index"] = int(row["window_index"])
                except (ValueError, TypeError):
                    ent_dict["window_index"] = None

            if has_group:
                ent_dict["group"] = row.get("group", "")

            entities.append(ent_dict)

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
# CHECKPOINTING HELPERS
# =============================================================================

def _entity_checkpoint_key(ent_data: Dict[str, Any]) -> str:
    """Create unique key for an entity for checkpoint deduplication."""
    entity = ent_data.get("entity", "")
    text_id = ent_data.get("text_id", "")
    window_index = ent_data.get("window_index")
    return f"{entity}|{text_id}|{window_index}"


def _append_to_checkpoint(checkpoint_path: Path, score: Any) -> None:
    """Append a scored entity to the checkpoint JSONL file."""
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(score.to_dict(include_runs=False)) + "\n")


def _load_checkpoint(
    checkpoint_path: Path,
    dimensions: list,
) -> tuple:
    """
    Load scored entities from checkpoint JSONL file.

    Args:
        checkpoint_path: Path to the checkpoint JSONL file.
        dimensions: List of DimensionDefinition objects for reconstruction.

    Returns:
        Tuple of (list of EntityScore objects, set of checkpoint keys).
    """
    from qualitative_analysis.entity.models import EntityScore, DimensionScore

    scores = []
    scored_keys = set()

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed checkpoint line {line_num}")
                continue

            key = _entity_checkpoint_key(data)
            scored_keys.add(key)

            # Reconstruct EntityScore from checkpoint data
            num_runs = data.get("num_runs", 1)
            dim_scores = {}

            for dim_def in dimensions:
                dim_key = dim_def.name.lower()
                # Collect run scores from flat columns
                run_scores = []
                for k in range(1, num_runs + 1):
                    run_key = f"{dim_key}_run{k}"
                    if run_key in data and data[run_key] is not None:
                        run_scores.append(data[run_key])

                if run_scores:
                    dim_scores[dim_key] = DimensionScore.from_scores(
                        dimension=dim_key,
                        scores=run_scores,
                        justification=data.get(f"{dim_key}_justification", ""),
                        scale_min=dim_def.scale_min,
                        scale_max=dim_def.scale_max,
                    )

            scores.append(EntityScore(
                entity=data.get("entity", ""),
                text_id=data.get("text_id", ""),
                context=data.get("context", ""),
                dimension_scores=dim_scores,
                num_runs=num_runs,
                processing_time_ms=data.get("processing_time_ms", 0),
                window_index=data.get("window_index"),
                group=data.get("group"),
            ))

    return scores, scored_keys


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
    parser.add_argument(
        "--scale-min",
        type=int,
        default=None,
        help="Minimum score value (default: auto-detect from score_metadata.json, typically 0).",
    )
    parser.add_argument(
        "--scale-max",
        type=int,
        default=None,
        help="Maximum score value (default: auto-detect from score_metadata.json, typically 100).",
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

    # Resolve scale
    scale = _resolve_scale(args, input_path.parent)

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
            scale_min=scale.scale_min,
            scale_max=scale.scale_max,
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
            scale_min=scale.scale_min,
            scale_max=scale.scale_max,
        )

    # Emit metadata file alongside the output image
    metadata_path = output_path.with_suffix('.metadata.json')
    _write_viz_metadata(metadata_path, args, {
        'command': 'entity viz',
        'input_csv': str(input_path),
        'output_path': str(output_path),
        'n_entities': len(entity_scores),
        'dimensions': dimension_names,
    })

    print(f"\nVisualization saved to: {output_path}")
    return 0


def _write_viz_metadata(
    metadata_path: Path,
    args: argparse.Namespace,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a JSON metadata file capturing the CLI arguments used."""
    from datetime import datetime as dt

    metadata: Dict[str, Any] = {
        'timestamp': dt.now().isoformat(),
        'cli_args': {},
    }

    # Capture all CLI args, converting Path objects to strings
    for key, value in vars(args).items():
        if key == 'func':
            continue
        if isinstance(value, Path):
            metadata['cli_args'][key] = str(value)
        else:
            metadata['cli_args'][key] = value

    if extra:
        metadata.update(extra)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)


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


# =============================================================================
# ENTITY COMPARISON COMMAND
# =============================================================================

def add_entity_compare_args(parser: argparse.ArgumentParser) -> None:
    """Add entity comparison arguments."""
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with scored entities (from 'qa entity score').",
    )
    parser.add_argument(
        "--participant-col",
        default="text_id",
        help="Column name for participant/text IDs (default: text_id).",
    )
    parser.add_argument(
        "--entity-col",
        default="entity",
        help="Column name for entities (default: entity).",
    )
    parser.add_argument(
        "--dimensions",
        default="social,ecological,technological",
        help="Dimensions to compare, comma-separated (default: social,ecological,technological).",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "aitchison", "emd"],
        default="euclidean",
        help="Distance metric for comparison (default: euclidean). Use 'aitchison' only for compositional data.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        default=False,
        help="Generate heatmaps and similarity maps for all distance metrics (euclidean, cosine, aitchison, emd).",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "distribution"],
        default="mean",
        help=(
            "How to aggregate entities per participant: "
            "'mean' compares centroids (single point per participant), "
            "'distribution' compares full entity distributions (EMD only, captures spread/shape). "
            "(default: mean)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates comparison_* subdirectory).",
    )
    parser.add_argument(
        "--viz",
        default="all",
        help=(
            "Visualizations to generate: 'all', 'faceted', 'overlaid', 'heatmap', "
            "'forest', 'similarity-map', or comma-separated list (default: all)."
        ),
    )
    parser.add_argument(
        "--participants",
        default=None,
        help="Specific participants to include, comma-separated (default: all).",
    )
    parser.add_argument(
        "--show-centroids",
        action="store_true",
        default=True,
        help="Show centroid markers on ternary plots (default: True).",
    )
    parser.add_argument(
        "--show-hull",
        action="store_true",
        help="Show convex hull on faceted plots.",
    )
    parser.add_argument(
        "--show-ellipses",
        action="store_true",
        help="Show 95%% confidence ellipses per participant on overlaid ternary.",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=4,
        help="Maximum columns in faceted grid (default: 4).",
    )
    parser.add_argument(
        "--similarity-method",
        choices=["mds", "umap"],
        default="mds",
        help="Method for similarity map: 'mds' (default) or 'umap' (requires umap-learn).",
    )
    parser.add_argument(
        "--all-projections",
        action="store_true",
        default=False,
        help="Generate similarity maps using both MDS and UMAP projections.",
    )

    # Group comparison arguments
    parser.add_argument(
        "--groups",
        default=None,
        help=(
            "Define groups inline: 'group1:pid1,pid2;group2:pid3,pid4'. "
            "Compare groups instead of individuals. Only one of --groups, "
            "--groups-file, or --group-by-col can be specified."
        ),
    )
    parser.add_argument(
        "--groups-file",
        default=None,
        help=(
            "Path to JSON or CSV file defining groups. "
            "JSON: {'group1': ['pid1', 'pid2'], ...}. "
            "CSV: columns 'participant_id' and 'group'."
        ),
    )
    parser.add_argument(
        "--group-by-col",
        default=None,
        help="Column name in input CSV to use for grouping participants.",
    )
    parser.add_argument(
        "--skip-individual-viz",
        action="store_true",
        help="When using groups, skip individual participant visualizations (only generate group visualizations).",
    )
    parser.add_argument(
        "--statistical-test",
        choices=["permutation", "none"],
        default="none",
        help=(
            "Statistical test for group comparison significance. "
            "'permutation': Run permutation test to compute p-value for effect size. "
            "'none': Skip statistical testing (default)."
        ),
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        help="Number of permutations for permutation test (default: 1000).",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also save individual ternary plot files per participant.",
    )
    parser.add_argument(
        "--no-numbers",
        action="store_true",
        help="Disable numbered labels on individual ternary plots. Entities are ordered by color spectrum in the legend.",
    )

    parser.add_argument(
        "--scale-min",
        type=int,
        default=None,
        help="Minimum score value (default: auto-detect from score_metadata.json, typically 0).",
    )
    parser.add_argument(
        "--scale-max",
        type=int,
        default=None,
        help="Maximum score value (default: auto-detect from score_metadata.json, typically 100).",
    )

    # Bayesian modeling arguments
    parser.add_argument(
        "--method",
        choices=["distance", "bayesian", "all"],
        default="distance",
        help=(
            "Analysis method: 'distance' (default, distance-based metrics), "
            "'bayesian' (hierarchical Bayesian model), or 'all' (both). "
            "Bayesian requires groups and run-level score data."
        ),
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains for Bayesian model (default: 4).",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=2000,
        help="Number of posterior draws per chain (default: 2000).",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="Number of tuning steps per chain (default: 1000).",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.95,
        help="Target acceptance rate for NUTS sampler (default: 0.95).",
    )
    parser.add_argument(
        "--rope-delta",
        type=float,
        default=None,
        help="ROPE half-width for practical significance (default: 5%% of scale range).",
    )
    parser.add_argument(
        "--sampler",
        choices=["nutpie", "nuts"],
        default="nutpie",
        help="MCMC sampler backend: 'nutpie' (faster, default) or 'nuts' (standard PyMC).",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save full InferenceData as netCDF files (can be large).",
    )

    # Advanced Bayesian analysis options
    parser.add_argument(
        "--loo-compare",
        action="store_true",
        help=(
            "Run LOO-CV model comparison between full (with group effects) "
            "and reduced (no group effects) models. Requires --method bayesian."
        ),
    )
    parser.add_argument(
        "--bayes-factor",
        action="store_true",
        help=(
            "Compute Savage-Dickey Bayes Factor for group effects = 0. "
            "Requires --method bayesian."
        ),
    )


# =============================================================================
# GROUP PARSING HELPERS
# =============================================================================


def parse_inline_groups(groups_str: str) -> Dict[str, List[str]]:
    """
    Parse inline group definitions.

    Format: 'group1:pid1,pid2;group2:pid3,pid4'

    Args:
        groups_str: String with group definitions.

    Returns:
        Dict mapping group names to participant ID lists.

    Raises:
        ValueError: If format is invalid.
    """
    groups = {}
    for group_def in groups_str.split(";"):
        group_def = group_def.strip()
        if not group_def:
            continue
        if ":" not in group_def:
            raise ValueError(
                f"Invalid group definition '{group_def}'. "
                "Expected format: 'group_name:pid1,pid2,...'"
            )
        name, members_str = group_def.split(":", 1)
        name = name.strip()
        members = [m.strip() for m in members_str.split(",") if m.strip()]
        if not members:
            raise ValueError(f"Group '{name}' has no members")
        groups[name] = members
    return groups


def load_groups_file(path: Path) -> Dict[str, List[str]]:
    """
    Load groups from JSON or CSV file.

    JSON format: {"group1": ["pid1", "pid2"], "group2": ["pid3", "pid4"]}
    CSV format: columns 'participant_id' (or 'text_id') and 'group'

    Args:
        path: Path to groups file.

    Returns:
        Dict mapping group names to participant ID lists.

    Raises:
        ValueError: If file format is unknown or invalid.
    """
    import csv
    import json

    path = Path(path)

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("JSON file must contain a dictionary")
        return {str(k): list(v) for k, v in data.items()}

    elif path.suffix.lower() == ".csv":
        groups: Dict[str, List[str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Find participant column
            fieldnames = reader.fieldnames or []
            pid_col = None
            for candidate in ["participant_id", "text_id", "id"]:
                if candidate in fieldnames:
                    pid_col = candidate
                    break
            if pid_col is None:
                raise ValueError(
                    "CSV must have 'participant_id', 'text_id', or 'id' column"
                )

            if "group" not in fieldnames:
                raise ValueError("CSV must have 'group' column")

            for row in reader:
                pid = row[pid_col]
                group = row["group"]
                if group not in groups:
                    groups[group] = []
                groups[group].append(pid)

        return groups

    else:
        raise ValueError(
            f"Unknown groups file format: {path.suffix}. Use .json or .csv"
        )


def extract_groups_from_data(
    input_path: Path,
    group_col: str,
    participant_col: str = "text_id",
) -> Dict[str, List[str]]:
    """
    Extract groups from a column in the input data CSV.

    Args:
        input_path: Path to input CSV file.
        group_col: Column name containing group assignments.
        participant_col: Column name for participant IDs.

    Returns:
        Dict mapping group names to participant ID lists.

    Raises:
        ValueError: If columns not found.
    """
    import csv

    groups: Dict[str, List[str]] = {}
    seen_participants: Dict[str, str] = {}  # Track pid -> group for deduplication

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        if participant_col not in fieldnames:
            raise ValueError(f"Participant column '{participant_col}' not found in data")
        if group_col not in fieldnames:
            raise ValueError(f"Group column '{group_col}' not found in data")

        for row in reader:
            pid = row[participant_col]
            group = row[group_col]

            # Skip if already seen (each participant belongs to one group)
            if pid in seen_participants:
                continue

            seen_participants[pid] = group
            if group not in groups:
                groups[group] = []
            groups[group].append(pid)

    return groups


async def run_entity_compare(args: argparse.Namespace) -> int:
    """
    Run entity comparison with the given arguments.

    This computes distances between participants and generates
    comparison visualizations (faceted ternary, overlaid ternary, heatmap).
    Supports both individual pairwise comparison and group-level comparison.

    Args:
        args: Parsed arguments namespace with comparison configuration

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.entity.comparison import (
        ParticipantComparison,
        GroupComparisonResult,
    )
    from qualitative_analysis.entity.comparison_viz import ComparisonVisualizer

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Check for mutually exclusive group options
    group_options = [
        args.groups is not None,
        args.groups_file is not None,
        args.group_by_col is not None,
    ]
    if sum(group_options) > 1:
        raise SystemExit(
            "Error: Only one of --groups, --groups-file, or --group-by-col can be specified."
        )

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"comparison_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse dimensions
    dimension_names = [d.strip().lower() for d in args.dimensions.split(",")]
    print(f"Comparing dimensions: {dimension_names}")

    # Load scores
    print(f"Loading scored entities from: {input_path}")
    comparison = ParticipantComparison()
    comparison.load_scores(
        input_path,
        participant_col=args.participant_col,
        entity_col=args.entity_col,
        dimensions=dimension_names,
    )

    n_participants = len(comparison.scores_by_participant)
    print(f"Loaded {n_participants} participants")

    if n_participants < 2:
        print("Need at least 2 participants for comparison.")
        return 1

    # Parse participants filter
    participants = None
    if args.participants:
        participants = [p.strip() for p in args.participants.split(",")]
        participants = [p for p in participants if p in comparison.scores_by_participant]
        print(f"Filtering to {len(participants)} participants: {participants}")

    # Parse group definitions (if any)
    groups = None
    if args.groups:
        print(f"\nParsing inline group definitions...")
        groups = parse_inline_groups(args.groups)
    elif args.groups_file:
        groups_file_path = Path(args.groups_file)
        if not groups_file_path.exists():
            raise SystemExit(f"Groups file not found: {groups_file_path}")
        print(f"\nLoading groups from: {groups_file_path}")
        groups = load_groups_file(groups_file_path)
    elif args.group_by_col:
        print(f"\nExtracting groups from column: {args.group_by_col}")
        groups = extract_groups_from_data(
            input_path, args.group_by_col, args.participant_col
        )

    # Aggregation mode
    aggregate_mode = getattr(args, 'aggregate', 'mean')
    if aggregate_mode == "distribution" and args.metric not in ("emd", "wasserstein"):
        print(f"\nNote: --aggregate=distribution is only meaningful with --metric=emd")
        print(f"      Using metric '{args.metric}' will compare centroids regardless.")

    # Handle group comparison
    group_result = None
    if groups:
        print(f"\nGroups defined: {list(groups.keys())}")
        for gname, members in groups.items():
            print(f"  {gname}: {len(members)} participants")

        # Set groups and get excluded participants
        excluded = comparison.set_groups(groups)

        # Compute group-level distances
        print(f"\nComputing group distances using {args.metric} metric (aggregate={aggregate_mode})...")
        group_result = comparison.compute_group_distances(
            metric=args.metric, aggregate=aggregate_mode
        )

        # Print group comparison summary
        print(group_result.summary_str())

        # Save group comparison results JSON
        group_results_path = output_dir / "group_comparison_results.json"
        with open(group_results_path, "w", encoding="utf-8") as f:
            json.dump(group_result.to_dict(), f, indent=2)
        print(f"\nSaved group results to: {group_results_path}")

        # Run statistical test if requested
        statistical_test = getattr(args, 'statistical_test', 'none')
        if statistical_test == "permutation":
            n_perms = getattr(args, 'n_permutations', 1000)
            print(f"\nRunning permutation test ({n_perms} permutations)...")
            perm_result = comparison.run_permutation_test(
                metric=args.metric,
                aggregate=aggregate_mode,
                n_permutations=n_perms,
            )
            print(f"  Observed effect size: {perm_result['observed_effect_size']:.3f}")
            print(f"  P-value: {perm_result['p_value']:.4f}")
            if perm_result['p_value'] < 0.05:
                print("  Result: Significant group differences (p < 0.05)")
            else:
                print("  Result: No significant group differences (p >= 0.05)")

            # Update group_result with permutation test p-value
            group_result.permutation_test_p = perm_result['p_value']

            # Re-save group results with p-value
            with open(group_results_path, "w", encoding="utf-8") as f:
                json.dump(group_result.to_dict(), f, indent=2)

            # Save full permutation test results
            perm_results_path = output_dir / "permutation_test_results.json"
            with open(perm_results_path, "w", encoding="utf-8") as f:
                # Don't save full null distribution to keep file size reasonable
                perm_summary = {
                    "observed_effect_size": perm_result['observed_effect_size'],
                    "p_value": perm_result['p_value'],
                    "n_permutations": perm_result['n_permutations'],
                    "null_distribution_summary": {
                        "mean": float(np.mean(perm_result['null_distribution'])),
                        "std": float(np.std(perm_result['null_distribution'])),
                        "min": float(np.min(perm_result['null_distribution'])),
                        "max": float(np.max(perm_result['null_distribution'])),
                        "percentile_95": float(np.percentile(perm_result['null_distribution'], 95)),
                    }
                }
                json.dump(perm_summary, f, indent=2)
            print(f"  Saved permutation test results to: {perm_results_path}")

    # Compute individual pairwise distances (always, for visualizations)
    print(f"\nComputing pairwise distances using {args.metric} metric (aggregate={aggregate_mode})...")
    result = comparison.compute_distances(metric=args.metric, aggregate=aggregate_mode)

    # Print individual comparison summary
    mode_desc = "centroid" if result.aggregate == "mean" else "distribution"
    print(f"\nIndividual Comparison Summary ({args.metric}, {mode_desc} comparison):")
    print(f"  Mean distance: {result.mean_distance:.4f}")
    print(f"  Median distance: {result.median_distance:.4f}")
    print(f"  Range: {result.min_distance:.4f} - {result.max_distance:.4f}")
    print(f"  Most similar: {result.most_similar_pair[0]} <-> {result.most_similar_pair[1]} ({result.most_similar_pair[2]:.4f})")
    print(f"  Most different: {result.most_different_pair[0]} <-> {result.most_different_pair[1]} ({result.most_different_pair[2]:.4f})")

    # Save comparison results JSON
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nSaved results to: {results_path}")

    # Determine which visualizations to generate
    viz_types = args.viz.lower().split(",")
    if "all" in viz_types:
        viz_types = ["faceted", "overlaid", "heatmap", "forest", "similarity-map"]

    # Check if individual visualizations should be skipped
    skip_individual = getattr(args, 'skip_individual_viz', False) and groups is not None

    # Resolve scale
    scale = _resolve_scale(args, input_path.parent)

    # Initialize visualizer
    visualizer = ComparisonVisualizer(
        comparison,
        scale_min=scale.scale_min,
        scale_max=scale.scale_max,
    )

    # Pass groups to visualizer if defined
    if groups:
        visualizer.groups = groups

    # Generate individual visualizations (unless skipped)
    if not skip_individual:
        if "faceted" in viz_types and len(dimension_names) == 3:
            print("\nGenerating faceted ternary plot...")
            faceted_path = output_dir / "faceted_ternary.png"
            visualizer.generate_faceted_ternary(
                output_path=faceted_path,
                dimension_names=dimension_names,
                participants=participants,
                max_cols=args.max_cols,
                show_centroid=args.show_centroids,
                show_convex_hull=args.show_hull,
                title="Multi-Participant Entity Comparison",
            )
            print(f"  Saved: {faceted_path}")

        if "overlaid" in viz_types and len(dimension_names) == 3:
            print("Generating overlaid ternary plot...")
            overlaid_path = output_dir / "overlaid_ternary.png"
            visualizer.generate_overlaid_ternary(
                output_path=overlaid_path,
                dimension_names=dimension_names,
                participants=participants,
                show_centroids=args.show_centroids,
                show_confidence_ellipses=getattr(args, 'show_ellipses', False),
                title="Entity Score Comparison (All Participants)",
            )
            print(f"  Saved: {overlaid_path}")

        if "heatmap" in viz_types:
            # Determine which metrics to generate heatmaps for
            all_metrics = ["euclidean", "cosine", "aitchison", "emd"]
            metrics_to_plot = all_metrics if getattr(args, 'all_metrics', False) else [args.metric]
            for m in metrics_to_plot:
                if m == args.metric:
                    m_result = result
                else:
                    print(f"Computing {m} distances...")
                    m_result = comparison.compute_distances(metric=m, aggregate=aggregate_mode)
                suffix = f"_{m}" if len(metrics_to_plot) > 1 else ""
                heatmap_path = output_dir / f"distance_heatmap{suffix}.png"
                heatmap_title = f"Participant Distance Matrix ({m.capitalize()}"
                if m_result.aggregate == "distribution":
                    heatmap_title += ", Distribution"
                heatmap_title += ")"
                print(f"Generating distance heatmap ({m})...")
                visualizer.generate_distance_heatmap(
                    m_result,
                    output_path=heatmap_path,
                    title=heatmap_title,
                )
                print(f"  Saved: {heatmap_path}")

        if "forest" in viz_types:
            print("Generating forest plots (normalized + raw)...")
            forest_norm_path = output_dir / "forest_plot_normalized.png"
            visualizer.generate_forest_plot(
                result,
                output_path=forest_norm_path,
                dimension_names=dimension_names,
                participants=participants,
                score_mode="normalized",
                groups=groups,
            )
            print(f"  Saved: {forest_norm_path}")

            forest_raw_path = output_dir / "forest_plot_raw.png"
            visualizer.generate_forest_plot(
                result,
                output_path=forest_raw_path,
                dimension_names=dimension_names,
                participants=participants,
                score_mode="raw",
                groups=groups,
            )
            print(f"  Saved: {forest_raw_path}")

        if "similarity-map" in viz_types:
            default_method = getattr(args, 'similarity_method', 'mds')
            all_metrics = ["euclidean", "cosine", "aitchison", "emd"]
            metrics_to_plot = all_metrics if getattr(args, 'all_metrics', False) else [args.metric]
            projections = ["mds", "umap"] if getattr(args, 'all_projections', False) else [default_method]
            multi_metric = len(metrics_to_plot) > 1
            multi_proj = len(projections) > 1
            for m in metrics_to_plot:
                if m == args.metric:
                    m_result = result
                else:
                    m_result = comparison.compute_distances(metric=m, aggregate=aggregate_mode)
                mode_label = "centroid" if getattr(m_result, 'aggregate', 'mean') == "mean" else "distribution"
                for proj in projections:
                    parts = []
                    if multi_metric:
                        parts.append(m)
                    if multi_proj:
                        parts.append(proj)
                    suffix = f"_{'_'.join(parts)}" if parts else ""
                    similarity_path = output_dir / f"similarity_map{suffix}.png"
                    print(f"Generating similarity map ({proj.upper()}, {m})...")
                    visualizer.generate_similarity_map(
                        m_result,
                        output_path=similarity_path,
                        method=proj,
                        title=f"Participant Similarity ({m.capitalize()}, {mode_label}, {proj.upper()})",
                    )
                    print(f"  Saved: {similarity_path}")

        # Generate individual per-participant ternary files
        if getattr(args, 'individual', False) and len(dimension_names) == 3:
            individual_dir = output_dir / "individual_ternary"
            individual_dir.mkdir(parents=True, exist_ok=True)
            print("Generating individual ternary plots...")
            individual_paths = visualizer.generate_individual_ternary(
                output_dir=individual_dir,
                dimension_names=dimension_names,
                participants=participants,
                marker_size=80,
                show_centroid=args.show_centroids,
                show_convex_hull=args.show_hull,
                show_numbers=not getattr(args, 'no_numbers', False),
            )
            print(f"  Saved {len(individual_paths)} individual plots to {individual_dir}")

    # Generate group-specific visualizations
    if groups:
        all_metrics = ["euclidean", "cosine", "aitchison", "emd"]
        metrics_to_plot = all_metrics if getattr(args, 'all_metrics', False) else [args.metric]

        # Group-colored similarity map
        if "similarity-map" in viz_types:
            default_method = getattr(args, 'similarity_method', 'mds')
            projections = ["mds", "umap"] if getattr(args, 'all_projections', False) else [default_method]
            multi_metric = len(metrics_to_plot) > 1
            multi_proj = len(projections) > 1
            for m in metrics_to_plot:
                if m == args.metric:
                    m_result = result
                else:
                    m_result = comparison.compute_distances(metric=m, aggregate=aggregate_mode)
                mode_label = "centroid" if getattr(m_result, 'aggregate', 'mean') == "mean" else "distribution"
                for proj in projections:
                    parts = []
                    if multi_metric:
                        parts.append(m)
                    if multi_proj:
                        parts.append(proj)
                    suffix = f"_{'_'.join(parts)}" if parts else ""
                    group_sim_path = output_dir / f"group_similarity_map{suffix}.png"
                    print(f"Generating group-colored similarity map ({proj.upper()}, {m})...")
                    visualizer.generate_similarity_map(
                        m_result,
                        output_path=group_sim_path,
                        method=proj,
                        title=f"Group Similarity ({m.capitalize()}, {mode_label}, {proj.upper()})",
                        groups=groups,
                    )
                    print(f"  Saved: {group_sim_path}")

        # Group-ordered heatmap
        if "heatmap" in viz_types:
            for m in metrics_to_plot:
                if m == args.metric:
                    m_result = result
                else:
                    m_result = comparison.compute_distances(metric=m, aggregate=aggregate_mode)
                suffix = f"_{m}" if len(metrics_to_plot) > 1 else ""
                heatmap_title = f"Group Distance Matrix ({m.capitalize()}"
                if m_result.aggregate == "distribution":
                    heatmap_title += ", Distribution"
                heatmap_title += ")"
                group_heatmap_path = output_dir / f"group_heatmap{suffix}.png"
                print(f"Generating group-ordered heatmap ({m})...")
                visualizer.generate_distance_heatmap(
                    m_result,
                    output_path=group_heatmap_path,
                    title=heatmap_title,
                    groups=groups,
                )
                print(f"  Saved: {group_heatmap_path}")

    # =========================================================================
    # BAYESIAN HIERARCHICAL MODEL
    # =========================================================================
    method = getattr(args, 'method', 'distance')
    if method in ("bayesian", "all"):
        if not groups:
            print("\nWarning: --method bayesian requires groups. Skipping Bayesian analysis.")
        else:
            # Check PyMC availability
            try:
                from qualitative_analysis.entity.bayesian import (
                    BayesianEntityModel,
                    BayesianVisualizer,
                )
            except ImportError:
                print(
                    "\nError: Bayesian modeling requires PyMC and ArviZ. Install with:\n"
                    '  pip install -e ".[bayes]"\n'
                    "Or:\n"
                    "  pip install 'pymc>=5.21' arviz nutpie"
                )
                if method == "bayesian":
                    return 1
                else:
                    print("Skipping Bayesian analysis (--method all).")
                    print(f"\nComparison complete. Output directory: {output_dir}")
                    return 0

            import pandas as pd

            print("\n" + "=" * 60)
            print("BAYESIAN HIERARCHICAL MODEL")
            print("=" * 60)

            bayesian_dir = output_dir / "bayesian"
            bayesian_dir.mkdir(parents=True, exist_ok=True)

            # Load the scored CSV as a DataFrame for Bayesian modeling
            scores_df = pd.read_csv(input_path)

            # Ensure group column exists
            group_col = getattr(args, 'group_by_col', None)
            if group_col and group_col in scores_df.columns:
                pass  # group column already in data
            else:
                # Add group column from groups dict
                pid_to_group = {}
                for gname, members in groups.items():
                    for pid in members:
                        pid_to_group[pid] = gname
                scores_df["group"] = scores_df[args.participant_col].map(pid_to_group)
                # Drop rows with no group assignment
                before = len(scores_df)
                scores_df = scores_df.dropna(subset=["group"])
                if len(scores_df) < before:
                    print(f"  Dropped {before - len(scores_df)} rows with no group assignment")
                group_col = "group"

            try:
                bayesian_model = BayesianEntityModel(
                    scores_df=scores_df,
                    dimensions=dimension_names,
                    groups=groups,
                    participant_col=args.participant_col,
                    entity_col=args.entity_col,
                    group_col=group_col,
                    scale_min=scale.scale_min,
                    scale_max=scale.scale_max,
                )

                # Fit all dimensions
                sampler = getattr(args, 'sampler', 'nutpie')
                bayesian_model.fit_all_dimensions(
                    chains=args.chains,
                    draws=args.draws,
                    tune=args.tune,
                    target_accept=args.target_accept,
                    sampler=sampler,
                )

                # Print convergence summary
                print("\nConvergence Summary:")
                all_converged = True
                for dim, res in bayesian_model.results.items():
                    d = res.diagnostics
                    status = "PASS" if res.converged else "FAIL"
                    if not res.converged:
                        all_converged = False
                    print(
                        f"  {dim}: {status} "
                        f"(R-hat={d['rhat_max']}, "
                        f"ESS_bulk={d['ess_bulk_min']:.0f}, "
                        f"ESS_tail={d['ess_tail_min']:.0f}, "
                        f"div={d['divergences']})"
                    )

                if not all_converged:
                    print(
                        "\nWarning: Some dimensions did not fully converge. "
                        "Consider increasing --tune or --target-accept."
                    )

                # Print group contrast highlights
                rope_delta = getattr(args, 'rope_delta', None)
                if rope_delta is None:
                    rope_delta = scale.rope_default()
                contrasts = bayesian_model.compute_group_contrasts(rope_delta=rope_delta)
                print(f"\nGroup Contrasts (population-level mean difference, {scale.scale_min}-{scale.scale_max} scale):")
                for dim, dim_contrasts in contrasts.items():
                    print(f"\n  {dim.upper()}:")
                    for pair_key, c in dim_contrasts.items():
                        direction = ">" if c["p_a_gt_b"] > 0.5 else "<"
                        p_dir = max(c["p_a_gt_b"], c["p_b_gt_a"])
                        print(
                            f"    {c['group_a']} vs {c['group_b']}: "
                            f"Δ = {c['mean_diff']:+.2f} "
                            f"[{c['hdi_lower']:+.2f}, {c['hdi_upper']:+.2f}], "
                            f"P({c['group_a']}{direction}{c['group_b']}) = {p_dir:.3f}"
                        )

                # Save all results
                save_trace = getattr(args, 'save_trace', False)
                bayesian_model.save_results(
                    bayesian_dir, rope_delta=rope_delta, save_trace=save_trace
                )

                # Optional LOO-CV model comparison
                if getattr(args, "loo_compare", False):
                    print("\nRunning LOO-CV model comparison...")
                    import json as _json
                    loo_results = {}
                    for dim in dimension_names:
                        loo_result = bayesian_model.compare_models(
                            dim,
                            chains=args.chains,
                            draws=args.draws,
                            tune=args.tune,
                            sampler=args.sampler,
                        )
                        loo_results[dim] = loo_result
                        print(
                            f"  {dim}: {loo_result['preferred_model']} model preferred "
                            f"(ΔELPD = {loo_result['elpd_diff']:.1f} ± {loo_result['se_diff']:.1f})"
                        )
                    with open(bayesian_dir / "loo_comparison.json", "w") as f:
                        _json.dump(loo_results, f, indent=2, default=str)
                    print(f"LOO-CV results saved to: {bayesian_dir / 'loo_comparison.json'}")

                # Optional Bayes Factor
                if getattr(args, "bayes_factor", False):
                    print("\nComputing Bayes Factors...")
                    import json as _json
                    bf_results = {}
                    for dim in dimension_names:
                        bf_result = bayesian_model.compute_bayes_factor(dim)
                        bf_results.update(bf_result)
                        for pair_key, pair_data in bf_result[dim].items():
                            bf10 = pair_data.get("bf10")
                            interp = pair_data.get("interpretation", "undefined")
                            print(f"  {dim} {pair_key}: BF10 = {bf10} ({interp})")
                    with open(bayesian_dir / "bayes_factors.json", "w") as f:
                        _json.dump(bf_results, f, indent=2, default=str)
                    print(f"Bayes Factor results saved to: {bayesian_dir / 'bayes_factors.json'}")

                # Generate visualizations
                print("\nGenerating Bayesian visualizations...")
                viz = BayesianVisualizer(bayesian_model)
                viz.generate_all(bayesian_dir, rope_delta=rope_delta)

                print(f"\nBayesian results saved to: {bayesian_dir}")

            except ValueError as e:
                print(f"\nBayesian model error: {e}")
                if method == "bayesian":
                    return 1
            except Exception as e:
                logger.exception("Bayesian model failed")
                print(f"\nBayesian model failed: {e}")
                if method == "bayesian":
                    return 1

    print(f"\nComparison complete. Output directory: {output_dir}")
    return 0


# =============================================================================
# ENTITY PREPARE-SCORING COMMAND
# =============================================================================

def add_entity_prepare_scoring_args(parser: argparse.ArgumentParser) -> None:
    """Add entity prepare-scoring arguments.

    This command generates a scoring input CSV with real context from windows files.
    It extracts the actual text where each entity was mentioned, linking entities
    back to their original source text rather than using placeholder context.
    """
    parser.add_argument(
        "windows_csv",
        help="Path to windows CSV file (from 'qa relationships detect' with --save-windows).",
    )
    parser.add_argument(
        "--nodes-csv",
        default=None,
        help="Optional path to graph nodes CSV to filter entities (from 'qa relationships graph').",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for scoring input CSV (default: <windows>_scoring_input.csv).",
    )
    parser.add_argument(
        "--context-mode",
        choices=["window", "sentence", "combined"],
        default="window",
        help="Context extraction mode: 'window' (full window text), 'sentence' (just the sentence), 'combined' (all windows) (default: window).",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=2000,
        help="Maximum context length in characters (default: 2000).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum entity frequency to include (default: 1).",
    )
    parser.add_argument(
        "--participants",
        default=None,
        help="Specific participants to include, comma-separated (default: all).",
    )


async def run_entity_prepare_scoring(args: argparse.Namespace) -> int:
    """
    Generate a scoring input CSV with real context from windows files.

    This extracts entity-context pairs from the relationship detection windows,
    ensuring that each entity has actual source text as context rather than
    placeholder text. This is essential for accurate entity scoring.

    Args:
        args: Parsed arguments namespace with configuration

    Returns:
        Exit code (0 for success)
    """
    windows_path = Path(args.windows_csv)
    if not windows_path.exists():
        raise SystemExit(f"Windows CSV not found: {windows_path}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = windows_path.parent / f"{windows_path.stem}_scoring_input.csv"

    print(f"Loading windows from: {windows_path}")

    # Parse participant filter
    participant_filter = None
    if args.participants:
        participant_filter = set(p.strip() for p in args.participants.split(","))
        print(f"Filtering to participants: {participant_filter}")

    # Load windows and extract entity-context pairs
    entity_contexts = _extract_entity_contexts_from_windows(
        windows_path=windows_path,
        context_mode=args.context_mode,
        max_context_length=args.max_context_length,
        participant_filter=participant_filter,
    )

    print(f"Extracted {len(entity_contexts)} entity-context pairs")

    # Optionally filter by nodes file
    if args.nodes_csv:
        nodes_path = Path(args.nodes_csv)
        if nodes_path.exists():
            print(f"Filtering by nodes file: {nodes_path}")
            entity_contexts = _filter_by_nodes(
                entity_contexts=entity_contexts,
                nodes_path=nodes_path,
                min_frequency=args.min_frequency,
            )
            print(f"After filtering: {len(entity_contexts)} entity-context pairs")
        else:
            print(f"Warning: Nodes file not found: {nodes_path}")

    if not entity_contexts:
        print("No entity-context pairs to write.")
        return 1

    # Write output CSV
    _write_scoring_input_csv(entity_contexts, output_path)

    # Print summary
    unique_entities = len(set(ec["entity"] for ec in entity_contexts))
    unique_participants = len(set(ec["text_id"] for ec in entity_contexts))
    print(f"\nScoring input prepared:")
    print(f"  Total pairs: {len(entity_contexts)}")
    print(f"  Unique entities: {unique_entities}")
    print(f"  Unique participants: {unique_participants}")
    print(f"  Output: {output_path}")

    return 0


def _extract_entity_contexts_from_windows(
    windows_path: Path,
    context_mode: str = "window",
    max_context_length: int = 2000,
    participant_filter: Optional[set] = None,
) -> List[Dict[str, str]]:
    """
    Extract entity-context pairs from a windows CSV file.

    Args:
        windows_path: Path to windows CSV (with text_id, window_index, window_text, entities_json)
        context_mode: How to extract context ('window', 'sentence', 'combined')
        max_context_length: Maximum length for context strings
        participant_filter: Optional set of participant IDs to include

    Returns:
        List of dicts with 'entity', 'context', 'text_id' keys
    """
    # Track entity contexts: {(entity, text_id): [contexts]}
    entity_context_map: Dict[tuple, List[str]] = {}

    with open(windows_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            return []

        # Check required columns
        required = ["text_id", "window_text", "entities_json"]
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Windows CSV missing required columns: {missing}")

        for row in reader:
            text_id = row.get("text_id", "").strip()
            window_text = row.get("window_text", "").strip()
            entities_json = row.get("entities_json", "[]")

            if not text_id or not window_text:
                continue

            # Apply participant filter
            if participant_filter and text_id not in participant_filter:
                continue

            # Parse entities
            try:
                entities = json.loads(entities_json)
            except json.JSONDecodeError:
                entities = []

            # Add context for each entity
            for entity in entities:
                entity = str(entity).strip()
                if not entity:
                    continue

                key = (entity.lower(), text_id)  # Use lowercase for dedup

                if key not in entity_context_map:
                    entity_context_map[key] = []

                entity_context_map[key].append(window_text)

    # Convert to output format based on context_mode
    results = []

    for (entity_lower, text_id), contexts in entity_context_map.items():
        # Use original case from first occurrence
        entity = entity_lower  # Could be improved to preserve original case

        if context_mode == "window":
            # Use first window where entity appeared
            context = contexts[0]
        elif context_mode == "sentence":
            # Extract sentence containing the entity from first window
            context = _extract_sentence_with_entity(contexts[0], entity)
        elif context_mode == "combined":
            # Combine all unique contexts
            unique_contexts = list(dict.fromkeys(contexts))  # Preserve order, remove dups
            context = " [...] ".join(unique_contexts)
        else:
            context = contexts[0]

        # Truncate if needed
        if len(context) > max_context_length:
            context = context[:max_context_length - 3] + "..."

        results.append({
            "entity": entity,
            "context": context,
            "text_id": text_id,
        })

    return results


def _extract_sentence_with_entity(text: str, entity: str) -> str:
    """Extract the sentence containing an entity from text."""
    import re

    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)

    entity_lower = entity.lower()
    for sentence in sentences:
        if entity_lower in sentence.lower():
            return sentence.strip()

    # If not found in any sentence, return full text
    return text


def _filter_by_nodes(
    entity_contexts: List[Dict[str, str]],
    nodes_path: Path,
    min_frequency: int = 1,
) -> List[Dict[str, str]]:
    """
    Filter entity-context pairs by a graph nodes file.

    Args:
        entity_contexts: List of entity-context dicts
        nodes_path: Path to graph nodes CSV (with id, frequency, source_text_ids)
        min_frequency: Minimum frequency to include

    Returns:
        Filtered list of entity-context dicts
    """
    # Load valid entities from nodes file
    valid_entities = set()
    entity_participants = {}  # entity -> set of valid participant IDs

    with open(nodes_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            entity_id = row.get("id", "").strip().lower()
            frequency = int(row.get("frequency", 1))
            source_text_ids = row.get("source_text_ids", "[]")

            if frequency >= min_frequency:
                valid_entities.add(entity_id)

                try:
                    participants = json.loads(source_text_ids)
                    entity_participants[entity_id] = set(participants)
                except json.JSONDecodeError:
                    entity_participants[entity_id] = set()

    # Filter entity contexts
    filtered = []
    for ec in entity_contexts:
        entity_lower = ec["entity"].lower()
        text_id = ec["text_id"]

        if entity_lower in valid_entities:
            # Check if this participant mentioned this entity
            valid_pids = entity_participants.get(entity_lower, set())
            if not valid_pids or text_id in valid_pids:
                filtered.append(ec)

    return filtered


def _write_scoring_input_csv(
    entity_contexts: List[Dict[str, str]],
    output_path: Path,
) -> None:
    """Write entity-context pairs to a scoring input CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["entity", "context", "text_id"])
        writer.writeheader()
        writer.writerows(entity_contexts)


# =============================================================================
# ENTITY COMPARE-VIZ COMMAND
# =============================================================================

def add_entity_compare_viz_args(parser: argparse.ArgumentParser) -> None:
    """Add entity comparison visualization arguments."""
    parser.add_argument(
        "input_csv",
        help="Path to input CSV with scored entities.",
    )
    parser.add_argument(
        "--participant-col",
        default="text_id",
        help="Column name for participant IDs (default: text_id).",
    )
    parser.add_argument(
        "--entity-col",
        default="entity",
        help="Column name for entities (default: entity).",
    )
    parser.add_argument(
        "--dimensions",
        default="social,ecological,technological",
        help="Dimensions to visualize, comma-separated (default: social,ecological,technological).",
    )
    parser.add_argument(
        "--viz",
        default="all",
        help=(
            "Visualizations to generate: 'all', 'faceted', 'overlaid', 'heatmap', "
            "'forest', 'similarity-map', or comma-separated list (default: all)."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "aitchison", "emd"],
        default="euclidean",
        help="Distance metric for heatmap/similarity-map (default: euclidean).",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        default=False,
        help="Generate heatmaps and similarity maps for all distance metrics (euclidean, cosine, aitchison, emd).",
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "distribution"],
        default="mean",
        help="Aggregation mode (default: mean).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates viz_* subdirectory).",
    )
    parser.add_argument(
        "--participants",
        default=None,
        help="Specific participants to include, comma-separated (default: all).",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Define groups inline: 'group1:pid1,pid2;group2:pid3,pid4'.",
    )
    parser.add_argument(
        "--groups-file",
        default=None,
        help="Path to JSON or CSV file defining groups.",
    )
    parser.add_argument(
        "--group-by-col",
        default=None,
        help="Column name in input CSV to use for grouping.",
    )
    parser.add_argument(
        "--show-centroids",
        action="store_true",
        default=True,
        help="Show centroid markers on ternary plots.",
    )
    parser.add_argument(
        "--show-hull",
        action="store_true",
        help="Show convex hull on faceted plots.",
    )
    parser.add_argument(
        "--show-ellipses",
        action="store_true",
        help="Show 95%% confidence ellipses per participant on overlaid ternary.",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=4,
        help="Maximum columns in faceted grid (default: 4).",
    )
    parser.add_argument(
        "--similarity-method",
        choices=["mds", "umap"],
        default="mds",
        help="Method for similarity map (default: mds).",
    )
    parser.add_argument(
        "--all-projections",
        action="store_true",
        default=False,
        help="Generate similarity maps using both MDS and UMAP projections.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also save individual ternary plot files per participant.",
    )
    parser.add_argument(
        "--no-numbers",
        action="store_true",
        help="Disable numbered labels on individual ternary plots. Entities are ordered by color spectrum in the legend.",
    )
    parser.add_argument(
        "--scale-min",
        type=int,
        default=None,
        help="Minimum score value (default: auto-detect from score_metadata.json, typically 0).",
    )
    parser.add_argument(
        "--scale-max",
        type=int,
        default=None,
        help="Maximum score value (default: auto-detect from score_metadata.json, typically 100).",
    )


async def run_entity_compare_viz(args: argparse.Namespace) -> int:
    """
    Run standalone comparison visualizations without re-computing distances.

    This generates visualizations from scored entity data. Useful for
    regenerating plots with different parameters without re-running
    the full comparison pipeline.

    Args:
        args: Parsed arguments namespace

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.entity.comparison import ParticipantComparison
    from qualitative_analysis.entity.comparison_viz import ComparisonVisualizer

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"viz_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse dimensions
    dimension_names = [d.strip().lower() for d in args.dimensions.split(",")]
    print(f"Visualizing dimensions: {dimension_names}")

    # Load scores
    print(f"Loading scored entities from: {input_path}")
    comparison = ParticipantComparison()
    comparison.load_scores(
        input_path,
        participant_col=args.participant_col,
        entity_col=args.entity_col,
        dimensions=dimension_names,
    )

    n_participants = len(comparison.scores_by_participant)
    print(f"Loaded {n_participants} participants")

    if n_participants < 2:
        print("Need at least 2 participants for comparison visualizations.")
        return 1

    # Parse participants filter
    participants = None
    if args.participants:
        participants = [p.strip() for p in args.participants.split(",")]
        participants = [p for p in participants if p in comparison.scores_by_participant]

    # Parse groups
    groups = None
    if args.groups:
        groups = parse_inline_groups(args.groups)
    elif args.groups_file:
        groups = load_groups_file(Path(args.groups_file))
    elif args.group_by_col:
        groups = extract_groups_from_data(
            input_path, args.group_by_col, args.participant_col
        )

    if groups:
        comparison.set_groups(groups)

    # Compute distances (needed for heatmap and similarity map)
    aggregate_mode = getattr(args, "aggregate", "mean")
    result = comparison.compute_distances(metric=args.metric, aggregate=aggregate_mode)

    # Determine which visualizations to generate
    viz_types = args.viz.lower().split(",")
    if "all" in viz_types:
        viz_types = ["faceted", "overlaid", "heatmap", "forest", "similarity-map"]

    # Resolve scale
    scale = _resolve_scale(args, input_path.parent)

    visualizer = ComparisonVisualizer(
        comparison,
        scale_min=scale.scale_min,
        scale_max=scale.scale_max,
    )
    if groups:
        visualizer.groups = groups

    generated = []

    if "faceted" in viz_types and len(dimension_names) == 3:
        print("Generating faceted ternary plot...")
        path = output_dir / "faceted_ternary.png"
        visualizer.generate_faceted_ternary(
            output_path=path,
            dimension_names=dimension_names,
            participants=participants,
            max_cols=args.max_cols,
            show_centroid=args.show_centroids,
            show_convex_hull=args.show_hull,
        )
        generated.append(path)

        # Generate individual per-participant ternary files
        if getattr(args, 'individual', False):
            individual_dir = output_dir / "individual_ternary"
            individual_dir.mkdir(parents=True, exist_ok=True)
            print("Generating individual ternary plots...")
            individual_paths = visualizer.generate_individual_ternary(
                output_dir=individual_dir,
                dimension_names=dimension_names,
                participants=participants,
                marker_size=80,
                show_centroid=args.show_centroids,
                show_convex_hull=args.show_hull,
                show_numbers=not getattr(args, 'no_numbers', False),
            )
            generated.extend(individual_paths)
            print(f"  Saved {len(individual_paths)} individual plots to {individual_dir}")

    if "overlaid" in viz_types and len(dimension_names) == 3:
        print("Generating overlaid ternary plot...")
        path = output_dir / "overlaid_ternary.png"
        visualizer.generate_overlaid_ternary(
            output_path=path,
            dimension_names=dimension_names,
            participants=participants,
            show_centroids=args.show_centroids,
            show_confidence_ellipses=getattr(args, 'show_ellipses', False),
        )
        generated.append(path)

    if "heatmap" in viz_types:
        all_metrics = ["euclidean", "cosine", "aitchison", "emd"]
        metrics_to_plot = all_metrics if getattr(args, 'all_metrics', False) else [args.metric]
        for m in metrics_to_plot:
            if m == args.metric:
                m_result = result
            else:
                print(f"Computing {m} distances...")
                m_result = comparison.compute_distances(metric=m, aggregate=aggregate_mode)
            suffix = f"_{m}" if len(metrics_to_plot) > 1 else ""
            path = output_dir / f"distance_heatmap{suffix}.png"
            print(f"Generating distance heatmap ({m})...")
            visualizer.generate_distance_heatmap(
                m_result,
                output_path=path,
                groups=groups,
            )
            generated.append(path)

    if "forest" in viz_types:
        print("Generating forest plots (normalized + raw)...")
        path_norm = output_dir / "forest_plot_normalized.png"
        visualizer.generate_forest_plot(
            result,
            output_path=path_norm,
            dimension_names=dimension_names,
            participants=participants,
            score_mode="normalized",
            groups=groups,
        )
        generated.append(path_norm)

        path_raw = output_dir / "forest_plot_raw.png"
        visualizer.generate_forest_plot(
            result,
            output_path=path_raw,
            dimension_names=dimension_names,
            participants=participants,
            score_mode="raw",
            groups=groups,
        )
        generated.append(path_raw)

    if "similarity-map" in viz_types:
        default_method = getattr(args, "similarity_method", "mds")
        all_metrics = ["euclidean", "cosine", "aitchison", "emd"]
        metrics_to_plot = all_metrics if getattr(args, 'all_metrics', False) else [args.metric]
        projections = ["mds", "umap"] if getattr(args, 'all_projections', False) else [default_method]
        multi_metric = len(metrics_to_plot) > 1
        multi_proj = len(projections) > 1
        for m in metrics_to_plot:
            if m == args.metric:
                m_result = result
            else:
                m_result = comparison.compute_distances(metric=m, aggregate=aggregate_mode)
            for proj in projections:
                parts = []
                if multi_metric:
                    parts.append(m)
                if multi_proj:
                    parts.append(proj)
                suffix = f"_{'_'.join(parts)}" if parts else ""
                path = output_dir / f"similarity_map{suffix}.png"
                print(f"Generating similarity map ({proj.upper()}, {m})...")
                visualizer.generate_similarity_map(
                    m_result,
                    output_path=path,
                    method=proj,
                    groups=groups,
                )
                generated.append(path)

    for p in generated:
        print(f"  Saved: {p}")

    # Emit metadata file in the output directory
    metadata_path = output_dir / "viz_metadata.json"
    _write_viz_metadata(metadata_path, args, {
        'command': 'entity compare-viz',
        'input_csv': str(input_path),
        'output_dir': str(output_dir),
        'n_participants': n_participants,
        'dimensions': dimension_names,
        'viz_types': viz_types,
        'groups': {k: v for k, v in groups.items()} if groups else None,
        'generated_files': [str(p) for p in generated],
    })

    print(f"\nVisualization complete. Output directory: {output_dir}")
    return 0


# =============================================================================
# ENTITY REPORT COMMAND
# =============================================================================

def add_entity_report_args(parser: argparse.ArgumentParser) -> None:
    """Add entity report generation arguments."""
    parser.add_argument(
        "input_csv",
        help="Path to input CSV with scored entities.",
    )
    parser.add_argument(
        "--participant-col",
        default="text_id",
        help="Column name for participant IDs (default: text_id).",
    )
    parser.add_argument(
        "--entity-col",
        default="entity",
        help="Column name for entities (default: entity).",
    )
    parser.add_argument(
        "--dimensions",
        default="social,ecological,technological",
        help="Dimensions to include, comma-separated (default: social,ecological,technological).",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "aitchison", "emd"],
        default="euclidean",
        help="Distance metric (default: euclidean).",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Define groups inline: 'group1:pid1,pid2;group2:pid3,pid4'.",
    )
    parser.add_argument(
        "--groups-file",
        default=None,
        help="Path to JSON or CSV file defining groups.",
    )
    parser.add_argument(
        "--group-by-col",
        default=None,
        help="Column name in input CSV to use for grouping.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for markdown report (default: <input>_report.md).",
    )
    parser.add_argument(
        "--bayesian-dir",
        default=None,
        help="Path to Bayesian results directory to include in report.",
    )


async def run_entity_report(args: argparse.Namespace) -> int:
    """
    Generate a markdown summary report of comparison findings.

    Args:
        args: Parsed arguments namespace

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.entity.comparison import ParticipantComparison
    from qualitative_analysis.entity_report import generate_comparison_report

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    dimension_names = [d.strip().lower() for d in args.dimensions.split(",")]

    # Load scores
    comparison = ParticipantComparison()
    comparison.load_scores(
        input_path,
        participant_col=args.participant_col,
        entity_col=args.entity_col,
        dimensions=dimension_names,
    )

    # Parse groups
    groups = None
    if args.groups:
        groups = parse_inline_groups(args.groups)
    elif args.groups_file:
        groups = load_groups_file(Path(args.groups_file))
    elif args.group_by_col:
        groups = extract_groups_from_data(
            input_path, args.group_by_col, args.participant_col
        )

    if groups:
        comparison.set_groups(groups)

    # Compute distances
    result = comparison.compute_distances(metric=args.metric, aggregate="mean")

    # Generate report
    report_md = generate_comparison_report(
        comparison=comparison,
        result=result,
        dimension_names=dimension_names,
        metric=args.metric,
        input_filename=input_path.name,
        groups=groups,
        bayesian_dir=getattr(args, "bayesian_dir", None),
    )

    # Write report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"Report saved to: {output_path}")
    return 0


# =============================================================================
# ENTITY AGREEMENT COMMAND (Krippendorff's Alpha)
# =============================================================================


def add_entity_agreement_args(parser: argparse.ArgumentParser) -> None:
    """Add entity agreement analysis arguments."""
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with scored entities (from 'qa entity score').",
    )
    parser.add_argument(
        "--mode",
        choices=["run", "participant", "both"],
        default="both",
        help=(
            "Agreement mode: 'run' (LLM run consistency), "
            "'participant' (inter-participant agreement), or 'both' (default)."
        ),
    )
    parser.add_argument(
        "--dimensions",
        default="social,ecological,technological",
        help="Dimensions to analyze, comma-separated (default: social,ecological,technological).",
    )
    parser.add_argument(
        "--participant-col",
        default="text_id",
        help="Column name for participant IDs (default: text_id).",
    )
    parser.add_argument(
        "--entity-col",
        default="entity",
        help="Column name for entities (default: entity).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence interval (default: 1000).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for CI (default: 0.95).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates agreement_* subdirectory).",
    )


async def run_entity_agreement(args: argparse.Namespace) -> int:
    """
    Run inter-rater agreement analysis.

    Computes Krippendorff's Alpha to measure agreement among LLM scoring runs
    (run-level) or among participants (participant-level).
    """
    import json
    from datetime import datetime

    from qualitative_analysis.entity.agreement import compute_agreement

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Parse dimensions
    dimension_names = [d.strip() for d in args.dimensions.split(",")]

    # Load data
    import pandas as pd
    scores_df = pd.read_csv(input_path)

    # Determine modes to run
    modes = []
    if args.mode in ("run", "both"):
        modes.append("run_level")
    if args.mode in ("participant", "both"):
        modes.append("participant_level")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_path.parent / f"agreement_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing Krippendorff's Alpha for {len(dimension_names)} dimensions...")
    print(f"Modes: {modes}")
    print()

    all_results = {}

    for mode in modes:
        print(f"--- {mode.replace('_', ' ').title()} ---")

        results = compute_agreement(
            scores_df=scores_df,
            dimensions=dimension_names,
            mode=mode,
            participant_col=args.participant_col,
            entity_col=args.entity_col,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
        )

        all_results[mode] = {dim: r.to_dict() for dim, r in results.items()}

        # Print results
        for dim, result in results.items():
            print(
                f"  {dim}: α = {result.alpha:.3f} "
                f"[{result.ci_lower:.3f}, {result.ci_upper:.3f}] "
                f"({result.interpretation}) "
                f"[n_units={result.n_units}, n_raters={result.n_raters}]"
            )
        print()

    # Save results
    output_path = output_dir / "agreement_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {output_path}")
    return 0


# =============================================================================
# ENTITY CLUSTER COMMAND
# =============================================================================


def add_entity_cluster_args(parser: argparse.ArgumentParser) -> None:
    """Add entity clustering and PCA arguments."""
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with scored entities (from 'qa entity score').",
    )
    parser.add_argument(
        "--method",
        choices=["hierarchical", "kmeans", "hdbscan", "all"],
        default="hierarchical",
        help="Clustering method (default: hierarchical).",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine", "aitchison"],
        default="euclidean",
        help="Distance metric for clustering (default: euclidean).",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Number of clusters (default: auto-select using silhouette scores).",
    )
    parser.add_argument(
        "--dimensions",
        default="social,ecological,technological",
        help="Dimensions to use, comma-separated (default: social,ecological,technological).",
    )
    parser.add_argument(
        "--participant-col",
        default="text_id",
        help="Column name for participant IDs (default: text_id).",
    )
    parser.add_argument(
        "--entity-col",
        default="entity",
        help="Column name for entities (default: entity).",
    )
    parser.add_argument(
        "--pca",
        action="store_true",
        help="Run PCA on participant score vectors.",
    )
    parser.add_argument(
        "--pca-clr",
        action="store_true",
        help="Apply CLR transform before PCA (only for compositional data).",
    )
    parser.add_argument(
        "--pca-n-components",
        type=int,
        default=None,
        help="Number of PCA components to retain (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates cluster_* subdirectory).",
    )
    parser.add_argument(
        "--viz",
        default="all",
        help="Visualizations: 'all', 'dendrogram', 'elbow', 'scatter', 'biplot', 'scree', or comma-separated list (default: all).",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Known groups for visualization coloring: 'group1:pid1,pid2;group2:pid3,pid4'.",
    )


async def run_entity_cluster(args: argparse.Namespace) -> int:
    """
    Run participant clustering and/or PCA analysis.

    Discovers natural groupings among participants based on their entity
    scoring patterns.
    """
    import json
    from datetime import datetime

    from qualitative_analysis.entity.comparison import ParticipantComparison
    from qualitative_analysis.entity.clustering import (
        cluster_participants,
        pca_on_scores,
        plot_dendrogram,
        plot_elbow,
        plot_cluster_scatter,
        plot_biplot,
        plot_scree,
    )

    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Parse dimensions
    dimension_names = [d.strip() for d in args.dimensions.split(",")]

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_path.parent / f"cluster_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load scores via ParticipantComparison
    comparison = ParticipantComparison()
    comparison.load_scores(
        str(input_path),
        participant_col=args.participant_col,
        entity_col=args.entity_col,
        dimensions=dimension_names,
    )

    print(f"Loaded {len(comparison.scores_by_participant)} participants")
    print(f"Dimensions: {dimension_names}")
    print()

    # Parse known groups for visualization
    groups = None
    if args.groups:
        groups = parse_inline_groups(args.groups)
        print(f"Known groups: {list(groups.keys())}")

    # Parse viz options
    viz_opts = [v.strip().lower() for v in args.viz.split(",")]
    if "all" in viz_opts:
        viz_opts = ["dendrogram", "elbow", "scatter", "biplot", "scree"]

    results = {}

    # Clustering
    methods = [args.method] if args.method != "all" else ["hierarchical", "kmeans"]
    if args.method == "all":
        try:
            import hdbscan  # noqa: F401
            methods.append("hdbscan")
        except ImportError:
            print("Note: hdbscan not installed, skipping HDBSCAN clustering")

    for method in methods:
        print(f"--- {method.upper()} Clustering ---")

        try:
            result = cluster_participants(
                comparison,
                method=method,
                metric=args.metric,
                n_clusters=args.n_clusters,
            )

            results[method] = result.to_dict()

            print(f"  Clusters: {result.n_clusters}")
            print(f"  Quality: {result.quality_metrics}")

            # Get centroids for visualization
            from qualitative_analysis.entity.clustering import _get_participant_centroids
            centroids, _ = _get_participant_centroids(comparison)

            # Generate visualizations
            if method == "hierarchical" and "dendrogram" in viz_opts:
                plot_dendrogram(result, output_dir / f"dendrogram_{method}.png")

            if method == "kmeans" and "elbow" in viz_opts:
                plot_elbow(result, output_dir / f"elbow_{method}.png")

            if "scatter" in viz_opts:
                plot_cluster_scatter(
                    result,
                    centroids,
                    output_dir / f"scatter_{method}.png",
                    dimension_names=dimension_names,
                )

            print()

        except Exception as e:
            print(f"  Error: {e}")
            print()

    # PCA
    if args.pca:
        print("--- PCA Analysis ---")

        pca_result = pca_on_scores(
            comparison,
            n_components=args.pca_n_components,
            use_clr=args.pca_clr,
            standardize=True,
        )

        results["pca"] = pca_result.to_dict()

        print(f"  Components: {len(pca_result.explained_variance_ratio)}")
        print(f"  Cumulative variance: {pca_result.cumulative_variance[-1]*100:.1f}%")
        for i, var in enumerate(pca_result.explained_variance_ratio):
            print(f"    PC{i+1}: {var*100:.1f}%")

        # Visualizations
        if "biplot" in viz_opts:
            plot_biplot(pca_result, output_dir / "pca_biplot.png", groups=groups)

        if "scree" in viz_opts:
            plot_scree(pca_result, output_dir / "pca_scree.png")

        print()

    # Save results
    output_path = output_dir / "cluster_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_dir}")
    return 0


# =============================================================================
# ENTITY DETECTION COMMAND
# =============================================================================

def add_entity_detect_args(parser: argparse.ArgumentParser) -> None:
    """
    Add entity detection arguments to a parser.

    This is used by the unified CLI via `qa entity detect`.
    """
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file with text to analyze.",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Column name for text content (default: text).",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Column name for text IDs (default: auto-generate).",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column name for group assignments (propagated through pipeline).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates timestamped dir).",
    )
    parser.add_argument(
        "--model",
        default="gpt-oss:120b",
        help="LLM model name (default: gpt-oss:120b).",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "mlx", "openai", "anthropic"],
        default="ollama",
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
        default=0.1,
        help="LLM temperature for generation (default: 0.1).",
    )
    # Windowing args (--window-size, --stride, --chunk-unit, --tokenizer, --no-windowing)
    from qualitative_analysis.core.cli_utils import add_common_windowing_args
    add_common_windowing_args(parser)

    parser.add_argument(
        "--prompt-version",
        type=int,
        default=1,
        help="Entity extraction prompt version (default: 1).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print LLM prompts and responses to terminal.",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="(Alias for --verbose) Print LLM prompts and responses to terminal.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of texts to process (for testing).",
    )


async def run_entity_detect(args: argparse.Namespace) -> int:
    """
    Run entity detection on input CSV.

    Extracts entities from text using LLM analysis and outputs:
    - entities.csv: One row per entity occurrence with context
    - entities_summary.csv: One row per text with entity list
    - detect_metadata.json: Processing metadata
    """
    from qualitative_analysis.entity.detector import EntityDetector
    from qualitative_analysis.core.providers import OllamaProvider

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = input_path.parent / f"entities_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM provider
    log_llm = getattr(args, 'verbose', False) or getattr(args, 'log_llm', False)
    if args.provider == "ollama":
        llm_provider = OllamaProvider(
            model_name=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            log_prompts=log_llm,
            log_responses=log_llm,
        )
    elif args.provider == "mlx":
        from qualitative_analysis.core.providers import MLXProvider
        llm_provider = MLXProvider(
            model_name=args.model,
            log_prompts=log_llm,
            log_responses=log_llm,
        )
    else:
        raise SystemExit(f"Provider '{args.provider}' not yet supported. Use 'ollama' or 'mlx'.")

    # Initialize detector
    detector = EntityDetector(
        llm_provider=llm_provider,
        window_size=args.window_size,
        stride=args.stride,
        prompt_version=args.prompt_version,
        chunk_unit=args.chunk_unit,
        tokenizer_name=args.tokenizer,
    )

    # Read input CSV
    print(f"Reading texts from: {input_path}")
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit:
        rows = rows[:args.limit]

    print(f"Processing {len(rows)} texts...")
    print(f"Model: {args.model} ({args.provider})")
    print(f"Windowing: {'disabled' if args.no_windowing else f'{args.window_size} {args.chunk_unit}, stride {args.stride}'}")
    print()

    # Build text_id → group mapping if group column specified
    group_col = getattr(args, 'group_col', None)
    has_groups = False
    text_id_to_group: Dict[str, str] = {}
    if group_col:
        for row in rows:
            tid = row.get(args.id_col, "") if args.id_col else ""
            grp = row.get(group_col, "")
            if tid and grp:
                text_id_to_group[tid] = grp
        has_groups = bool(text_id_to_group)
        if has_groups:
            groups_found = sorted(set(text_id_to_group.values()))
            print(f"Group column '{group_col}': {len(groups_found)} groups ({', '.join(groups_found)})")
        else:
            print(f"Warning: --group-col '{group_col}' specified but no group values found")

    # Process each row
    all_results = []
    all_entity_rows = []

    for i, row in enumerate(rows):
        text = row.get(args.text_col, "")
        text_id = row.get(args.id_col, f"text_{i}") if args.id_col else f"text_{i}"

        if not text.strip():
            logger.warning(f"Empty text for {text_id}, skipping")
            continue

        print(f"Processing {text_id} ({i+1}/{len(rows)})...", end=" ", flush=True)

        result = await detector.detect(
            text=text,
            text_id=text_id,
            use_windowing=not args.no_windowing,
        )
        all_results.append(result)

        print(f"found {len(result.entities)} entities")

        for ec in result.entity_contexts:
            entity_row = {
                "text_id": text_id,
                "entity": ec["entity"],
                "context": ec["context"],
                "window_index": ec["window_index"],
            }
            if has_groups:
                entity_row["group"] = text_id_to_group.get(text_id, "")
            all_entity_rows.append(entity_row)

    # Write entities.csv (one row per entity occurrence)
    fieldnames = ["text_id", "entity", "context", "window_index"]
    if has_groups:
        fieldnames.append("group")
    entities_path = output_dir / "entities.csv"
    with open(entities_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entity_rows)

    # Write entities_summary.csv (one row per text)
    summary_fieldnames = ["text_id", "entity_count", "entities_json"]
    if has_groups:
        summary_fieldnames.append("group")
    summary_path = output_dir / "entities_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for result in all_results:
            summary_row = {
                "text_id": result.text_id,
                "entity_count": len(result.entities),
                "entities_json": json.dumps(result.entities),
            }
            if has_groups:
                summary_row["group"] = text_id_to_group.get(result.text_id, "")
            writer.writerow(summary_row)

    # Compute unique entities across all texts
    unique_entities = set(r["entity"].lower() for r in all_entity_rows)

    # Write metadata
    metadata = {
        "input_file": str(input_path),
        "model": args.model,
        "provider": args.provider,
        "prompt_version": args.prompt_version,
        "windowing": not args.no_windowing,
        "window_size": args.window_size if not args.no_windowing else None,
        "stride": args.stride if not args.no_windowing else None,
        "total_texts": len(all_results),
        "total_entity_occurrences": len(all_entity_rows),
        "unique_entities": len(unique_entities),
        "timestamp": datetime.now().isoformat(),
        "package_version": PACKAGE_VERSION,
    }
    with open(output_dir / "detect_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print()
    print("=" * 50)
    print("Entity Detection Complete")
    print("=" * 50)
    print(f"Texts processed:           {len(all_results)}")
    print(f"Total entity occurrences:  {len(all_entity_rows)}")
    print(f"Unique entities:           {len(unique_entities)}")
    print()
    print(f"Output files:")
    print(f"  {entities_path}")
    print(f"  {summary_path}")
    print(f"  {output_dir / 'detect_metadata.json'}")

    return 0
