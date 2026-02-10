"""
CLI for running relationship extraction and normalization on CSV files.

This module can be used standalone via `qualitative-relationships` command (deprecated)
or through the unified CLI via `qa relationships detect` and `qa relationships normalize`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from qualitative_analysis.core.cli_utils import (
    add_common_csv_args,
    add_common_llm_args,
    add_common_windowing_args,
    emit_deprecation_warning,
    PACKAGE_VERSION,
)
from qualitative_analysis.core.providers import OllamaProvider
from qualitative_analysis.relationships.detector import RelationshipDetector
from qualitative_analysis.relationships.models import Relationship
from qualitative_analysis.relationships.normalizer import RelationshipNormalizer

OUTPUT_CHOICES = {
    "summary",
    "relationships",
    "windows",
    "summary+relationships",
    "summary+windows",
    "relationships+windows",
    "summary+relationships+windows",
    "all",
}


def _parse_output_selection(value: str) -> set[str]:
    if value == "all":
        return {"summary", "relationships", "windows"}
    parts = value.split("+")
    invalid = [
        part for part in parts if part not in {"summary", "relationships", "windows"}
    ]
    if invalid:
        raise ValueError(f"Invalid output selection: {value}")
    return set(parts)


def _parse_entities(raw_value: str | None) -> List[str]:
    if not raw_value:
        return []
    value = raw_value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    delimiter = ";" if ";" in value else ","
    if delimiter in value:
        return [part.strip() for part in value.split(delimiter) if part.strip()]
    return [value]


def add_relationships_detect_args(parser: argparse.ArgumentParser) -> None:
    """
    Add relationship detection arguments to a parser.

    This is used by both the standalone CLI and the unified CLI.
    """
    # Shared CSV args: input_csv, --text-col, --id-col
    add_common_csv_args(parser)
    # Shared LLM args: --model, --base-url, --timeout, --log-llm
    add_common_llm_args(parser)
    # Shared windowing args: --window-size, --stride, --chunk-unit, --tokenizer, --no-windowing
    add_common_windowing_args(parser)

    # Relationship-specific args
    parser.add_argument(
        "--entities-col",
        default=None,
        help="Column name for pre-identified entities (optional).",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column name for group assignments (e.g., treatment vs control).",
    )
    parser.add_argument(
        "--output",
        default="all",
        choices=sorted(OUTPUT_CHOICES),
        help="Output format: summary, relationships, windows, combinations, or all (default: all).",
    )
    parser.add_argument(
        "--no-windows",
        action="store_true",
        help="Disable windows output (shorthand for --output summary+relationships).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: output/ alongside input file).",
    )
    parser.add_argument(
        "--strategy",
        default="two_pass",
        choices=["two_pass", "one_pass"],
        help="Relationship extraction strategy (default: two_pass).",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama"],
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N texts (for testing).",
    )
    # Summary args
    parser.add_argument(
        "--summary-buffer-size",
        type=int,
        default=3,
        help="Number of window summaries to keep for context.",
    )
    parser.add_argument(
        "--summary-prompt-version",
        type=int,
        default=1,
        help="Window summary prompt version.",
    )
    parser.add_argument(
        "--summary-min-windows",
        type=int,
        default=2,
        help="Minimum number of windows required before summaries run.",
    )
    parser.add_argument(
        "--no-summaries",
        action="store_true",
        help="Disable window summarization entirely.",
    )
    parser.add_argument(
        "--include-summaries",
        action="store_true",
        help="Include window summaries in relationship prompts.",
    )
    # Prompt version args
    parser.add_argument(
        "--prompt-version",
        type=int,
        default=None,
        help="Relationship prompt version (strategy-dependent).",
    )
    parser.add_argument(
        "--entity-prompt-version",
        type=int,
        default=1,
        help="Entity extraction prompt version (two-pass only, default: 1).",
    )
    parser.add_argument(
        "--context-buffer-size",
        type=int,
        default=0,
        help="Number of prior windows to keep for entity context (0 disables).",
    )
    parser.add_argument(
        "--coref",
        action="store_true",
        help="Enable LLM-based coreference resolution before extraction.",
    )
    parser.add_argument(
        "--coref-prompt-version",
        type=int,
        default=1,
        help="Coreference resolution prompt version.",
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments (for standalone usage)."""
    parser = argparse.ArgumentParser(
        description="Run relationship extraction on a CSV file."
    )
    add_relationships_detect_args(parser)
    return parser.parse_args()


def _resolve_output_dir(input_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        # Default to 'output' subdirectory next to the input file
        path = input_path.parent / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, Any]:
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


async def run_relationships_detect(args: argparse.Namespace) -> int:
    """
    Run relationship detection with the given arguments.

    This is the main detection logic, callable from both standalone and unified CLI.

    Args:
        args: Parsed arguments namespace with detection configuration

    Returns:
        Exit code (0 for success)
    """
    start_time = datetime.now()

    if args.prompt_version is None:
        args.prompt_version = 1 if args.strategy == "two_pass" else 2
    if args.no_summaries and getattr(args, "include_summaries", False):
        args.include_summaries = False
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Handle --no-windows flag
    if getattr(args, "no_windows", False):
        args.output = "summary+relationships"

    output_selection = _parse_output_selection(args.output)
    write_summary = "summary" in output_selection
    write_relationships = "relationships" in output_selection
    write_windows = "windows" in output_selection

    output_dir = _resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{input_path.stem}_relationships"

    window_size = args.window_size
    stride = args.stride
    if args.no_windowing:
        window_size = 1_000_000
        stride = 1_000_000

    # Build LLM provider in CLI (DI pattern)
    provider = getattr(args, "provider", "ollama")
    if provider == "ollama":
        llm_provider = OllamaProvider(
            model_name=args.model,
            base_url=args.base_url,
            timeout=getattr(args, "timeout", 60.0),
            temperature=getattr(args, "temperature", 0.1),
            log_prompts=args.log_llm,
            log_responses=args.log_llm,
        )
    else:
        raise SystemExit(f"Provider '{provider}' not yet supported. Use 'ollama'.")

    detector = RelationshipDetector(
        llm_provider=llm_provider,
        strategy=args.strategy,
        prompt_version=args.prompt_version,
        entity_prompt_version=args.entity_prompt_version,
        relationship_prompt_version=args.prompt_version,
        coref_prompt_version=args.coref_prompt_version,
        window_size=window_size,
        stride=stride,
        chunk_unit=args.chunk_unit,
        tokenizer_name=args.tokenizer,
        summary_buffer_size=args.summary_buffer_size,
        summary_prompt_version=args.summary_prompt_version,
        enable_summaries=not args.no_summaries,
        include_summaries_in_prompt=getattr(args, "include_summaries", False),
        summary_min_windows=args.summary_min_windows,
        return_windows=write_windows,
        context_buffer_size=args.context_buffer_size,
        coref_resolution=args.coref,
    )

    # Read all rows into memory so we can show progress as (N/M)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no headers.")

        if args.text_col not in reader.fieldnames:
            raise SystemExit(
                f"Text column '{args.text_col}' not found in CSV headers: {reader.fieldnames}"
            )
        if args.id_col and args.id_col not in reader.fieldnames:
            raise SystemExit(
                f"ID column '{args.id_col}' not found in CSV headers: {reader.fieldnames}"
            )
        if getattr(args, "entities_col", None) and args.entities_col not in reader.fieldnames:
            raise SystemExit(
                f"Entities column '{args.entities_col}' not found in CSV headers: {reader.fieldnames}"
            )
        group_col = getattr(args, "group_col", None)
        if group_col and group_col not in reader.fieldnames:
            raise SystemExit(
                f"Group column '{group_col}' not found in CSV headers: {reader.fieldnames}"
            )

        rows = list(reader)

    # Apply --limit
    if getattr(args, "limit", None):
        rows = rows[: args.limit]

    # Build group mapping
    text_id_to_group: Dict[str, str] = {}
    if group_col:
        for i, row in enumerate(rows, start=1):
            tid = row.get(args.id_col) if args.id_col else str(i)
            text_id_to_group[tid] = row.get(group_col, "")
        groups = set(text_id_to_group.values()) - {""}
        if groups:
            print(f"Found {len(groups)} groups: {', '.join(sorted(groups))}")

    print(f"Processing {len(rows)} texts with strategy={args.strategy}...")

    # Determine CSV fieldnames (add group column if present)
    summary_fields = [
        "text_id", "text", "entity_count", "relationship_count",
        "window_count", "entities_json", "strategy", "error",
    ]
    edge_fields = [
        "text_id", "window_index", "source", "target", "type", "description",
    ]
    window_fields = [
        "text_id", "window_index", "window_text", "summary",
        "relationship_count", "entities_json", "relationships_json",
    ]
    if group_col:
        summary_fields.insert(1, "group")
        edge_fields.insert(1, "group")
        window_fields.insert(1, "group")

    summary_writer = None
    summary_handle = None
    relationships_writer = None
    relationships_handle = None
    windows_writer = None
    windows_handle = None

    if write_summary:
        summary_path = run_dir / f"{output_prefix}_summary_{timestamp}.csv"
        summary_writer, summary_handle = _writer(summary_path, summary_fields)

    if write_relationships:
        relationships_path = run_dir / f"{output_prefix}_edges_{timestamp}.csv"
        relationships_writer, relationships_handle = _writer(relationships_path, edge_fields)

    if write_windows:
        windows_path = run_dir / f"{output_prefix}_windows_{timestamp}.csv"
        windows_writer, windows_handle = _writer(windows_path, window_fields)

    # Accumulators for metadata
    total_relationships = 0
    all_unique_entities: set[str] = set()
    errors = 0

    try:
        for i, row in enumerate(rows, start=1):
            text_id = row.get(args.id_col) if args.id_col else str(i)
            text = row.get(args.text_col) or ""
            group = text_id_to_group.get(text_id, "") if group_col else ""
            entities_input = (
                _parse_entities(row.get(args.entities_col))
                if getattr(args, "entities_col", None)
                else []
            )

            error = ""
            result = None
            try:
                result = await detector.detect(
                    text,
                    current_entities=entities_input,
                    text_id=text_id,
                )
            except Exception as exc:
                error = str(exc)
                errors += 1

            if result:
                rel_count = result.metadata.get("relationship_count", 0)
                ent_count = result.metadata.get("entity_count", 0)
                total_relationships += rel_count
                all_unique_entities.update(e.lower() for e in result.entities)
                print(f"Processing {text_id} ({i}/{len(rows)})... "
                      f"found {ent_count} entities, {rel_count} relationships")
            else:
                print(f"Processing {text_id} ({i}/{len(rows)})... ERROR - {error}")

            if write_summary and summary_writer:
                row_data: Dict[str, Any] = {
                    "text_id": text_id,
                    "text": text,
                    "entity_count": result.metadata.get("entity_count") if result else "",
                    "relationship_count": result.metadata.get("relationship_count") if result else "",
                    "window_count": result.metadata.get("window_count") if result else "",
                    "entities_json": json.dumps(result.entities) if result else "",
                    "strategy": result.metadata.get("strategy") if result else "",
                    "error": error,
                }
                if group_col:
                    row_data["group"] = group
                summary_writer.writerow(row_data)

            if result and write_relationships and relationships_writer:
                for rel in result.relationships:
                    edge_row: Dict[str, Any] = {
                        "text_id": text_id,
                        "window_index": rel.window_index,
                        "source": rel.source,
                        "target": rel.target,
                        "type": rel.type,
                        "description": rel.description,
                    }
                    if group_col:
                        edge_row["group"] = group
                    relationships_writer.writerow(edge_row)

            if result and write_windows and windows_writer:
                for window in result.metadata.get("windows", []):
                    win_row: Dict[str, Any] = {
                        "text_id": text_id,
                        "window_index": window.get("window_index", ""),
                        "window_text": window.get("window_text", ""),
                        "summary": window.get("summary", ""),
                        "relationship_count": window.get("relationship_count", ""),
                        "entities_json": json.dumps(window.get("entities", [])),
                        "relationships_json": json.dumps(window.get("relationships", [])),
                    }
                    if group_col:
                        win_row["group"] = group
                    windows_writer.writerow(win_row)

    finally:
        for h in [summary_handle, relationships_handle, windows_handle]:
            if h:
                h.close()

    # Write detect_metadata.json
    metadata = {
        "input_file": str(input_path),
        "model": args.model,
        "provider": provider,
        "temperature": getattr(args, "temperature", 0.1),
        "strategy": args.strategy,
        "prompt_version": args.prompt_version,
        "entity_prompt_version": args.entity_prompt_version,
        "windowing": not args.no_windowing,
        "window_size": window_size if not args.no_windowing else None,
        "stride": stride if not args.no_windowing else None,
        "chunk_unit": args.chunk_unit,
        "group_col": group_col,
        "total_texts": len(rows),
        "total_relationships": total_relationships,
        "unique_entities": len(all_unique_entities),
        "errors": errors,
        "timestamp": start_time.isoformat(),
        "duration_seconds": round((datetime.now() - start_time).total_seconds(), 2),
        "package_version": PACKAGE_VERSION,
    }
    metadata_path = run_dir / "detect_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Collect output paths
    output_paths = [str(metadata_path)]
    if write_summary:
        output_paths.append(str(summary_path))
    if write_relationships:
        output_paths.append(str(relationships_path))
    if write_windows:
        output_paths.append(str(windows_path))

    # Final summary banner
    print()
    print("=" * 50)
    print("Relationship Detection Complete")
    print("=" * 50)
    print(f"  Texts processed:      {len(rows)}")
    print(f"  Total relationships:  {total_relationships}")
    print(f"  Unique entities:      {len(all_unique_entities)}")
    if errors:
        print(f"  Errors:               {errors}")
    print()
    print("Output files:")
    for p in output_paths:
        print(f"  {p}")

    return 0


# =============================================================================
# NORMALIZE COMMAND
# =============================================================================

def add_relationships_normalize_args(parser: argparse.ArgumentParser) -> None:
    """
    Add relationship normalization arguments to a parser.

    This is used by the unified CLI via `qa relationships normalize`.
    """
    parser.add_argument("input_csv", help="Path to input CSV file with relationships.")
    parser.add_argument(
        "--source-col",
        default="source",
        help="Column name for source entities (default: source).",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Column name for target entities (default: target).",
    )
    parser.add_argument(
        "--type-col",
        default="type",
        help="Column name for relationship types (default: type).",
    )
    parser.add_argument(
        "--description-col",
        default="description",
        help="Column name for descriptions (default: description).",
    )
    parser.add_argument(
        "--text-id-col",
        default="text_id",
        help="Column name for text IDs (default: text_id).",
    )
    parser.add_argument(
        "--window-index-col",
        default="window_index",
        help="Column name for window indices (default: window_index).",
    )
    parser.add_argument(
        "--entity-threshold",
        default="moderate",
        help="Entity clustering threshold: conservative (0.85), moderate (0.75), aggressive (0.60), or float.",
    )
    parser.add_argument(
        "--type-threshold",
        default="moderate",
        help="Type clustering threshold: conservative (0.85), moderate (0.75), aggressive (0.60), or float.",
    )
    parser.add_argument(
        "--canonical-method",
        default="shortest",
        choices=["shortest", "frequent", "representative", "llm"],
        help="Method for selecting canonical labels (default: shortest).",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model for embeddings.",
    )
    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="LLM model for canonical_method=llm (default: qwen3).",
    )
    parser.add_argument(
        "--no-normalize-entities",
        action="store_true",
        help="Skip entity normalization (only normalize types).",
    )
    parser.add_argument(
        "--no-normalize-types",
        action="store_true",
        help="Skip type normalization (only normalize entities).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates normalize_* subdirectory).",
    )
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json", "both"],
        help="Output format (default: csv).",
    )
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Save entity and type mapping files separately.",
    )


async def run_relationships_normalize(args: argparse.Namespace) -> int:
    """
    Run relationship normalization with the given arguments.

    This normalizes entities and relationship types using semantic clustering,
    then merges relationships with matching (source, type, target) tuples.

    Args:
        args: Parsed arguments namespace with normalization configuration

    Returns:
        Exit code (0 for success)
    """
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create normalize_* subdirectory next to input or in parent run directory
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"normalize_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read relationships from CSV (supports raw, normalized, and causal formats)
    print(f"Reading relationships from: {input_path}")
    relationships = _read_causal_input_csv(
        input_path,
        source_col=args.source_col,
        target_col=args.target_col,
        type_col=args.type_col,
        description_col=args.description_col,
        text_id_col=args.text_id_col,
        window_index_col=args.window_index_col,
    )

    if not relationships:
        print("No relationships found in input CSV.")
        return 1

    print(f"Loaded {len(relationships)} relationships")

    # Parse thresholds
    entity_threshold = _parse_threshold(args.entity_threshold)
    type_threshold = _parse_threshold(args.type_threshold)

    # Create normalizer
    normalizer = RelationshipNormalizer(
        embedding_model=args.embedding_model,
    )

    # Run normalization
    print(f"Normalizing with entity_threshold={entity_threshold}, type_threshold={type_threshold}")
    result = normalizer.normalize(
        relationships,
        entity_threshold=entity_threshold,
        type_threshold=type_threshold,
        canonical_method=args.canonical_method,
        normalize_entities=not args.no_normalize_entities,
        normalize_types=not args.no_normalize_types,
    )

    print(
        f"Normalization complete: "
        f"{len(result.entity_clusters)} entity clusters, "
        f"{len(result.type_clusters)} type clusters, "
        f"{len(relationships)} → {len(result.normalized_relationships)} relationships"
    )

    # Write outputs
    output_prefix = input_path.stem

    if args.output_format in ("csv", "both"):
        csv_path = output_dir / f"{output_prefix}_normalized.csv"
        _write_normalized_csv(result.normalized_relationships, csv_path)
        print(f"Wrote normalized CSV: {csv_path}")

    if args.output_format in ("json", "both"):
        json_path = output_dir / f"{output_prefix}_normalization_result.json"
        normalizer.save(result, json_path)
        print(f"Wrote normalization result JSON: {json_path}")

    if args.save_maps:
        entity_map_path = output_dir / f"{output_prefix}_entity_map.json"
        type_map_path = output_dir / f"{output_prefix}_type_map.json"

        with open(entity_map_path, "w", encoding="utf-8") as f:
            json.dump(result.entity_mapping, f, indent=2)
        print(f"Wrote entity map: {entity_map_path}")

        with open(type_map_path, "w", encoding="utf-8") as f:
            json.dump(result.type_mapping, f, indent=2)
        print(f"Wrote type map: {type_map_path}")

    return 0


def _parse_threshold(value: str) -> float | str:
    """Parse threshold value (float or preset name)."""
    try:
        return float(value)
    except ValueError:
        return value  # Return as preset name


def _read_relationships_csv(
    path: Path,
    source_col: str = "source",
    target_col: str = "target",
    type_col: str = "type",
    description_col: str = "description",
    text_id_col: str = "text_id",
    window_index_col: str = "window_index",
) -> List[Relationship]:
    """Read relationships from a CSV file."""
    relationships = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        # Check required columns
        if source_col not in reader.fieldnames:
            raise SystemExit(f"Source column '{source_col}' not found. Available: {reader.fieldnames}")
        if target_col not in reader.fieldnames:
            raise SystemExit(f"Target column '{target_col}' not found. Available: {reader.fieldnames}")
        if type_col not in reader.fieldnames:
            raise SystemExit(f"Type column '{type_col}' not found. Available: {reader.fieldnames}")

        for row in reader:
            source = row.get(source_col, "").strip()
            target = row.get(target_col, "").strip()
            rel_type = row.get(type_col, "").strip()

            if not source or not target or not rel_type:
                continue  # Skip incomplete rows

            description = row.get(description_col, "") if description_col in reader.fieldnames else ""
            text_id = row.get(text_id_col, "") if text_id_col in reader.fieldnames else ""
            window_index_str = row.get(window_index_col, "0") if window_index_col in reader.fieldnames else "0"

            try:
                window_index = int(window_index_str) if window_index_str else 0
            except ValueError:
                window_index = 0

            relationships.append(Relationship(
                source=source,
                target=target,
                type=rel_type,
                description=description,
                text_id=text_id,
                window_index=window_index,
            ))

    return relationships


def _write_normalized_csv(
    relationships: List,
    path: Path,
) -> None:
    """Write normalized relationships to CSV."""
    if not relationships:
        return

    fieldnames = [
        "source",
        "target",
        "type",
        "original_source",
        "original_target",
        "original_type",
        "description",
        "count",
        "confidence",
        "window_indices",
        "text_ids",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rel in relationships:
            writer.writerow({
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "original_source": rel.original_source,
                "original_target": rel.original_target,
                "original_type": rel.original_type,
                "description": rel.description,
                "count": rel.count,
                "confidence": rel.confidence,
                "window_indices": json.dumps(rel.window_indices),
                "text_ids": json.dumps(rel.text_ids),
            })


# =============================================================================
# CAUSAL ANALYSIS COMMAND
# =============================================================================

def add_relationships_causal_args(parser: argparse.ArgumentParser) -> None:
    """
    Add causal analysis arguments to a parser.

    This is used by the unified CLI via `qa relationships causal`.
    """
    parser.add_argument("input_csv", help="Path to input CSV file with relationships.")
    parser.add_argument(
        "--source-col",
        default="source",
        help="Column name for source entities (default: source).",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Column name for target entities (default: target).",
    )
    parser.add_argument(
        "--type-col",
        default="type",
        help="Column name for relationship types (default: type).",
    )
    parser.add_argument(
        "--description-col",
        default="description",
        help="Column name for descriptions/evidence (default: description).",
    )
    parser.add_argument(
        "--descriptions-col",
        default=None,
        help="Column name for JSON array of descriptions (for normalized relationships).",
    )
    parser.add_argument(
        "--text-id-col",
        default="text_id",
        help="Column name for text IDs (default: text_id).",
    )
    parser.add_argument(
        "--window-index-col",
        default="window_index",
        help="Column name for window indices (default: window_index).",
    )
    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="LLM model for causal analysis.",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama"],
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3).",
    )
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=3,
        help="Maximum evidence snippets per relationship (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates causal_* subdirectory).",
    )
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json", "both"],
        help="Output format (default: csv).",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Print LLM prompts and responses to the terminal.",
    )
    parser.add_argument(
        "--causal-only",
        action="store_true",
        help="Only output relationships classified as causal.",
    )


async def run_relationships_causal(args: argparse.Namespace) -> int:
    """
    Run causal analysis with the given arguments.

    This analyzes relationships to determine if they represent causal
    relationships and classifies their attributes (polarity, certainty, etc.).

    Args:
        args: Parsed arguments namespace with causal analysis configuration

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.core.providers import OllamaProvider
    from qualitative_analysis.relationships.causal import CausalAnalyzer

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"causal_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read relationships from CSV
    print(f"Reading relationships from: {input_path}")
    relationships = _read_causal_input_csv(
        input_path,
        source_col=args.source_col,
        target_col=args.target_col,
        type_col=args.type_col,
        description_col=args.description_col,
        descriptions_col=args.descriptions_col,
        text_id_col=args.text_id_col,
        window_index_col=args.window_index_col,
    )

    if not relationships:
        print("No relationships found in input CSV.")
        return 1

    print(f"Loaded {len(relationships)} relationships")

    # Create LLM provider
    llm_provider = OllamaProvider(
        model_name=args.model,
        base_url=args.base_url,
        log_prompts=args.log_llm,
        log_responses=args.log_llm,
    )

    # Create analyzer
    analyzer = CausalAnalyzer(temperature=args.temperature)

    # Progress callback
    def on_progress(current: int, total: int) -> None:
        print(f"Progress: {current}/{total} relationships analyzed")

    # Run analysis
    print(f"Analyzing causal attributes with model: {args.model}")
    result = await analyzer.analyze(
        relationships,
        llm_provider,
        max_evidence=args.max_evidence,
        on_progress=on_progress,
    )

    # Print statistics
    stats = result.stats
    print("\nCausal Analysis Results:")
    print(f"  Total relationships: {stats.get('total', 0)}")
    print(f"  Causal: {stats.get('causal_count', 0)}")
    print(f"  Non-causal: {stats.get('non_causal_count', 0)}")

    if stats.get('causal_count', 0) > 0:
        print("\n  Polarity distribution:")
        print(f"    Positive: {stats.get('positive_count', 0)}")
        print(f"    Negative: {stats.get('negative_count', 0)}")
        print(f"    Neutral: {stats.get('neutral_count', 0)}")
        print("\n  Certainty distribution:")
        print(f"    Certain: {stats.get('certain_count', 0)}")
        print(f"    Likely: {stats.get('likely_count', 0)}")
        print(f"    Possible: {stats.get('possible_count', 0)}")
        print("\n  Explicit vs Implicit:")
        print(f"    Explicit: {stats.get('explicit_count', 0)}")
        print(f"    Implicit: {stats.get('implicit_count', 0)}")

    # Filter if causal_only
    output_rels = result.causal_relationships
    if args.causal_only:
        output_rels = [r for r in output_rels if r.causal.is_causal]
        print(f"\nFiltered to {len(output_rels)} causal relationships")

    # Write outputs
    output_prefix = input_path.stem

    if args.output_format in ("csv", "both"):
        csv_path = output_dir / f"{output_prefix}_causal.csv"
        _write_causal_csv(output_rels, csv_path)
        print(f"\nWrote causal CSV: {csv_path}")

    if args.output_format in ("json", "both"):
        json_path = output_dir / f"{output_prefix}_causal_result.json"
        analyzer.save(result, json_path)
        print(f"Wrote causal result JSON: {json_path}")

    return 0


def _read_causal_input_csv(
    path: Path,
    source_col: str = "source",
    target_col: str = "target",
    type_col: str = "type",
    description_col: str = "description",
    descriptions_col: Optional[str] = None,
    text_id_col: str = "text_id",
    window_index_col: str = "window_index",
) -> List:
    """Read relationships from CSV for causal analysis.

    Handles both raw Relationship format and NormalizedRelationship format.
    """
    from qualitative_analysis.relationships.models import (
        Relationship,
        NormalizedRelationship,
    )

    relationships = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        # Check required columns
        if source_col not in reader.fieldnames:
            raise SystemExit(f"Source column '{source_col}' not found. Available: {reader.fieldnames}")
        if target_col not in reader.fieldnames:
            raise SystemExit(f"Target column '{target_col}' not found. Available: {reader.fieldnames}")
        if type_col not in reader.fieldnames:
            raise SystemExit(f"Type column '{type_col}' not found. Available: {reader.fieldnames}")

        # Detect if this is normalized relationship format
        is_normalized = "original_source" in reader.fieldnames or "count" in reader.fieldnames

        for row in reader:
            source = row.get(source_col, "").strip()
            target = row.get(target_col, "").strip()
            rel_type = row.get(type_col, "").strip()

            if not source or not target or not rel_type:
                continue

            description = row.get(description_col, "") if description_col in reader.fieldnames else ""

            # Handle descriptions array (for normalized relationships)
            descriptions = []
            if descriptions_col and descriptions_col in reader.fieldnames:
                try:
                    descriptions = json.loads(row.get(descriptions_col, "[]"))
                except json.JSONDecodeError:
                    descriptions = [description] if description else []
            elif description:
                descriptions = [description]

            text_id = row.get(text_id_col, "") if text_id_col in reader.fieldnames else ""

            # Handle window indices (could be JSON array or single value)
            window_indices = []
            if "window_indices" in reader.fieldnames:
                try:
                    window_indices = json.loads(row.get("window_indices", "[]"))
                except json.JSONDecodeError:
                    pass
            elif window_index_col in reader.fieldnames:
                try:
                    window_indices = [int(row.get(window_index_col, "0"))]
                except ValueError:
                    window_indices = [0]

            # Handle text_ids array
            text_ids = []
            if "text_ids" in reader.fieldnames:
                try:
                    text_ids = json.loads(row.get("text_ids", "[]"))
                except json.JSONDecodeError:
                    text_ids = [text_id] if text_id else []
            elif text_id:
                text_ids = [text_id]

            if is_normalized:
                relationships.append(NormalizedRelationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    original_source=row.get("original_source", source),
                    original_target=row.get("original_target", target),
                    original_type=row.get("original_type", rel_type),
                    description=description,
                    descriptions=descriptions,
                    count=int(row.get("count", "1")) if "count" in reader.fieldnames else 1,
                    window_indices=window_indices,
                    text_ids=text_ids,
                ))
            else:
                relationships.append(Relationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    description=description,
                    text_id=text_id,
                    window_index=window_indices[0] if window_indices else 0,
                ))

    return relationships


def _write_causal_csv(relationships: List, path: Path) -> None:
    """Write causal analysis results to CSV."""
    if not relationships:
        return

    fieldnames = [
        "source",
        "target",
        "type",
        "description",
        "is_causal",
        "polarity",
        "certainty",
        "explicit_vs_implicit",
        "reasoning",
        "original_source",
        "original_target",
        "original_type",
        "count",
        "window_indices",
        "text_ids",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rel in relationships:
            writer.writerow({
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "description": rel.description,
                "is_causal": rel.causal.is_causal,
                "polarity": rel.causal.polarity or "",
                "certainty": rel.causal.certainty or "",
                "explicit_vs_implicit": rel.causal.explicit_vs_implicit or "",
                "reasoning": rel.causal.reasoning,
                "original_source": rel.original_source,
                "original_target": rel.original_target,
                "original_type": rel.original_type,
                "count": rel.count,
                "window_indices": json.dumps(rel.window_indices),
                "text_ids": json.dumps(rel.text_ids),
            })


# =============================================================================
# GRAPH GENERATION COMMAND
# =============================================================================

def add_relationships_graph_args(parser: argparse.ArgumentParser) -> None:
    """
    Add graph generation arguments to a parser.

    This is used by the unified CLI via `qa relationships graph`.
    """
    parser.add_argument("input_csv", help="Path to input CSV file with relationships.")
    parser.add_argument(
        "--source-col",
        default="source",
        help="Column name for source entities (default: source).",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Column name for target entities (default: target).",
    )
    parser.add_argument(
        "--type-col",
        default="type",
        help="Column name for relationship types (default: type).",
    )
    parser.add_argument(
        "--description-col",
        default="description",
        help="Column name for descriptions (default: description).",
    )
    parser.add_argument(
        "--text-id-col",
        default="text_id",
        help="Column name for text IDs (default: text_id).",
    )
    parser.add_argument(
        "--window-index-col",
        default="window_index",
        help="Column name for window indices (default: window_index).",
    )
    parser.add_argument(
        "--layout",
        default="spring",
        choices=["spring", "circular", "kamada_kawai", "semantic", "hierarchical", "community"],
        help="Graph layout algorithm. 'community' detects groups via Louvain (default: spring).",
    )
    parser.add_argument(
        "--layout-spacing",
        type=float,
        default=1.0,
        help="Multiplier for node spacing in layouts (default: 1.0).",
    )
    parser.add_argument(
        "--color-by",
        default="polarity",
        choices=["polarity", "type", "none"],
        help="Edge coloring mode for causal graphs (default: polarity).",
    )
    parser.add_argument(
        "--labels",
        default="all",
        help="Label display mode: all, none, top:N, pagerank:N (default: all).",
    )
    parser.add_argument(
        "--scope",
        default="aggregate",
        help="Graph scope: aggregate, individual, or text:ID (default: aggregate).",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        default=True,
        help="Show legend for colors and styles (default: True).",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide legend.",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Embedding model for semantic layout (default: Qwen/Qwen3-Embedding-0.6B). Alternative: 'all-MiniLM-L6-v2' (faster).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.3,
        help="UMAP min_dist parameter (default: 0.3). Higher values spread nodes more for better clustering.",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=int,
        default=1,
        help="Minimum edge weight to include (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates graph_* subdirectory).",
    )
    parser.add_argument(
        "--output-format",
        default="csv,json",
        help="Output formats, comma-separated: csv, json, png, gexf (default: csv,json).",
    )
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="Show relationship types on edges in PNG output.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PNG output (default: 150).",
    )
    parser.add_argument(
        "--figsize",
        default="16,12",
        help="Figure size for PNG as 'width,height' in inches (default: 16,12).",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Custom title for PNG visualization.",
    )
    parser.add_argument(
        "--no-causal",
        action="store_true",
        help="Exclude causal attributes from graph (included by default).",
    )
    parser.add_argument(
        "--cluster-nodes",
        action="store_true",
        help="Cluster nodes by semantic similarity and color by cluster (requires --layout semantic).",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum nodes per cluster when using --cluster-nodes (default: 3).",
    )
    parser.add_argument(
        "--no-cluster-hulls",
        action="store_true",
        help="Hide convex hull regions around clusters.",
    )
    parser.add_argument(
        "--cluster-label-min-size",
        type=int,
        default=5,
        help="Minimum cluster size to show label (reduces clutter). Default: 5.",
    )
    parser.add_argument(
        "--cluster-label-model",
        default=None,
        help="LLM model to generate descriptive labels for clusters (e.g., 'qwen3:8b'). Requires --cluster-nodes.",
    )
    parser.add_argument(
        "--cluster-label-base-url",
        default="http://localhost:11434",
        help="Base URL for cluster label LLM provider (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--show-node-roles",
        action="store_true",
        help="Show different node shapes by role: triangle for source-only (causes), square for target-only (effects), circle for both (mediators).",
    )


async def run_relationships_graph(args: argparse.Namespace) -> int:
    """
    Run graph generation with the given arguments.

    This generates a relationship graph with node metrics (degree, betweenness,
    PageRank) and exports to multiple formats.

    Args:
        args: Parsed arguments namespace with graph generation configuration

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.relationships.graph import RelationshipGraph

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"graph_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read relationships from CSV
    print(f"Reading relationships from: {input_path}")
    relationships = _read_graph_input_csv(
        input_path,
        source_col=args.source_col,
        target_col=args.target_col,
        type_col=args.type_col,
        description_col=args.description_col,
        text_id_col=args.text_id_col,
        window_index_col=args.window_index_col,
    )

    if not relationships:
        print("No relationships found in input CSV.")
        return 1

    print(f"Loaded {len(relationships)} relationships")

    # Parse output formats
    formats = [f.strip().lower() for f in args.output_format.split(",")]
    valid_formats = {"csv", "json", "png", "gexf"}
    formats = [f for f in formats if f in valid_formats]
    if not formats:
        formats = ["csv", "json"]

    # Parse figure size
    try:
        figsize = tuple(int(x.strip()) for x in args.figsize.split(","))
        if len(figsize) != 2:
            figsize = (16, 12)
    except ValueError:
        figsize = (16, 12)

    # Determine legend visibility
    show_legend = args.show_legend and not args.no_legend

    # Handle scope
    graph = RelationshipGraph()
    include_causal = not args.no_causal

    # Determine clustering options
    cluster_nodes = args.cluster_nodes
    show_cluster_hulls = not args.no_cluster_hulls

    # Get cluster labeling options
    cluster_label_model = args.cluster_label_model
    cluster_label_base_url = args.cluster_label_base_url
    show_node_roles = args.show_node_roles

    if args.scope == "individual":
        # Build individual graphs per text_id
        print(f"Building individual graphs per text_id...")
        results = graph.build_individual_graphs(
            relationships,
            output_dir,
            formats=formats,
            layout=args.layout,
            embedding_model=args.embedding_model,
            umap_min_dist=args.umap_min_dist,
            color_by=args.color_by,
            labels_mode=args.labels,
            show_legend=show_legend,
            layout_spacing=args.layout_spacing,
            min_edge_weight=args.min_edge_weight,
            cluster_nodes=cluster_nodes,
            min_cluster_size=args.min_cluster_size,
            show_cluster_hulls=show_cluster_hulls,
            cluster_label_min_size=args.cluster_label_min_size,
            cluster_label_model=cluster_label_model,
            cluster_label_base_url=cluster_label_base_url,
            show_node_roles=show_node_roles,
        )
        print(f"\nBuilt {len(results)} individual graphs:")
        for text_id, paths in results.items():
            print(f"  {text_id}: {len(paths)} files")
        return 0

    elif args.scope.startswith("text:"):
        # Filter to specific text_id
        target_text_id = args.scope[5:]  # Remove "text:" prefix
        filtered_rels = []
        for rel in relationships:
            text_ids = []
            if hasattr(rel, "text_ids") and rel.text_ids:
                text_ids = rel.text_ids
            elif hasattr(rel, "text_id") and rel.text_id:
                text_ids = [rel.text_id]
            if target_text_id in text_ids:
                filtered_rels.append(rel)

        if not filtered_rels:
            print(f"No relationships found for text_id: {target_text_id}")
            return 1

        print(f"Filtered to {len(filtered_rels)} relationships for text_id: {target_text_id}")
        relationships = filtered_rels

    # Build aggregate graph
    print(f"Building graph with min_edge_weight={args.min_edge_weight}")
    graph_data = graph.build(
        relationships,
        include_causal=include_causal,
        min_edge_weight=args.min_edge_weight,
    )

    # Print summary
    metrics = graph_data.metrics_summary
    print(f"\nGraph Summary:")
    print(f"  Nodes: {metrics.get('node_count', 0)}")
    print(f"  Edges: {metrics.get('edge_count', 0)}")
    print(f"  Density: {metrics.get('density', 0):.4f}")
    print(f"  Average degree: {metrics.get('average_degree', 0):.2f}")
    if metrics.get('connected_components'):
        print(f"  Connected components: {metrics.get('connected_components')}")
    if metrics.get('top_nodes_by_pagerank'):
        print(f"  Top nodes (PageRank): {', '.join(metrics['top_nodes_by_pagerank'][:5])}")

    # Export
    output_prefix = input_path.stem
    saved_paths = []

    if "json" in formats:
        json_path = output_dir / f"{output_prefix}_graph.json"
        graph.to_json(json_path)
        saved_paths.append(str(json_path))
        print(f"\nWrote graph JSON: {json_path}")

    if "csv" in formats:
        nodes_path, edges_path = graph.to_csv(output_dir, f"{output_prefix}_graph")
        saved_paths.extend([nodes_path, edges_path])
        print(f"Wrote nodes CSV: {nodes_path}")
        print(f"Wrote edges CSV: {edges_path}")

    if "png" in formats:
        png_path = output_dir / f"{output_prefix}_graph.png"
        graph.to_png(
            png_path,
            layout=args.layout,
            embedding_model=args.embedding_model,
            umap_min_dist=args.umap_min_dist,
            figsize=figsize,
            dpi=args.dpi,
            show_edge_labels=args.show_edge_labels,
            title=args.title,
            color_by=args.color_by,
            labels_mode=args.labels,
            show_legend=show_legend,
            layout_spacing=args.layout_spacing,
            cluster_nodes=cluster_nodes,
            min_cluster_size=args.min_cluster_size,
            show_cluster_hulls=show_cluster_hulls,
            cluster_label_min_size=args.cluster_label_min_size,
            cluster_label_model=cluster_label_model,
            cluster_label_base_url=cluster_label_base_url,
            show_node_roles=show_node_roles,
        )
        saved_paths.append(str(png_path))
        print(f"Wrote graph PNG: {png_path}")

    if "gexf" in formats:
        gexf_path = output_dir / f"{output_prefix}_graph.gexf"
        graph.to_gexf(gexf_path)
        saved_paths.append(str(gexf_path))
        print(f"Wrote graph GEXF: {gexf_path}")

    return 0


def _read_graph_input_csv(
    path: Path,
    source_col: str = "source",
    target_col: str = "target",
    type_col: str = "type",
    description_col: str = "description",
    text_id_col: str = "text_id",
    window_index_col: str = "window_index",
) -> List:
    """Read relationships from CSV for graph generation.

    Handles raw, normalized, and causal relationship formats.
    """
    from qualitative_analysis.relationships.models import (
        Relationship,
        NormalizedRelationship,
        CausalRelationship,
        CausalAttributes,
    )

    relationships = []

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        # Check required columns
        if source_col not in reader.fieldnames:
            raise SystemExit(f"Source column '{source_col}' not found. Available: {reader.fieldnames}")
        if target_col not in reader.fieldnames:
            raise SystemExit(f"Target column '{target_col}' not found. Available: {reader.fieldnames}")
        if type_col not in reader.fieldnames:
            raise SystemExit(f"Type column '{type_col}' not found. Available: {reader.fieldnames}")

        # Detect format
        is_causal = "is_causal" in reader.fieldnames
        is_normalized = "original_source" in reader.fieldnames or "count" in reader.fieldnames

        for row in reader:
            source = row.get(source_col, "").strip()
            target = row.get(target_col, "").strip()
            rel_type = row.get(type_col, "").strip()

            if not source or not target or not rel_type:
                continue

            description = row.get(description_col, "") if description_col in reader.fieldnames else ""
            text_id = row.get(text_id_col, "") if text_id_col in reader.fieldnames else ""

            # Handle arrays
            window_indices = []
            if "window_indices" in reader.fieldnames:
                try:
                    window_indices = json.loads(row.get("window_indices", "[]"))
                except json.JSONDecodeError:
                    pass
            elif window_index_col in reader.fieldnames:
                try:
                    window_indices = [int(row.get(window_index_col, "0"))]
                except ValueError:
                    window_indices = [0]

            text_ids = []
            if "text_ids" in reader.fieldnames:
                try:
                    text_ids = json.loads(row.get("text_ids", "[]"))
                except json.JSONDecodeError:
                    text_ids = [text_id] if text_id else []
            elif text_id:
                text_ids = [text_id]

            if is_causal:
                # Parse causal attributes
                is_causal_val = row.get("is_causal", "").lower() in ("true", "1", "yes")
                causal = CausalAttributes(
                    is_causal=is_causal_val,
                    polarity=row.get("polarity") or None,
                    certainty=row.get("certainty") or None,
                    explicit_vs_implicit=row.get("explicit_vs_implicit") or None,
                    reasoning=row.get("reasoning", ""),
                )
                relationships.append(CausalRelationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    description=description,
                    causal=causal,
                    original_source=row.get("original_source", source),
                    original_target=row.get("original_target", target),
                    original_type=row.get("original_type", rel_type),
                    count=int(row.get("count", "1")) if "count" in reader.fieldnames else 1,
                    window_indices=window_indices,
                    text_ids=text_ids,
                ))
            elif is_normalized:
                relationships.append(NormalizedRelationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    original_source=row.get("original_source", source),
                    original_target=row.get("original_target", target),
                    original_type=row.get("original_type", rel_type),
                    description=description,
                    count=int(row.get("count", "1")) if "count" in reader.fieldnames else 1,
                    window_indices=window_indices,
                    text_ids=text_ids,
                ))
            else:
                relationships.append(Relationship(
                    source=source,
                    target=target,
                    type=rel_type,
                    description=description,
                    text_id=text_id,
                    window_index=window_indices[0] if window_indices else 0,
                ))

    return relationships


# =============================================================================
# VERIFICATION COMMAND
# =============================================================================

def add_relationships_verify_args(parser: argparse.ArgumentParser) -> None:
    """
    Add relationship verification arguments to a parser.

    This is used by the unified CLI via `qa relationships verify`.
    """
    parser.add_argument("input_csv", help="Path to input CSV file with relationships.")
    parser.add_argument(
        "source_csv",
        nargs="?",
        default=None,
        help="Path to CSV file with source texts (text_id, text columns). Optional if --windows is provided.",
    )
    parser.add_argument(
        "--windows",
        dest="windows_csv",
        default=None,
        help="Path to windows CSV file (text_id, window_index, window_text). Preferred over source texts for accurate verification.",
    )
    parser.add_argument(
        "--window-text-col",
        default="window_text",
        help="Column name for window text in windows CSV (default: window_text).",
    )
    parser.add_argument(
        "--source-col",
        default="source",
        help="Column name for source entities (default: source).",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Column name for target entities (default: target).",
    )
    parser.add_argument(
        "--type-col",
        default="type",
        help="Column name for relationship types (default: type).",
    )
    parser.add_argument(
        "--description-col",
        default="description",
        help="Column name for descriptions (default: description).",
    )
    parser.add_argument(
        "--text-id-col",
        default="text_id",
        help="Column name for text IDs in relationships CSV (default: text_id).",
    )
    parser.add_argument(
        "--window-index-col",
        default="window_index",
        help="Column name for window indices (default: window_index).",
    )
    parser.add_argument(
        "--source-text-id-col",
        default="text_id",
        help="Column name for text IDs in source texts CSV (default: text_id).",
    )
    parser.add_argument(
        "--source-text-col",
        default="text",
        help="Column name for text content in source texts CSV (default: text).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence to accept a relationship (default: 0.7).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Maximum relationships per LLM call (default: 20).",
    )
    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="LLM model for verification.",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama"],
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1 for consistent verification).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: creates verify_* subdirectory).",
    )
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "json", "both"],
        help="Output format (default: csv).",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Print LLM prompts and responses to the terminal.",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Only output relationships that passed verification.",
    )
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Include rejected relationships in a separate output file.",
    )


async def run_relationships_verify(args: argparse.Namespace) -> int:
    """
    Run relationship verification with the given arguments.

    This verifies extracted relationships by re-prompting the LLM with the
    original source texts to determine if each relationship is actually
    supported by the text.

    Supports two modes:
    1. Window-level verification (preferred): Uses --windows CSV with window_text
    2. Document-level verification: Uses source_csv with full document text

    Args:
        args: Parsed arguments namespace with verification configuration

    Returns:
        Exit code (0 for success)
    """
    from qualitative_analysis.core.providers import OllamaProvider
    from qualitative_analysis.relationships.verifier import RelationshipVerifier

    input_path = Path(args.input_csv)

    # Validate inputs - need either windows or source texts
    windows_path = Path(args.windows_csv) if args.windows_csv else None
    source_path = Path(args.source_csv) if args.source_csv else None

    if not input_path.exists():
        raise SystemExit(f"Relationships CSV not found: {input_path}")

    if not windows_path and not source_path:
        raise SystemExit("Must provide either source_csv or --windows (or both)")

    if windows_path and not windows_path.exists():
        raise SystemExit(f"Windows CSV not found: {windows_path}")
    if source_path and not source_path.exists():
        raise SystemExit(f"Source texts CSV not found: {source_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parent = input_path.parent
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        output_dir = parent / f"verify_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read relationships from CSV (supports both raw and normalized formats)
    print(f"Reading relationships from: {input_path}")
    relationships = _read_causal_input_csv(
        input_path,
        source_col=args.source_col,
        target_col=args.target_col,
        type_col=args.type_col,
        description_col=args.description_col,
        text_id_col=args.text_id_col,
        window_index_col=args.window_index_col,
    )

    if not relationships:
        print("No relationships found in input CSV.")
        return 1

    print(f"Loaded {len(relationships)} relationships")

    # Read window texts if provided (preferred)
    window_texts: Dict[tuple, str] = {}
    if windows_path:
        print(f"Reading window texts from: {windows_path}")
        window_texts = _read_windows_csv(
            windows_path,
            text_id_col=args.source_text_id_col,
            window_index_col=args.window_index_col,
            window_text_col=args.window_text_col,
        )
        print(f"Loaded {len(window_texts)} windows (window-level verification)")

    # Read source texts as fallback
    source_texts: Dict[str, str] = {}
    if source_path:
        print(f"Reading source texts from: {source_path}")
        source_texts = _read_source_texts_csv(
            source_path,
            text_id_col=args.source_text_id_col,
            text_col=args.source_text_col,
        )
        print(f"Loaded {len(source_texts)} source texts (document-level fallback)")

    if not source_texts and not window_texts:
        print("No source texts or window texts found in CSVs.")
        return 1

    # Create LLM provider
    llm_provider = OllamaProvider(
        model_name=args.model,
        base_url=args.base_url,
        log_prompts=args.log_llm,
        log_responses=args.log_llm,
    )

    # Create verifier
    verifier = RelationshipVerifier(
        temperature=args.temperature,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
    )

    # Progress callback
    def on_progress(current: int, total: int) -> None:
        print(f"Progress: {current}/{total} relationships verified")

    # Run verification
    mode = "window-level" if window_texts else "document-level"
    print(f"Verifying relationships with model: {args.model}")
    print(f"Verification mode: {mode}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    result = await verifier.verify(
        relationships,
        source_texts=source_texts,
        llm_provider=llm_provider,
        confidence_threshold=args.confidence_threshold,
        on_progress=on_progress,
        window_texts=window_texts,
    )

    # Print statistics
    stats = result.statistics
    print("\nVerification Results:")
    print(f"  Total relationships: {stats.get('total', 0)}")
    print(f"  Verified: {stats.get('verified_count', 0)}")
    print(f"  Rejected: {stats.get('rejected_count', 0)}")
    print(f"  Verification rate: {stats.get('verification_rate', 0):.1f}%")
    print(f"  Average confidence: {stats.get('avg_confidence', 0):.3f}")

    # Write outputs
    output_prefix = input_path.stem

    # Determine what to output
    if args.verified_only:
        output_rels = result.verified_relationships
        suffix = "_verified"
    else:
        output_rels = result.all_verifications
        suffix = "_all"

    if args.output_format in ("csv", "both"):
        csv_path = output_dir / f"{output_prefix}{suffix}.csv"
        _write_verified_csv(output_rels, csv_path)
        print(f"\nWrote verified CSV: {csv_path}")

        if args.include_rejected and not args.verified_only:
            rejected_path = output_dir / f"{output_prefix}_rejected.csv"
            _write_verified_csv(result.rejected_relationships, rejected_path)
            print(f"Wrote rejected CSV: {rejected_path}")

    if args.output_format in ("json", "both"):
        json_path = output_dir / f"{output_prefix}_verification_result.json"
        verifier.save(result, json_path)
        print(f"Wrote verification result JSON: {json_path}")

    return 0


def _read_source_texts_csv(
    path: Path,
    text_id_col: str = "text_id",
    text_col: str = "text",
) -> Dict[str, str]:
    """Read source texts from CSV into a dict mapping text_id to text."""
    source_texts = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        if text_id_col not in reader.fieldnames:
            raise SystemExit(f"Text ID column '{text_id_col}' not found. Available: {reader.fieldnames}")
        if text_col not in reader.fieldnames:
            raise SystemExit(f"Text column '{text_col}' not found. Available: {reader.fieldnames}")

        for row in reader:
            text_id = row.get(text_id_col, "").strip()
            text = row.get(text_col, "").strip()
            if text_id and text:
                source_texts[text_id] = text

    return source_texts


def _read_windows_csv(
    path: Path,
    text_id_col: str = "text_id",
    window_index_col: str = "window_index",
    window_text_col: str = "window_text",
) -> Dict[tuple, str]:
    """Read windows from CSV into a dict mapping (text_id, window_index) to window text."""
    from typing import Tuple
    window_texts: Dict[Tuple[str, int], str] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}

        # Check required columns
        if text_id_col not in reader.fieldnames:
            raise SystemExit(f"Text ID column '{text_id_col}' not found. Available: {reader.fieldnames}")
        if window_index_col not in reader.fieldnames:
            raise SystemExit(f"Window index column '{window_index_col}' not found. Available: {reader.fieldnames}")
        if window_text_col not in reader.fieldnames:
            raise SystemExit(f"Window text column '{window_text_col}' not found. Available: {reader.fieldnames}")

        for row in reader:
            text_id = row.get(text_id_col, "").strip()
            window_text = row.get(window_text_col, "").strip()
            try:
                window_index = int(row.get(window_index_col, "0"))
            except ValueError:
                window_index = 0

            if text_id and window_text:
                window_texts[(text_id, window_index)] = window_text

    return window_texts


def _write_verified_csv(relationships: List, path: Path) -> None:
    """Write verified relationships to CSV."""
    if not relationships:
        return

    fieldnames = [
        "source",
        "target",
        "type",
        "description",
        "verified",
        "verification_confidence",
        "verification_note",
        "text_id",
        "window_index",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rel in relationships:
            writer.writerow({
                "source": rel.source,
                "target": rel.target,
                "type": rel.type,
                "description": rel.description,
                "verified": rel.verified,
                "verification_confidence": rel.verification_confidence,
                "verification_note": rel.verification_note,
                "text_id": rel.text_id,
                "window_index": rel.window_index,
            })


def main() -> None:
    """
    Legacy entry point for standalone CLI.

    DEPRECATED: Use `qa relationships detect` instead.
    """
    emit_deprecation_warning("qualitative-relationships", "qa relationships detect")
    args = _parse_args()
    raise SystemExit(asyncio.run(run_relationships_detect(args)))


if __name__ == "__main__":
    main()
