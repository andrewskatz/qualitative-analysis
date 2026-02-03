"""
CLI for running figurative language detection on CSV files.

This module can be used standalone via `qualitative-analysis` command (deprecated)
or through the unified CLI via `qa figurative detect`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set

from qualitative_analysis.core.cli_utils import emit_deprecation_warning
from qualitative_analysis.figurative.detector import FigurativeDetector
from qualitative_analysis.figurative.models import DetectionCheckpoint

OUTPUT_CHOICES = {
    "summary",
    "instances",
    "windows",
    "summary+instances",
    "summary+windows",
    "instances+windows",
    "summary+instances+windows",
    "all",
}


def _parse_output_selection(value: str) -> set[str]:
    if value == "all":
        return {"summary", "instances", "windows"}
    parts = value.split("+")
    invalid = [part for part in parts if part not in {"summary", "instances", "windows"}]
    if invalid:
        raise ValueError(f"Invalid output selection: {value}")
    return set(parts)


def add_figurative_detect_args(parser: argparse.ArgumentParser) -> None:
    """
    Add figurative detection arguments to a parser.
    
    This is used by both the standalone CLI and the unified CLI.
    """
    parser.add_argument("input_csv", help="Path to input CSV file.")
    parser.add_argument("--id-col", default=None, help="Column name for IDs (optional).")
    parser.add_argument("--text-col", default="text", help="Column name for text.")
    parser.add_argument(
        "--output",
        default="summary",
        choices=sorted(OUTPUT_CHOICES),
        help="Output format: summary, instances, windows, combinations, or all.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: output/).",
    )
    parser.add_argument("--model", default="qwen3:30b-a3b-instruct-2507-q4_K_M", help="Ollama model name.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Print LLM prompts and responses to the terminal.",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout seconds.")
    parser.add_argument("--window-size", type=int, default=3, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=2, help="Sliding window stride.")
    parser.add_argument(
        "--chunk-unit",
        default="sentences",
        choices=["sentences", "tokens"],
        help="Chunking unit for windowing (sentences or tokens).",
    )
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="Tokenizer name for token chunking (tiktoken encoding).",
    )
    parser.add_argument(
        "--no-windowing",
        action="store_true",
        help="Disable windowing and analyze each text as a single window.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold.")
    parser.add_argument(
        "--context-window",
        type=int,
        default=5,
        help="Number of previous window summaries to include as context (default: 5).",
    )
    parser.add_argument("--prompt-version", type=int, default=1, help="Prompt version.")
    parser.add_argument(
        "--types",
        default=None,
        help=(
            "Comma-separated figurative types to detect (e.g., 'metaphor,analogy'). "
            "Valid types: metaphor, simile, personification, hyperbole, idiom, irony, "
            "extended_metaphor, analogy, other. Default: all types."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file path for resumable processing.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N texts (default: 10).",
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments (for standalone usage)."""
    parser = argparse.ArgumentParser(
        description="Run figurative language detection on a CSV file."
    )
    add_figurative_detect_args(parser)
    return parser.parse_args()


def _resolve_output_dir(input_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        # Default to 'output' directory in the package root (qualitative-analysis/)
        # From cli.py: parent=qualitative_analysis, parent.parent=src, parent.parent.parent=qualitative-analysis
        path = Path(__file__).parent.parent.parent / "output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, Any]:
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


def _writer_append(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, Any]:
    """Open a CSV file for appending (no header written)."""
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    return writer, handle


def _load_checkpoint(path: Path) -> Optional[DetectionCheckpoint]:
    """Load checkpoint from file if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DetectionCheckpoint.from_dict(data)
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def _save_checkpoint(
    path: Path,
    processed_ids: Set[str],
    config: dict,
    input_file: str,
    total_texts: int,
) -> None:
    """Save checkpoint to file."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = DetectionCheckpoint(
        processed_text_ids=list(processed_ids),
        config=config,
        timestamp=datetime.now().isoformat(),
        input_file=input_file,
        total_texts=total_texts,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint.to_dict(), f, indent=2)
    print(f"\n[Checkpoint saved: {len(processed_ids)}/{total_texts} texts processed]")


async def run_figurative_detect(args: argparse.Namespace) -> int:
    """
    Run figurative language detection with the given arguments.
    
    This is the main detection logic, callable from both standalone and unified CLI.
    
    Args:
        args: Parsed arguments namespace with detection configuration
    
    Returns:
        Exit code (0 for success)
    """
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_selection = _parse_output_selection(args.output)
    write_summary = "summary" in output_selection
    write_instances = "instances" in output_selection
    write_windows = "windows" in output_selection

    output_dir = _resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Checkpoint handling
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    checkpoint_interval = args.checkpoint_interval
    checkpoint: Optional[DetectionCheckpoint] = None
    processed_ids: Set[str] = set()
    resuming = False
    
    if checkpoint_path:
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint:
            processed_ids = set(checkpoint.processed_text_ids)
            resuming = True
            print(f"Resuming from checkpoint: {len(processed_ids)} texts already processed")
    
    # Determine run directory and timestamp
    if resuming and checkpoint:
        # Extract timestamp from checkpoint config if available
        timestamp = checkpoint.config.get("timestamp", datetime.now().strftime("%Y%m%d-%H%M"))
        run_dir = output_dir / f"run_{timestamp}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = output_dir / f"run_{timestamp}"
    
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    output_prefix = f"{input_path.stem}_figurative"

    # Stats accumulator
    stats = {
        "texts_processed": len(processed_ids) if resuming else 0,
        "windows_analyzed": 0,
        "instances_found": 0,
        "errors": 0,
        "skipped_from_checkpoint": len(processed_ids) if resuming else 0,
    }

    window_size = args.window_size
    stride = args.stride
    if args.no_windowing:
        window_size = 1_000_000
        stride = 1_000_000

    # Parse figurative types if specified
    figurative_types = None
    if args.types:
        figurative_types = [t.strip() for t in args.types.split(",") if t.strip()]
        print(f"Filtering for figurative types: {figurative_types}")

    detector = FigurativeDetector(
        model_name=args.model,
        provider="ollama",
        provider_config={
            "base_url": args.base_url,
            "timeout": args.timeout,
            "log_prompts": args.log_llm,
            "log_responses": args.log_llm,
        },
        strategy="two_step",
        threshold=args.threshold,
        prompt_version=args.prompt_version,
        window_size=window_size,
        stride=stride,
        chunk_unit=args.chunk_unit,
        tokenizer_name=args.tokenizer,
        return_windows=write_windows,
        figurative_types=figurative_types,
        summary_buffer_size=args.context_window,
    )

    summary_writer = None
    summary_handle = None
    instances_writer = None
    instances_handle = None
    windows_writer = None
    windows_handle = None

    # Define fieldnames for each output type
    summary_fieldnames = [
        "text_id", "text", "contains_figurative", "confidence",
        "instance_count", "window_count", "strategy", "error",
    ]
    instances_fieldnames = [
        "text_id", "window_index", "window_text", "instance_text",
        "type", "confidence", "explanation", "context_dependent",
        "start_char", "end_char",
    ]
    windows_fieldnames = [
        "text_id", "window_index", "window_text", "has_figurative",
        "confidence", "instances_count", "summary",
    ]

    if write_summary:
        summary_path = run_dir / f"{output_prefix}_summary_{timestamp}.csv"
        if resuming and summary_path.exists():
            summary_writer, summary_handle = _writer_append(summary_path, summary_fieldnames)
        else:
            summary_writer, summary_handle = _writer(summary_path, summary_fieldnames)

    if write_instances:
        instances_path = run_dir / f"{output_prefix}_instances_{timestamp}.csv"
        if resuming and instances_path.exists():
            instances_writer, instances_handle = _writer_append(instances_path, instances_fieldnames)
        else:
            instances_writer, instances_handle = _writer(instances_path, instances_fieldnames)

    if write_windows:
        windows_path = run_dir / f"{output_prefix}_windows_{timestamp}.csv"
        if resuming and windows_path.exists():
            windows_writer, windows_handle = _writer_append(windows_path, windows_fieldnames)
        else:
            windows_writer, windows_handle = _writer(windows_path, windows_fieldnames)


    try:
        # First pass: count total rows for progress tracking
        with input_path.open("r", newline="", encoding="utf-8") as handle:
            total_rows = sum(1 for _ in csv.DictReader(handle))
        print(f"Processing {total_rows} texts...")
        
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

            for index, row in enumerate(reader, start=1):
                text_id = row.get(args.id_col) if args.id_col else str(index)
                text = row.get(args.text_col) or ""

                # Skip already-processed texts (from checkpoint)
                if text_id in processed_ids:
                    continue

                # Progress display: Text N of M
                print(f"\r[Text {index}/{total_rows}] Processing '{text_id[:30]}...'", end="", flush=True)

                error = ""
                result = None
                try:
                    result = await detector.detect(text)
                except Exception as exc:
                    error = str(exc)
                    stats["errors"] += 1

                # Update stats and track processed
                stats["texts_processed"] += 1
                processed_ids.add(text_id)
                
                if result:
                    window_count = result.metadata.get("window_count", 1)
                    instance_count = len(result.instances)
                    stats["windows_analyzed"] += window_count
                    stats["instances_found"] += instance_count
                    # Show completion info
                    print(f"\r[Text {index}/{total_rows}] '{text_id[:30]}': {window_count} windows, {instance_count} instances found")
                else:
                    print(f"\r[Text {index}/{total_rows}] '{text_id[:30]}': Error - {error[:50]}")


                if write_summary and summary_writer:
                    summary_writer.writerow(
                        {
                            "text_id": text_id,
                            "text": text,
                            "contains_figurative": getattr(result, "contains_figurative", ""),
                            "confidence": getattr(result, "confidence", ""),
                            "instance_count": len(result.instances) if result else "",
                            "window_count": result.metadata.get("window_count") if result else "",
                            "strategy": result.metadata.get("strategy") if result else "",
                            "error": error,
                        }
                    )

                if result and write_instances and instances_writer:
                    # Build window_text lookup from metadata
                    window_texts = {}
                    for window in result.metadata.get("windows", []):
                        w_idx = window.get("window_index")
                        if w_idx is not None:
                            window_texts[w_idx] = window.get("window_text", "")
                    
                    for instance in result.instances:
                        w_idx = getattr(instance, "window_index", 0)
                        instances_writer.writerow(
                            {
                                "text_id": text_id,
                                "window_index": w_idx,
                                "window_text": window_texts.get(w_idx, ""),
                                "instance_text": getattr(instance, "text", ""),
                                "type": getattr(instance, "type", ""),
                                "confidence": getattr(instance, "confidence", ""),
                                "explanation": getattr(instance, "explanation", ""),
                                "context_dependent": getattr(instance, "context_dependent", ""),
                                "start_char": getattr(instance, "start_char", ""),
                                "end_char": getattr(instance, "end_char", ""),
                            }
                        )

                if result and write_windows and windows_writer:
                    for window in result.metadata.get("windows", []):
                        windows_writer.writerow(
                            {
                                "text_id": text_id,
                                "window_index": window.get("window_index", ""),
                                "window_text": window.get("window_text", ""),
                                "has_figurative": window.get("has_figurative", ""),
                                "confidence": window.get("confidence", ""),
                                "instances_count": window.get("instances_count", ""),
                                "summary": window.get("summary", ""),
                            }
                        )

                # Save checkpoint periodically
                if checkpoint_path and stats["texts_processed"] % checkpoint_interval == 0:
                    # Flush CSV files before checkpoint
                    for handle in [summary_handle, instances_handle, windows_handle]:
                        if handle:
                            handle.flush()
                    _save_checkpoint(
                        checkpoint_path,
                        processed_ids,
                        {"timestamp": timestamp},
                        str(input_path),
                        total_rows,
                    )

    finally:
        for handle in [summary_handle, instances_handle, windows_handle]:
            if handle:
                handle.close()

    if write_summary:
        print(f"Wrote summary CSV: {summary_path}")
    if write_instances:
        print(f"Wrote instances CSV: {instances_path}")
    if write_windows:
        print(f"Wrote windows CSV: {windows_path}")

    # Write run metadata JSON
    end_time = datetime.now()
    metadata = {
        "run_id": f"run_{timestamp}",
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "cli_args": {
            "input_csv": str(input_path),
            "text_col": args.text_col,
            "id_col": args.id_col,
            "output": args.output,
            "model": args.model,
            "base_url": args.base_url,
            "window_size": window_size,
            "stride": stride,
            "chunk_unit": args.chunk_unit,
            "tokenizer": args.tokenizer,
            "no_windowing": args.no_windowing,
            "threshold": args.threshold,
            "context_window": args.context_window,
            "prompt_version": args.prompt_version,
            "types": figurative_types,
            "log_llm": args.log_llm,
        },
        "stats": stats,
        "outputs": {
            "summary": str(summary_path) if write_summary else None,
            "instances": str(instances_path) if write_instances else None,
            "windows": str(windows_path) if write_windows else None,
        },
        "package_version": "0.1.0",
    }
    
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote run metadata: {metadata_path}")

    # Delete checkpoint file on successful completion
    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed checkpoint file: {checkpoint_path}")

    return 0


def main() -> None:
    """
    Legacy entry point for standalone CLI.
    
    DEPRECATED: Use `qa figurative detect` instead.
    """
    emit_deprecation_warning("qualitative-analysis", "qa figurative detect")
    args = _parse_args()
    raise SystemExit(asyncio.run(run_figurative_detect(args)))


if __name__ == "__main__":
    main()
