"""
Shared CLI utilities for the qualitative-analysis package.

This module provides common argument patterns and utilities used across
all CLI commands to ensure consistency.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Package version - should match pyproject.toml
PACKAGE_VERSION = "0.1.0"


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common LLM-related arguments to a parser.
    
    Adds: --model, --base-url, --timeout, --log-llm
    """
    parser.add_argument(
        "--model",
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        help="LLM model name (default: qwen3:30b-a3b-instruct-2507-q4_K_M)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--log-llm",
        action="store_true",
        help="Print LLM prompts and responses to the terminal",
    )


def add_common_output_args(
    parser: argparse.ArgumentParser,
    default_output: str = "summary",
    output_choices: Optional[set[str]] = None,
) -> None:
    """
    Add common output-related arguments to a parser.
    
    Adds: --output-dir, --output
    """
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside input file)",
    )
    if output_choices:
        parser.add_argument(
            "--output",
            default=default_output,
            choices=sorted(output_choices),
            help=f"Output format (default: {default_output})",
        )


def add_common_windowing_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common text windowing arguments to a parser.
    
    Adds: --window-size, --stride, --chunk-unit, --tokenizer, --no-windowing
    """
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Sliding window size (default: 3)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Sliding window stride (default: 2)",
    )
    parser.add_argument(
        "--chunk-unit",
        default="sentences",
        choices=["sentences", "tokens"],
        help="Chunking unit for windowing (default: sentences)",
    )
    parser.add_argument(
        "--tokenizer",
        default="cl100k_base",
        help="Tokenizer name for token chunking (default: cl100k_base)",
    )
    parser.add_argument(
        "--no-windowing",
        action="store_true",
        help="Disable windowing and analyze each text as a single window",
    )


def add_common_csv_args(parser: argparse.ArgumentParser) -> None:
    """
    Add common CSV input arguments to a parser.
    
    Adds: input_csv (positional), --text-col, --id-col
    """
    parser.add_argument(
        "input_csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Column name for text content (default: text)",
    )
    parser.add_argument(
        "--id-col",
        default=None,
        help="Column name for row IDs (default: use row number)",
    )


def create_run_directory(
    output_dir: Path,
    prefix: str = "run",
) -> tuple[Path, str]:
    """
    Create a timestamped run directory.
    
    Args:
        output_dir: Parent directory for the run
        prefix: Prefix for the run directory name
    
    Returns:
        Tuple of (run_directory_path, timestamp_string)
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = output_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def write_run_metadata(
    run_dir: Path,
    *,
    step: str,
    cli_args: Dict[str, Any],
    stats: Dict[str, Any],
    outputs: Dict[str, Any],
    start_time: datetime,
    end_time: Optional[datetime] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write standardized run metadata JSON.
    
    Args:
        run_dir: Directory to write metadata to
        step: Name of the analysis step (e.g., "figurative_detection")
        cli_args: Dictionary of CLI arguments used
        stats: Dictionary of statistics from the run
        outputs: Dictionary mapping output types to file paths
        start_time: When the run started
        end_time: When the run ended (default: now)
        extra: Additional metadata to include
    
    Returns:
        Path to the written metadata file
    """
    if end_time is None:
        end_time = datetime.now()
    
    timestamp = start_time.strftime("%Y%m%d-%H%M")
    
    metadata = {
        "run_id": f"{step}_{timestamp}",
        "step": step,
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": round((end_time - start_time).total_seconds(), 2),
        "cli_args": cli_args,
        "stats": stats,
        "outputs": outputs,
        "package_version": PACKAGE_VERSION,
    }
    
    if extra:
        metadata.update(extra)
    
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


def create_progress_callback(
    total: int,
    prefix: str = "Processing",
) -> Callable[[int, int], None]:
    """
    Create a progress callback for consistent progress display.
    
    Args:
        total: Total number of items to process
        prefix: Prefix for progress messages
    
    Returns:
        Callback function that takes (current, total) arguments
    """
    def callback(current: int, total: int) -> None:
        pct = 100 * current / total if total > 0 else 0
        print(f"\r{prefix}: {current}/{total} ({pct:.1f}%)", end="", flush=True)
    
    return callback


def resolve_output_dir(input_path: Path, output_dir: Optional[str]) -> Path:
    """
    Resolve the output directory from input path and optional override.
    
    Args:
        input_path: Path to the input file
        output_dir: Optional explicit output directory
    
    Returns:
        Resolved Path for output directory
    """
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return input_path.parent


import re

def resolve_nested_output_dir(
    input_path: Path,
    step_name: str,
    output_dir: Optional[str] = None,
) -> tuple[Path, str]:
    """
    Resolve output directory for pipeline steps (map, normalize, graph).
    
    If the input file is inside a 'run_*' directory, creates a nested structure:
      run_YYYYMMDD-HHMM/
        └── <step_name>/
            └── <step_name>_YYYYMMDD-HHMM/
    
    If output_dir is explicitly provided, uses that instead.
    If input is not in a run_* directory, creates output in input's parent.
    
    Args:
        input_path: Path to the input file
        step_name: Name of the step (e.g., "map", "normalize", "graph")
        output_dir: Optional explicit output directory override
    
    Returns:
        Tuple of (output_directory, timestamp_string)
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    
    # If explicit output dir provided, use it
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return out_path, timestamp
    
    # Check if input is in a run_* directory
    parent_run_dir = _find_parent_run_dir(input_path)
    
    if parent_run_dir:
        # Create nested structure: run_*/step_name/step_name_timestamp/
        step_dir = parent_run_dir / step_name / f"{step_name}_{timestamp}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir, timestamp
    else:
        # Fallback: create in input's parent directory
        step_dir = input_path.parent / f"{step_name}_{timestamp}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir, timestamp


def _find_parent_run_dir(path: Path) -> Optional[Path]:
    """
    Walk up from path to find a parent directory matching 'run_*' pattern.
    
    Args:
        path: Starting path to search from
    
    Returns:
        The run_* directory Path if found, None otherwise
    """
    # Pattern for run directories: run_YYYYMMDD-HHMM
    run_pattern = re.compile(r"^run_\d{8}-\d{4}$")
    
    current = path.parent  # Start from parent of the file
    
    # Walk up the directory tree (limit to 5 levels to avoid infinite loops)
    for _ in range(5):
        if run_pattern.match(current.name):
            return current
        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent
    
    return None


def emit_deprecation_warning(old_command: str, new_command: str) -> None:
    """
    Emit a deprecation warning for old CLI entry points.
    
    Args:
        old_command: The deprecated command name
        new_command: The new command to use instead
    """
    warnings.warn(
        f"'{old_command}' is deprecated and will be removed in a future release. "
        f"Please use '{new_command}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    # Also print to stderr for visibility
    print(
        f"\n⚠️  DEPRECATION WARNING: '{old_command}' is deprecated. "
        f"Use '{new_command}' instead.\n",
        file=sys.stderr,
    )


def get_provider_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build provider config dictionary from parsed arguments.
    
    Expects args to have: base_url, timeout (optional), log_llm
    
    Returns:
        Dictionary suitable for passing to detector/extractor classes
    """
    config = {
        "base_url": args.base_url,
        "log_prompts": args.log_llm,
        "log_responses": args.log_llm,
    }
    if hasattr(args, "timeout"):
        config["timeout"] = args.timeout
    return config
