"""
CLI for running figurative language detection on CSV files.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from qualitative_analysis.figurative.detector import FigurativeDetector

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run figurative language detection on a CSV file."
    )
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
        help="Output directory (default: alongside input CSV).",
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
    parser.add_argument("--prompt-version", type=int, default=1, help="Prompt version.")
    return parser.parse_args()


def _resolve_output_dir(input_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return input_path.parent


def _writer(path: Path, fieldnames: list[str]) -> tuple[csv.DictWriter, Any]:
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return writer, handle


async def _run() -> int:
    args = _parse_args()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_selection = _parse_output_selection(args.output)
    write_summary = "summary" in output_selection
    write_instances = "instances" in output_selection
    write_windows = "windows" in output_selection

    output_dir = _resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{input_path.stem}_figurative"

    window_size = args.window_size
    stride = args.stride
    if args.no_windowing:
        window_size = 1_000_000
        stride = 1_000_000

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
    )

    summary_writer = None
    summary_handle = None
    instances_writer = None
    instances_handle = None
    windows_writer = None
    windows_handle = None

    if write_summary:
        summary_path = run_dir / f"{output_prefix}_summary_{timestamp}.csv"
        summary_writer, summary_handle = _writer(
            summary_path,
            [
                "text_id",
                "text",
                "contains_figurative",
                "confidence",
                "instance_count",
                "window_count",
                "strategy",
                "error",
            ],
        )

    if write_instances:
        instances_path = run_dir / f"{output_prefix}_instances_{timestamp}.csv"
        instances_writer, instances_handle = _writer(
            instances_path,
            [
                "text_id",
                "window_index",
                "instance_text",
                "type",
                "confidence",
                "explanation",
                "context_dependent",
                "start_char",
                "end_char",
            ],
        )

    if write_windows:
        windows_path = run_dir / f"{output_prefix}_windows_{timestamp}.csv"
        windows_writer, windows_handle = _writer(
            windows_path,
            [
                "text_id",
                "window_index",
                "window_text",
                "has_figurative",
                "confidence",
                "instances_count",
                "summary",
            ],
        )

    try:
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

                error = ""
                result = None
                try:
                    result = await detector.detect(text)
                except Exception as exc:
                    error = str(exc)

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
                    for instance in result.instances:
                        instances_writer.writerow(
                            {
                                "text_id": text_id,
                                "window_index": getattr(instance, "window_index", ""),
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

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
