"""
CLI for running relationship extraction on CSV files.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any, List

from qualitative_analysis.relationships.detector import RelationshipDetector

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run relationship extraction on a CSV file."
    )
    parser.add_argument("input_csv", help="Path to input CSV file.")
    parser.add_argument("--id-col", default=None, help="Column name for IDs (optional).")
    parser.add_argument("--text-col", default="text", help="Column name for text.")
    parser.add_argument(
        "--entities-col",
        default=None,
        help="Column name for pre-identified entities (optional).",
    )
    parser.add_argument(
        "--output",
        default="summary",
        choices=sorted(OUTPUT_CHOICES),
        help="Output format: summary, relationships, windows, combinations, or all.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside input CSV).",
    )
    parser.add_argument(
        "--strategy",
        default="two_pass",
        choices=["two_pass", "one_pass"],
        help="Relationship extraction strategy (default: two_pass).",
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
        "--summary-buffer-size",
        type=int,
        default=5,
        help="Number of window summaries to keep for context.",
    )
    parser.add_argument(
        "--summary-prompt-version",
        type=int,
        default=1,
        help="Window summary prompt version.",
    )
    parser.add_argument(
        "--include-summaries",
        action="store_true",
        help="Include window summaries in relationship prompts.",
    )
    parser.add_argument(
        "--no-windowing",
        action="store_true",
        help="Disable windowing and analyze each text as a single window.",
    )
    parser.add_argument(
        "--prompt-version",
        type=int,
        default=None,
        help="Relationship prompt version (strategy-dependent).",
    )
    parser.add_argument(
        "--entity-prompt-version",
        type=int,
        default=3,
        help="Entity extraction prompt version (two-pass only).",
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
    if args.prompt_version is None:
        args.prompt_version = 1 if args.strategy == "two_pass" else 2
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    output_selection = _parse_output_selection(args.output)
    write_summary = "summary" in output_selection
    write_relationships = "relationships" in output_selection
    write_windows = "windows" in output_selection

    output_dir = _resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = f"{input_path.stem}_relationships"

    window_size = args.window_size
    stride = args.stride
    if args.no_windowing:
        window_size = 1_000_000
        stride = 1_000_000

    detector = RelationshipDetector(
        model_name=args.model,
        provider="ollama",
        provider_config={
            "base_url": args.base_url,
            "timeout": args.timeout,
            "log_prompts": args.log_llm,
            "log_responses": args.log_llm,
        },
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
        include_summaries_in_prompt=args.include_summaries,
        return_windows=write_windows,
        context_buffer_size=args.context_buffer_size,
        coref_resolution=args.coref,
    )

    summary_writer = None
    summary_handle = None
    relationships_writer = None
    relationships_handle = None
    windows_writer = None
    windows_handle = None

    if write_summary:
        summary_path = output_dir / f"{output_prefix}_summary.csv"
        summary_writer, summary_handle = _writer(
            summary_path,
            [
                "text_id",
                "text",
                "entity_count",
                "relationship_count",
                "window_count",
                "entities_json",
                "strategy",
                "error",
            ],
        )

    if write_relationships:
        relationships_path = output_dir / f"{output_prefix}_edges.csv"
        relationships_writer, relationships_handle = _writer(
            relationships_path,
            [
                "text_id",
                "window_index",
                "source",
                "target",
                "type",
                "description",
            ],
        )

    if write_windows:
        windows_path = output_dir / f"{output_prefix}_windows.csv"
        windows_writer, windows_handle = _writer(
            windows_path,
            [
                "text_id",
                "window_index",
                "window_text",
                "summary",
                "relationship_count",
                "entities_json",
                "relationships_json",
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

            if args.entities_col and args.entities_col not in reader.fieldnames:
                raise SystemExit(
                    f"Entities column '{args.entities_col}' not found in CSV headers: {reader.fieldnames}"
                )

            for index, row in enumerate(reader, start=1):
                text_id = row.get(args.id_col) if args.id_col else str(index)
                text = row.get(args.text_col) or ""
                entities_input = (
                    _parse_entities(row.get(args.entities_col))
                    if args.entities_col
                    else []
                )

                error = ""
                result = None
                try:
                    result = await detector.detect(text, current_entities=entities_input)
                except Exception as exc:
                    error = str(exc)

                if write_summary and summary_writer:
                    summary_writer.writerow(
                        {
                            "text_id": text_id,
                            "text": text,
                            "entity_count": result.metadata.get("entity_count") if result else "",
                            "relationship_count": result.metadata.get("relationship_count") if result else "",
                            "window_count": result.metadata.get("window_count") if result else "",
                            "entities_json": json.dumps(result.entities) if result else "",
                            "strategy": result.metadata.get("strategy") if result else "",
                            "error": error,
                        }
                    )

                if result and write_relationships and relationships_writer:
                    for rel in result.relationships:
                        relationships_writer.writerow(
                            {
                                "text_id": text_id,
                                "window_index": rel.window_index,
                                "source": rel.source,
                                "target": rel.target,
                                "type": rel.type,
                                "description": rel.description,
                            }
                        )

                if result and write_windows and windows_writer:
                    for window in result.metadata.get("windows", []):
                        windows_writer.writerow(
                            {
                                "text_id": text_id,
                                "window_index": window.get("window_index", ""),
                                "window_text": window.get("window_text", ""),
                                "summary": window.get("summary", ""),
                                "relationship_count": window.get("relationship_count", ""),
                                "entities_json": json.dumps(window.get("entities", [])),
                                "relationships_json": json.dumps(window.get("relationships", [])),
                            }
                        )

    finally:
        for handle in [summary_handle, relationships_handle, windows_handle]:
            if handle:
                handle.close()

    if write_summary:
        print(f"Wrote summary CSV: {summary_path}")
    if write_relationships:
        print(f"Wrote relationships CSV: {relationships_path}")
    if write_windows:
        print(f"Wrote windows CSV: {windows_path}")

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
