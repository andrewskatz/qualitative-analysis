"""
CLI for decision/factor extraction, normalization, aggregation, and visualization.

Used through the unified CLI: `qa decisions detect`, `qa decisions normalize`, etc.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from qualitative_analysis.core.cli_utils import (
    add_common_csv_args,
    add_common_llm_args,
    add_common_windowing_args,
    add_provider_args,
    auto_detect_csv_columns,
    create_llm_provider,
    write_run_metadata,
    PACKAGE_VERSION,
)

logger = logging.getLogger(__name__)


# =====================================================================
# detect command
# =====================================================================


def add_decisions_detect_args(parser: argparse.ArgumentParser) -> None:
    """Add decision detection arguments."""
    add_common_csv_args(parser)
    add_common_llm_args(parser)
    add_provider_args(parser)
    add_common_windowing_args(parser)

    # Strategy
    parser.add_argument(
        "--strategy",
        default="semantic_two_pass",
        choices=["one_pass", "semantic_two_pass", "context_aware"],
        help="Extraction strategy (default: semantic_two_pass).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3).",
    )

    # Prompt versions
    parser.add_argument(
        "--decisions-prompt-version",
        type=int,
        default=5,
        help="Decision extraction prompt version (default: 5).",
    )
    parser.add_argument(
        "--factors-prompt-version",
        type=int,
        default=5,
        help="Factor extraction prompt version (default: 5).",
    )
    parser.add_argument(
        "--combined-prompt-version",
        type=int,
        default=2,
        help="Combined (single-pass) prompt version (default: 2).",
    )
    parser.add_argument(
        "--summary-prompt-version",
        type=int,
        default=1,
        help="Window summary prompt version (default: 1).",
    )

    # Custom prompt file overrides
    parser.add_argument(
        "--decisions-prompt",
        default=None,
        help="Custom prompt file for decision extraction (overrides version).",
    )
    parser.add_argument(
        "--factors-prompt",
        default=None,
        help="Custom prompt file for factor extraction (overrides version).",
    )
    parser.add_argument(
        "--summary-prompt",
        default=None,
        help="Custom prompt file for window summarization (overrides version).",
    )

    # Embedding
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model for semantic search/dedup (default: all-MiniLM-L6-v2).",
    )

    # Semantic two-pass specific
    parser.add_argument(
        "--top-k-windows",
        type=int,
        default=3,
        help="Top-K windows per decision for semantic search (default: 3).",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.5,
        help="Minimum similarity threshold for semantic search (default: 0.5).",
    )

    # Deduplication
    parser.add_argument(
        "--dedup-method",
        choices=["exact", "semantic"],
        default="semantic",
        help="Deduplication method (default: semantic).",
    )
    parser.add_argument(
        "--decision-dedup-threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for decision dedup (default: 0.9).",
    )
    parser.add_argument(
        "--factor-dedup-threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for factor dedup (default: 0.8).",
    )

    # Validation
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable rule-based decision validation.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable causal clause stripping from decisions.",
    )

    # Checkpointing
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        default=False,
        help="Disable checkpointing. By default, progress is saved after each text and resumed on restart.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: output/ alongside input file).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N texts (for testing).",
    )
    parser.add_argument(
        "--group-col",
        default=None,
        help="Column name for group assignments.",
    )


async def run_decisions_detect(args: argparse.Namespace) -> int:
    """Run decision/factor extraction."""
    from qualitative_analysis.decisions.extractor import DecisionExtractor
    from qualitative_analysis.decisions.models import (
        DecisionConfig,
        DecisionExtractionResult,
    )
    from qualitative_analysis.decisions.aggregator import aggregate_decisions

    start_time = datetime.now()
    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Windowing
    window_size = args.window_size
    stride = args.stride
    if args.no_windowing:
        window_size = 1_000_000
        stride = 1_000_000

    # Build LLM provider
    llm_provider = create_llm_provider(args)

    # Build config
    config = DecisionConfig(
        strategy=args.strategy,
        temperature=getattr(args, "temperature", 0.3),
        validate_decisions=not args.no_validation,
        normalize_decisions=not args.no_normalize,
        dedup_method=args.dedup_method,
        decision_dedup_threshold=args.decision_dedup_threshold,
        factor_dedup_threshold=args.factor_dedup_threshold,
        top_k_windows=args.top_k_windows,
        min_similarity=args.min_similarity,
    )

    # Build extractor
    extractor = DecisionExtractor(
        llm_provider=llm_provider,
        strategy=args.strategy,
        config=config,
        window_size=window_size,
        stride=stride,
        chunk_unit=args.chunk_unit,
        tokenizer_name=args.tokenizer,
        decisions_prompt_version=args.decisions_prompt_version,
        factors_prompt_version=args.factors_prompt_version,
        combined_prompt_version=args.combined_prompt_version,
        summary_prompt_version=args.summary_prompt_version,
        decisions_prompt_path=args.decisions_prompt,
        factors_prompt_path=args.factors_prompt,
        summary_prompt_path=args.summary_prompt,
        embedding_model=args.embedding_model,
    )

    # Read CSV with auto-detection
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SystemExit("Input CSV has no headers.")

        text_col, id_col = auto_detect_csv_columns(
            list(reader.fieldnames),
            text_col=args.text_col,
            id_col=args.id_col,
        )

        if text_col not in reader.fieldnames:
            raise SystemExit(
                f"Text column '{text_col}' not found in CSV headers: "
                f"{list(reader.fieldnames)}"
            )
        rows = list(reader)

    if args.limit:
        rows = rows[: args.limit]

    total = len(rows)

    # Checkpoint setup
    use_checkpoint = not getattr(args, "no_checkpoint", False)
    output_prefix = input_path.stem
    checkpoint_path = run_dir / f"{output_prefix}_checkpoint.jsonl"
    checkpoint_results: List[DecisionExtractionResult] = []

    if use_checkpoint and checkpoint_path.exists():
        checkpoint_results, scored_keys = _load_checkpoint(checkpoint_path)
        if scored_keys:
            print(f"\nResuming from checkpoint: {len(scored_keys)} texts already processed")
            rows = [
                r for r in rows
                if _text_checkpoint_key(r, text_col, id_col, rows.index(r) + 1)
                not in scored_keys
            ]
            skipped = total - len(rows)
            print(f"Skipping {skipped} already-processed texts, {len(rows)} remaining")
    else:
        scored_keys = set()

    remaining = len(rows)

    # Track errors across both paths
    errors = 0

    # All already processed?
    if not rows and checkpoint_results:
        print("\nAll texts already processed in checkpoint. Finalizing output...")
        all_results = checkpoint_results
    elif not rows:
        print("No texts to process.")
        return 1
    else:
        provider_name = getattr(args, "provider", "ollama")
        print(f"\nProcessing {remaining} texts with strategy '{args.strategy}'...")
        print(f"Model: {args.model} ({provider_name})")
        print(f"Temperature: {getattr(args, 'temperature', 0.3)}")
        if use_checkpoint:
            print(f"Checkpoint: {checkpoint_path}")
        print()

        all_results = list(checkpoint_results)
        errors = 0
        n_already = len(checkpoint_results)

        for i, row in enumerate(rows, start=1):
            row_idx = total - remaining + i
            text_id = row.get(id_col, str(row_idx)) if id_col else str(row_idx)
            text = row.get(text_col) or ""

            overall = n_already + i
            print(f"\r  [{overall}/{total}] text_id={text_id}", end="", flush=True)

            try:
                result = await extractor.extract(text, text_id=text_id)
                all_results.append(result)
                # Save checkpoint immediately after each text
                if use_checkpoint:
                    _append_to_checkpoint(checkpoint_path, result)
            except Exception as exc:
                print(f"\n  Error on text_id={text_id}: {exc}")
                errors += 1

        print()  # newline after progress

        # Close LLM provider connection
        if hasattr(llm_provider, "close"):
            await llm_provider.close()

    # Aggregate
    aggregation = aggregate_decisions(all_results)

    # Write outputs
    outputs: Dict[str, str] = {}

    # 1. Full results JSON
    results_path = run_dir / "decisions_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    outputs["results"] = str(results_path)

    # 2. Decisions CSV
    decisions_csv_path = run_dir / "decisions.csv"
    decision_rows = []
    for r in all_results:
        decision_rows.extend(r.to_flat_decisions_rows())
    if decision_rows:
        with open(decisions_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=decision_rows[0].keys())
            writer.writeheader()
            writer.writerows(decision_rows)
        outputs["decisions_csv"] = str(decisions_csv_path)

    # 3. Factors CSV
    factors_csv_path = run_dir / "factors.csv"
    factor_rows = []
    for r in all_results:
        factor_rows.extend(r.to_flat_factors_rows())
    if factor_rows:
        with open(factors_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=factor_rows[0].keys())
            writer.writeheader()
            writer.writerows(factor_rows)
        outputs["factors_csv"] = str(factors_csv_path)

    # 4. Aggregation JSON
    agg_path = run_dir / "aggregation.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregation.to_dict(), f, indent=2)
    outputs["aggregation"] = str(agg_path)

    # 5. Run metadata
    cli_args_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if not callable(v)
    }
    stats = {
        "total_texts": total,
        "texts_processed": len(all_results),
        "errors": errors,
        "total_decisions": aggregation.total_decisions,
        "unique_decisions": aggregation.unique_decisions,
        "total_factors": aggregation.total_factors,
        "unique_factors": aggregation.unique_factors,
        "polarity_distribution": aggregation.polarity_distribution,
    }
    write_run_metadata(
        run_dir,
        step="decisions_detect",
        cli_args=cli_args_dict,
        stats=stats,
        outputs=outputs,
        start_time=start_time,
    )

    # Clean up checkpoint on success
    if use_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint file removed (results saved to output files)")

    # Summary
    print(f"\nResults saved to: {run_dir}")
    print(f"  Texts processed: {len(all_results)}/{total} (errors: {errors})")
    print(f"  Total decisions: {aggregation.total_decisions} ({aggregation.unique_decisions} unique)")
    print(f"  Total factors: {aggregation.total_factors} ({aggregation.unique_factors} unique)")
    if aggregation.polarity_distribution:
        dist = aggregation.polarity_distribution
        print(f"  Polarity: supporting={dist.get('supporting', 0)}, "
              f"opposing={dist.get('opposing', 0)}, neutral={dist.get('neutral', 0)}")

    return 0


# =====================================================================
# Checkpoint helpers
# =====================================================================


def _text_checkpoint_key(
    row: Dict[str, Any],
    text_col: str,
    id_col: str | None,
    row_num: int,
) -> str:
    """Create a unique key for a text row for checkpoint deduplication."""
    if id_col and id_col in row:
        return row[id_col]
    return str(row_num)


def _append_to_checkpoint(
    checkpoint_path: Path,
    result,
) -> None:
    """Append a single DecisionExtractionResult to the checkpoint JSONL file."""
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def _load_checkpoint(
    checkpoint_path: Path,
) -> tuple:
    """
    Load processed texts from checkpoint JSONL file.

    Returns:
        Tuple of (list of DecisionExtractionResult, set of text_id keys).
    """
    from qualitative_analysis.decisions.models import DecisionExtractionResult

    results = []
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

            result = DecisionExtractionResult.from_dict(data)
            results.append(result)
            scored_keys.add(result.text_id)

    return results, scored_keys


# =====================================================================
# normalize command
# =====================================================================


def add_decisions_normalize_args(parser: argparse.ArgumentParser) -> None:
    """Add decision normalization arguments."""
    parser.add_argument(
        "input_json",
        help="Path to decisions_results.json from detect step.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold for clustering decisions (default: 0.85).",
    )
    parser.add_argument(
        "--factor-threshold",
        type=float,
        default=0.80,
        help="Cosine similarity threshold for clustering factors (default: 0.80).",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="Embedding model (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside input file).",
    )


async def run_decisions_normalize(args: argparse.Namespace) -> int:
    """Run decision/factor normalization."""
    from qualitative_analysis.decisions.models import DecisionExtractionResult
    from qualitative_analysis.decisions.normalizer import DecisionNormalizer

    start_time = datetime.now()
    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    # Load results
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    results = [DecisionExtractionResult.from_dict(r) for r in raw]

    # Normalize
    normalizer = DecisionNormalizer(
        embedding_model=args.embedding_model,
        decision_threshold=args.decision_threshold,
        factor_threshold=args.factor_threshold,
    )
    norm_result = normalizer.normalize(results)

    # Output
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_path = out_dir / "normalization_result.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_result.to_dict(), f, indent=2)

    print(f"Normalization result saved to: {norm_path}")
    print(f"  Decision clusters: {len(norm_result.decision_clusters)}")
    print(f"  Factor clusters: {len(norm_result.factor_clusters)}")

    return 0


# =====================================================================
# aggregate command
# =====================================================================


def add_decisions_aggregate_args(parser: argparse.ArgumentParser) -> None:
    """Add decision aggregation arguments."""
    parser.add_argument(
        "input_json",
        help="Path to decisions_results.json from detect step.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: alongside input file).",
    )


async def run_decisions_aggregate(args: argparse.Namespace) -> int:
    """Run decision/factor aggregation."""
    from qualitative_analysis.decisions.models import DecisionExtractionResult
    from qualitative_analysis.decisions.aggregator import aggregate_decisions

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    results = [DecisionExtractionResult.from_dict(r) for r in raw]

    aggregation = aggregate_decisions(results)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = out_dir / "aggregation.json"
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregation.to_dict(), f, indent=2)

    print(f"Aggregation saved to: {agg_path}")
    print(f"  Total decisions: {aggregation.total_decisions} ({aggregation.unique_decisions} unique)")
    print(f"  Total factors: {aggregation.total_factors} ({aggregation.unique_factors} unique)")
    if aggregation.polarity_distribution:
        print(f"  Polarity: {aggregation.polarity_distribution}")

    return 0


# =====================================================================
# viz command
# =====================================================================


def add_decisions_viz_args(parser: argparse.ArgumentParser) -> None:
    """Add decision visualization arguments."""
    parser.add_argument(
        "input_json",
        help="Path to decisions_results.json from detect step.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for plots (default: alongside input file).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top items to show in frequency charts (default: 20).",
    )
    parser.add_argument(
        "--plot-types",
        nargs="+",
        default=["all"],
        choices=["frequency", "polarity", "network", "all"],
        help="Types of plots to generate (default: all).",
    )


async def run_decisions_viz(args: argparse.Namespace) -> int:
    """Generate decision/factor visualizations."""
    from qualitative_analysis.decisions.models import DecisionExtractionResult
    from qualitative_analysis.decisions.aggregator import aggregate_decisions
    from qualitative_analysis.decisions.visualizer import DecisionVisualizer

    input_path = Path(args.input_json)
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    results = [DecisionExtractionResult.from_dict(r) for r in raw]

    aggregation = aggregate_decisions(results)

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = input_path.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    viz = DecisionVisualizer()
    plot_types = set(args.plot_types)
    do_all = "all" in plot_types

    generated = {}

    if do_all or "frequency" in plot_types:
        generated["decision_frequency"] = viz.plot_decision_frequency(
            aggregation, str(out_dir / "decision_frequency.png"), top_n=args.top_n
        )
        generated["factor_frequency"] = viz.plot_factor_frequency(
            aggregation, str(out_dir / "factor_frequency.png"), top_n=args.top_n
        )

    if do_all or "polarity" in plot_types:
        generated["polarity_distribution"] = viz.plot_polarity_distribution(
            aggregation, str(out_dir / "polarity_distribution.png")
        )
        generated["polarity_by_decision"] = viz.plot_factor_polarity_by_decision(
            results, str(out_dir / "polarity_by_decision.png")
        )

    if do_all or "network" in plot_types:
        generated["network"] = viz.plot_decision_factor_network(
            results, str(out_dir / "decision_factor_network.png")
        )

    print(f"Visualizations saved to: {out_dir}")
    for name, path in generated.items():
        print(f"  {name}: {path}")

    return 0
