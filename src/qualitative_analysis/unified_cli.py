"""
Unified CLI for the qualitative-analysis package.

Entry point: `qa`

This provides a single, consistent interface for all qualitative analysis
commands, organized by analysis type (figurative, relationships, etc.).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Callable, Optional

from qualitative_analysis.core.cli_utils import PACKAGE_VERSION


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="qa",
        description="Qualitative Analysis CLI - LLM-powered text analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qa figurative detect transcripts.csv --model gpt-oss:120b
  qa fig map instances.csv --multi-level
  qa relationships detect transcripts.csv --entities "Company,Person"
  
For more information on a specific command:
  qa figurative --help
  qa fig detect --help
""",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {PACKAGE_VERSION}",
    )
    
    subparsers = parser.add_subparsers(
        dest="command",
        title="analysis types",
        description="Available analysis types",
        metavar="COMMAND",
    )
    
    # Figurative language analysis
    _add_figurative_commands(subparsers)

    # Relationship extraction
    _add_relationships_commands(subparsers)

    # Entity analysis
    _add_entity_commands(subparsers)

    # Decision/factor analysis
    _add_decisions_commands(subparsers)

    return parser


def _add_figurative_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add figurative language analysis subcommands."""
    # Import here to avoid circular imports
    from qualitative_analysis.cli import (
        add_figurative_detect_args,
        run_figurative_detect,
    )
    from qualitative_analysis.figurative.domains_cli import (
        add_map_args,
        add_normalize_args,
        add_graph_args,
        add_pipeline_args,
        run_map,
        run_normalize,
        run_graph,
        run_pipeline,
    )
    
    fig_parser = subparsers.add_parser(
        "figurative",
        aliases=["fig"],
        help="Figurative language analysis (detection, domain mapping, graphing)",
        description="Analyze text for figurative language use and map conceptual domains.",
    )
    
    fig_subs = fig_parser.add_subparsers(
        dest="action",
        title="commands",
        description="Available figurative language commands",
        metavar="ACTION",
    )
    
    # detect subcommand
    detect_parser = fig_subs.add_parser(
        "detect",
        help="Detect figurative language in text",
        description="Analyze CSV text data to identify metaphors, analogies, and other figurative language.",
    )
    add_figurative_detect_args(detect_parser)
    detect_parser.set_defaults(func=lambda args: asyncio.run(run_figurative_detect(args)))
    
    # map subcommand
    map_parser = fig_subs.add_parser(
        "map",
        help="Map source/target domains from figurative instances",
        description="Extract conceptual domains from detected figurative language instances.",
    )
    add_map_args(map_parser)
    map_parser.set_defaults(func=lambda args: asyncio.run(run_map(args)))
    
    # normalize subcommand
    normalize_parser = fig_subs.add_parser(
        "normalize",
        help="Normalize/cluster domain labels",
        description="Cluster semantically similar domains and generate canonical labels.",
    )
    add_normalize_args(normalize_parser)
    normalize_parser.set_defaults(func=run_normalize)
    
    # graph subcommand
    graph_parser = fig_subs.add_parser(
        "graph",
        help="Generate domain relationship graph",
        description="Create a graph of source→target domain relationships.",
    )
    add_graph_args(graph_parser)
    graph_parser.set_defaults(func=run_graph)
    
    # pipeline subcommand
    pipeline_parser = fig_subs.add_parser(
        "pipeline",
        help="Run full pipeline: detect → map → normalize → graph",
        description="Run the complete figurative language analysis pipeline.",
    )
    add_pipeline_args(pipeline_parser)
    pipeline_parser.set_defaults(func=lambda args: asyncio.run(run_pipeline(args)))


def _add_relationships_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add relationship extraction subcommands."""
    from qualitative_analysis.relationships_cli import (
        add_relationships_detect_args,
        run_relationships_detect,
        add_relationships_normalize_args,
        run_relationships_normalize,
        add_relationships_causal_args,
        run_relationships_causal,
        add_relationships_graph_args,
        run_relationships_graph,
        add_relationships_verify_args,
        run_relationships_verify,
    )

    rel_parser = subparsers.add_parser(
        "relationships",
        aliases=["rel"],
        help="Entity-relationship extraction and analysis",
        description="Extract and analyze relationships between entities in text.\n\n"
                    "Recommended pipeline order:\n"
                    "  detect → verify → causal → normalize → graph",
    )

    rel_subs = rel_parser.add_subparsers(
        dest="action",
        title="commands",
        description="Available relationship commands",
        metavar="ACTION",
    )

    # detect subcommand
    detect_parser = rel_subs.add_parser(
        "detect",
        help="Detect entity relationships in text",
        description="Analyze CSV text data to identify entities and their relationships.",
    )
    add_relationships_detect_args(detect_parser)
    detect_parser.set_defaults(func=lambda args: asyncio.run(run_relationships_detect(args)))

    # normalize subcommand
    normalize_parser = rel_subs.add_parser(
        "normalize",
        help="Normalize/consolidate entities and relationship types",
        description="Cluster semantically similar entities and relationship types, then merge duplicate relationships.",
    )
    add_relationships_normalize_args(normalize_parser)
    normalize_parser.set_defaults(func=lambda args: asyncio.run(run_relationships_normalize(args)))

    # causal subcommand
    causal_parser = rel_subs.add_parser(
        "causal",
        help="Analyze relationships for causal attributes",
        description="Classify relationships as causal/non-causal and extract polarity, certainty, and explicit/implicit attributes.",
    )
    add_relationships_causal_args(causal_parser)
    causal_parser.set_defaults(func=lambda args: asyncio.run(run_relationships_causal(args)))

    # graph subcommand
    graph_parser = rel_subs.add_parser(
        "graph",
        help="Generate relationship network graph",
        description="Build a relationship graph with node metrics (degree, betweenness, PageRank) and export to multiple formats.",
    )
    add_relationships_graph_args(graph_parser)
    graph_parser.set_defaults(func=lambda args: asyncio.run(run_relationships_graph(args)))

    # verify subcommand
    verify_parser = rel_subs.add_parser(
        "verify",
        help="Verify relationships against source texts",
        description="Use LLM second-pass verification to filter out unsupported or hallucinated relationships.",
    )
    add_relationships_verify_args(verify_parser)
    verify_parser.set_defaults(func=lambda args: asyncio.run(run_relationships_verify(args)))


def _add_entity_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add entity analysis subcommands."""
    from qualitative_analysis.entity_cli import (
        add_entity_detect_args,
        run_entity_detect,
        add_entity_score_args,
        run_entity_score,
        add_entity_consolidate_args,
        run_entity_consolidate,
        add_entity_viz_args,
        run_entity_viz,
        add_entity_compare_args,
        run_entity_compare,
        add_entity_compare_viz_args,
        run_entity_compare_viz,
        add_entity_report_args,
        run_entity_report,
        add_entity_prepare_scoring_args,
        run_entity_prepare_scoring,
        add_entity_agreement_args,
        run_entity_agreement,
        add_entity_cluster_args,
        run_entity_cluster,
    )

    entity_parser = subparsers.add_parser(
        "entity",
        aliases=["ent"],
        help="Entity analysis (scoring, consolidation, visualization)",
        description="Score, consolidate, and visualize entities extracted from text.",
    )

    entity_subs = entity_parser.add_subparsers(
        dest="action",
        title="commands",
        description="Available entity commands",
        metavar="ACTION",
    )

    # detect subcommand
    detect_parser = entity_subs.add_parser(
        "detect",
        help="Detect entities in text",
        description="Extract entities from text using LLM analysis. Outputs CSV ready for 'qa entity score'.",
    )
    add_entity_detect_args(detect_parser)
    detect_parser.set_defaults(func=lambda args: asyncio.run(run_entity_detect(args)))

    # score subcommand
    score_parser = entity_subs.add_parser(
        "score",
        help="Score entities along multiple dimensions",
        description="Score entities using LLM analysis with uncertainty quantification through multiple runs.",
    )
    add_entity_score_args(score_parser)
    score_parser.set_defaults(func=lambda args: asyncio.run(run_entity_score(args)))

    # consolidate subcommand (placeholder)
    consolidate_parser = entity_subs.add_parser(
        "consolidate",
        help="Consolidate/deduplicate entities (coming soon)",
        description="Semantic deduplication of entities using embeddings.",
    )
    add_entity_consolidate_args(consolidate_parser)
    consolidate_parser.set_defaults(func=lambda args: asyncio.run(run_entity_consolidate(args)))

    # viz subcommand
    viz_parser = entity_subs.add_parser(
        "viz",
        help="Visualize scored entities",
        description="Generate ternary plots and radar charts for scored entities.",
    )
    add_entity_viz_args(viz_parser)
    viz_parser.set_defaults(func=lambda args: asyncio.run(run_entity_viz(args)))

    # compare subcommand
    compare_parser = entity_subs.add_parser(
        "compare",
        help="Compare entity scores across participants",
        description="Compute distances between participants and generate comparison visualizations.",
    )
    add_entity_compare_args(compare_parser)
    compare_parser.set_defaults(func=lambda args: asyncio.run(run_entity_compare(args)))

    # compare-viz subcommand
    compare_viz_parser = entity_subs.add_parser(
        "compare-viz",
        help="Generate comparison visualizations without re-running analysis",
        description="Standalone visualization generation from scored entity data.",
    )
    add_entity_compare_viz_args(compare_viz_parser)
    compare_viz_parser.set_defaults(func=lambda args: asyncio.run(run_entity_compare_viz(args)))

    # report subcommand
    report_parser = entity_subs.add_parser(
        "report",
        help="Generate markdown summary report of comparison findings",
        description="Auto-generate a markdown report summarizing comparison results, group contrasts, and ICC decomposition.",
    )
    add_entity_report_args(report_parser)
    report_parser.set_defaults(func=lambda args: asyncio.run(run_entity_report(args)))

    # prepare-scoring subcommand
    prep_score_parser = entity_subs.add_parser(
        "prepare-scoring",
        aliases=["prep"],
        help="Generate scoring input CSV with real context from windows",
        description="Extract entity-context pairs from relationship detection windows, ensuring actual source text is used as context rather than placeholders.",
    )
    add_entity_prepare_scoring_args(prep_score_parser)
    prep_score_parser.set_defaults(func=lambda args: asyncio.run(run_entity_prepare_scoring(args)))

    # agreement subcommand
    agreement_parser = entity_subs.add_parser(
        "agreement",
        help="Compute inter-rater agreement (Krippendorff's Alpha)",
        description="Measure agreement among LLM scoring runs (run-level) or among participants (participant-level) using Krippendorff's Alpha.",
    )
    add_entity_agreement_args(agreement_parser)
    agreement_parser.set_defaults(func=lambda args: asyncio.run(run_entity_agreement(args)))

    # cluster subcommand
    cluster_parser = entity_subs.add_parser(
        "cluster",
        help="Cluster participants by scoring patterns",
        description="Discover natural groupings among participants using hierarchical, k-means, or HDBSCAN clustering. Optionally run PCA for dimensionality reduction.",
    )
    add_entity_cluster_args(cluster_parser)
    cluster_parser.set_defaults(func=lambda args: asyncio.run(run_entity_cluster(args)))


def _add_decisions_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add decision/factor analysis subcommands."""
    from qualitative_analysis.decisions_cli import (
        add_decisions_detect_args,
        run_decisions_detect,
        add_decisions_normalize_args,
        run_decisions_normalize,
        add_decisions_aggregate_args,
        run_decisions_aggregate,
        add_decisions_viz_args,
        run_decisions_viz,
    )

    dec_parser = subparsers.add_parser(
        "decisions",
        aliases=["dec"],
        help="Decision/factor extraction and analysis",
        description="Extract decisions and their supporting/opposing factors from text.\n\n"
                    "Recommended pipeline order:\n"
                    "  detect → normalize → aggregate → viz",
    )

    dec_subs = dec_parser.add_subparsers(
        dest="action",
        title="commands",
        description="Available decision commands",
        metavar="ACTION",
    )

    # detect subcommand
    detect_parser = dec_subs.add_parser(
        "detect",
        help="Extract decisions and factors from text",
        description="Analyze CSV text data to identify decisions and their supporting/opposing factors.",
    )
    add_decisions_detect_args(detect_parser)
    detect_parser.set_defaults(func=lambda args: asyncio.run(run_decisions_detect(args)))

    # normalize subcommand
    normalize_parser = dec_subs.add_parser(
        "normalize",
        help="Normalize/cluster decisions and factors",
        description="Cluster semantically similar decisions and factors across texts.",
    )
    add_decisions_normalize_args(normalize_parser)
    normalize_parser.set_defaults(func=lambda args: asyncio.run(run_decisions_normalize(args)))

    # aggregate subcommand
    aggregate_parser = dec_subs.add_parser(
        "aggregate",
        help="Aggregate decision/factor statistics",
        description="Compute frequency counts, polarity distribution, and cross-text coverage.",
    )
    add_decisions_aggregate_args(aggregate_parser)
    aggregate_parser.set_defaults(func=lambda args: asyncio.run(run_decisions_aggregate(args)))

    # viz subcommand
    viz_parser = dec_subs.add_parser(
        "viz",
        help="Generate decision/factor visualizations",
        description="Create frequency charts, polarity distributions, and decision-factor network graphs.",
    )
    add_decisions_viz_args(viz_parser)
    viz_parser.set_defaults(func=lambda args: asyncio.run(run_decisions_viz(args)))


def main() -> int:
    """Main entry point for the unified CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    # No command specified
    if args.command is None:
        parser.print_help()
        return 0
    
    # Command specified but no action
    if not hasattr(args, "func"):
        # Find the subparser for this command and print its help
        parser.parse_args([args.command, "--help"])
        return 0
    
    # Run the command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
