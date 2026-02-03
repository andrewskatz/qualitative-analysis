"""
Entity comparison report generation.

Generates markdown summary reports from comparison results, group analyses,
and optional Bayesian model outputs. Separated from CLI logic to enable
programmatic report generation and testing.

Example usage:
    from qualitative_analysis.entity.comparison import ParticipantComparison
    from qualitative_analysis.entity_report import generate_comparison_report

    comparison = ParticipantComparison()
    comparison.load_scores("scored_entities.csv")
    result = comparison.compute_distances(metric="euclidean")

    report_md = generate_comparison_report(
        comparison=comparison,
        result=result,
        dimension_names=["social", "ecological", "technological"],
        metric="euclidean",
        input_filename="scored_entities.csv",
    )
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from qualitative_analysis.core.cli_utils import PACKAGE_VERSION

logger = logging.getLogger(__name__)


def generate_comparison_report(
    comparison: Any,
    result: Any,
    dimension_names: List[str],
    metric: str,
    input_filename: str = "input.csv",
    groups: Optional[Dict[str, List[str]]] = None,
    group_result: Optional[Any] = None,
    bayesian_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Generate a markdown summary report of comparison findings.

    Args:
        comparison: ParticipantComparison with loaded scores.
        result: ComparisonResult from compute_distances().
        dimension_names: List of dimension names.
        metric: Distance metric used.
        input_filename: Name of the input file (for display).
        groups: Optional group definitions.
        group_result: Optional precomputed GroupComparisonResult.
            If groups are provided but group_result is not, it will be computed.
        bayesian_dir: Optional path to directory with Bayesian results JSON files.

    Returns:
        Markdown report as a string.
    """
    n_participants = len(comparison.scores_by_participant)
    pids = list(comparison.scores_by_participant.keys())

    lines = []
    lines.append("# Entity Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Input:** `{input_filename}`")
    lines.append(f"**Dimensions:** {', '.join(d.capitalize() for d in dimension_names)}")
    lines.append(f"**Participants:** {n_participants}")
    lines.append(f"**Distance Metric:** {metric.capitalize()}")
    lines.append("")

    # --- Summary ---
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Mean distance | {result.mean_distance:.4f} |")
    lines.append(f"| Median distance | {result.median_distance:.4f} |")
    lines.append(f"| Min distance | {result.min_distance:.4f} |")
    lines.append(f"| Max distance | {result.max_distance:.4f} |")
    lines.append(f"| Most similar | {result.most_similar_pair[0]} <-> {result.most_similar_pair[1]} ({result.most_similar_pair[2]:.4f}) |")
    lines.append(f"| Most different | {result.most_different_pair[0]} <-> {result.most_different_pair[1]} ({result.most_different_pair[2]:.4f}) |")
    lines.append("")

    # --- Per-participant dimension means ---
    lines.append("## Participant Dimension Means")
    lines.append("")

    header = "| Participant | " + " | ".join(d.capitalize() for d in dimension_names) + " |"
    sep = "|------------|" + "|".join("-" * (len(d) + 2) for d in dimension_names) + "|"
    lines.append(header)
    lines.append(sep)

    for pid in sorted(pids):
        scores = comparison.scores_by_participant[pid]
        dim_means = {}
        for dim in dimension_names:
            vals = [s[dimension_names.index(dim)] for s in scores]
            dim_means[dim] = np.mean(vals)
        row = f"| {pid} | " + " | ".join(f"{dim_means[d]:.1f}" for d in dimension_names) + " |"
        lines.append(row)

    lines.append("")

    # --- Group comparison ---
    if groups:
        lines.append("## Group Comparison")
        lines.append("")

        if group_result is None:
            group_result = comparison.compute_group_distances(
                metric=metric, aggregate="mean"
            )
        lines.append(group_result.summary_str())
        lines.append("")

    # --- Bayesian results ---
    if bayesian_dir:
        bayesian_path = Path(bayesian_dir)
        contrasts_file = bayesian_path / "group_contrasts.json"
        icc_file = bayesian_path / "icc_decomposition.json"

        if contrasts_file.exists():
            lines.append("## Bayesian Group Contrasts")
            lines.append("")

            with open(contrasts_file) as f:
                contrasts = json.load(f)

            for dim, dim_contrasts in contrasts.items():
                lines.append(f"### {dim.capitalize()}")
                lines.append("")
                lines.append("| Comparison | Mean Δ | 94% HDI | P(direction) |")
                lines.append("|-----------|--------|---------|--------------|")

                for pair_key, c in dim_contrasts.items():
                    p_dir = max(c.get("p_a_gt_b", 0), c.get("p_b_gt_a", 0))
                    lines.append(
                        f"| {c['group_a']} vs {c['group_b']} "
                        f"| {c['mean_diff']:+.2f} "
                        f"| [{c['hdi_3%']:+.2f}, {c['hdi_97%']:+.2f}] "
                        f"| {p_dir:.3f} |"
                    )
                lines.append("")

        if icc_file.exists():
            lines.append("## Variance Decomposition (ICC)")
            lines.append("")
            lines.append("| Dimension | Group | Entity | Participant | Run |")
            lines.append("|-----------|-------|--------|-------------|-----|")

            with open(icc_file) as f:
                icc = json.load(f)

            for dim, icc_data in icc.items():
                lines.append(
                    f"| {dim.capitalize()} "
                    f"| {icc_data['icc_group']['mean']:.1%} "
                    f"| {icc_data['icc_entity']['mean']:.1%} "
                    f"| {icc_data['icc_participant']['mean']:.1%} "
                    f"| {icc_data['icc_run']['mean']:.1%} |"
                )
            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by qualitative-analysis v{PACKAGE_VERSION}*")

    return "\n".join(lines)
