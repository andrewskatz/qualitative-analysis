"""
Matplotlib-based visualizations for decision/factor analysis.

Generates:
- Decision frequency bar charts
- Factor frequency bar charts
- Polarity distribution charts
- Decision-factor bipartite network graphs
- Per-decision polarity breakdown charts
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AggregationResult, DecisionExtractionResult

logger = logging.getLogger(__name__)


class DecisionVisualizer:
    """Generate matplotlib visualizations for decision/factor data."""

    # Polarity color scheme
    POLARITY_COLORS = {
        "supporting": "#2ecc71",
        "opposing": "#e74c3c",
        "neutral": "#95a5a6",
    }

    def plot_decision_frequency(
        self,
        aggregation: AggregationResult,
        output_path: str,
        top_n: int = 20,
        title: str = "Decision Frequency",
    ) -> Path:
        """
        Horizontal bar chart of most frequent decisions.

        Args:
            aggregation: Aggregation result with frequency data.
            output_path: Path for the output image.
            top_n: Number of top decisions to show.
            title: Chart title.

        Returns:
            Path to the saved image.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        freq = aggregation.decision_frequency
        if not freq:
            logger.warning("No decisions to plot")
            return Path(output_path)

        # Sort by frequency, take top-N
        sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        labels = [self._truncate(d, 60) for d, _ in sorted_items]
        counts = [c for _, c in sorted_items]

        fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
        bars = ax.barh(range(len(labels)), counts, color="#3498db", edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(title, fontsize=14, fontweight="bold")

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        out = Path(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved decision frequency chart: {out}")
        return out

    def plot_factor_frequency(
        self,
        aggregation: AggregationResult,
        output_path: str,
        top_n: int = 20,
        title: str = "Factor Frequency",
    ) -> Path:
        """
        Horizontal bar chart of most frequent factors.

        Args:
            aggregation: Aggregation result with frequency data.
            output_path: Path for the output image.
            top_n: Number of top factors to show.
            title: Chart title.

        Returns:
            Path to the saved image.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        freq = aggregation.factor_frequency
        if not freq:
            logger.warning("No factors to plot")
            return Path(output_path)

        sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        labels = [self._truncate(f, 60) for f, _ in sorted_items]
        counts = [c for _, c in sorted_items]

        fig, ax = plt.subplots(figsize=(12, max(6, len(labels) * 0.4)))
        bars = ax.barh(range(len(labels)), counts, color="#e67e22", edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(title, fontsize=14, fontweight="bold")

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(count),
                va="center",
                fontsize=8,
            )

        plt.tight_layout()
        out = Path(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved factor frequency chart: {out}")
        return out

    def plot_polarity_distribution(
        self,
        aggregation: AggregationResult,
        output_path: str,
        title: str = "Factor Polarity Distribution",
    ) -> Path:
        """
        Stacked bar chart showing polarity distribution.

        Args:
            aggregation: Aggregation result with polarity data.
            output_path: Path for the output image.
            title: Chart title.

        Returns:
            Path to the saved image.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        dist = aggregation.polarity_distribution
        if not dist:
            logger.warning("No polarity data to plot")
            return Path(output_path)

        polarities = ["supporting", "opposing", "neutral"]
        counts = [dist.get(p, 0) for p in polarities]
        colors = [self.POLARITY_COLORS[p] for p in polarities]
        total = sum(counts) or 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        bars = ax1.bar(polarities, counts, color=colors, edgecolor="white", width=0.6)
        ax1.set_ylabel("Count")
        ax1.set_title(title, fontsize=13, fontweight="bold")
        for bar, count in zip(bars, counts):
            pct = count / total * 100
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{count} ({pct:.0f}%)",
                ha="center",
                fontsize=10,
            )

        # Pie chart
        nonzero = [(p, c) for p, c in zip(polarities, counts) if c > 0]
        if nonzero:
            pie_labels, pie_counts = zip(*nonzero)
            pie_colors = [self.POLARITY_COLORS[p] for p in pie_labels]
            ax2.pie(
                pie_counts,
                labels=pie_labels,
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.set_title("Proportions", fontsize=13, fontweight="bold")

        plt.tight_layout()
        out = Path(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved polarity distribution chart: {out}")
        return out

    def plot_decision_factor_network(
        self,
        results: List[DecisionExtractionResult],
        output_path: str,
        min_frequency: int = 1,
        title: str = "Decision-Factor Network",
    ) -> Path:
        """
        Bipartite network graph: decisions connected to factors.

        Decision nodes in blue, factor nodes in orange.
        Edge color by polarity (green=supporting, red=opposing, grey=neutral).
        Node size by frequency.

        Args:
            results: List of extraction results.
            output_path: Path for the output image.
            min_frequency: Minimum factor frequency to include.
            title: Chart title.

        Returns:
            Path to the saved image.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            import networkx as nx
        except ImportError:
            logger.error("networkx is required for network plots. Install with: pip install networkx")
            return Path(output_path)

        G = nx.Graph()

        # Collect decision-factor links with polarity
        decision_freq: Counter = Counter()
        factor_freq: Counter = Counter()
        edges: List[Dict[str, Any]] = []

        for r in results:
            for d in r.decisions:
                decision_freq[d.text] += 1
                for f in d.factors:
                    factor_freq[f.text] += 1
                    edges.append(
                        {
                            "decision": d.text,
                            "factor": f.text,
                            "polarity": f.polarity,
                        }
                    )

        # Filter by min_frequency
        included_factors = {
            f for f, c in factor_freq.items() if c >= min_frequency
        }

        if not edges:
            logger.warning("No decision-factor links to plot")
            return Path(output_path)

        # Add nodes
        for d_text in decision_freq:
            G.add_node(d_text, node_type="decision", freq=decision_freq[d_text])

        for edge in edges:
            if edge["factor"] not in included_factors:
                continue
            f_text = edge["factor"]
            if f_text not in G:
                G.add_node(f_text, node_type="factor", freq=factor_freq[f_text])
            G.add_edge(
                edge["decision"],
                edge["factor"],
                polarity=edge["polarity"],
            )

        if len(G.nodes) == 0:
            logger.warning("No nodes in network after filtering")
            return Path(output_path)

        # Layout
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)

        # Node properties
        decision_nodes = [
            n for n, d in G.nodes(data=True) if d.get("node_type") == "decision"
        ]
        factor_nodes = [
            n for n, d in G.nodes(data=True) if d.get("node_type") == "factor"
        ]

        decision_sizes = [
            300 + G.nodes[n].get("freq", 1) * 200 for n in decision_nodes
        ]
        factor_sizes = [
            200 + G.nodes[n].get("freq", 1) * 150 for n in factor_nodes
        ]

        # Edge colors by polarity
        edge_colors = [
            self.POLARITY_COLORS.get(
                G.edges[e].get("polarity", "neutral"), "#95a5a6"
            )
            for e in G.edges
        ]

        fig, ax = plt.subplots(figsize=(16, 12))

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color=edge_colors, alpha=0.6, width=1.5
        )

        # Draw decision nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=decision_nodes,
            node_size=decision_sizes,
            node_color="#3498db",
            alpha=0.9,
            ax=ax,
        )

        # Draw factor nodes
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=factor_nodes,
            node_size=factor_sizes,
            node_color="#e67e22",
            alpha=0.7,
            ax=ax,
        )

        # Labels
        labels = {n: self._truncate(n, 25) for n in G.nodes}
        nx.draw_networkx_labels(
            G, pos, labels, font_size=7, ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
                   markersize=10, label="Decision"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22",
                   markersize=10, label="Factor"),
            Line2D([0], [0], color="#2ecc71", linewidth=2, label="Supporting"),
            Line2D([0], [0], color="#e74c3c", linewidth=2, label="Opposing"),
            Line2D([0], [0], color="#95a5a6", linewidth=2, label="Neutral"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        plt.tight_layout()
        out = Path(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved decision-factor network: {out}")
        return out

    def plot_factor_polarity_by_decision(
        self,
        results: List[DecisionExtractionResult],
        output_path: str,
        top_n_decisions: int = 10,
        title: str = "Factor Polarity by Decision",
    ) -> Path:
        """
        Grouped bar chart showing polarity breakdown per decision.

        Args:
            results: List of extraction results.
            output_path: Path for the output image.
            top_n_decisions: Number of top decisions to show.
            title: Chart title.

        Returns:
            Path to the saved image.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Collect polarity counts per decision
        decision_polarity: Dict[str, Counter] = {}

        for r in results:
            for d in r.decisions:
                if d.text not in decision_polarity:
                    decision_polarity[d.text] = Counter()
                for f in d.factors:
                    decision_polarity[d.text][f.polarity] += 1

        if not decision_polarity:
            logger.warning("No polarity data to plot")
            return Path(output_path)

        # Sort by total factor count, take top-N
        sorted_decisions = sorted(
            decision_polarity.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True,
        )[:top_n_decisions]

        labels = [self._truncate(d, 45) for d, _ in sorted_decisions]
        supporting = [c.get("supporting", 0) for _, c in sorted_decisions]
        opposing = [c.get("opposing", 0) for _, c in sorted_decisions]
        neutral = [c.get("neutral", 0) for _, c in sorted_decisions]

        x = np.arange(len(labels))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, max(6, len(labels) * 0.5)))

        ax.barh(x - width, supporting, width, label="Supporting",
                color=self.POLARITY_COLORS["supporting"])
        ax.barh(x, opposing, width, label="Opposing",
                color=self.POLARITY_COLORS["opposing"])
        ax.barh(x + width, neutral, width, label="Neutral",
                color=self.POLARITY_COLORS["neutral"])

        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Factor Count")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()

        plt.tight_layout()
        out = Path(output_path)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved polarity by decision chart: {out}")
        return out

    def generate_all(
        self,
        results: List[DecisionExtractionResult],
        aggregation: AggregationResult,
        output_dir: str,
    ) -> Dict[str, Path]:
        """
        Generate all visualization types to a directory.

        Args:
            results: Extraction results.
            aggregation: Aggregation result.
            output_dir: Directory for output images.

        Returns:
            Dict mapping plot name to file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        outputs: Dict[str, Path] = {}

        outputs["decision_frequency"] = self.plot_decision_frequency(
            aggregation, str(out / "decision_frequency.png")
        )
        outputs["factor_frequency"] = self.plot_factor_frequency(
            aggregation, str(out / "factor_frequency.png")
        )
        outputs["polarity_distribution"] = self.plot_polarity_distribution(
            aggregation, str(out / "polarity_distribution.png")
        )
        outputs["decision_factor_network"] = self.plot_decision_factor_network(
            results, str(out / "decision_factor_network.png")
        )
        outputs["polarity_by_decision"] = self.plot_factor_polarity_by_decision(
            results, str(out / "polarity_by_decision.png")
        )

        return outputs

    @staticmethod
    def _truncate(text: str, max_len: int = 50) -> str:
        """Truncate text for labels."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
