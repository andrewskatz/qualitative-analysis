"""
Visualization utilities for multi-participant entity score comparisons.

Provides faceted ternary plots, overlaid ternary plots, distance heatmaps,
forest plots, and similarity maps for comparing entity scores across
participants.

Example usage:
    from qualitative_analysis.entity.comparison import ParticipantComparison
    from qualitative_analysis.entity.comparison_viz import ComparisonVisualizer

    comparison = ParticipantComparison()
    comparison.load_scores(scores_csv_path)
    result = comparison.compute_distances(metric="euclidean")

    viz = ComparisonVisualizer(comparison)
    viz.generate_distance_heatmap(result, output_path="heatmap.png")
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ComparisonVisualizer:
    """
    Visualization utilities for multi-participant comparisons.

    Provides faceted ternary plots, overlaid ternary plots, and
    distance heatmaps for comparing entity scores across participants.
    """

    def __init__(self, comparison: "ParticipantComparison"):
        """
        Initialize with a loaded ParticipantComparison.

        Args:
            comparison: ParticipantComparison with loaded scores.
        """
        self.comparison = comparison
        self.dimension_colors = {
            "social": "#1f77b4",       # Blue
            "ecological": "#2ca02c",   # Green
            "technological": "#d62728" # Red
        }

        # Extended color palette for participants
        self.participant_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        ]

    def _barycentric_to_cartesian(
        self,
        a: float,
        b: float,
        c: float
    ) -> Tuple[float, float]:
        """
        Convert barycentric coordinates to cartesian coordinates.

        Args:
            a: Coordinate for top vertex
            b: Coordinate for bottom-left vertex
            c: Coordinate for bottom-right vertex

        Returns:
            Tuple of (x, y) cartesian coordinates
        """
        # Equilateral triangle vertices
        vertices = np.array([
            [0, 0],               # Bottom-left
            [1, 0],               # Bottom-right
            [0.5, np.sqrt(3)/2]   # Top
        ])

        x = a * vertices[2, 0] + b * vertices[0, 0] + c * vertices[1, 0]
        y = a * vertices[2, 1] + b * vertices[0, 1] + c * vertices[1, 1]

        return x, y

    def _draw_triangle(
        self,
        ax,
        dimension_names: List[str],
        show_labels: bool = True,
        show_grid: bool = True,
    ) -> None:
        """Draw ternary triangle with optional labels and grid."""
        import matplotlib.pyplot as plt

        # Triangle outline
        vertices = np.array([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [0, 0]
        ])
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=1.5)

        if show_labels and len(dimension_names) >= 3:
            label_offset = 0.06
            # RGB channel colors: dim[0]=Social→Blue, dim[1]=Ecological→Green, dim[2]=Technological→Red
            dim_label_colors = [
                (0.0, 0.0, 0.85),   # dim[0] Social → Blue
                (0.0, 0.60, 0.0),   # dim[1] Ecological → Green (darker for readability)
                (0.85, 0.0, 0.0),   # dim[2] Technological → Red
            ]

            # Bottom-left (dimension 2)
            ax.text(0, -label_offset, dimension_names[2][:4].upper(),
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    color=dim_label_colors[2])

            # Bottom-right (dimension 0)
            ax.text(1, -label_offset, dimension_names[0][:4].upper(),
                    ha='center', va='top', fontsize=8, fontweight='bold',
                    color=dim_label_colors[0])

            # Top (dimension 1)
            ax.text(0.5, np.sqrt(3)/2 + label_offset, dimension_names[1][:4].upper(),
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=dim_label_colors[1])

        if show_grid:
            for i in [2, 4, 6, 8]:
                alpha = 0.2

                # Lines parallel to each edge
                x1, y1 = self._barycentric_to_cartesian(1-i/10, i/10, 0)
                x2, y2 = self._barycentric_to_cartesian(0, i/10, 1-i/10)
                ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

                x1, y1 = self._barycentric_to_cartesian(i/10, 0, 1-i/10)
                x2, y2 = self._barycentric_to_cartesian(i/10, 1-i/10, 0)
                ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

                x1, y1 = self._barycentric_to_cartesian(0, 1-i/10, i/10)
                x2, y2 = self._barycentric_to_cartesian(1-i/10, 0, i/10)
                ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to sum to 1 for ternary plotting."""
        scores = np.asarray(scores, dtype=np.float64)
        scores = np.clip(scores, 0, None)  # Ensure non-negative

        if scores.ndim == 1:
            total = scores.sum()
            return scores / total if total > 0 else np.ones(len(scores)) / len(scores)
        else:
            totals = scores.sum(axis=1, keepdims=True)
            totals = np.where(totals == 0, 1, totals)
            return scores / totals

    def generate_faceted_ternary(
        self,
        output_path: Optional[Union[str, Path]] = None,
        dimension_names: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        groups: Optional[Dict[str, List[str]]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        max_cols: int = 4,
        marker_size: int = 60,
        show_centroid: bool = True,
        show_convex_hull: bool = False,
        title: Optional[str] = None,
    ):
        """
        Generate faceted ternary plots, one per participant.

        Creates a grid of ternary plots showing each participant's entity
        scores, allowing visual comparison of scoring patterns.

        Args:
            output_path: Path to save PNG file (optional).
            dimension_names: List of 3 dimension names. Uses loaded if None.
            participants: Specific participants to include. All if None.
            groups: Optional dict mapping group names to participant ID lists.
                    When provided, participant titles are colored by group.
            figsize: Figure size. Auto-calculated if None.
            max_cols: Maximum columns in grid.
            marker_size: Size of entity markers.
            show_centroid: Show centroid marker for each participant.
            show_convex_hull: Draw convex hull around entities.
            title: Overall plot title.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        # Validate dimensions
        dims = dimension_names or self.comparison.dimension_names
        if len(dims) != 3:
            raise ValueError(f"Faceted ternary requires exactly 3 dimensions, got {len(dims)}")

        # Select participants
        if participants:
            pids = [p for p in participants if p in self.comparison.scores_by_participant]
        else:
            pids = list(self.comparison.scores_by_participant.keys())

        if not pids:
            raise ValueError("No participants to plot")

        # Build pid → group color mapping
        # Fall back to groups stored on the visualizer instance
        groups = groups or getattr(self, 'groups', None)
        pid_to_group_color: Dict[str, str] = {}
        if groups:
            for g_idx, (group_name, members) in enumerate(groups.items()):
                color = self.participant_colors[g_idx % len(self.participant_colors)]
                for member in members:
                    pid_to_group_color[member] = color

        n_participants = len(pids)
        n_cols = min(max_cols, n_participants)
        n_rows = (n_participants + n_cols - 1) // n_cols

        # Calculate figure size
        if figsize is None:
            fig_width = 3.5 * n_cols
            fig_height = 3.5 * n_rows + (0.5 if title else 0)
            figsize = (fig_width, fig_height)

        fig = plt.figure(figsize=figsize)

        # Add space for title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.2)

        for idx, pid in enumerate(pids):
            row = idx // n_cols
            col = idx % n_cols

            ax = fig.add_subplot(gs[row, col])

            # Get scores for this participant
            scores = self.comparison.scores_by_participant[pid]
            normalized = self._normalize_scores(scores)

            # Convert to cartesian coordinates
            xs, ys = [], []
            for score in normalized:
                x, y = self._barycentric_to_cartesian(score[1], score[2], score[0])
                xs.append(x)
                ys.append(y)

            xs = np.array(xs)
            ys = np.array(ys)

            # Draw triangle
            self._draw_triangle(ax, dims, show_labels=True, show_grid=True)

            # Draw convex hull if requested
            if show_convex_hull and len(xs) >= 3:
                try:
                    from scipy.spatial import ConvexHull

                    points = np.column_stack([xs, ys])
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])  # Close hull
                    ax.fill(hull_points[:, 0], hull_points[:, 1],
                           alpha=0.1, color='gray')
                    ax.plot(hull_points[:, 0], hull_points[:, 1],
                           'k--', alpha=0.3, linewidth=1)
                except Exception as e:
                    logger.debug(f"Could not compute convex hull for {pid}: {e}")

            # RGB color from dimensional profile + per-entity sizes
            colors = []
            sizes = []
            mean_scores_per_entity = scores.mean(axis=1)

            for eidx, score in enumerate(normalized):
                # Map normalized proportions to RGB channels:
                #   dims[0] (Social) → Blue, dims[1] (Ecological) → Green, dims[2] (Technological) → Red
                max_norm = max(score)
                if max_norm > 0:
                    s = 0.85 / max_norm
                    colors.append((
                        min(1.0, score[2] * s),  # R ← Technological
                        min(1.0, score[1] * s),  # G ← Ecological
                        min(1.0, score[0] * s),  # B ← Social
                    ))
                else:
                    colors.append((0.5, 0.5, 0.5))
                # Size encodes magnitude on absolute 0-100 scale
                abs_frac = max(0, min(mean_scores_per_entity[eidx] / 100.0, 1.0))
                sizes.append(marker_size * (0.15 + 2.35 * abs_frac))

            # Plot entities with size encoding magnitude
            ax.scatter(xs, ys, s=sizes, c=colors,
                      edgecolors='white', linewidths=0.5, alpha=0.7)

            # Plot centroid
            if show_centroid:
                centroid_scores = self._normalize_scores(scores.mean(axis=0))
                cx, cy = self._barycentric_to_cartesian(
                    centroid_scores[1], centroid_scores[2], centroid_scores[0]
                )
                ax.scatter([cx], [cy], s=150, c='black', marker='X',
                          edgecolors='white', linewidths=2, zorder=10)

            # Configure subplot
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
            ax.set_aspect('equal')
            ax.axis('off')

            # Truncate long participant IDs, color by group if available
            display_name = pid[:20] + "..." if len(pid) > 20 else pid
            title_color = pid_to_group_color.get(pid, 'black')
            ax.set_title(f"{display_name}\n(n={len(scores)})", fontsize=9, pad=5,
                        color=title_color, fontweight='bold' if title_color != 'black' else 'normal')

        # Add group legend if groups are defined
        if groups and pid_to_group_color:
            from matplotlib.lines import Line2D
            group_handles = []
            for g_idx, (group_name, members) in enumerate(groups.items()):
                color = self.participant_colors[g_idx % len(self.participant_colors)]
                group_handles.append(
                    Line2D([0], [0], marker='s', color='w',
                           markerfacecolor=color, markersize=10,
                           markeredgecolor='black', markeredgewidth=0.5,
                           label=f"{group_name} (n={len(members)})")
                )
            fig.legend(
                handles=group_handles, loc='lower center',
                ncol=len(groups), fontsize=9, framealpha=0.9,
                title="Groups", title_fontsize=10,
                bbox_to_anchor=(0.5, -0.01),
            )

        plt.tight_layout()
        # Make room for group legend at bottom
        if groups and pid_to_group_color:
            fig.subplots_adjust(bottom=0.05)

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved faceted ternary plot to {output_path}")
            return None

        return fig

    def generate_individual_ternary(
        self,
        output_dir: Union[str, Path],
        dimension_names: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        marker_size: int = 80,
        show_centroid: bool = True,
        show_convex_hull: bool = False,
        show_numbers: bool = True,
    ) -> List[Path]:
        """
        Generate individual ternary plot files, one per participant.

        Each plot includes an entity legend sidebar with color-matched
        circle swatches showing the entity's dimensional RGB blend.

        Args:
            output_dir: Directory to save individual PNG files.
            dimension_names: List of 3 dimension names.
            participants: Specific participants to include. All if None.
            marker_size: Size of entity markers.
            show_centroid: Show centroid marker.
            show_convex_hull: Draw convex hull around entities.
            show_numbers: If True, show numbered labels on the plot and
                number-prefixed legend. If False, skip plot labels and
                sort legend entries by color spectrum (hue).

        Returns:
            List of saved file paths.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        dims = dimension_names or self.comparison.dimension_names
        if len(dims) != 3:
            raise ValueError(f"Ternary requires exactly 3 dimensions, got {len(dims)}")

        if participants:
            pids = [p for p in participants if p in self.comparison.scores_by_participant]
        else:
            pids = list(self.comparison.scores_by_participant.keys())

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        for pid in pids:
            scores = self.comparison.scores_by_participant[pid]
            normalized = self._normalize_scores(scores)
            n_entities = len(scores)

            # Get entity names for this participant
            entity_names = self.comparison.entity_names_by_participant.get(pid, [])
            if not entity_names:
                entity_names = [f"Entity {i+1}" for i in range(n_entities)]

            # Determine legend panel width based on entity count
            if n_entities <= 20:
                legend_ratio = 1.2
            elif n_entities <= 50:
                legend_ratio = 1.8
            else:
                legend_ratio = 2.5

            fig = plt.figure(figsize=(6 + 3 * legend_ratio / 1.2, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, legend_ratio], figure=fig)
            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])
            legend_ax.axis('off')

            # Convert to cartesian
            xs, ys = [], []
            for score in normalized:
                x, y = self._barycentric_to_cartesian(score[1], score[2], score[0])
                xs.append(x)
                ys.append(y)

            xs = np.array(xs)
            ys = np.array(ys)

            # Draw triangle
            self._draw_triangle(ax, dims, show_labels=True, show_grid=True)

            # Convex hull
            if show_convex_hull and len(xs) >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    pts = np.column_stack([xs, ys])
                    hull = ConvexHull(pts)
                    hull_pts = pts[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])
                    ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                           alpha=0.1, color='gray')
                    ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                           'k--', alpha=0.3, linewidth=1)
                except Exception as e:
                    logger.debug(f"Could not compute convex hull for {pid}: {e}")

            # RGB color + size encoding per entity
            colors = []
            sizes = []
            mean_scores_per_entity = scores.mean(axis=1)

            for eidx, score in enumerate(normalized):
                max_norm = max(score)
                if max_norm > 0:
                    s = 0.85 / max_norm
                    colors.append((
                        min(1.0, score[2] * s),
                        min(1.0, score[1] * s),
                        min(1.0, score[0] * s),
                    ))
                else:
                    colors.append((0.5, 0.5, 0.5))
                abs_frac = max(0, min(mean_scores_per_entity[eidx] / 100.0, 1.0))
                sizes.append(marker_size * (0.15 + 2.35 * abs_frac))

            ax.scatter(xs, ys, s=sizes, c=colors,
                      edgecolors='black', linewidths=0.8, alpha=0.8)

            # Add numbered labels on the plot (only if show_numbers is True)
            if show_numbers:
                base_offset = 0.05
                placed_labels = []
                min_separation = 0.04

                for eidx in range(n_entities):
                    x, y = xs[eidx], ys[eidx]
                    num = eidx + 1

                    # Find non-colliding position for number label
                    best_pos = None
                    for offset_mult in [1.0, 1.5, 2.0, 2.5]:
                        offset = base_offset * offset_mult
                        candidates = [
                            (0, offset, 'center', 'bottom'),
                            (offset, 0, 'left', 'center'),
                            (0, -offset, 'center', 'top'),
                            (-offset, 0, 'right', 'center'),
                        ]
                        for dx, dy, ha, va in candidates:
                            lx, ly = x + dx, y + dy
                            collision = any(
                                np.sqrt((lx - px)**2 + (ly - py)**2) < min_separation
                                for px, py in placed_labels
                            )
                            if not collision:
                                best_pos = (lx, ly, ha, va)
                                break
                        if best_pos:
                            break
                    if not best_pos:
                        best_pos = (x, y + base_offset * 2.5, 'center', 'bottom')

                    lx, ly, ha, va = best_pos
                    placed_labels.append((lx, ly))

                    ax.plot([x, lx], [y, ly], color='gray', linewidth=0.4, alpha=0.5)
                    ax.text(lx, ly, str(num), fontsize=7, ha='center', va='center',
                           bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="gray", alpha=0.8),
                           zorder=10)

            # Centroid
            if show_centroid:
                centroid_scores = self._normalize_scores(scores.mean(axis=0))
                cx, cy = self._barycentric_to_cartesian(
                    centroid_scores[1], centroid_scores[2], centroid_scores[0]
                )
                ax.scatter([cx], [cy], s=150, c='black', marker='X',
                          edgecolors='white', linewidths=2, zorder=10)

            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"{pid}\n(n={n_entities})", fontsize=11, pad=10)

            # Subtitle
            ax.text(
                0.5, -0.06,
                "Position: relative proportions | Color: dimensional blend | Size: score magnitude",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=7, color='gray', style='italic'
            )

            # Build entity legend with colored swatches
            legend_elements = []
            legend_labels = []

            if show_numbers:
                # Numbered order
                order = list(range(n_entities))
            else:
                # Sort by hue (color spectrum) for easier visual matching
                import colorsys
                hues = []
                for c in colors:
                    h, s, v = colorsys.rgb_to_hsv(c[0], c[1], c[2])
                    hues.append((h, s, v))
                order = sorted(range(n_entities), key=lambda i: (hues[i][0], -hues[i][1], -hues[i][2]))

            for eidx in order:
                legend_elements.append(plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=colors[eidx],
                    markeredgecolor='black', markeredgewidth=0.5,
                    markersize=8, alpha=0.9
                ))
                name = entity_names[eidx] if eidx < len(entity_names) else f"Entity {eidx+1}"
                if show_numbers:
                    legend_labels.append(f"{eidx+1}. {name}")
                else:
                    legend_labels.append(name)

            # Layout columns based on entity count
            if n_entities <= 25:
                ncol, fontsize = 1, 8
            elif n_entities <= 50:
                ncol, fontsize = 2, 7
            else:
                ncol, fontsize = 3, 6

            legend_ax.legend(
                legend_elements, legend_labels,
                loc='center left', fontsize=fontsize,
                frameon=True, framealpha=0.9,
                title="Entities", title_fontsize=10,
                ncol=ncol, handletextpad=0.5,
                borderpad=0.8, labelspacing=0.4,
            )

            plt.tight_layout()

            # Sanitize filename
            safe_name = pid.replace("/", "_").replace("\\", "_").replace(" ", "_")
            path = output_dir / f"{safe_name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            saved_paths.append(path)

        logger.info(f"Saved {len(saved_paths)} individual ternary plots to {output_dir}")
        return saved_paths

    def generate_overlaid_ternary(
        self,
        output_path: Optional[Union[str, Path]] = None,
        dimension_names: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
        marker_size: int = 80,
        show_centroids: bool = True,
        show_confidence_ellipses: bool = False,
        show_legend: bool = True,
        connect_same_entities: bool = False,
        title: str = "Multi-Participant Entity Comparison",
    ):
        """
        Generate overlaid ternary plot with all participants.

        Shows all participants on a single ternary plot with different
        colors, making it easy to see overall patterns and differences.

        Args:
            output_path: Path to save PNG file (optional).
            dimension_names: List of 3 dimension names.
            participants: Specific participants to include.
            figsize: Figure size.
            marker_size: Size of entity markers.
            show_centroids: Show centroid marker for each participant.
            show_confidence_ellipses: Show 95% confidence ellipses around
                each participant's entity distribution (requires >= 3 entities).
            show_legend: Show participant legend.
            connect_same_entities: Draw lines connecting same entity across participants.
            title: Plot title.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib.patches import Ellipse

        # Validate dimensions
        dims = dimension_names or self.comparison.dimension_names
        if len(dims) != 3:
            raise ValueError(f"Ternary plot requires exactly 3 dimensions, got {len(dims)}")

        # Select participants
        if participants:
            pids = [p for p in participants if p in self.comparison.scores_by_participant]
        else:
            pids = list(self.comparison.scores_by_participant.keys())

        if not pids:
            raise ValueError("No participants to plot")

        # Create figure with legend space
        if show_legend:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])
            legend_ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            legend_ax = None

        # Draw triangle
        self._draw_triangle(ax, dims, show_labels=True, show_grid=True)

        # Track entities for connecting lines
        entity_positions: Dict[str, List[Tuple[float, float, str]]] = {}

        legend_handles = []
        legend_labels = []

        for pid_idx, pid in enumerate(pids):
            color = self.participant_colors[pid_idx % len(self.participant_colors)]

            scores = self.comparison.scores_by_participant[pid]
            normalized = self._normalize_scores(scores)

            # Convert to cartesian
            xs, ys = [], []
            for i, score in enumerate(normalized):
                x, y = self._barycentric_to_cartesian(score[1], score[2], score[0])
                xs.append(x)
                ys.append(y)

                # Track for connecting lines
                if connect_same_entities and i < len(self.comparison.entity_names):
                    entity = self.comparison.entity_names[i]
                    if entity not in entity_positions:
                        entity_positions[entity] = []
                    entity_positions[entity].append((x, y, pid))

            # Compute per-entity sizes for magnitude encoding (absolute 0-100 scale)
            scores_raw = self.comparison.scores_by_participant[pid]
            entity_mean_scores = scores_raw.mean(axis=1)
            point_sizes = [
                marker_size * (0.15 + 2.35 * max(0, min(ms / 100.0, 1.0)))
                for ms in entity_mean_scores
            ]

            # Plot entities with size encoding magnitude
            scatter = ax.scatter(xs, ys, s=point_sizes, c=[color], alpha=0.6,
                               edgecolors='white', linewidths=0.5, label=pid)

            # Plot centroid
            cx, cy = None, None
            if show_centroids:
                centroid_scores = self._normalize_scores(scores.mean(axis=0))
                cx, cy = self._barycentric_to_cartesian(
                    centroid_scores[1], centroid_scores[2], centroid_scores[0]
                )
                ax.scatter([cx], [cy], s=200, c=[color], marker='X',
                          edgecolors='black', linewidths=2, zorder=10)

            # Draw 95% confidence ellipse around the entity scatter
            if show_confidence_ellipses and len(xs) >= 3:
                xs_arr, ys_arr = np.array(xs), np.array(ys)
                cov = np.cov(xs_arr, ys_arr)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]

                # 95% confidence: chi2(2) = 5.991
                chi2_val = 5.991
                width = 2 * np.sqrt(chi2_val * max(eigenvalues[0], 1e-10))
                height = 2 * np.sqrt(chi2_val * max(eigenvalues[1], 1e-10))
                angle = np.degrees(np.arctan2(
                    eigenvectors[1, 0], eigenvectors[0, 0]
                ))

                ellipse_cx = cx if cx is not None else np.mean(xs_arr)
                ellipse_cy = cy if cy is not None else np.mean(ys_arr)

                ellipse = Ellipse(
                    xy=(ellipse_cx, ellipse_cy),
                    width=width, height=height, angle=angle,
                    facecolor=color, alpha=0.1,
                    edgecolor=color, linewidth=1.5, linestyle="--",
                )
                ax.add_patch(ellipse)

            # For legend
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color, markersize=10))
            legend_labels.append(f"{pid} (n={len(scores)})")

        # Draw connecting lines between same entities
        if connect_same_entities:
            for entity, positions in entity_positions.items():
                if len(positions) > 1:
                    for i in range(len(positions) - 1):
                        x1, y1, _ = positions[i]
                        x2, y2, _ = positions[i + 1]
                        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)

        # Configure plot
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.15, np.sqrt(3)/2 + 0.15)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=14, pad=15)

        # Add legend
        if show_legend and legend_ax:
            legend_ax.legend(legend_handles, legend_labels,
                           loc='center left', fontsize=9,
                           title="Participants", title_fontsize=10,
                           frameon=True, framealpha=0.9)

        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved overlaid ternary plot to {output_path}")
            return None

        return fig

    def generate_distance_heatmap(
        self,
        result: "ComparisonResult",
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 8),
        show_values: bool = True,
        show_dendrogram: bool = True,
        cmap: str = "RdYlBu_r",
        title: Optional[str] = None,
        groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Generate distance matrix heatmap with optional dendrogram or group ordering.

        Args:
            result: ComparisonResult with distance matrix.
            output_path: Path to save PNG file (optional).
            figsize: Figure size.
            show_values: Show distance values in cells.
            show_dendrogram: Show hierarchical clustering dendrogram.
            cmap: Colormap name.
            title: Plot title.
            groups: Optional dict mapping group names to participant IDs for ordering.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt

        n = len(result.participant_ids)

        # If groups are provided, order by group membership instead of dendrogram
        if groups:
            # Build ordered list: all members of group1, then group2, etc.
            ordered_ids = []
            group_boundaries = []  # Store indices where groups change

            for group_name, members in groups.items():
                group_start = len(ordered_ids)
                for pid in members:
                    if pid in result.participant_ids:
                        ordered_ids.append(pid)
                if len(ordered_ids) > group_start:
                    group_boundaries.append((group_start, len(ordered_ids), group_name))

            # Add any participants not in groups at the end
            for pid in result.participant_ids:
                if pid not in ordered_ids:
                    ordered_ids.append(pid)

            # Build index mapping
            order = [result.participant_ids.index(pid) for pid in ordered_ids]
            ordered_matrix = result.distance_matrix[np.ix_(order, order)]

            # Use wider figure to accommodate group labels on right
            group_figsize = (figsize[0] + 2, figsize[1])
            fig, ax_heatmap = plt.subplots(figsize=group_figsize)

        elif show_dendrogram and n >= 3:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform

            fig = plt.figure(figsize=figsize)

            # Create grid for dendrogram + heatmap
            gs = fig.add_gridspec(2, 2, width_ratios=[0.2, 1], height_ratios=[0.2, 1],
                                 hspace=0.02, wspace=0.02)

            ax_dendro_top = fig.add_subplot(gs[0, 1])
            ax_dendro_left = fig.add_subplot(gs[1, 0])
            ax_heatmap = fig.add_subplot(gs[1, 1])

            # Compute linkage
            condensed = squareform(result.distance_matrix)
            linkage_matrix = linkage(condensed, method="average")

            # Top dendrogram
            dendro = dendrogram(linkage_matrix, ax=ax_dendro_top, orientation='top',
                              no_labels=True)
            ax_dendro_top.axis('off')

            # Left dendrogram
            dendrogram(linkage_matrix, ax=ax_dendro_left, orientation='left',
                      no_labels=True)
            ax_dendro_left.axis('off')

            # Reorder matrix by dendrogram
            order = dendro['leaves']
            ordered_matrix = result.distance_matrix[np.ix_(order, order)]
            ordered_ids = [result.participant_ids[i] for i in order]
            group_boundaries = []

        else:
            fig, ax_heatmap = plt.subplots(figsize=figsize)
            ordered_matrix = result.distance_matrix
            ordered_ids = result.participant_ids
            order = list(range(n))
            group_boundaries = []

        # Plot heatmap
        im = ax_heatmap.imshow(ordered_matrix, cmap=cmap, aspect='equal')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
        cbar.set_label(f'{result.metric.capitalize()} Distance', fontsize=10)

        # Build group color mapping for labels
        group_colors = {}
        pid_to_group = {}
        if groups and group_boundaries:
            # Use consistent colors from participant_colors
            for g_idx, (start, end, group_name) in enumerate(group_boundaries):
                group_colors[group_name] = self.participant_colors[g_idx % len(self.participant_colors)]
                for pid in ordered_ids[start:end]:
                    pid_to_group[pid] = group_name

            # Draw boundary lines around group blocks
            for start, end, group_name in group_boundaries:
                ax_heatmap.axhline(y=start - 0.5, xmin=0, xmax=1, color='black', linewidth=2)
                ax_heatmap.axhline(y=end - 0.5, xmin=0, xmax=1, color='black', linewidth=2)
                ax_heatmap.axvline(x=start - 0.5, ymin=0, ymax=1, color='black', linewidth=2)
                ax_heatmap.axvline(x=end - 0.5, ymin=0, ymax=1, color='black', linewidth=2)

            # Add group labels on TOP of the heatmap (above the columns)
            for start, end, group_name in group_boundaries:
                mid = (start + end - 1) / 2  # Center of group columns
                ax_heatmap.text(
                    mid,
                    -1.5,  # Position above the heatmap
                    group_name,
                    fontsize=11,
                    fontweight='bold',
                    va='bottom',
                    ha='center',
                    color=group_colors[group_name],
                    clip_on=False,
                )

        # Add labels
        ax_heatmap.set_xticks(range(len(ordered_ids)))
        ax_heatmap.set_yticks(range(len(ordered_ids)))

        # Use full participant IDs (adjust font size based on number of participants)
        label_fontsize = 8 if len(ordered_ids) <= 10 else 7 if len(ordered_ids) <= 15 else 6

        ax_heatmap.set_xticklabels(ordered_ids, rotation=45, ha='right', fontsize=label_fontsize)
        ax_heatmap.set_yticklabels(ordered_ids, fontsize=label_fontsize)

        # Color-code the tick labels by group membership
        if groups and pid_to_group:
            # Color x-axis labels
            for tick_label in ax_heatmap.get_xticklabels():
                pid = tick_label.get_text()
                if pid in pid_to_group:
                    tick_label.set_color(group_colors[pid_to_group[pid]])
            # Color y-axis labels
            for tick_label in ax_heatmap.get_yticklabels():
                pid = tick_label.get_text()
                if pid in pid_to_group:
                    tick_label.set_color(group_colors[pid_to_group[pid]])

        # Add values to cells (use perceptual luminance for text contrast)
        if show_values and len(ordered_ids) <= 15:
            import matplotlib.cm as mcm
            cmap_obj = mcm.get_cmap(cmap)
            vmin, vmax = im.get_clim()
            for i in range(len(ordered_ids)):
                for j in range(len(ordered_ids)):
                    value = ordered_matrix[i, j]
                    # Map value to colormap RGBA, compute perceptual luminance
                    norm_val = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    rgba = cmap_obj(norm_val)
                    luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    text_color = 'white' if luminance < 0.5 else 'black'
                    ax_heatmap.text(j, i, f'{value:.2f}',
                                   ha='center', va='center',
                                   color=text_color, fontsize=7)

        # Title
        plot_title = title or f"Participant Distance Matrix ({result.metric.capitalize()})"
        if show_dendrogram and n >= 3 and not groups:
            fig.suptitle(plot_title, fontsize=12, fontweight='bold', y=0.98)
        else:
            ax_heatmap.set_title(plot_title, fontsize=12, pad=10)

        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved distance heatmap to {output_path}")
            return None

        return fig

    def generate_forest_plot(
        self,
        result: Optional["ComparisonResult"] = None,
        output_path: Optional[Union[str, Path]] = None,
        dimension_names: Optional[List[str]] = None,
        participants: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 8),
        ci_level: float = 0.95,
        group_by: str = "dimension",
        title: Optional[str] = None,
        score_mode: str = "normalized",
        groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Generate forest plot showing per-dimension scores with confidence intervals.

        Displays each participant's mean score with bootstrapped confidence intervals
        for each dimension, allowing visual comparison of scoring patterns.

        Args:
            result: ComparisonResult (optional, not used but kept for consistency).
            output_path: Path to save PNG file (optional).
            dimension_names: List of dimension names to display.
            participants: Specific participants to include.
            figsize: Figure size.
            ci_level: Confidence interval level (default 0.95 = 95% CI).
            group_by: How to organize plot - "dimension" (default) or "participant".
            title: Plot title.
            score_mode: "normalized" for proportions (sum=1) or "raw" for
                absolute scores (0-100 scale).
            groups: Optional dict mapping group names to participant ID lists
                for coloring labels by group membership.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt

        use_raw = score_mode == "raw"

        # Get dimensions
        dims = dimension_names or self.comparison.dimension_names
        n_dims = len(dims)

        # Select participants
        if participants:
            pids = [p for p in participants if p in self.comparison.scores_by_participant]
        else:
            pids = list(self.comparison.scores_by_participant.keys())

        if not pids:
            raise ValueError("No participants to plot")

        n_participants = len(pids)

        # Entity count per participant
        pid_entity_count = {pid: len(self.comparison.scores_by_participant[pid]) for pid in pids}

        # Build group color mapping
        groups = groups or getattr(self, 'groups', None)
        group_colors = {}
        pid_to_group = {}
        group_boundaries = []  # [(boundary_y_position, ...)] for separator lines
        if groups:
            for g_idx, (group_name, members) in enumerate(groups.items()):
                group_colors[group_name] = self.participant_colors[g_idx % len(self.participant_colors)]
                for member in members:
                    pid_to_group[member] = group_name

            # Reorder pids by group membership so group members are adjacent
            ordered_pids = []
            for group_name, members in groups.items():
                group_members = [p for p in pids if p in members]
                if ordered_pids and group_members:
                    group_boundaries.append(len(ordered_pids))
                ordered_pids.extend(group_members)
            # Append any participants not in any group
            ungrouped = [p for p in pids if p not in pid_to_group]
            if ungrouped and ordered_pids:
                group_boundaries.append(len(ordered_pids))
            ordered_pids.extend(ungrouped)
            pids = ordered_pids
            n_participants = len(pids)

        # Compute means and CIs for each participant-dimension pair
        stats = {}  # {(pid, dim_idx): (mean, ci_low, ci_high)}

        for pid in pids:
            raw_scores = self.comparison.scores_by_participant[pid]  # shape: (n_entities, n_dims)
            if use_raw:
                scores = raw_scores
            else:
                # Normalize each entity's scores to sum to 1
                scores = self._normalize_scores(raw_scores)
            n_entities = len(scores)

            for dim_idx in range(n_dims):
                dim_scores = scores[:, dim_idx]
                mean_val = np.mean(dim_scores)

                # Bootstrap confidence interval (seeded for reproducibility)
                if n_entities >= 2:
                    n_bootstrap = 1000
                    rng = np.random.default_rng(42)
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        sample = rng.choice(dim_scores, size=n_entities, replace=True)
                        bootstrap_means.append(np.mean(sample))

                    alpha = 1 - ci_level
                    ci_low = np.percentile(bootstrap_means, alpha / 2 * 100)
                    ci_high = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
                else:
                    # Single entity - no CI
                    ci_low = ci_high = mean_val

                if use_raw:
                    ci_low = max(0.0, min(100.0, ci_low))
                    ci_high = max(0.0, min(100.0, ci_high))
                else:
                    ci_low = max(0.0, min(1.0, ci_low))
                    ci_high = max(0.0, min(1.0, ci_high))

                stats[(pid, dim_idx)] = (mean_val, ci_low, ci_high)

        # Axis configuration based on score mode
        if use_raw:
            x_label = 'Score'
            x_lim = (0, 100)
            ref_line_x = 50
            subtitle_text = "Raw scores (0–100 scale) | Dashed line = midpoint (50)"
        else:
            x_label = 'Proportion'
            x_lim = (0, 1)
            ref_line_x = 1 / n_dims
            subtitle_text = "Scores normalized to proportions (sum = 1) | Dashed line = equal distribution"

        # Create figure (scale height with number of participants/rows)
        if group_by == "dimension":
            # One subplot per dimension, participants on y-axis
            scaled_height = max(figsize[1], 0.4 * n_participants + 2)
            fig, axes = plt.subplots(1, n_dims, figsize=(figsize[0], scaled_height), sharey=True)
            if n_dims == 1:
                axes = [axes]

            for dim_idx, ax in enumerate(axes):
                dim_name = dims[dim_idx] if dim_idx < len(dims) else f"Dim {dim_idx}"

                y_positions = list(range(n_participants))
                means = []
                ci_lows = []
                ci_highs = []

                for pid in pids:
                    mean_val, ci_low, ci_high = stats[(pid, dim_idx)]
                    means.append(mean_val)
                    ci_lows.append(ci_low)
                    ci_highs.append(ci_high)

                # Convert to error bar format (ensure non-negative)
                errors = [
                    [max(0, m - cl) for m, cl in zip(means, ci_lows)],
                    [max(0, ch - m) for m, ch in zip(ci_highs, means)]
                ]

                # Color by dimension
                color = self.dimension_colors.get(dim_name.lower(), self.participant_colors[dim_idx % len(self.participant_colors)])

                ax.errorbar(
                    means, y_positions,
                    xerr=errors,
                    fmt='o',
                    color=color,
                    capsize=3,
                    capthick=1.5,
                    markersize=6,
                    linewidth=1.5
                )

                ax.set_xlabel(x_label, fontsize=10)
                ax.set_title(dim_name.capitalize(), fontsize=11, fontweight='bold')
                ax.set_xlim(*x_lim)
                ax.axvline(x=ref_line_x, color='gray', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)

                # Add group separator lines
                if groups and group_boundaries:
                    for boundary in group_boundaries:
                        ax.axhline(y=boundary - 0.5, color='gray', linestyle='--',
                                   alpha=0.5, linewidth=1.0)

                if dim_idx == 0:
                    ax.set_yticks(y_positions)
                    y_labels = [f"{pid}  (n={pid_entity_count[pid]})" for pid in pids]
                    label_fontsize = 8 if max(len(p) for p in pids) > 20 else 9
                    ax.set_yticklabels(y_labels, fontsize=label_fontsize)
                    # Color labels by group membership
                    if groups and pid_to_group:
                        for tick_label in ax.get_yticklabels():
                            # Extract pid from label (before the "  (n=" part)
                            pid_text = tick_label.get_text().split("  (n=")[0]
                            if pid_text in pid_to_group:
                                tick_label.set_color(group_colors[pid_to_group[pid_text]])
                                tick_label.set_fontweight('bold')

        else:
            # group_by == "participant": One subplot per participant, dimensions on y-axis
            n_cols = min(3, n_participants)
            n_rows = (n_participants + n_cols - 1) // n_cols
            scaled_height = max(figsize[1], 2.5 * n_rows)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], scaled_height), sharex=True, sharey=True)
            axes = np.array(axes).flatten()

            for pid_idx, pid in enumerate(pids):
                ax = axes[pid_idx]

                y_positions = list(range(n_dims))
                means = []
                ci_lows = []
                ci_highs = []

                for dim_idx in range(n_dims):
                    mean_val, ci_low, ci_high = stats[(pid, dim_idx)]
                    means.append(mean_val)
                    ci_lows.append(ci_low)
                    ci_highs.append(ci_high)

                errors = [
                    [max(0, m - cl) for m, cl in zip(means, ci_lows)],
                    [max(0, ch - m) for m, ch in zip(ci_highs, means)]
                ]

                # Color by dimension
                colors = [self.dimension_colors.get(dims[i].lower(), self.participant_colors[i]) for i in range(n_dims)]

                for i, (y, m, err_l, err_h, c) in enumerate(zip(y_positions, means, errors[0], errors[1], colors)):
                    ax.errorbar(
                        m, y,
                        xerr=[[err_l], [err_h]],
                        fmt='o',
                        color=c,
                        capsize=3,
                        capthick=1.5,
                        markersize=6,
                        linewidth=1.5
                    )

                ax.set_xlim(*x_lim)
                ax.axvline(x=ref_line_x, color='gray', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                title_color = 'black'
                if groups and pid in pid_to_group:
                    title_color = group_colors[pid_to_group[pid]]
                ax.set_title(f"{pid}\n(n={pid_entity_count[pid]})", fontsize=10,
                             color=title_color,
                             fontweight='bold' if title_color != 'black' else 'normal')

                if pid_idx % n_cols == 0:
                    ax.set_yticks(y_positions)
                    y_labels = [d[:10].capitalize() for d in dims]
                    ax.set_yticklabels(y_labels, fontsize=9)

                if pid_idx >= (n_rows - 1) * n_cols:
                    ax.set_xlabel(x_label, fontsize=10)

            # Hide unused subplots
            for idx in range(n_participants, len(axes)):
                axes[idx].axis('off')

        # Overall title with subtitle
        if title:
            plot_title = title
        elif use_raw:
            plot_title = f"Dimension Scores by Participant — Raw ({int(ci_level*100)}% CI)"
        else:
            plot_title = f"Dimension Scores by Participant — Proportions ({int(ci_level*100)}% CI)"
        fig.suptitle(plot_title, fontsize=12, fontweight='bold', y=1.04)
        fig.text(0.5, 1.01, subtitle_text,
                 ha='center', va='top', fontsize=9, color='gray', style='italic',
                 transform=fig.transFigure)
        plt.tight_layout()

        # Add group legend if groups are defined
        if groups and pid_to_group:
            from matplotlib.lines import Line2D
            group_handles = []
            for g_idx, (group_name, members) in enumerate(groups.items()):
                group_handles.append(
                    Line2D([0], [0], marker='s', color='w',
                           markerfacecolor=group_colors[group_name], markersize=10,
                           markeredgecolor='black', markeredgewidth=0.5,
                           label=f"{group_name} (n={len(members)})")
                )
            fig.legend(
                handles=group_handles, loc='lower center',
                ncol=len(groups), fontsize=9, framealpha=0.9,
                title="Groups", title_fontsize=10,
                bbox_to_anchor=(0.5, -0.06),
            )
            fig.subplots_adjust(bottom=0.12)

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved forest plot ({score_mode}) to {output_path}")
            return None

        return fig

    def generate_similarity_map(
        self,
        result: "ComparisonResult",
        output_path: Optional[Union[str, Path]] = None,
        method: str = "mds",
        figsize: Tuple[int, int] = (10, 8),
        show_labels: bool = True,
        title: Optional[str] = None,
        groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Generate 2D similarity map using MDS or UMAP dimensionality reduction.

        Projects participants into 2D space based on their distance matrix,
        so similar participants appear close together.

        Args:
            result: ComparisonResult with distance_matrix.
            output_path: Path to save PNG file (optional).
            method: Dimensionality reduction method - "mds" (default) or "umap".
            figsize: Figure size.
            show_labels: Show participant ID labels.
            title: Plot title.
            groups: Optional dict mapping group names to participant IDs for coloring.

        Returns:
            Matplotlib figure if output_path is None, otherwise None.
        """
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from sklearn.manifold import MDS

        if result is None or result.distance_matrix is None:
            raise ValueError("ComparisonResult with distance_matrix required")

        n = len(result.participant_ids)
        if n < 2:
            raise ValueError("Need at least 2 participants for similarity map")

        # Compute 2D embedding
        actual_method = method  # Track which method was actually used
        if method == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=2,
                    metric="precomputed",
                    n_neighbors=min(15, n - 1),
                    min_dist=0.1,
                    random_state=42
                )
                embedding = reducer.fit_transform(result.distance_matrix)
            except ImportError:
                logger.warning("UMAP not available, falling back to MDS")
                print("  Warning: UMAP not installed, falling back to MDS. Install umap-learn for UMAP support.")
                actual_method = "mds"
            except Exception as e:
                logger.warning(f"UMAP failed ({e}), falling back to MDS")
                print(f"  Warning: UMAP failed ({e}), falling back to MDS.")
                actual_method = "mds"

        if actual_method == "mds":
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                normalized_stress="auto"
            )
            embedding = mds.fit_transform(result.distance_matrix)

        # Build group color mapping if groups provided
        group_colors = {}
        pid_to_group = {}
        if groups:
            group_names = list(groups.keys())
            for g_idx, (group_name, members) in enumerate(groups.items()):
                group_colors[group_name] = self.participant_colors[g_idx % len(self.participant_colors)]
                for pid in members:
                    pid_to_group[pid] = group_name

        # Create figure with optional legend for groups
        if groups:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1], figure=fig)
            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])
            legend_ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=figsize)
            legend_ax = None

        # Plot participants
        for i, pid in enumerate(result.participant_ids):
            if groups and pid in pid_to_group:
                group_name = pid_to_group[pid]
                color = group_colors[group_name]
            else:
                color = self.participant_colors[i % len(self.participant_colors)]

            ax.scatter(
                embedding[i, 0], embedding[i, 1],
                s=150,
                c=color,
                alpha=0.7,
                edgecolors='white',
                linewidth=1.5,
                zorder=2
            )

        # Add labels with collision avoidance
        if show_labels:
            placed_labels = []
            # Compute data-space separation threshold from embedding range
            x_range = embedding[:, 0].max() - embedding[:, 0].min() if n > 1 else 1.0
            y_range = embedding[:, 1].max() - embedding[:, 1].min() if n > 1 else 1.0
            min_sep = max(x_range, y_range) * 0.04

            for i, pid in enumerate(result.participant_ids):
                label = pid[:20] + "..." if len(pid) > 20 else pid
                px, py = embedding[i, 0], embedding[i, 1]

                # Try multiple offset directions to avoid overlap
                offset_pts = [(8, 8), (-8, 8), (8, -8), (-8, -8),
                              (12, 0), (-12, 0), (0, 12), (0, -12)]
                best_offset = offset_pts[0]
                best_min_dist = -1

                for ox, oy in offset_pts:
                    # Approximate label position in data coords
                    lx = px + ox * x_range / 300
                    ly = py + oy * y_range / 300
                    if placed_labels:
                        dists = [np.sqrt((lx - plx)**2 + (ly - ply)**2)
                                 for plx, ply in placed_labels]
                        min_dist = min(dists)
                    else:
                        min_dist = float('inf')
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_offset = (ox, oy)

                placed_labels.append((px + best_offset[0] * x_range / 300,
                                      py + best_offset[1] * y_range / 300))
                ax.annotate(
                    label,
                    (px, py),
                    xytext=best_offset,
                    textcoords='offset points',
                    fontsize=9,
                    alpha=0.8,
                    zorder=3
                )

        # Draw convex hulls around groups
        if groups:
            try:
                from scipy.spatial import ConvexHull

                for group_name, members in groups.items():
                    # Get indices of members in this group
                    group_indices = [
                        result.participant_ids.index(pid)
                        for pid in members
                        if pid in result.participant_ids
                    ]

                    if len(group_indices) >= 3:
                        points = embedding[group_indices]
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])  # Close hull
                        ax.fill(
                            hull_points[:, 0], hull_points[:, 1],
                            alpha=0.15,
                            color=group_colors[group_name],
                            zorder=0
                        )
                        ax.plot(
                            hull_points[:, 0], hull_points[:, 1],
                            '--',
                            alpha=0.4,
                            color=group_colors[group_name],
                            linewidth=1.5,
                            zorder=1
                        )
            except Exception as e:
                logger.debug(f"Could not draw convex hulls: {e}")

        # Style
        ax.set_xlabel(f'{actual_method.upper()} Dimension 1', fontsize=10)
        ax.set_ylabel(f'{actual_method.upper()} Dimension 2', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')

        # Add legend for groups
        if groups and legend_ax:
            legend_handles = []
            legend_labels = []
            for group_name, members in groups.items():
                n_members = len([m for m in members if m in result.participant_ids])
                legend_handles.append(
                    plt.Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=group_colors[group_name],
                              markersize=12, alpha=0.7)
                )
                legend_labels.append(f"{group_name} (n={n_members})")
            legend_ax.legend(
                legend_handles, legend_labels,
                loc='center left',
                fontsize=10,
                title="Groups",
                title_fontsize=11,
                frameon=True,
                framealpha=0.9
            )
        elif not groups:
            # Add distance annotations for closest/furthest pairs
            if n >= 2:
                most_similar = result.most_similar_pair
                most_different = result.most_different_pair

                if most_similar:
                    idx1 = result.participant_ids.index(most_similar[0])
                    idx2 = result.participant_ids.index(most_similar[1])
                    ax.plot(
                        [embedding[idx1, 0], embedding[idx2, 0]],
                        [embedding[idx1, 1], embedding[idx2, 1]],
                        'g-', alpha=0.5, linewidth=2, zorder=1,
                        label=f'Most similar: {most_similar[2]:.3f}'
                    )

                if most_different:
                    idx1 = result.participant_ids.index(most_different[0])
                    idx2 = result.participant_ids.index(most_different[1])
                    ax.plot(
                        [embedding[idx1, 0], embedding[idx2, 0]],
                        [embedding[idx1, 1], embedding[idx2, 1]],
                        'r--', alpha=0.5, linewidth=2, zorder=1,
                        label=f'Most different: {most_different[2]:.3f}'
                    )

                ax.legend(loc='best', fontsize=9)

        # Title
        aggregate_mode = getattr(result, 'aggregate', 'mean')
        mode_label = "centroid" if aggregate_mode == "mean" else "distribution"
        plot_title = title or f"Participant Similarity Map ({result.metric.capitalize()}, {mode_label}, {actual_method.upper()})"
        ax.set_title(plot_title, fontsize=12, fontweight='bold', pad=10)

        plt.tight_layout()

        # Save or return
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                output_path,
                dpi=150,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            plt.close(fig)
            logger.info(f"Saved similarity map to {output_path}")
            return None

        return fig
