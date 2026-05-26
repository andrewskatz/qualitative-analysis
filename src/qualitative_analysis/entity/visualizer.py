"""
Entity visualization module for the qualitative-analysis package.

This module provides functionality for visualizing entity scores using:
- Ternary plots (for 3-dimension frameworks like SETS)
- Radar charts (for multi-dimension comparisons)

Example usage:
    from qualitative_analysis.entity import EntityVisualizer

    visualizer = EntityVisualizer()

    # Generate ternary plot
    visualizer.generate_ternary_plot(
        entity_scores=scores,
        dimension_names=["Social", "Ecological", "Technological"],
        output_path="ternary.png",
        title="Entity Classifications (SETS)"
    )

    # Generate radar chart
    visualizer.generate_radar_chart(
        entity_scores=scores,
        dimension_names=["Dim1", "Dim2", "Dim3", "Dim4"],
        output_path="radar.png"
    )
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CLI usage

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from qualitative_analysis.entity.models import EntityScore, EntityScoreResult

logger = logging.getLogger(__name__)


class EntityVisualizer:
    """
    Visualization service for entity scores.

    Provides methods for generating ternary plots and radar charts
    from entity scoring results.
    """

    def __init__(self):
        """Initialize the visualizer with default dimension colors."""
        self.dimension_colors = {
            "social": "#1f77b4",       # Blue
            "ecological": "#2ca02c",   # Green
            "technological": "#d62728" # Red
        }

        # Extended color palette for additional dimensions
        self.extended_colors = [
            "#1f77b4",  # Blue
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#ff7f0e",  # Orange
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf",  # Cyan
        ]

    def generate_ternary_plot(
        self,
        entity_scores: Union[List[Dict[str, Any]], EntityScoreResult],
        dimension_names: List[str],
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Entity Classifications (Ternary)",
        figsize: Tuple[int, int] = (14, 10),
        marker_size: int = 100,
        show_labels: bool = True,
        use_numbered_labels: bool = True,
        max_labels: int = 200,
        uncertainty_style: str = "opacity",
        scale_min: int = 0,
        scale_max: int = 100,
    ) -> Optional[plt.Figure]:
        """
        Generate a ternary plot for entity scores.

        Ternary plots are ideal for visualizing entities scored on exactly
        3 dimensions (like the SETS framework: Social, Ecological, Technological).

        Args:
            entity_scores: List of entity score dictionaries or EntityScoreResult
            dimension_names: List of exactly 3 dimension names
            output_path: Path to save PNG file (optional)
            title: Plot title
            figsize: Figure size as (width, height) in inches
            marker_size: Size of the markers
            show_labels: Whether to show entity labels
            use_numbered_labels: Use numbered labels with legend (better for many entities)
            max_labels: Maximum entities to label (skip labels if exceeded)
            uncertainty_style: How to visualize uncertainty ("opacity" or "none")

        Returns:
            Matplotlib figure if output_path is None, otherwise None

        Raises:
            ValueError: If dimension_names doesn't have exactly 3 dimensions
        """
        if len(dimension_names) != 3:
            raise ValueError(f"Ternary plots require exactly 3 dimensions, got {len(dimension_names)}")

        # Convert EntityScoreResult to list of dicts
        scores_list = self._normalize_input(entity_scores)

        if not scores_list:
            logger.warning("No entity scores provided")
            return None

        logger.info(f"Generating ternary plot for {len(scores_list)} entities")

        # Prepare data
        prepared_data = self._prepare_ternary_data(scores_list, dimension_names)

        # Create figure with space for legend if using numbered labels
        if use_numbered_labels and show_labels:
            fig = plt.figure(figsize=figsize)

            # Adjust legend width based on number of entities
            num_entities = len(scores_list)
            if num_entities <= 15:
                legend_width_ratio = 1.0
            elif num_entities <= 40:
                legend_width_ratio = 1.5
            elif num_entities <= 80:
                legend_width_ratio = 2.0
            else:
                legend_width_ratio = 2.5

            gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, legend_width_ratio], figure=fig)
            ax = fig.add_subplot(gs[0])
            legend_ax = fig.add_subplot(gs[1])
            legend_ax.axis('off')
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            legend_ax = None

        # Plot entities
        points = []
        for idx, entity_data in enumerate(prepared_data, start=1):
            entity = entity_data['entity']
            dim_values = entity_data['normalized_scores']

            # Convert to cartesian coordinates
            # Map: dim[0] → bottom-right, dim[1] → top, dim[2] → bottom-left
            # This matches the vertex labels in _draw_triangle
            x, y = self._barycentric_to_cartesian(
                dim_values[1],  # top vertex
                dim_values[2],  # bottom-left vertex
                dim_values[0]   # bottom-right vertex
            )

            # RGB color from dimensional profile
            # Map normalized proportions to color channels:
            #   Social → Blue, Ecological → Green, Technological → Red
            # Brighten so dominant channel reaches ~0.85 for visibility
            norm = entity_data['normalized_scores']
            max_norm = max(norm)
            if max_norm > 0:
                scale = 0.85 / max_norm
                color = (
                    min(1.0, norm[2] * scale),  # R ← Technological
                    min(1.0, norm[1] * scale),  # G ← Ecological
                    min(1.0, norm[0] * scale),  # B ← Social
                )
            else:
                color = (0.5, 0.5, 0.5)

            # Calculate opacity based on uncertainty (CV)
            # Linear mapping: CV=0 → alpha=0.95, CV=1 → alpha=0.3
            alpha = 0.8
            if uncertainty_style == "opacity":
                cv = entity_data.get('avg_cv', 0)
                alpha = max(0.3, 0.95 - 0.65 * min(cv, 1.0))

            # Calculate marker size based on mean score (magnitude encoding)
            # Use absolute scale so size is meaningful across plots
            # Scale from marker_size * 0.15 (score=min) to marker_size * 2.5 (score=max)
            mean_score = entity_data['mean_score']
            scale_range = scale_max - scale_min
            abs_fraction = max(0, min((mean_score - scale_min) / scale_range, 1.0))
            point_size = marker_size * (0.15 + 2.35 * abs_fraction)

            # Plot the point
            ax.scatter(
                x, y,
                s=point_size,
                c=[color],
                edgecolors='black',
                alpha=alpha,
                linewidths=1.5
            )

            points.append({
                'entity': entity,
                'x': x,
                'y': y,
                'color': color,
                'alpha': alpha,
                'size': point_size,
                'mean_score': mean_score,
                'index': idx
            })

        # Add labels
        if show_labels and len(points) <= max_labels:
            if use_numbered_labels:
                self._add_numbered_labels(ax, points, legend_ax)
            else:
                self._add_direct_labels(ax, points)
        elif show_labels:
            logger.warning(f"Skipping labels: {len(points)} entities exceeds max_labels={max_labels}")

        # Draw triangle
        self._draw_triangle(ax, dimension_names)

        # Add dimension legend
        self._add_dimension_legend(ax, dimension_names)

        # Add size legend showing magnitude encoding
        self._add_size_legend(ax, marker_size, scale_min=scale_min, scale_max=scale_max)

        # Configure plot
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title, fontsize=14, pad=20)

        # Add subtitle explaining visual encoding
        subtitle_parts = ["Position: relative proportions", "Color: dimensional blend (RGB)", "Size: score magnitude"]
        if uncertainty_style == "opacity":
            subtitle_parts.append("Opacity: confidence")
        ax.text(
            0.5, -0.08, " | ".join(subtitle_parts),
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='gray', style='italic'
        )

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
            logger.info(f"Saved ternary plot to {output_path}")
            return None

        return fig

    def generate_radar_chart(
        self,
        entity_scores: Union[List[Dict[str, Any]], EntityScoreResult],
        dimension_names: List[str],
        output_path: Optional[Union[str, Path]] = None,
        title: str = "Entity Scores (Radar)",
        figsize: Tuple[int, int] = (10, 10),
        max_entities: int = 10,
        show_legend: bool = True,
        scale_min: int = 0,
        scale_max: int = 100,
    ) -> Optional[plt.Figure]:
        """
        Generate a radar chart for entity scores.

        Radar charts can visualize entities across any number of dimensions,
        showing the profile of each entity as a polygon.

        Args:
            entity_scores: List of entity score dictionaries or EntityScoreResult
            dimension_names: List of dimension names
            output_path: Path to save PNG file (optional)
            title: Plot title
            figsize: Figure size as (width, height) in inches
            max_entities: Maximum number of entities to show (most frequent first)
            show_legend: Whether to show entity legend

        Returns:
            Matplotlib figure if output_path is None, otherwise None
        """
        # Convert EntityScoreResult to list of dicts
        scores_list = self._normalize_input(entity_scores)

        if not scores_list:
            logger.warning("No entity scores provided")
            return None

        # Limit entities
        if len(scores_list) > max_entities:
            scores_list = scores_list[:max_entities]
            logger.info(f"Limiting radar chart to top {max_entities} entities")

        logger.info(f"Generating radar chart for {len(scores_list)} entities")

        # Prepare angles
        num_dims = len(dimension_names)
        angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # Plot each entity
        for i, score_data in enumerate(scores_list):
            entity = score_data.get('entity', f'Entity {i+1}')

            # Extract scores
            values = []
            for dim in dimension_names:
                dim_key = dim.lower()
                values.append(
                    self._extract_dimension_value(
                        score_data,
                        dim_key,
                        entity=entity,
                        prefer_cv=False,
                    )[0]
                )

            values += values[:1]  # Close the polygon

            # Get color
            color = self.extended_colors[i % len(self.extended_colors)]

            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, label=entity, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        # Configure axes
        ax.set_theta_offset(np.pi / 2)  # Start from top
        ax.set_theta_direction(-1)  # Clockwise
        ax.set_thetagrids(np.degrees(angles[:-1]), dimension_names)
        ax.set_ylim(scale_min, scale_max)
        r = scale_max - scale_min
        ticks = [scale_min + i * r / 5 for i in range(1, 6)]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f'{t:.0f}' for t in ticks])
        ax.grid(True)

        # Title and legend
        plt.title(title, size=14, y=1.08)
        if show_legend:
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

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
            logger.info(f"Saved radar chart to {output_path}")
            return None

        return fig

    def _normalize_input(
        self,
        entity_scores: Union[List[Dict[str, Any]], EntityScoreResult]
    ) -> List[Dict[str, Any]]:
        """Convert various input formats to a list of score dictionaries."""
        if isinstance(entity_scores, EntityScoreResult):
            return [score.to_flat_dict() for score in entity_scores.scores]
        elif isinstance(entity_scores, list):
            result = []
            for item in entity_scores:
                if isinstance(item, EntityScore):
                    result.append(item.to_flat_dict())
                elif isinstance(item, dict):
                    result.append(item)
            return result
        return []

    def _prepare_ternary_data(
        self,
        scores_list: List[Dict[str, Any]],
        dimension_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Prepare entity scores for ternary plotting."""
        prepared = []

        for score_data in scores_list:
            entity = score_data.get('entity', 'Unknown')

            # Extract raw scores for each dimension
            raw_scores = []
            cv_values = []

            for dim in dimension_names:
                dim_key = dim.lower()
                score, cv = self._extract_dimension_value(
                    score_data,
                    dim_key,
                    entity=entity,
                    prefer_cv=True,
                )
                raw_scores.append(score)
                cv_values.append(cv if cv else 0)

            # Normalize scores to sum to 1 for ternary position
            total = sum(raw_scores)
            if total > 0:
                normalized = [s / total for s in raw_scores]
            else:
                normalized = [1.0 / len(dimension_names)] * len(dimension_names)

            # Mean raw score for magnitude encoding (point size)
            mean_score = np.mean(raw_scores)

            # Average CV for uncertainty visualization (opacity)
            avg_cv = np.mean(cv_values) if cv_values else 0

            prepared.append({
                'entity': entity,
                'raw_scores': raw_scores,
                'normalized_scores': normalized,
                'mean_score': mean_score,
                'avg_cv': avg_cv
            })

        return prepared

    def _extract_dimension_value(
        self,
        score_data: Dict[str, Any],
        dim_key: str,
        entity: str,
        prefer_cv: bool,
    ) -> Tuple[float, float]:
        """Extract a dimension score from supported input shapes or raise."""
        score: Optional[float] = None
        cv = 0.0

        if f'{dim_key}_mean' in score_data:
            score = score_data[f'{dim_key}_mean']
            cv = score_data.get(f'{dim_key}_cv', 0) or 0
        elif 'dimensions' in score_data and dim_key in score_data['dimensions']:
            dim_data = score_data['dimensions'][dim_key]
            if isinstance(dim_data, dict):
                if 'mean' in dim_data:
                    score = dim_data['mean']
                elif 'score' in dim_data:
                    score = dim_data['score']
                if prefer_cv:
                    cv = dim_data.get('cv', dim_data.get('coefficient_of_variation', 0)) or 0
            else:
                score = dim_data
        elif dim_key in score_data:
            score = score_data[dim_key]

        if score is None:
            raise ValueError(
                f"Missing score for dimension '{dim_key}' while visualizing entity '{entity}'"
            )

        try:
            return float(score), float(cv)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid score for dimension '{dim_key}' while visualizing entity '{entity}'"
            ) from exc

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
        ax: plt.Axes,
        dimension_names: List[str]
    ) -> None:
        """Draw the ternary plot triangle with grid lines and labels."""
        # Triangle outline
        vertices = np.array([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3)/2],
            [0, 0]
        ])
        ax.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)

        # Dimension labels at corners
        # RGB channel colors: dim[0]=Social→Blue, dim[1]=Ecological→Green, dim[2]=Technological→Red
        dim_label_colors = [
            (0.0, 0.0, 0.85),   # dim[0] Social → Blue
            (0.0, 0.60, 0.0),   # dim[1] Ecological → Green (darker for readability)
            (0.85, 0.0, 0.0),   # dim[2] Technological → Red
        ]
        label_offset = 0.08

        # Bottom-left (dimension 2 - typically Technological)
        ax.text(0, -label_offset, dimension_names[2].capitalize(),
                ha='center', va='top', fontsize=12, fontweight='bold',
                color=dim_label_colors[2])
        ax.text(0, -label_offset - 0.03, '(100%)',
                ha='center', va='top', fontsize=9, color='gray')

        # Bottom-right (dimension 0 - typically Social)
        ax.text(1, -label_offset, dimension_names[0].capitalize(),
                ha='center', va='top', fontsize=12, fontweight='bold',
                color=dim_label_colors[0])
        ax.text(1, -label_offset - 0.03, '(100%)',
                ha='center', va='top', fontsize=9, color='gray')

        # Top (dimension 1 - typically Ecological)
        ax.text(0.5, np.sqrt(3)/2 + label_offset, dimension_names[1].capitalize(),
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=dim_label_colors[1])
        ax.text(0.5, np.sqrt(3)/2 + label_offset + 0.03, '(100%)',
                ha='center', va='bottom', fontsize=9, color='gray')

        # Grid lines
        for i in range(1, 10):
            alpha = 0.3 if i % 2 == 0 else 0.15

            # Lines parallel to bottom edge
            x1, y1 = self._barycentric_to_cartesian(1-i/10, i/10, 0)
            x2, y2 = self._barycentric_to_cartesian(0, i/10, 1-i/10)
            ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

            # Lines parallel to left edge
            x1, y1 = self._barycentric_to_cartesian(i/10, 0, 1-i/10)
            x2, y2 = self._barycentric_to_cartesian(i/10, 1-i/10, 0)
            ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

            # Lines parallel to right edge
            x1, y1 = self._barycentric_to_cartesian(0, 1-i/10, i/10)
            x2, y2 = self._barycentric_to_cartesian(1-i/10, 0, i/10)
            ax.plot([x1, x2], [y1, y2], 'k:', alpha=alpha, linewidth=0.5)

    def _add_direct_labels(
        self,
        ax: plt.Axes,
        points: List[Dict[str, Any]]
    ) -> None:
        """Add entity labels directly next to points."""
        label_offset = 0.03

        for point in points:
            entity = point['entity']
            x, y = point['x'], point['y']

            # Position based on location in triangle
            if y > 0.6:  # Top
                ha, va = 'center', 'bottom'
                dx, dy = 0, label_offset
            elif x < 0.3:  # Left
                ha, va = 'right', 'center'
                dx, dy = -label_offset, 0
            elif x > 0.7:  # Right
                ha, va = 'left', 'center'
                dx, dy = label_offset, 0
            else:  # Center
                ha, va = 'left', 'bottom'
                dx, dy = label_offset, label_offset

            ax.annotate(
                entity,
                (x, y),
                xytext=(x + dx, y + dy),
                fontsize=7,
                ha=ha,
                va=va,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.5)
            )

    def _add_numbered_labels(
        self,
        ax: plt.Axes,
        points: List[Dict[str, Any]],
        legend_ax: Optional[plt.Axes]
    ) -> None:
        """Add numbered labels to points with a legend."""
        base_offset = 0.06
        label_radius = 0.03
        min_separation = label_radius * 3.0

        placed_labels = []
        point_positions = [(p['x'], p['y']) for p in points]

        for point in points:
            x, y = point['x'], point['y']
            idx = point['index']

            best_pos = None
            for offset_mult in [1.0, 1.5, 2.0, 2.5]:
                offset = base_offset * offset_mult

                candidates = [
                    (0, offset, 'center', 'bottom'),
                    (offset, 0, 'left', 'center'),
                    (0, -offset, 'center', 'top'),
                    (-offset, 0, 'right', 'center'),
                    (offset * 0.7, offset * 0.7, 'left', 'bottom'),
                    (-offset * 0.7, offset * 0.7, 'right', 'bottom'),
                    (offset * 0.7, -offset * 0.7, 'left', 'top'),
                    (-offset * 0.7, -offset * 0.7, 'right', 'top'),
                ]

                for dx, dy, ha, va in candidates:
                    label_x, label_y = x + dx, y + dy

                    # Check collisions
                    collision = False
                    for placed_x, placed_y in placed_labels:
                        if np.sqrt((label_x - placed_x)**2 + (label_y - placed_y)**2) < min_separation:
                            collision = True
                            break

                    if not collision:
                        for px, py in point_positions:
                            if px == x and py == y:
                                continue
                            if np.sqrt((label_x - px)**2 + (label_y - py)**2) < min_separation * 0.8:
                                collision = True
                                break

                    if not collision:
                        best_pos = (label_x, label_y, ha, va)
                        break

                if best_pos:
                    break

            if not best_pos:
                best_pos = (x, y + base_offset * 2.5, 'center', 'bottom')

            label_x, label_y, ha, va = best_pos
            placed_labels.append((label_x, label_y))

            # Draw connector line
            ax.plot([x, label_x], [y, label_y], color='gray', linestyle='-', linewidth=0.5, alpha=0.7)

            # Add number
            ax.text(
                label_x, label_y, str(idx),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="gray", alpha=0.8),
                zorder=10
            )

        # Create legend
        if legend_ax is not None:
            legend_elements = []
            legend_labels = []

            for point in points:
                legend_elements.append(plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=point['color'],
                    markersize=8, alpha=point['alpha']
                ))
                legend_labels.append(f"{point['index']}. {point['entity']}")

            # Determine columns
            n = len(legend_labels)
            if n <= 15:
                ncol, fontsize = 1, 9
            elif n <= 40:
                ncol, fontsize = 2, 8
            elif n <= 80:
                ncol, fontsize = 3, 7
            else:
                ncol, fontsize = 4, 6

            legend_ax.legend(
                legend_elements, legend_labels,
                loc='center left', fontsize=fontsize,
                frameon=True, framealpha=0.8,
                title="Entities", title_fontsize=10,
                ncol=ncol
            )

    def _add_dimension_legend(
        self,
        ax: plt.Axes,
        dimension_names: List[str]
    ) -> None:
        """Add a legend showing dimension-to-RGB color mapping."""
        # RGB corner colors: each dimension maps to one channel
        # Order matches dimension_names: [0]=Social→Blue, [1]=Ecological→Green, [2]=Technological→Red
        rgb_corners = [
            (0.0, 0.0, 0.85),   # Social → Blue
            (0.0, 0.85, 0.0),   # Ecological → Green
            (0.85, 0.0, 0.0),   # Technological → Red
        ]

        legend_elements = []
        for i, dim in enumerate(dimension_names):
            legend_elements.append(
                plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor=rgb_corners[i], markersize=10,
                    markeredgecolor='black', markeredgewidth=1,
                    label=f"{dim.capitalize()}"
                )
            )

        # Add as first legend, then use add_artist so size legend can coexist
        dim_legend = ax.legend(
            handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10,
            title="Color channels", title_fontsize=9
        )
        ax.add_artist(dim_legend)

    def _add_size_legend(
        self,
        ax: plt.Axes,
        base_marker_size: int,
        scale_min: int = 0,
        scale_max: int = 100,
    ) -> None:
        """Add a legend showing point size to mean score mapping."""
        # Show 3 reference sizes at 25%, 50%, 75% of scale range
        r = scale_max - scale_min
        levels = [
            (scale_min + 0.25 * r, f"{scale_min + 0.25 * r:.0f}"),
            (scale_min + 0.50 * r, f"{scale_min + 0.50 * r:.0f}"),
            (scale_min + 0.75 * r, f"{scale_min + 0.75 * r:.0f}"),
        ]

        legend_elements = []
        for score_val, label in levels:
            abs_fraction = (score_val - scale_min) / r
            size = base_marker_size * (0.15 + 2.35 * abs_fraction)
            legend_elements.append(
                plt.scatter(
                    [], [], s=size, c='gray', edgecolors='black',
                    linewidths=0.5, alpha=0.6,
                    label=label
                )
            )

        ax.legend(
            handles=legend_elements, loc='lower right',
            title="Mean Score", title_fontsize=9,
            fontsize=8, framealpha=0.9,
            labelspacing=1.5, borderpad=1.0
        )
