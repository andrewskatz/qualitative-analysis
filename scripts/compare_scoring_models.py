#!/usr/bin/env python3
"""
Cross-model comparison of factorial scoring experiments.

Aggregates per-model analysis tables and produces publication-ready
comparison figures and a summary CSV.

Usage:
    python scripts/compare_scoring_models.py
    python scripts/compare_scoring_models.py --output-dir output/cross-model-comparison
    python scripts/compare_scoring_models.py --exp-dirs output/scoring-experiment-qwen3-30b output/scoring-experiment-qwen3.5-122b
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Defaults ────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"
DIMENSIONS = ["social", "ecological", "technological"]
DIM_COLORS = {
    "social": "#1f77b4",
    "ecological": "#2ca02c",
    "technological": "#d62728",
}
SCALES = ["0-100", "1-10", "1-5"]
PROMPTS = ["v2", "v3", "v4"]
PROMPT_LABELS = {"v2": "Score→Justify", "v3": "Justify→Score", "v4": "No CoT"}

DPI = 150
RNG_SEED = 42


def discover_experiments(output_root: Path) -> list[Path]:
    """Find all scoring-experiment-* directories with analysis results."""
    dirs = sorted(
        list(output_root.glob("scoring-experiment-*"))
        + [d for d in [output_root / "scoring-experiment"] if d.is_dir()]
    )
    valid = []
    for d in dirs:
        tables_dir = d / "analysis" / "tables"
        if tables_dir.exists() and (tables_dir / "table05_reliability.csv").exists():
            valid.append(d)
    return valid


def extract_model_name(exp_dir: Path) -> str:
    """Extract model name from experiment manifest."""
    manifest = exp_dir / "experiment_manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            data = json.load(f)
        return data.get("parameters", {}).get("model", exp_dir.name)
    return exp_dir.name.replace("scoring-experiment-", "")


def short_model_name(name: str) -> str:
    """Create a short display name from full model string."""
    # Strip quantization suffixes for display
    short = name.split(":")[0] if ":" in name else name
    # Keep the parameter count if present
    parts = name.split(":")
    if len(parts) > 1:
        tag = parts[1]
        # Extract size info (e.g., "122b", "4b", "30b")
        for segment in tag.split("-"):
            if segment.endswith("b") and segment[:-1].replace(".", "").isdigit():
                short = f"{short}-{segment}"
                break
    return short


def load_model_tables(exp_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all analysis tables for one model."""
    tables_dir = exp_dir / "analysis" / "tables"
    tables = {}
    for csv_path in sorted(tables_dir.glob("table*.csv")):
        key = csv_path.stem
        tables[key] = pd.read_csv(csv_path)
    return tables


def save_figure(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════
# COMPARISON FIGURES
# ══════════════════════════════════════════════════════════════════════

def fig_reliability_comparison(all_data, fig_dir):
    """Bar chart: Krippendorff's alpha by model, dimension, best condition (0-100/v2)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Krippendorff's Alpha by Model (0-100 / v2 condition)",
                 fontsize=14, fontweight="bold")

    models = list(all_data.keys())
    x = np.arange(len(models))
    bar_width = 0.6

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        alphas = []
        ci_lows = []
        ci_highs = []
        for model in models:
            t5 = all_data[model]["table05_reliability"]
            row = t5[(t5["scale"] == "0-100") & (t5["prompt"] == "v2") & (t5["dimension"] == dim)]
            if len(row) > 0:
                a = row.iloc[0]["alpha"]
                cl = row.iloc[0]["alpha_ci_low"]
                ch = row.iloc[0]["alpha_ci_high"]
            else:
                a, cl, ch = 0, 0, 0
            alphas.append(a)
            ci_lows.append(a - cl)
            ci_highs.append(ch - a)

        bars = ax.barh(x, alphas, bar_width, color=DIM_COLORS[dim], alpha=0.7,
                       xerr=[ci_lows, ci_highs], capsize=3, ecolor="gray")
        ax.set_xlim(0.5, 1.0)
        ax.set_yticks(x)
        ax.set_yticklabels(models if ax_i == 0 else [], fontsize=9)
        ax.set_xlabel("Krippendorff's Alpha")
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.axvline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="α=0.80")
        ax.grid(axis="x", alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_xm01_reliability_by_model.png")


def fig_alpha_heatmap(all_data, fig_dir):
    """Heatmap: alpha across all models x conditions for each dimension."""
    models = list(all_data.keys())
    conditions = [f"{s}/{p}" for s in SCALES for p in PROMPTS]

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, len(models) * 0.7 + 2)))
    fig.suptitle("Krippendorff's Alpha: All Models × Conditions", fontsize=14, fontweight="bold")

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        matrix = np.zeros((len(models), len(conditions)))
        for mi, model in enumerate(models):
            t5 = all_data[model]["table05_reliability"]
            for ci, cond in enumerate(conditions):
                scale, prompt = cond.split("/")
                row = t5[(t5["scale"] == scale) & (t5["prompt"] == prompt) & (t5["dimension"] == dim)]
                if len(row) > 0:
                    matrix[mi, ci] = row.iloc[0]["alpha"]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.6, vmax=1.0, aspect="auto")
        for mi in range(len(models)):
            for ci in range(len(conditions)):
                val = matrix[mi, ci]
                color = "white" if val < 0.75 else "black"
                ax.text(ci, mi, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if ax_i == 0 else [], fontsize=8)
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        plt.colorbar(im, ax=ax, shrink=0.8, label="α")

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_xm02_alpha_heatmap.png")


def fig_cv_comparison(all_data, fig_dir):
    """Box plot: CV distributions by model for 0-100/v2."""
    models = list(all_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Score CV Distribution by Model (0-100 / v2)",
                 fontsize=14, fontweight="bold")

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        t5 = []
        for model in models:
            row = all_data[model]["table05_reliability"]
            row = row[(row["scale"] == "0-100") & (row["prompt"] == "v2") & (row["dimension"] == dim)]
            if len(row) > 0:
                t5.append(row.iloc[0]["avg_cv"])
            else:
                t5.append(np.nan)

        bars = ax.barh(range(len(models)), t5, color=DIM_COLORS[dim], alpha=0.7)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models if ax_i == 0 else [], fontsize=9)
        ax.set_xlabel("Mean CV")
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_xm03_cv_by_model.png")


def fig_rank_correlation_comparison(all_data, fig_dir):
    """Average within-model rank correlation (Spearman rho) across all condition pairs."""
    models = list(all_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Mean Spearman ρ Across Condition Pairs (Within-Model Rank Stability)",
                 fontsize=14, fontweight="bold")

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        mean_rhos = []
        rho_ranges = []
        for model in models:
            t6 = all_data[model]["table06_rank_correlations"]
            dim_rows = t6[t6["dimension"] == dim]
            rhos = dim_rows["spearman_rho"].values
            mean_rhos.append(np.mean(rhos))
            rho_ranges.append((np.min(rhos), np.max(rhos)))

        y = np.arange(len(models))
        ax.barh(y, mean_rhos, 0.6, color=DIM_COLORS[dim], alpha=0.7)
        for i, (rmin, rmax) in enumerate(rho_ranges):
            ax.plot([rmin, rmax], [i, i], color="black", linewidth=1.5, marker="|", markersize=8)

        ax.set_xlim(0.4, 1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(models if ax_i == 0 else [], fontsize=9)
        ax.set_xlabel("Spearman ρ (mean ± range)")
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.axvline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_xm04_rank_stability.png")


def fig_pareto_frontier(all_data, fig_dir):
    """Pareto frontier: quality index vs processing time across all models."""
    models = list(all_data.keys())

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Quality–Speed Pareto Frontier (0-100/v2 condition)",
                 fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        t8 = all_data[model]["table08_efficiency"]
        row = t8[(t8["scale"] == "0-100") & (t8["prompt"] == "v2")]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        time_s = r["mean_time_ms"] / 1000
        qi = r["quality_index"]
        ax.scatter(time_s, qi, s=120, color=colors[i], zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(model, (time_s, qi), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, color=colors[i])

    # Also plot v4 for each model (speed option)
    for i, model in enumerate(models):
        t8 = all_data[model]["table08_efficiency"]
        row = t8[(t8["scale"] == "0-100") & (t8["prompt"] == "v4")]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        time_s = r["mean_time_ms"] / 1000
        qi = r["quality_index"]
        ax.scatter(time_s, qi, s=60, color=colors[i], zorder=4, marker="^",
                   edgecolors="black", linewidth=0.5, alpha=0.6)
        ax.annotate(f"{model} (v4)", (time_s, qi), textcoords="offset points",
                    xytext=(8, -8), fontsize=6.5, color=colors[i], alpha=0.7)

    # Compute and draw Pareto frontier for v2 points
    v2_points = []
    for model in models:
        t8 = all_data[model]["table08_efficiency"]
        row = t8[(t8["scale"] == "0-100") & (t8["prompt"] == "v2")]
        if len(row) > 0:
            r = row.iloc[0]
            v2_points.append((r["mean_time_ms"] / 1000, r["quality_index"]))

    if v2_points:
        # Sort by time ascending
        v2_points.sort()
        pareto = [v2_points[0]]
        for t, q in v2_points[1:]:
            if q > pareto[-1][1]:
                pareto.append((t, q))
        if len(pareto) > 1:
            px, py = zip(*pareto)
            ax.plot(px, py, "k--", linewidth=1, alpha=0.4, label="Pareto frontier (v2)")

    ax.set_xlabel("Processing Time per Entity (seconds)", fontsize=11)
    ax.set_ylabel("Quality Index (geometric mean of α and ρ)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.75, 1.0)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_xm05_pareto_frontier.png")


def fig_entropy_comparison(all_data, fig_dir):
    """Bar chart: Shannon entropy by model for 0-100/v2 condition."""
    models = list(all_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Shannon Entropy by Model (0-100 / v2 — higher = more discriminating)",
                 fontsize=14, fontweight="bold")

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        entropies = []
        for model in models:
            t7 = all_data[model]["table07_resolution"]
            row = t7[(t7["scale"] == "0-100") & (t7["prompt"] == "v2") & (t7["dimension"] == dim)]
            if len(row) > 0:
                entropies.append(row.iloc[0]["entropy_bits"])
            else:
                entropies.append(0)

        y = np.arange(len(models))
        ax.barh(y, entropies, 0.6, color=DIM_COLORS[dim], alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(models if ax_i == 0 else [], fontsize=9)
        ax.set_xlabel("Shannon Entropy (bits)")
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_xm06_entropy_comparison.png")


def fig_ceiling_compression(all_data, fig_dir):
    """Ceiling % comparison across models for 1-5 scale."""
    models = list(all_data.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Ceiling Compression: % Entities at Ceiling (1-5 / v2)",
                 fontsize=14, fontweight="bold")

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        ceil_pcts = []
        for model in models:
            t7 = all_data[model]["table07_resolution"]
            row = t7[(t7["scale"] == "1-5") & (t7["prompt"] == "v2") & (t7["dimension"] == dim)]
            if len(row) > 0:
                ceil_pcts.append(row.iloc[0]["ceiling_pct"])
            else:
                ceil_pcts.append(0)

        y = np.arange(len(models))
        colors = ["#d62728" if c > 50 else "#ff7f0e" if c > 30 else "#2ca02c" for c in ceil_pcts]
        ax.barh(y, ceil_pcts, 0.6, color=colors, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(models if ax_i == 0 else [], fontsize=9)
        ax.set_xlabel("% at Ceiling")
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_xm07_ceiling_compression.png")


def build_summary_table(all_data) -> pd.DataFrame:
    """Build a single summary CSV with one row per model."""
    rows = []
    for model, tables in all_data.items():
        t5 = tables["table05_reliability"]
        t6 = tables["table06_rank_correlations"]
        t7 = tables["table07_resolution"]
        t8 = tables["table08_efficiency"]

        # Best condition (0-100/v2)
        best = t5[(t5["scale"] == "0-100") & (t5["prompt"] == "v2")]
        alpha_by_dim = {}
        cv_by_dim = {}
        for dim in DIMENSIONS:
            r = best[best["dimension"] == dim]
            if len(r) > 0:
                alpha_by_dim[dim] = r.iloc[0]["alpha"]
                cv_by_dim[dim] = r.iloc[0]["avg_cv"]

        # Overall alpha range
        all_alpha = t5["alpha"].values
        alpha_min = float(np.min(all_alpha))
        alpha_max = float(np.max(all_alpha))

        # Mean rank correlation
        all_rho = t6["spearman_rho"].values
        mean_rho = float(np.mean(all_rho))
        min_rho = float(np.min(all_rho))
        max_rho = float(np.max(all_rho))

        # Quality index (from efficiency table, 0-100/v2)
        eff_row = t8[(t8["scale"] == "0-100") & (t8["prompt"] == "v2")]
        qi = eff_row.iloc[0]["quality_index"] if len(eff_row) > 0 else np.nan
        time_ms = eff_row.iloc[0]["mean_time_ms"] if len(eff_row) > 0 else np.nan

        # Entropy for 0-100/v2
        entropy_by_dim = {}
        for dim in DIMENSIONS:
            r = t7[(t7["scale"] == "0-100") & (t7["prompt"] == "v2") & (t7["dimension"] == dim)]
            if len(r) > 0:
                entropy_by_dim[dim] = r.iloc[0]["entropy_bits"]

        # Ceiling compression on 1-5/v2
        ceil_by_dim = {}
        for dim in DIMENSIONS:
            r = t7[(t7["scale"] == "1-5") & (t7["prompt"] == "v2") & (t7["dimension"] == dim)]
            if len(r) > 0:
                ceil_by_dim[dim] = r.iloc[0]["ceiling_pct"]

        rows.append({
            "model": model,
            "alpha_social": alpha_by_dim.get("social"),
            "alpha_ecological": alpha_by_dim.get("ecological"),
            "alpha_technological": alpha_by_dim.get("technological"),
            "alpha_min_all": alpha_min,
            "alpha_max_all": alpha_max,
            "cv_social": cv_by_dim.get("social"),
            "cv_ecological": cv_by_dim.get("ecological"),
            "cv_technological": cv_by_dim.get("technological"),
            "mean_rho": round(mean_rho, 4),
            "min_rho": round(min_rho, 4),
            "max_rho": round(max_rho, 4),
            "quality_index": round(qi, 4) if not np.isnan(qi) else None,
            "time_ms_per_entity": round(time_ms, 0) if not np.isnan(time_ms) else None,
            "entropy_social": entropy_by_dim.get("social"),
            "entropy_ecological": entropy_by_dim.get("ecological"),
            "entropy_technological": entropy_by_dim.get("technological"),
            "ceiling_pct_social_1_5": ceil_by_dim.get("social"),
            "ceiling_pct_eco_1_5": ceil_by_dim.get("ecological"),
            "ceiling_pct_tech_1_5": ceil_by_dim.get("technological"),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model comparison of scoring experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dirs", nargs="*", default=None,
        help="Explicit experiment directories to compare. If omitted, auto-discovers all.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: output/cross-model-comparison).",
    )
    args = parser.parse_args()

    # Discover experiments
    if args.exp_dirs:
        exp_dirs = [Path(d).resolve() for d in args.exp_dirs]
    else:
        exp_dirs = discover_experiments(OUTPUT_ROOT)

    if not exp_dirs:
        raise SystemExit("No scoring experiments found.")

    # Output setup
    out_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_ROOT / "cross-model-comparison"
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"  Models: {len(exp_dirs)}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}\n")

    # Load all model data
    all_data = {}
    for exp_dir in exp_dirs:
        model_name = extract_model_name(exp_dir)
        short = short_model_name(model_name)
        print(f"  Loading {short} from {exp_dir.name}")
        all_data[short] = load_model_tables(exp_dir)

    print(f"\n  Loaded {len(all_data)} models\n")

    # Generate comparison figures
    print("=== Generating Cross-Model Figures ===")
    fig_reliability_comparison(all_data, fig_dir)
    fig_alpha_heatmap(all_data, fig_dir)
    fig_cv_comparison(all_data, fig_dir)
    fig_rank_correlation_comparison(all_data, fig_dir)
    fig_pareto_frontier(all_data, fig_dir)
    fig_entropy_comparison(all_data, fig_dir)
    fig_ceiling_compression(all_data, fig_dir)

    # Generate summary table
    print("\n=== Generating Summary Table ===")
    summary = build_summary_table(all_data)
    summary = summary.sort_values("quality_index", ascending=False)
    summary.to_csv(tbl_dir / "cross_model_summary.csv", index=False)
    print("  Saved cross_model_summary.csv")

    # Print summary to terminal
    print(f"\n{'='*60}")
    print("  MODEL RANKING (by Quality Index, 0-100/v2)")
    print(f"{'='*60}")
    for _, row in summary.iterrows():
        qi = row["quality_index"]
        rho = row["mean_rho"]
        time_s = row["time_ms_per_entity"] / 1000 if row["time_ms_per_entity"] else 0
        print(f"  {row['model']:30s}  Q={qi:.3f}  ρ={rho:.3f}  {time_s:6.1f}s/entity")

    print(f"\n{'='*60}")
    print(f"  COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"  Figures: {fig_dir}")
    print(f"  Tables:  {tbl_dir}")


if __name__ == "__main__":
    main()
