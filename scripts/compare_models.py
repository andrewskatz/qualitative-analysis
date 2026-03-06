#!/usr/bin/env python3
"""
Cross-model comparison of 3x3 factorial scoring experiments.

Compares 4 LLM models that each completed identical factorial designs
(3 scales x 3 prompts x 3 SETS dimensions, 450 entities, 3 runs).

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --output-dir custom/path
"""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Experiment Registry ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_DIR = BASE_DIR / "cross-model-comparison"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"

EXPERIMENTS = {
    "GPT-OSS 120B": {
        "dir": "scoring-experiment",
        "model_id": "gpt-oss:120b",
        "provider": "ollama",
        "color": "#1b9e77",   # teal
        "marker": "o",
    },
    "Qwen3 30B": {
        "dir": "scoring-experiment-qwen3-30b",
        "model_id": "qwen3:30b-a3b-instruct-2507-q4_K_M",
        "provider": "ollama",
        "color": "#d95f02",   # orange
        "marker": "s",
    },
    "Qwen3.5 35B": {
        "dir": "scoring-experiment-qwen35-35b",
        "model_id": "mlx-community/Qwen3.5-35B-A3B-8bit",
        "provider": "mlx",
        "color": "#e7298a",   # pink
        "marker": "D",
    },
    "Qwen3 80B": {
        "dir": "scoring-experiment-qwen3-80b",
        "model_id": "qwen3-next:80b-a3b-instruct-q4_K_M",
        "provider": "ollama",
        "color": "#7570b3",   # purple
        "marker": "^",
    },
}

# Ordered by approximate active parameters (ascending)
MODEL_ORDER = ["GPT-OSS 120B", "Qwen3 30B", "Qwen3.5 35B", "Qwen3 80B"]

DIMENSIONS = ["social", "ecological", "technological"]
DIM_COLORS = {
    "social": "#1f77b4",
    "ecological": "#2ca02c",
    "technological": "#d62728",
}
SCALES = [("0-100", 0, 100), ("1-10", 1, 10), ("1-5", 1, 5)]
SCALE_LABELS = [s[0] for s in SCALES]
PROMPTS = ["v2", "v3", "v4"]
PROMPT_LABELS = {"v2": "Score\u2192Justify", "v3": "Justify\u2192Score", "v4": "No CoT"}
REFERENCE_CONDITION = ("0-100", "v2")

RNG_SEED = 42
N_BOOT = 10000
DPI = 150

# 9 condition labels in canonical order
CONDITION_LABELS = [f"{sl}/{p}" for sl in SCALE_LABELS for p in PROMPTS]
CONDITION_KEYS = [(sl, p) for sl in SCALE_LABELS for p in PROMPTS]


# ── Utility Functions ────────────────────────────────────────────────

def save_figure(fig, name):
    """Save figure to FIG_DIR at publication DPI."""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {path.name}")


def condition_key(scale_label, prompt):
    return f"{scale_label}_{prompt}"


def condition_label(scale_label, prompt):
    return f"{scale_label}/{prompt}"


# ── Data Loading ─────────────────────────────────────────────────────

def load_analysis_tables(exp_dir):
    """Load all 8 analysis tables from one experiment directory."""
    tbl_dir = exp_dir / "analysis" / "tables"
    table_files = {
        "table01": "table01_descriptives.csv",
        "table02": "table02_scale_effects.csv",
        "table03": "table03_prompt_effects.csv",
        "table04": "table04_interaction.csv",
        "table05": "table05_reliability.csv",
        "table06": "table06_rank_correlations.csv",
        "table07": "table07_resolution.csv",
        "table08": "table08_efficiency.csv",
    }
    tables = {}
    for key, fname in table_files.items():
        path = tbl_dir / fname
        if path.exists():
            tables[key] = pd.read_csv(path)
        else:
            print(f"  WARNING: Missing {path}")
    return tables


def load_manifest(exp_dir):
    """Load experiment manifest JSON."""
    path = exp_dir / "experiment_manifest.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_raw_scores(exp_dir, scale_label, prompt):
    """Load raw entity scores CSV for one condition.

    Adds composite entity_key and normalized _frac columns.
    """
    s_map = {"0-100": (0, 100), "1-10": (1, 10), "1-5": (1, 5)}
    s_min, s_max = s_map[scale_label]
    dir_name = f"scale-{s_min}-{s_max}_prompt-{prompt}"
    cond_dir = exp_dir / dir_name
    csv_files = list(cond_dir.glob("*_scores.csv"))
    if not csv_files:
        return None
    df = pd.read_csv(csv_files[0])
    # Composite key for cross-model alignment
    df["entity_key"] = (
        df["entity"].astype(str) + "|"
        + df["text_id"].astype(str) + "|"
        + df["window_index"].astype(str)
    )
    # Normalize to [0, 1]
    s_range = s_max - s_min
    for dim in DIMENSIONS:
        mean_col = f"{dim}_mean"
        if mean_col in df.columns:
            df[f"{dim}_mean_frac"] = (df[mean_col] - s_min) / s_range
    return df


def load_all_experiments():
    """Load all experiments' analysis tables and manifests.

    Returns (all_tables, all_manifests) dicts keyed by model label.
    Only includes experiments whose directories exist.
    """
    all_tables = {}
    all_manifests = {}
    for label in MODEL_ORDER:
        info = EXPERIMENTS[label]
        exp_dir = BASE_DIR / info["dir"]
        if not exp_dir.exists():
            print(f"  WARNING: Experiment not found: {exp_dir}")
            continue
        print(f"  Loading {label} from {info['dir']}/")
        all_tables[label] = load_analysis_tables(exp_dir)
        all_manifests[label] = load_manifest(exp_dir)
    return all_tables, all_manifests


def load_cross_model_scores(scale_label, prompt):
    """Load raw scores from all models for one condition, aligned by entity_key.

    Returns dict[model_label -> DataFrame] with only common entities.
    """
    model_dfs = {}
    for label in MODEL_ORDER:
        info = EXPERIMENTS[label]
        exp_dir = BASE_DIR / info["dir"]
        if not exp_dir.exists():
            continue
        df = load_raw_scores(exp_dir, scale_label, prompt)
        if df is not None:
            model_dfs[label] = df

    if len(model_dfs) < 2:
        return model_dfs

    # Inner join on entity_key
    common_keys = set.intersection(*(set(df["entity_key"]) for df in model_dfs.values()))
    for label in model_dfs:
        model_dfs[label] = (
            model_dfs[label][model_dfs[label]["entity_key"].isin(common_keys)]
            .sort_values("entity_key")
            .reset_index(drop=True)
        )
    return model_dfs


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: SCORE DISTRIBUTION COMPARISON
# ══════════════════════════════════════════════════════════════════════

def section1_score_distributions(all_tables):
    """Compare score distributions across models."""
    print("\n=== Section 1: Score Distribution Comparison ===")
    models = [m for m in MODEL_ORDER if m in all_tables]
    n_models = len(models)

    # ── Fig01: Mean score comparison (grouped bar chart) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    bar_width = 0.8 / n_models
    x = np.arange(len(CONDITION_KEYS))

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        for m_idx, model in enumerate(models):
            t01 = all_tables[model].get("table01")
            if t01 is None:
                continue
            means = []
            for sl, pr in CONDITION_KEYS:
                row = t01[(t01["scale"] == sl) & (t01["prompt"] == pr) & (t01["dimension"] == dim)]
                means.append(row["mean"].values[0] if len(row) > 0 else np.nan)
            offset = (m_idx - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset, means, bar_width * 0.9,
                color=EXPERIMENTS[model]["color"],
                label=model if dim_idx == 0 else None,
                edgecolor="white", linewidth=0.5,
            )
        ax.set_title(dim.capitalize(), fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        if dim_idx == 0:
            ax.set_ylabel("Normalized Mean Score [0, 1]")
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4)

    axes[0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Mean Score Comparison Across Models", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig01_mean_score_comparison")

    # ── Fig02: Ceiling effects heatmap ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        matrix = []
        for model in models:
            t01 = all_tables[model].get("table01")
            if t01 is None:
                matrix.append([np.nan] * len(CONDITION_KEYS))
                continue
            row_vals = []
            for sl, pr in CONDITION_KEYS:
                row = t01[(t01["scale"] == sl) & (t01["prompt"] == pr) & (t01["dimension"] == dim)]
                row_vals.append(row["ceiling_pct"].values[0] if len(row) > 0 else np.nan)
            matrix.append(row_vals)
        matrix = np.array(matrix)
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(60, np.nanmax(matrix)))
        ax.set_title(f"{dim.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(CONDITION_LABELS)))
        ax.set_xticklabels(CONDITION_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=9)
        # Annotate cells
        for i in range(n_models):
            for j in range(len(CONDITION_KEYS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val > 30 else "black"
                    ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7, color=color)
        if dim_idx == 2:
            fig.colorbar(im, ax=ax, label="Ceiling %", shrink=0.8)

    fig.suptitle("Ceiling Effects by Model, Condition, and Dimension", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig02_ceiling_effects_heatmap")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: RELIABILITY COMPARISON
# ══════════════════════════════════════════════════════════════════════

def section2_reliability(all_tables):
    """Compare reliability metrics across models."""
    print("\n=== Section 2: Reliability Comparison ===")
    models = [m for m in MODEL_ORDER if m in all_tables]
    n_models = len(models)

    # ── Fig03: Reliability forest plot ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 12), sharey=True)
    y_pos = 0
    y_ticks = []
    y_labels = []

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        y_pos = 0
        y_ticks_dim = []
        y_labels_dim = []

        for ci, (sl, pr) in enumerate(CONDITION_KEYS):
            for m_idx, model in enumerate(models):
                t05 = all_tables[model].get("table05")
                if t05 is None:
                    continue
                row = t05[(t05["scale"] == sl) & (t05["prompt"] == pr) & (t05["dimension"] == dim)]
                if len(row) == 0:
                    continue
                alpha = row["alpha"].values[0]
                ci_low = row["alpha_ci_low"].values[0]
                ci_high = row["alpha_ci_high"].values[0]
                ax.errorbar(
                    alpha, y_pos, xerr=[[alpha - ci_low], [ci_high - alpha]],
                    fmt=EXPERIMENTS[model]["marker"], color=EXPERIMENTS[model]["color"],
                    markersize=5, capsize=2, linewidth=1,
                    label=model if (ci == 0 and dim_idx == 0) else None,
                )
                if m_idx == n_models // 2:
                    y_ticks_dim.append(y_pos)
                    y_labels_dim.append(condition_label(sl, pr))
                y_pos += 1
            y_pos += 0.5  # gap between conditions

        ax.set_yticks(y_ticks_dim)
        ax.set_yticklabels(y_labels_dim, fontsize=8)
        ax.set_title(f"{dim.capitalize()}", fontsize=12, fontweight="bold")
        ax.axvline(0.80, color="green", linestyle="--", alpha=0.5, label="0.80" if dim_idx == 0 else None)
        ax.axvline(0.67, color="orange", linestyle="--", alpha=0.5, label="0.67" if dim_idx == 0 else None)
        ax.set_xlim(0.6, 1.0)
        ax.invert_yaxis()
        if dim_idx == 0:
            ax.set_xlabel("Krippendorff's Alpha")

    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Reliability (Krippendorff's Alpha) by Model and Condition", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "fig03_reliability_forest")

    # ── Fig04: Alpha summary by scale ──
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8 / n_models
    x = np.arange(len(SCALE_LABELS))

    for m_idx, model in enumerate(models):
        t05 = all_tables[model].get("table05")
        if t05 is None:
            continue
        means = []
        errs_low = []
        errs_high = []
        for sl in SCALE_LABELS:
            subset = t05[t05["scale"] == sl]["alpha"]
            mean_val = subset.mean()
            means.append(mean_val)
            errs_low.append(mean_val - subset.min())
            errs_high.append(subset.max() - mean_val)
        offset = (m_idx - (n_models - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, bar_width * 0.9,
            yerr=[errs_low, errs_high],
            color=EXPERIMENTS[model]["color"], label=model,
            edgecolor="white", linewidth=0.5, capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_LABELS, fontsize=11)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Krippendorff's Alpha (mean across prompts & dims)")
    ax.set_ylim(0.7, 1.0)
    ax.axhline(0.80, color="green", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    ax.set_title("Average Reliability by Scale and Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig04_alpha_summary")

    # ── Fig05: SNR comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    bar_width = 0.8 / n_models
    x = np.arange(len(CONDITION_KEYS))

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        for m_idx, model in enumerate(models):
            t05 = all_tables[model].get("table05")
            if t05 is None:
                continue
            vals = []
            for sl, pr in CONDITION_KEYS:
                row = t05[(t05["scale"] == sl) & (t05["prompt"] == pr) & (t05["dimension"] == dim)]
                vals.append(row["snr"].values[0] if len(row) > 0 else np.nan)
            offset = (m_idx - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset, vals, bar_width * 0.9,
                color=EXPERIMENTS[model]["color"],
                label=model if dim_idx == 0 else None,
                edgecolor="white", linewidth=0.5,
            )
        ax.set_title(f"{dim.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_LABELS, rotation=45, ha="right", fontsize=8)
        if dim_idx == 0:
            ax.set_ylabel("Signal-to-Noise Ratio")

    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("SNR by Model, Condition, and Dimension", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig05_snr_comparison")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: RANK CONSISTENCY
# ══════════════════════════════════════════════════════════════════════

def section3_rank_consistency(all_tables):
    """Compare rank ordering agreement across models."""
    print("\n=== Section 3: Rank Consistency ===")
    models = [m for m in MODEL_ORDER if m in all_tables]
    n_models = len(models)

    # ── Fig06: Cross-model rank correlations for reference condition ──
    ref_sl, ref_pr = REFERENCE_CONDITION
    model_dfs = load_cross_model_scores(ref_sl, ref_pr)
    available_models = [m for m in models if m in model_dfs]

    cross_rhos = {}  # (model_a, model_b) -> {dim: rho}

    if len(available_models) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        n_avail = len(available_models)

        for dim_idx, dim in enumerate(DIMENSIONS):
            ax = axes[dim_idx]
            rho_matrix = np.ones((n_avail, n_avail))

            for i, m_a in enumerate(available_models):
                for j, m_b in enumerate(available_models):
                    if i >= j:
                        continue
                    vals_a = model_dfs[m_a][f"{dim}_mean_frac"].values
                    vals_b = model_dfs[m_b][f"{dim}_mean_frac"].values
                    rho, _ = sp_stats.spearmanr(vals_a, vals_b)
                    rho_matrix[i, j] = rho
                    rho_matrix[j, i] = rho
                    pair = (m_a, m_b)
                    if pair not in cross_rhos:
                        cross_rhos[pair] = {}
                    cross_rhos[pair][dim] = rho

            im = ax.imshow(rho_matrix, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="equal")
            ax.set_title(f"{dim.capitalize()}", fontsize=12, fontweight="bold")
            ax.set_xticks(range(n_avail))
            ax.set_xticklabels(available_models, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n_avail))
            ax.set_yticklabels(available_models, fontsize=8)
            # Annotate
            for i in range(n_avail):
                for j in range(n_avail):
                    color = "white" if rho_matrix[i, j] < 0.75 else "black"
                    ax.text(j, i, f"{rho_matrix[i, j]:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold" if i == j else "normal", color=color)
            if dim_idx == 2:
                fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)

        fig.suptitle(
            f"Cross-Model Rank Correlations (Reference: {ref_sl}/{ref_pr})",
            fontsize=14, fontweight="bold", y=1.02
        )
        fig.tight_layout()
        save_figure(fig, "fig06_cross_model_rank_correlations")
    else:
        print("  Skipping Fig06: insufficient raw score data")

    # ── Fig07: Within-model rank stability (box plot of table06 rho values) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = []
    box_labels = []
    box_colors = []
    for model in models:
        t06 = all_tables[model].get("table06")
        if t06 is None:
            continue
        box_data.append(t06["spearman_rho"].values)
        box_labels.append(model)
        box_colors.append(EXPERIMENTS[model]["color"])

    bp = ax.boxplot(
        box_data, labels=box_labels, patch_artist=True,
        widths=0.5, showfliers=False,
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median_line in bp["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(1.5)

    # Jittered strip plot overlay
    rng = np.random.default_rng(RNG_SEED)
    for i, data in enumerate(box_data):
        jitter = rng.uniform(-0.15, 0.15, size=len(data))
        ax.scatter(
            np.full_like(data, i + 1) + jitter, data,
            color=box_colors[i], alpha=0.15, s=8, zorder=2,
        )

    ax.set_ylabel("Spearman rho (within-model, cross-condition)")
    ax.set_title("Within-Model Rank Stability Across Conditions", fontsize=13, fontweight="bold")
    ax.axhline(0.9, color="green", linestyle="--", alpha=0.4, label="rho = 0.90")
    ax.axhline(0.8, color="orange", linestyle="--", alpha=0.4, label="rho = 0.80")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig07_within_model_rank_stability")

    return cross_rhos


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: SCALE & PROMPT SENSITIVITY
# ══════════════════════════════════════════════════════════════════════

def section4_sensitivity(all_tables):
    """Compare sensitivity to scale and prompt manipulations across models."""
    print("\n=== Section 4: Scale & Prompt Sensitivity ===")
    models = [m for m in MODEL_ORDER if m in all_tables]
    n_models = len(models)

    # ── Fig08: Scale effect sizes (forest plot) ──
    _plot_effect_forest(
        all_tables, models, "table02",
        "fig08_scale_effect_comparison",
        "Scale Effect Sizes by Model (Mean Difference with 95% CI)"
    )

    # ── Fig09: Prompt effect sizes (forest plot) ──
    _plot_effect_forest(
        all_tables, models, "table03",
        "fig09_prompt_effect_comparison",
        "Prompt Effect Sizes by Model (Mean Difference with 95% CI)"
    )


def _plot_effect_forest(all_tables, models, table_key, fig_name, title):
    """Generic forest plot for scale or prompt effects."""
    n_models = len(models)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)

    for dim_idx, dim in enumerate(DIMENSIONS):
        ax = axes[dim_idx]
        y_pos = 0
        y_ticks = []
        y_labels = []

        # Get comparison labels from first available model
        comparisons = []
        for model in models:
            tbl = all_tables[model].get(table_key)
            if tbl is not None:
                wilcoxon_rows = tbl[(tbl["test"] == "Wilcoxon") & (tbl["dimension"] == dim)]
                comparisons = wilcoxon_rows["comparison"].unique().tolist()
                break

        for comp in comparisons:
            for m_idx, model in enumerate(models):
                tbl = all_tables[model].get(table_key)
                if tbl is None:
                    continue
                row = tbl[(tbl["test"] == "Wilcoxon") & (tbl["dimension"] == dim)
                          & (tbl["comparison"] == comp)]
                if len(row) == 0:
                    continue
                mean_diff = row["mean_diff"].values[0]
                ci_low = row["ci_low"].values[0]
                ci_high = row["ci_high"].values[0]
                ax.errorbar(
                    mean_diff, y_pos,
                    xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
                    fmt=EXPERIMENTS[model]["marker"],
                    color=EXPERIMENTS[model]["color"],
                    markersize=6, capsize=3, linewidth=1.2,
                    label=model if (dim_idx == 0 and y_pos < n_models) else None,
                )
                y_pos += 1

            # Label at center of model group
            center = y_pos - n_models / 2 - 0.5
            y_ticks.append(center)
            y_labels.append(comp)
            y_pos += 1  # gap

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.axvline(0, color="grey", linestyle="-", alpha=0.4)
        ax.set_title(f"{dim.capitalize()}", fontsize=12, fontweight="bold")
        ax.invert_yaxis()
        if dim_idx == 0:
            ax.set_xlabel("Mean Difference (normalized [0, 1])")

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, fig_name)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: EFFICIENCY & SPEED
# ══════════════════════════════════════════════════════════════════════

def section5_efficiency(all_tables, all_manifests):
    """Compare processing speed and quality across models."""
    print("\n=== Section 5: Efficiency & Speed ===")
    models = [m for m in MODEL_ORDER if m in all_tables]
    n_models = len(models)

    # ── Fig10: Speed comparison (grouped bar) ──
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.8 / n_models
    x = np.arange(len(CONDITION_KEYS))

    for m_idx, model in enumerate(models):
        t08 = all_tables[model].get("table08")
        if t08 is None:
            continue
        times = []
        for sl, pr in CONDITION_KEYS:
            row = t08[(t08["scale"] == sl) & (t08["prompt"] == pr)]
            times.append(row["median_time_ms"].values[0] / 1000 if len(row) > 0 else np.nan)
        offset = (m_idx - (n_models - 1) / 2) * bar_width
        ax.bar(
            x + offset, times, bar_width * 0.9,
            color=EXPERIMENTS[model]["color"], label=model,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_LABELS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Median Processing Time per Entity (seconds)")
    ax.set_title("Processing Speed by Model and Condition", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "fig10_speed_comparison")

    # ── Fig11: Pareto frontier (quality vs speed) ──
    fig, ax = plt.subplots(figsize=(10, 8))

    for model in models:
        t08 = all_tables[model].get("table08")
        if t08 is None:
            continue
        times = t08["median_time_ms"].values / 1000
        quality = t08["quality_index"].values
        ax.scatter(
            times, quality,
            marker=EXPERIMENTS[model]["marker"],
            color=EXPERIMENTS[model]["color"],
            s=80, label=model, zorder=3, edgecolors="white", linewidths=0.5,
        )

    # Global Pareto frontier
    all_points = []
    for model in models:
        t08 = all_tables[model].get("table08")
        if t08 is None:
            continue
        for _, row in t08.iterrows():
            all_points.append((row["median_time_ms"] / 1000, row["quality_index"], model,
                               condition_label(row["scale"], row["prompt"])))

    if all_points:
        all_points.sort(key=lambda p: p[0])
        pareto = []
        best_quality = -1
        for pt in all_points:
            if pt[1] > best_quality:
                pareto.append(pt)
                best_quality = pt[1]
        if len(pareto) > 1:
            px = [p[0] for p in pareto]
            py = [p[1] for p in pareto]
            ax.step(px, py, where="post", color="black", linestyle="--", alpha=0.5,
                    linewidth=1.5, label="Pareto frontier")
        # Annotate Pareto-optimal points
        for pt in pareto:
            ax.annotate(
                f"{pt[2]}\n{pt[3]}", (pt[0], pt[1]),
                textcoords="offset points", xytext=(8, 5), fontsize=7, alpha=0.7,
            )

    ax.set_xlabel("Median Processing Time per Entity (seconds)")
    ax.set_ylabel("Quality Index")
    ax.set_title("Quality vs Speed Pareto Analysis", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig11_pareto_frontier")


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: INTER-MODEL AGREEMENT
# ══════════════════════════════════════════════════════════════════════

def section6_agreement(all_tables, cross_rhos):
    """Produce inter-model agreement tables and supplementary figure."""
    print("\n=== Section 6: Inter-Model Agreement ===")
    models = [m for m in MODEL_ORDER if m in all_tables]

    # ── T02: Cross-model rank correlations (reference condition) ──
    if cross_rhos:
        rows = []
        for (m_a, m_b), dim_rhos in cross_rhos.items():
            row = {"model_1": m_a, "model_2": m_b}
            rho_vals = []
            for dim in DIMENSIONS:
                rho = dim_rhos.get(dim, np.nan)
                row[f"{dim}_rho"] = round(rho, 4)
                rho_vals.append(rho)
            row["mean_rho"] = round(np.nanmean(rho_vals), 4)
            rows.append(row)
        t02_df = pd.DataFrame(rows)
        t02_path = TBL_DIR / "table_t02_cross_model_ranks.csv"
        t02_df.to_csv(t02_path, index=False)
        print(f"  Saved {t02_path.name}")

        # ── T04: Dimension-level agreement summary ──
        rows = []
        for dim in DIMENSIONS:
            rhos = [v.get(dim, np.nan) for v in cross_rhos.values()]
            rhos = [r for r in rhos if not np.isnan(r)]
            pairs = list(cross_rhos.keys())
            if rhos:
                min_idx = np.argmin(rhos)
                max_idx = np.argmax(rhos)
                rows.append({
                    "dimension": dim,
                    "mean_cross_model_rho": round(np.mean(rhos), 4),
                    "min_rho": round(min(rhos), 4),
                    "max_rho": round(max(rhos), 4),
                    "min_pair": f"{pairs[min_idx][0]} vs {pairs[min_idx][1]}",
                    "max_pair": f"{pairs[max_idx][0]} vs {pairs[max_idx][1]}",
                })
        if rows:
            t04_df = pd.DataFrame(rows)
            t04_path = TBL_DIR / "table_t04_dimension_agreement.csv"
            t04_df.to_csv(t04_path, index=False)
            print(f"  Saved {t04_path.name}")

    # ── Fig12: Agreement heatmap for all conditions ──
    # Compute cross-model Spearman rho for each condition
    ref_sl, ref_pr = REFERENCE_CONDITION
    available_models = [m for m in models if (BASE_DIR / EXPERIMENTS[m]["dir"]).exists()]
    model_pairs = list(combinations(available_models, 2))

    if len(model_pairs) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for dim_idx, dim in enumerate(DIMENSIONS):
            ax = axes[dim_idx]
            # Matrix: rows = model pairs, cols = conditions
            matrix = np.full((len(model_pairs), len(CONDITION_KEYS)), np.nan)
            pair_labels = [f"{a[:8]} vs {b[:8]}" for a, b in model_pairs]

            for ci, (sl, pr) in enumerate(CONDITION_KEYS):
                mdfs = load_cross_model_scores(sl, pr)
                for pi, (m_a, m_b) in enumerate(model_pairs):
                    if m_a in mdfs and m_b in mdfs:
                        va = mdfs[m_a][f"{dim}_mean_frac"].values
                        vb = mdfs[m_b][f"{dim}_mean_frac"].values
                        if len(va) > 2 and len(vb) > 2:
                            rho, _ = sp_stats.spearmanr(va, vb)
                            matrix[pi, ci] = rho

            im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.5, vmax=1.0, aspect="auto")
            ax.set_title(f"{dim.capitalize()}", fontsize=12, fontweight="bold")
            ax.set_xticks(range(len(CONDITION_LABELS)))
            ax.set_xticklabels(CONDITION_LABELS, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(model_pairs)))
            ax.set_yticklabels(pair_labels, fontsize=7)
            # Annotate
            for i in range(len(model_pairs)):
                for j in range(len(CONDITION_KEYS)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = "white" if val < 0.75 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
            if dim_idx == 2:
                fig.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)

        fig.suptitle("Cross-Model Entity Rank Agreement by Condition", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()
        save_figure(fig, "fig12_cross_model_agreement_heatmap")


# ══════════════════════════════════════════════════════════════════════
# OUTPUT TABLES
# ══════════════════════════════════════════════════════════════════════

def generate_tables(all_tables, all_manifests):
    """Generate summary comparison tables."""
    print("\n=== Generating Summary Tables ===")
    models = [m for m in MODEL_ORDER if m in all_tables]

    # ── T01: Model overview ──
    rows = []
    for model in models:
        manifest = all_manifests.get(model)
        t05 = all_tables[model].get("table05")
        t08 = all_tables[model].get("table08")
        t01 = all_tables[model].get("table01")
        row = {
            "model": model,
            "model_id": EXPERIMENTS[model]["model_id"],
            "provider": EXPERIMENTS[model]["provider"],
            "n_entities": int(t01["n"].iloc[0]) if t01 is not None else None,
            "total_duration_hrs": round(manifest["summary"]["total_duration_seconds"] / 3600, 1) if manifest else None,
            "avg_alpha": round(t05["alpha"].mean(), 4) if t05 is not None else None,
            "avg_quality_index": round(t08["quality_index"].mean(), 4) if t08 is not None else None,
            "avg_snr": round(t05["snr"].mean(), 2) if t05 is not None else None,
        }
        rows.append(row)
    t01_df = pd.DataFrame(rows)
    t01_path = TBL_DIR / "table_t01_model_overview.csv"
    t01_df.to_csv(t01_path, index=False)
    print(f"  Saved {t01_path.name}")

    # ── T03: Best condition per model ──
    rows = []
    for model in models:
        t08 = all_tables[model].get("table08")
        if t08 is None:
            continue
        best_idx = t08["quality_index"].idxmax()
        best = t08.loc[best_idx]
        rows.append({
            "model": model,
            "best_condition": condition_label(best["scale"], best["prompt"]),
            "quality_index": round(best["quality_index"], 4),
            "median_time_s": round(best["median_time_ms"] / 1000, 1),
            "avg_alpha": round(best["avg_alpha"], 4),
        })
    t03_df = pd.DataFrame(rows)
    t03_path = TBL_DIR / "table_t03_best_condition.csv"
    t03_df.to_csv(t03_path, index=False)
    print(f"  Saved {t03_path.name}")

    return t01_df, t03_df


# ══════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════

def generate_summary_report(all_tables, all_manifests, cross_rhos, t01_df, t03_df):
    """Generate markdown summary report."""
    print("\n=== Generating Summary Report ===")
    models = [m for m in MODEL_ORDER if m in all_tables]
    L = []

    L.append("# Cross-Model Comparison Report")
    L.append("")
    L.append("## Experiment Overview")
    L.append("")
    L.append("Four LLM models completed identical 3x3 factorial scoring experiments")
    L.append("(3 scales x 3 prompt versions x 3 SETS dimensions, 3 runs per entity).")
    L.append("")

    # Model overview table
    L.append("### Models")
    L.append("")
    L.append("| Model | Provider | Entities | Duration (hrs) | Avg Alpha | Avg Quality | Avg SNR |")
    L.append("|-------|----------|----------|----------------|-----------|-------------|---------|")
    for _, row in t01_df.iterrows():
        L.append(f"| {row['model']} | {row['provider']} | {row['n_entities']} | "
                 f"{row['total_duration_hrs']} | {row['avg_alpha']:.4f} | "
                 f"{row['avg_quality_index']:.4f} | {row['avg_snr']:.1f} |")
    L.append("")

    # Key findings
    L.append("## Key Findings")
    L.append("")

    # 1. Score distributions
    L.append("### 1. Score Distributions")
    L.append("")
    for model in models:
        t01 = all_tables[model].get("table01")
        if t01 is None:
            continue
        social_mean = t01[t01["dimension"] == "social"]["mean"].mean()
        ceiling_max = t01["ceiling_pct"].max()
        L.append(f"- **{model}**: Social mean = {social_mean:.3f}, max ceiling = {ceiling_max:.1f}%")
    L.append("")

    # 2. Reliability
    L.append("### 2. Reliability")
    L.append("")
    for model in models:
        t05 = all_tables[model].get("table05")
        if t05 is None:
            continue
        avg_alpha = t05["alpha"].mean()
        min_alpha = t05["alpha"].min()
        max_alpha = t05["alpha"].max()
        L.append(f"- **{model}**: Alpha = {avg_alpha:.3f} (range: {min_alpha:.3f}-{max_alpha:.3f})")
    L.append("")

    # 3. Cross-model agreement
    if cross_rhos:
        L.append("### 3. Cross-Model Rank Agreement (Reference: 0-100/v2)")
        L.append("")
        L.append("| Model Pair | Social | Ecological | Technological | Mean |")
        L.append("|------------|--------|------------|---------------|------|")
        for (m_a, m_b), dim_rhos in cross_rhos.items():
            soc = dim_rhos.get("social", np.nan)
            eco = dim_rhos.get("ecological", np.nan)
            tec = dim_rhos.get("technological", np.nan)
            mean_r = np.nanmean([soc, eco, tec])
            L.append(f"| {m_a} vs {m_b} | {soc:.3f} | {eco:.3f} | {tec:.3f} | {mean_r:.3f} |")
        L.append("")

    # 4. Best conditions
    L.append("### 4. Best Condition Per Model")
    L.append("")
    L.append("| Model | Best Condition | Quality Index | Time (s) | Alpha |")
    L.append("|-------|---------------|---------------|----------|-------|")
    for _, row in t03_df.iterrows():
        L.append(f"| {row['model']} | {row['best_condition']} | "
                 f"{row['quality_index']:.4f} | {row['median_time_s']:.1f} | {row['avg_alpha']:.4f} |")
    L.append("")

    # 5. Efficiency
    L.append("### 5. Efficiency")
    L.append("")
    for model in models:
        t08 = all_tables[model].get("table08")
        manifest = all_manifests.get(model)
        if t08 is None:
            continue
        avg_time_s = t08["median_time_ms"].mean() / 1000
        total_hrs = manifest["summary"]["total_duration_seconds"] / 3600 if manifest else 0
        L.append(f"- **{model}**: Avg {avg_time_s:.1f}s/entity, total {total_hrs:.1f} hrs")
    L.append("")

    # Caveats
    L.append("## Caveats")
    L.append("")
    L.append("- Processing times reflect different hardware configurations and runtime environments")
    L.append("  (Ollama vs MLX, different quantization levels). Speed comparisons should be")
    L.append("  interpreted cautiously.")
    L.append("- Cross-model rank correlations are descriptive, not inferential.")
    L.append("- Qwen3.5 35B had 1 entity scoring failure (449/450 entities in some conditions).")
    L.append("")

    # Output inventory
    L.append("## Output Inventory")
    L.append("")
    L.append("### Figures")
    L.append("")
    L.append("| Figure | Description |")
    L.append("|--------|-------------|")
    L.append("| Fig 01 | Mean score comparison across models |")
    L.append("| Fig 02 | Ceiling effects heatmap |")
    L.append("| Fig 03 | Reliability forest plot (alpha with CIs) |")
    L.append("| Fig 04 | Average alpha by scale and model |")
    L.append("| Fig 05 | SNR comparison |")
    L.append("| Fig 06 | Cross-model rank correlations (reference condition) |")
    L.append("| Fig 07 | Within-model rank stability |")
    L.append("| Fig 08 | Scale effect size comparison |")
    L.append("| Fig 09 | Prompt effect size comparison |")
    L.append("| Fig 10 | Processing speed comparison |")
    L.append("| Fig 11 | Quality vs speed Pareto analysis |")
    L.append("| Fig 12 | Cross-model agreement heatmap (all conditions) |")
    L.append("")
    L.append("### Tables")
    L.append("")
    L.append("| Table | Description |")
    L.append("|-------|-------------|")
    L.append("| T01 | Model overview (provider, duration, reliability, quality) |")
    L.append("| T02 | Cross-model rank correlations for reference condition |")
    L.append("| T03 | Best condition per model |")
    L.append("| T04 | Dimension-level cross-model agreement summary |")
    L.append("")

    # Methods
    L.append("## Methods Notes")
    L.append("")
    L.append("- **Normalization:** All scores converted to [0, 1] fraction via `(score - scale_min) / scale_range`")
    L.append("- **Cross-model alignment:** Entity-level comparisons use composite key")
    L.append("  `entity|text_id|window_index` with inner join across models")
    L.append("- **Rank correlations:** Spearman rho (scipy.stats.spearmanr)")
    L.append("- **Quality index:** Geometric mean of avg Krippendorff's alpha and avg Spearman rho")
    L.append("  vs reference condition (from per-model analysis)")

    report_path = OUT_DIR / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Saved {report_path.name}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    global OUT_DIR, FIG_DIR, TBL_DIR

    parser = argparse.ArgumentParser(
        description="Cross-model comparison of factorial scoring experiments"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: output/cross-model-comparison)"
    )
    args = parser.parse_args()

    if args.output_dir:
        OUT_DIR = Path(args.output_dir).resolve()
    FIG_DIR = OUT_DIR / "figures"
    TBL_DIR = OUT_DIR / "tables"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CROSS-MODEL COMPARISON ANALYSIS")
    print("=" * 60)

    # Load all experiments
    print("\nLoading experiments...")
    all_tables, all_manifests = load_all_experiments()

    if len(all_tables) < 2:
        print("ERROR: Need at least 2 experiments to compare.")
        return

    print(f"\n  Found {len(all_tables)} experiments: {', '.join(all_tables.keys())}")

    # Run analysis sections
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        section1_score_distributions(all_tables)
        section2_reliability(all_tables)
        cross_rhos = section3_rank_consistency(all_tables)
        section4_sensitivity(all_tables)
        section5_efficiency(all_tables, all_manifests)
        section6_agreement(all_tables, cross_rhos)
        t01_df, t03_df = generate_tables(all_tables, all_manifests)
        generate_summary_report(all_tables, all_manifests, cross_rhos, t01_df, t03_df)

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TBL_DIR}")
    print(f"  Report:  {OUT_DIR / 'summary_report.md'}")


if __name__ == "__main__":
    main()
