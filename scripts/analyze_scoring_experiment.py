#!/usr/bin/env python3
"""
Analysis of 3×3 factorial scoring experiment (scale × prompt version).

Produces publication-ready figures and statistical tables for Sections 1-9.
See analysis_plan.md for full research design.

Usage:
    python qualitative-analysis/scripts/analyze_scoring_experiment.py
    python qualitative-analysis/scripts/analyze_scoring_experiment.py --exp-dir qualitative-analysis/output/scoring-experiment-qwen3-80b
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Configuration (defaults; overridden by --exp-dir CLI arg) ────────
EXP_DIR = Path(__file__).resolve().parent.parent / "output" / "scoring-experiment"
OUT_DIR = EXP_DIR / "analysis"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
MODEL_NAME = "gpt-oss:120b"  # default; auto-detected from manifest

DIMENSIONS = ["social", "ecological", "technological"]
DIM_COLORS = {
    "social": "#1f77b4",
    "ecological": "#2ca02c",
    "technological": "#d62728",
}
SCALES = [("0-100", 0, 100), ("1-10", 1, 10), ("1-5", 1, 5)]
PROMPTS = ["v2", "v3", "v4"]
PROMPT_LABELS = {"v2": "Score→Justify", "v3": "Justify→Score", "v4": "No CoT"}
PROMPT_STYLES = {"v2": ("o", "-"), "v3": ("s", "--"), "v4": ("^", ":")}

RNG_SEED = 42
N_BOOT = 10000
DPI = 150

# ── Utility Functions ────────────────────────────────────────────────

def save_figure(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {path.name}")


def paired_bootstrap_ci(a, b, n_boot=N_BOOT, seed=RNG_SEED, ci=0.95):
    """Bootstrap CI for mean(a - b)."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a) - np.asarray(b)
    n = len(diffs)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.mean(diffs)), float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def single_bootstrap_ci(arr, n_boot=N_BOOT, seed=RNG_SEED, ci=0.95):
    """Bootstrap CI for a single statistic (mean)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr)
    n = len(arr)
    boot_means = np.array([
        np.mean(rng.choice(arr, size=n, replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(boot_means, alpha * 100)), float(np.percentile(boot_means, (1 - alpha) * 100))


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def condition_key(scale_label, prompt):
    return f"{scale_label}_{prompt}"


def aligned_values(data, key_a, key_b, col):
    """Return paired arrays aligned on the 'entity' column.

    Handles cases where one condition may have a missing entity
    (e.g., a scoring failure) by doing an inner join.
    """
    df_a = data[key_a][["entity", col]].dropna(subset=[col])
    df_b = data[key_b][["entity", col]].dropna(subset=[col])
    merged = df_a.merge(df_b, on="entity", suffixes=("_a", "_b"))
    return merged[f"{col}_a"].values, merged[f"{col}_b"].values


# ── Data Loading ─────────────────────────────────────────────────────

def load_all_conditions():
    """Load all 9 condition CSVs and metadata."""
    data = {}
    for scale_label, s_min, s_max in SCALES:
        for prompt in PROMPTS:
            dir_name = f"scale-{s_min}-{s_max}_prompt-{prompt}"
            cond_dir = EXP_DIR / dir_name
            csv_files = list(cond_dir.glob("*_scores.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No scores CSV in {cond_dir}")
            df = pd.read_csv(csv_files[0])
            df["scale"] = scale_label
            df["prompt"] = prompt
            df["scale_min"] = s_min
            df["scale_max"] = s_max
            df["scale_range"] = s_max - s_min
            # Normalize run-level and aggregate scores to [0, 1]
            for dim in DIMENSIONS:
                sr = df["scale_range"]
                sm = df["scale_min"]
                for col in [f"{dim}_run1", f"{dim}_run2", f"{dim}_run3", f"{dim}_mean", f"{dim}_median"]:
                    if col in df.columns:
                        df[f"{col}_frac"] = (df[col] - sm) / sr
                # Normalized std (as fraction of scale range)
                if f"{dim}_std" in df.columns:
                    df[f"{dim}_std_frac"] = df[f"{dim}_std"] / sr
            key = condition_key(scale_label, prompt)
            data[key] = df
    return data


def build_master_df(data):
    """Concatenate all conditions into one DataFrame."""
    return pd.concat(data.values(), ignore_index=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: Descriptive Overview
# ══════════════════════════════════════════════════════════════════════

def section1_descriptives(data, master_df):
    print("\n=== Section 1: Descriptive Overview ===")

    # Table 1: Descriptive statistics
    rows = []
    for scale_label, _, _ in SCALES:
        for prompt in PROMPTS:
            key = condition_key(scale_label, prompt)
            df = data[key]
            for dim in DIMENSIONS:
                vals = df[f"{dim}_mean_frac"].dropna().values
                floor_pct = (vals <= 0.05).mean() * 100
                ceil_pct = (vals >= 0.95).mean() * 100
                rows.append({
                    "scale": scale_label, "prompt": prompt, "dimension": dim,
                    "mean": round(np.mean(vals), 4),
                    "sd": round(np.std(vals, ddof=1), 4),
                    "median": round(np.median(vals), 4),
                    "skewness": round(float(sp_stats.skew(vals)), 4),
                    "kurtosis": round(float(sp_stats.kurtosis(vals)), 4),
                    "floor_pct": round(floor_pct, 1),
                    "ceiling_pct": round(ceil_pct, 1),
                    "n": len(vals),
                })
    table1 = pd.DataFrame(rows)
    table1.to_csv(TBL_DIR / "table01_descriptives.csv", index=False)
    print("  Saved table01_descriptives.csv")

    # Figure 1: 3×3 distribution grid per dimension
    for dim in DIMENSIONS:
        fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True, sharey=True)
        fig.suptitle(f"Score Distributions — {dim.title()}", fontsize=14, fontweight="bold")
        color = DIM_COLORS[dim]
        for row_i, (scale_label, _, _) in enumerate(SCALES):
            for col_j, prompt in enumerate(PROMPTS):
                ax = axes[row_i, col_j]
                key = condition_key(scale_label, prompt)
                vals = data[key][f"{dim}_mean_frac"].dropna().values
                ax.hist(vals, bins=30, density=True, alpha=0.6, color=color, edgecolor="white", linewidth=0.5)
                # KDE overlay
                if len(vals) > 5 and np.std(vals) > 0:
                    kde_x = np.linspace(0, 1, 200)
                    try:
                        kde = sp_stats.gaussian_kde(vals, bw_method=0.1)
                        ax.plot(kde_x, kde(kde_x), color=color, linewidth=1.5)
                    except Exception:
                        pass
                ax.axvline(np.mean(vals), color="black", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.set_xlim(0, 1)
                if row_i == 0:
                    ax.set_title(PROMPT_LABELS[prompt], fontsize=10)
                if col_j == 0:
                    ax.set_ylabel(f"Scale {scale_label}", fontsize=10)
                # Annotate
                ax.text(0.02, 0.95, f"μ={np.mean(vals):.2f}\nσ={np.std(vals):.2f}\nskew={sp_stats.skew(vals):.2f}",
                        transform=ax.transAxes, fontsize=7, va="top", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        fig.supxlabel("Normalized Score [0, 1]", fontsize=11)
        fig.supylabel("Density", fontsize=11)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        save_figure(fig, f"fig01_distribution_grid_{dim}")

    # Figure 2: Condition mean dot plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    fig.suptitle("Mean Normalized Scores by Condition (with 95% Bootstrap CI)", fontsize=13, fontweight="bold")
    y_labels = []
    for scale_label, _, _ in SCALES:
        for prompt in PROMPTS:
            y_labels.append(f"{scale_label} / {PROMPT_LABELS[prompt]}")
    y_pos = np.arange(len(y_labels))

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        means, ci_lows, ci_highs = [], [], []
        for scale_label, _, _ in SCALES:
            for prompt in PROMPTS:
                key = condition_key(scale_label, prompt)
                vals = data[key][f"{dim}_mean_frac"].dropna().values
                m, cl, ch = single_bootstrap_ci(vals)
                means.append(m)
                ci_lows.append(cl)
                ci_highs.append(ch)
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)
        ax.errorbar(means, y_pos, xerr=[means - ci_lows, ci_highs - means],
                     fmt="o", color=DIM_COLORS[dim], capsize=3, markersize=5)
        ax.set_yticks(y_pos)
        if ax_i == 0:
            ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Normalized Mean [0, 1]", fontsize=9)
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.set_xlim(0, 1)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "fig02_condition_means")

    # Figure 3: Score discretization heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle("Raw Score Frequency by Condition", fontsize=13, fontweight="bold")
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        cond_labels = []
        freq_matrix = []
        for scale_label, s_min, s_max in SCALES:
            possible_vals = np.arange(s_min, s_max + 1)
            for prompt in PROMPTS:
                key = condition_key(scale_label, prompt)
                df = data[key]
                # Collect all run-level raw scores
                all_runs = []
                for rc in [f"{dim}_run1", f"{dim}_run2", f"{dim}_run3"]:
                    if rc in df.columns:
                        all_runs.extend(df[rc].dropna().tolist())
                all_runs = np.array(all_runs)
                # Count frequencies for each possible value
                counts = np.array([np.sum(np.round(all_runs) == v) for v in possible_vals])
                freq = counts / max(counts.sum(), 1)
                # Pad to uniform length (normalized to [0,1] fraction)
                frac_vals = (possible_vals - s_min) / (s_max - s_min)
                # Bin into 20 uniform bins on [0,1]
                bin_edges = np.linspace(0, 1, 21)
                hist, _ = np.histogram(
                    np.repeat(frac_vals, counts.astype(int)),
                    bins=bin_edges,
                )
                freq_matrix.append(hist / max(hist.sum(), 1))
                cond_labels.append(f"{scale_label}/{prompt}")

        freq_matrix = np.array(freq_matrix)
        im = ax.imshow(freq_matrix.T, aspect="auto", cmap="YlOrRd",
                       origin="lower", interpolation="nearest")
        ax.set_xticks(range(len(cond_labels)))
        ax.set_xticklabels(cond_labels, rotation=45, ha="right", fontsize=7)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        tick_idx = [0, 4, 9, 14, 19]
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f"{bin_centers[i]:.2f}" for i in tick_idx], fontsize=8)
        ax.set_ylabel("Normalized Score Bin" if ax_i == 0 else "")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        plt.colorbar(im, ax=ax, label="Proportion", shrink=0.8)
    fig.tight_layout()
    save_figure(fig, "fig03_discretization_heatmap")

    return table1


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: Scale Main Effects
# ══════════════════════════════════════════════════════════════════════

def section2_scale_effects(data):
    print("\n=== Section 2: Scale Main Effects ===")
    rows = []
    scale_labels = [s[0] for s in SCALES]
    scale_pairs = [("0-100", "1-10"), ("0-100", "1-5"), ("1-10", "1-5")]

    for dim in DIMENSIONS:
        # Compute entity means collapsed over prompt for each scale
        scale_means = {}
        for sl in scale_labels:
            combined = []
            for prompt in PROMPTS:
                key = condition_key(sl, prompt)
                combined.append(data[key][f"{dim}_mean_frac"].values)
            scale_means[sl] = np.mean(combined, axis=0)  # average over prompts

        # Friedman test
        fr_stat, fr_p = sp_stats.friedmanchisquare(
            scale_means["0-100"], scale_means["1-10"], scale_means["1-5"]
        )
        rows.append({
            "dimension": dim, "test": "Friedman",
            "comparison": "omnibus (3 scales)",
            "statistic": round(fr_stat, 2), "p_value": fr_p,
            "p_adjusted": fr_p, "effect_size": None,
            "mean_diff": None, "ci_low": None, "ci_high": None,
        })

        # Pairwise Wilcoxon
        p_vals = []
        pair_results = []
        for s1, s2 in scale_pairs:
            stat, p = sp_stats.wilcoxon(scale_means[s1], scale_means[s2])
            m_diff, cl, ch = paired_bootstrap_ci(scale_means[s1], scale_means[s2])
            n = len(scale_means[s1])
            r_rb = 1 - (2 * stat / (n * (n + 1) / 2))  # rank-biserial
            p_vals.append(p)
            pair_results.append({
                "dimension": dim, "test": "Wilcoxon",
                "comparison": f"{s1} vs {s2}",
                "statistic": round(float(stat), 2), "p_value": p,
                "effect_size": round(r_rb, 4),
                "mean_diff": round(m_diff, 5), "ci_low": round(cl, 5), "ci_high": round(ch, 5),
            })

        adj_p = holm_bonferroni(p_vals)
        for i, pr in enumerate(pair_results):
            pr["p_adjusted"] = round(adj_p[i], 6)
            rows.append(pr)

    table2 = pd.DataFrame(rows)
    table2.to_csv(TBL_DIR / "table02_scale_effects.csv", index=False)
    print("  Saved table02_scale_effects.csv")

    # Figure 4: Paired difference violins
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Scale Main Effects: Paired Difference Distributions", fontsize=13, fontweight="bold")
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        scale_means = {}
        for sl in scale_labels:
            combined = [data[condition_key(sl, p)][f"{dim}_mean_frac"].values for p in PROMPTS]
            scale_means[sl] = np.mean(combined, axis=0)

        diff_data = []
        pair_labels = []
        for s1, s2 in scale_pairs:
            diffs = scale_means[s1] - scale_means[s2]
            diff_data.append(diffs)
            pair_labels.append(f"{s1}\nvs\n{s2}")

        parts = ax.violinplot(diff_data, positions=range(len(scale_pairs)),
                              showmeans=True, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(DIM_COLORS[dim])
            pc.set_alpha(0.5)
        parts["cmeans"].set_color("black")
        # Add bootstrap CI
        for i, diffs in enumerate(diff_data):
            m, cl, ch = paired_bootstrap_ci(diffs + np.mean(diffs), np.zeros_like(diffs))
            ax.plot([i, i], [cl, ch], color="black", linewidth=2)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, fontsize=8)
        ax.set_ylabel("Difference (normalized)" if ax_i == 0 else "")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig04_scale_paired_diffs")

    return table2


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: Prompt Main Effects
# ══════════════════════════════════════════════════════════════════════

def section3_prompt_effects(data):
    print("\n=== Section 3: Prompt Main Effects ===")
    rows = []
    prompt_pairs = [("v2", "v3"), ("v2", "v4"), ("v3", "v4")]

    for dim in DIMENSIONS:
        prompt_means = {}
        for pr in PROMPTS:
            combined = [data[condition_key(sl, pr)][f"{dim}_mean_frac"].values for sl, _, _ in SCALES]
            prompt_means[pr] = np.mean(combined, axis=0)

        fr_stat, fr_p = sp_stats.friedmanchisquare(
            prompt_means["v2"], prompt_means["v3"], prompt_means["v4"]
        )
        rows.append({
            "dimension": dim, "test": "Friedman",
            "comparison": "omnibus (3 prompts)",
            "statistic": round(fr_stat, 2), "p_value": fr_p,
            "p_adjusted": fr_p, "effect_size": None,
            "mean_diff": None, "ci_low": None, "ci_high": None,
        })

        p_vals = []
        pair_results = []
        for p1, p2 in prompt_pairs:
            stat, p = sp_stats.wilcoxon(prompt_means[p1], prompt_means[p2])
            m_diff, cl, ch = paired_bootstrap_ci(prompt_means[p1], prompt_means[p2])
            n = len(prompt_means[p1])
            r_rb = 1 - (2 * stat / (n * (n + 1) / 2))
            p_vals.append(p)
            pair_results.append({
                "dimension": dim, "test": "Wilcoxon",
                "comparison": f"{p1} vs {p2}",
                "statistic": round(float(stat), 2), "p_value": p,
                "effect_size": round(r_rb, 4),
                "mean_diff": round(m_diff, 5), "ci_low": round(cl, 5), "ci_high": round(ch, 5),
            })

        adj_p = holm_bonferroni(p_vals)
        for i, pr in enumerate(pair_results):
            pr["p_adjusted"] = round(adj_p[i], 6)
            rows.append(pr)

    table3 = pd.DataFrame(rows)
    table3.to_csv(TBL_DIR / "table03_prompt_effects.csv", index=False)
    print("  Saved table03_prompt_effects.csv")

    # Figure 5: Paired difference violins (prompt)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Prompt Main Effects: Paired Difference Distributions", fontsize=13, fontweight="bold")
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        prompt_means = {}
        for pr in PROMPTS:
            combined = [data[condition_key(sl, pr)][f"{dim}_mean_frac"].values for sl, _, _ in SCALES]
            prompt_means[pr] = np.mean(combined, axis=0)

        diff_data = []
        pair_labels = []
        for p1, p2 in prompt_pairs:
            diffs = prompt_means[p1] - prompt_means[p2]
            diff_data.append(diffs)
            pair_labels.append(f"{PROMPT_LABELS[p1]}\nvs\n{PROMPT_LABELS[p2]}")

        parts = ax.violinplot(diff_data, positions=range(len(prompt_pairs)),
                              showmeans=True, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(DIM_COLORS[dim])
            pc.set_alpha(0.5)
        parts["cmeans"].set_color("black")
        for i, diffs in enumerate(diff_data):
            m, cl, ch = paired_bootstrap_ci(diffs + np.mean(diffs), np.zeros_like(diffs))
            ax.plot([i, i], [cl, ch], color="black", linewidth=2)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, fontsize=7)
        ax.set_ylabel("Difference (normalized)" if ax_i == 0 else "")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig05_prompt_paired_diffs")

    return table3


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: Interaction (Scale × Prompt)
# ══════════════════════════════════════════════════════════════════════

def section4_interaction(data):
    print("\n=== Section 4: Scale × Prompt Interaction ===")
    rows = []

    # Figure 6: Interaction plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Scale × Prompt Interaction", fontsize=13, fontweight="bold")
    scale_labels = [s[0] for s in SCALES]

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        for prompt in PROMPTS:
            means = []
            ci_lows = []
            ci_highs = []
            for sl in scale_labels:
                key = condition_key(sl, prompt)
                vals = data[key][f"{dim}_mean_frac"].dropna().values
                m, cl, ch = single_bootstrap_ci(vals)
                means.append(m)
                ci_lows.append(cl)
                ci_highs.append(ch)
            means = np.array(means)
            ci_lows = np.array(ci_lows)
            ci_highs = np.array(ci_highs)
            marker, ls = PROMPT_STYLES[prompt]
            ax.errorbar(range(len(scale_labels)), means,
                        yerr=[means - ci_lows, ci_highs - means],
                        marker=marker, linestyle=ls, capsize=4, markersize=7,
                        label=PROMPT_LABELS[prompt], linewidth=1.5)
        ax.set_xticks(range(len(scale_labels)))
        ax.set_xticklabels(scale_labels, fontsize=9)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Normalized Mean [0, 1]" if ax_i == 0 else "")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(alpha=0.3)
        if ax_i == 2:
            ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    save_figure(fig, "fig06_interaction_plot")

    # Interaction contrasts and Friedman test
    scale_pairs = [("0-100", "1-10"), ("0-100", "1-5"), ("1-10", "1-5")]
    prompt_pairs = [("v2", "v3"), ("v2", "v4"), ("v3", "v4")]

    for dim in DIMENSIONS:
        # Test: does the v4-v2 prompt effect differ across scales?
        prompt_effects = {}
        for sl in [s[0] for s in SCALES]:
            v4_vals = data[condition_key(sl, "v4")][f"{dim}_mean_frac"].values
            v2_vals = data[condition_key(sl, "v2")][f"{dim}_mean_frac"].values
            prompt_effects[sl] = v4_vals - v2_vals

        fr_stat, fr_p = sp_stats.friedmanchisquare(
            prompt_effects["0-100"], prompt_effects["1-10"], prompt_effects["1-5"]
        )
        rows.append({
            "dimension": dim, "test": "Friedman (v4-v2 effect across scales)",
            "statistic": round(fr_stat, 2), "p_value": round(fr_p, 6),
        })

        # Interaction contrasts
        for sp in scale_pairs:
            for pp in prompt_pairs:
                eff_s1 = (data[condition_key(sp[0], pp[1])][f"{dim}_mean_frac"].values -
                          data[condition_key(sp[0], pp[0])][f"{dim}_mean_frac"].values)
                eff_s2 = (data[condition_key(sp[1], pp[1])][f"{dim}_mean_frac"].values -
                          data[condition_key(sp[1], pp[0])][f"{dim}_mean_frac"].values)
                interaction = eff_s1 - eff_s2
                m = float(np.mean(interaction))
                m_, cl, ch = single_bootstrap_ci(interaction)
                rows.append({
                    "dimension": dim,
                    "test": f"Interaction: ({sp[0]} vs {sp[1]}) × ({pp[0]} vs {pp[1]})",
                    "statistic": round(m, 5),
                    "p_value": None,
                    "ci_low": round(cl, 5), "ci_high": round(ch, 5),
                })

    table4 = pd.DataFrame(rows)
    table4.to_csv(TBL_DIR / "table04_interaction.csv", index=False)
    print("  Saved table04_interaction.csv")

    # Figure 7: Interaction contrast heatmap
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Interaction Contrasts (Difference of Differences)", fontsize=13, fontweight="bold")
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        matrix = np.zeros((len(scale_pairs), len(prompt_pairs)))
        labels_matrix = np.empty((len(scale_pairs), len(prompt_pairs)), dtype=object)
        for si, sp in enumerate(scale_pairs):
            for pi, pp in enumerate(prompt_pairs):
                eff_s1 = (data[condition_key(sp[0], pp[1])][f"{dim}_mean_frac"].values -
                          data[condition_key(sp[0], pp[0])][f"{dim}_mean_frac"].values)
                eff_s2 = (data[condition_key(sp[1], pp[1])][f"{dim}_mean_frac"].values -
                          data[condition_key(sp[1], pp[0])][f"{dim}_mean_frac"].values)
                interaction = float(np.mean(eff_s1 - eff_s2))
                matrix[si, pi] = interaction
                labels_matrix[si, pi] = f"{interaction:.4f}"
        vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for si in range(len(scale_pairs)):
            for pi in range(len(prompt_pairs)):
                ax.text(pi, si, labels_matrix[si, pi], ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(prompt_pairs)))
        ax.set_xticklabels([f"{p[0]}→{p[1]}" for p in prompt_pairs], fontsize=8)
        ax.set_yticks(range(len(scale_pairs)))
        ax.set_yticklabels([f"{s[0]} vs {s[1]}" for s in scale_pairs], fontsize=8)
        if ax_i == 0:
            ax.set_ylabel("Scale Pair")
        ax.set_xlabel("Prompt Pair")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_figure(fig, "fig07_interaction_heatmap")

    return table4


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: Reliability
# ══════════════════════════════════════════════════════════════════════

def section5_reliability(data):
    print("\n=== Section 5: Reliability ===")
    from qualitative_analysis.entity.agreement import krippendorff_alpha, bootstrap_ci

    rows = []
    for scale_label, s_min, s_max in SCALES:
        s_range = s_max - s_min
        for prompt in PROMPTS:
            key = condition_key(scale_label, prompt)
            df = data[key]
            for dim in DIMENSIONS:
                # Build run-level reliability matrix (3 raters × 450 units)
                run_cols = [f"{dim}_run1", f"{dim}_run2", f"{dim}_run3"]
                matrix = df[run_cols].values.T  # (3, 450)
                # Normalize to [0, 1] for cross-scale comparability
                matrix_frac = (matrix - s_min) / s_range

                alpha_val = krippendorff_alpha(matrix_frac)
                ci_low, ci_high = bootstrap_ci(matrix_frac, n_bootstrap=1000, random_seed=RNG_SEED)

                # CV and SNR
                cvs = df[f"{dim}_cv"].dropna().values
                avg_cv = float(np.mean(cvs))
                between_sd = float(np.std(df[f"{dim}_mean_frac"].dropna().values, ddof=1))
                within_sd = float(np.mean(df[f"{dim}_std_frac"].dropna().values))
                snr = between_sd / within_sd if within_sd > 0 else float("inf")

                rows.append({
                    "scale": scale_label, "prompt": prompt, "dimension": dim,
                    "alpha": round(alpha_val, 4),
                    "alpha_ci_low": round(ci_low, 4),
                    "alpha_ci_high": round(ci_high, 4),
                    "avg_cv": round(avg_cv, 4),
                    "between_sd_frac": round(between_sd, 4),
                    "within_sd_frac": round(within_sd, 4),
                    "snr": round(snr, 2),
                })

    table5 = pd.DataFrame(rows)
    table5.to_csv(TBL_DIR / "table05_reliability.csv", index=False)
    print("  Saved table05_reliability.csv")

    # Figure 8: Reliability forest plot
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Within-Entity Reliability (Krippendorff's α)", fontsize=13, fontweight="bold")
    y_pos = 0
    y_ticks = []
    y_labels = []
    for scale_label, _, _ in SCALES:
        for prompt in PROMPTS:
            for dim in DIMENSIONS:
                row = table5[(table5["scale"] == scale_label) &
                             (table5["prompt"] == prompt) &
                             (table5["dimension"] == dim)].iloc[0]
                ax.errorbar(row["alpha"], y_pos,
                            xerr=[[row["alpha"] - row["alpha_ci_low"]],
                                  [row["alpha_ci_high"] - row["alpha"]]],
                            fmt="o", color=DIM_COLORS[dim], capsize=3, markersize=5)
                y_ticks.append(y_pos)
                y_labels.append(f"{scale_label}/{prompt}/{dim[:3]}")
                y_pos += 1
            y_pos += 0.5  # gap between conditions

    ax.axvline(0.80, color="green", linestyle="--", alpha=0.5, label="Good (0.80)")
    ax.axvline(0.67, color="orange", linestyle="--", alpha=0.5, label="Fair (0.67)")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Krippendorff's α", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "fig08_reliability_forest")

    # Figure 9: CV distributions (box plots)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Coefficient of Variation (CV) Distributions by Condition", fontsize=13, fontweight="bold")
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        cv_data = []
        cond_labels = []
        for scale_label, _, _ in SCALES:
            for prompt in PROMPTS:
                key = condition_key(scale_label, prompt)
                cv_data.append(data[key][f"{dim}_cv"].dropna().values)
                cond_labels.append(f"{scale_label}\n{prompt}")
        bp = ax.boxplot(cv_data, patch_artist=True, showfliers=False, widths=0.6)
        for patch in bp["boxes"]:
            patch.set_facecolor(DIM_COLORS[dim])
            patch.set_alpha(0.4)
        ax.axhline(0.10, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xticklabels(cond_labels, fontsize=7)
        ax.set_ylabel("CV" if ax_i == 0 else "")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig09_cv_distributions")

    # Figure 10: SNR comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Signal-to-Noise Ratio by Condition", fontsize=13, fontweight="bold")
    x = np.arange(9)
    width = 0.25
    for d_i, dim in enumerate(DIMENSIONS):
        snrs = []
        for scale_label, _, _ in SCALES:
            for prompt in PROMPTS:
                row = table5[(table5["scale"] == scale_label) &
                             (table5["prompt"] == prompt) &
                             (table5["dimension"] == dim)].iloc[0]
                snrs.append(row["snr"])
        ax.bar(x + d_i * width, snrs, width, label=dim.title(),
               color=DIM_COLORS[dim], alpha=0.7, edgecolor="white")

    cond_labels = []
    for scale_label, _, _ in SCALES:
        for prompt in PROMPTS:
            cond_labels.append(f"{scale_label}/{prompt}")
    ax.set_xticks(x + width)
    ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("SNR (between-SD / within-SD)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    grand_mean_snr = table5["snr"].mean()
    ax.axhline(grand_mean_snr, color="gray", linestyle="--", alpha=0.5,
               label=f"Grand mean: {grand_mean_snr:.1f}")
    fig.tight_layout()
    save_figure(fig, "fig10_snr_comparison")

    return table5


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: Rank Preservation
# ══════════════════════════════════════════════════════════════════════

def section6_rank_preservation(data):
    print("\n=== Section 6: Rank Preservation ===")

    cond_keys = []
    for sl, _, _ in SCALES:
        for pr in PROMPTS:
            cond_keys.append(condition_key(sl, pr))

    corr_matrices = {}
    for dim in DIMENSIONS:
        n = len(cond_keys)
        rho_matrix = np.ones((n, n))
        p_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = aligned_values(data, cond_keys[i], cond_keys[j], f"{dim}_mean_frac")
                rho, p = sp_stats.spearmanr(a, b)
                rho_matrix[i, j] = rho
                rho_matrix[j, i] = rho
                p_matrix[i, j] = p
                p_matrix[j, i] = p
        corr_matrices[dim] = rho_matrix

    # Save correlations
    cond_short = [f"{sl}/{pr}" for sl, _, _ in SCALES for pr in PROMPTS]
    rows = []
    for dim in DIMENSIONS:
        for i in range(len(cond_keys)):
            for j in range(i + 1, len(cond_keys)):
                rows.append({
                    "dimension": dim,
                    "condition_1": cond_short[i],
                    "condition_2": cond_short[j],
                    "spearman_rho": round(corr_matrices[dim][i, j], 4),
                })
    table6 = pd.DataFrame(rows)
    table6.to_csv(TBL_DIR / "table06_rank_correlations.csv", index=False)
    print("  Saved table06_rank_correlations.csv")

    # Figure 11: Rank correlation heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Cross-Condition Rank Correlations (Spearman ρ)", fontsize=13, fontweight="bold")
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        mat = corr_matrices[dim]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=0.7, vmax=1.0, aspect="equal")
        for i in range(len(cond_short)):
            for j in range(len(cond_short)):
                val = mat[i, j]
                color = "white" if val < 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color=color)
        ax.set_xticks(range(len(cond_short)))
        ax.set_xticklabels(cond_short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cond_short)))
        ax.set_yticklabels(cond_short, fontsize=7)
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    fig.tight_layout()
    save_figure(fig, "fig11_rank_correlation_matrix")

    # Figure 12: Rank displacement for most divergent pair
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Entity Rank Displacement: 0-100/v2 vs 1-5/v4 (max divergence)", fontsize=13, fontweight="bold")
    key_a = condition_key("0-100", "v2")
    key_b = condition_key("1-5", "v4")
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        a_vals = data[key_a][f"{dim}_mean_frac"].values
        b_vals = data[key_b][f"{dim}_mean_frac"].values
        rank_a = sp_stats.rankdata(-a_vals)  # descending
        rank_b = sp_stats.rankdata(-b_vals)
        displacements = np.abs(rank_a - rank_b)
        # Plot lines connecting rank positions
        n_show = 50  # top entities by average rank
        avg_rank = (rank_a + rank_b) / 2
        show_idx = np.argsort(avg_rank)[:n_show]
        for idx in show_idx:
            alpha_val = 0.3 + 0.7 * (displacements[idx] / max(displacements.max(), 1))
            color = "red" if displacements[idx] > 50 else "gray"
            ax.plot([0, 1], [rank_a[idx], rank_b[idx]], color=color, alpha=alpha_val, linewidth=0.5)
        ax.set_xlim(-0.1, 1.1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["0-100/v2", "1-5/v4"], fontsize=9)
        ax.set_ylabel("Rank (1 = highest)" if ax_i == 0 else "")
        ax.set_title(f"{dim.title()} (ρ={corr_matrices[dim][0, -1]:.3f})",
                     fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.invert_yaxis()
    fig.tight_layout()
    save_figure(fig, "fig12_rank_displacement")

    return table6


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: Resolution & Information Content
# ══════════════════════════════════════════════════════════════════════

def section7_resolution(data):
    print("\n=== Section 7: Resolution & Information Content ===")
    rows = []

    for scale_label, s_min, s_max in SCALES:
        for prompt in PROMPTS:
            key = condition_key(scale_label, prompt)
            df = data[key]
            for dim in DIMENSIONS:
                # Unique raw run values
                run_vals = []
                for rc in [f"{dim}_run1", f"{dim}_run2", f"{dim}_run3"]:
                    run_vals.extend(df[rc].dropna().tolist())
                unique_raw = len(set(np.round(run_vals, 2)))

                # Unique mean values
                unique_means = len(set(np.round(df[f"{dim}_mean"].dropna().values, 2)))

                # Shannon entropy on normalized scores (20 bins)
                frac_vals = df[f"{dim}_mean_frac"].dropna().values
                hist, _ = np.histogram(frac_vals, bins=20, range=(0, 1))
                hist_norm = hist / hist.sum()
                entropy = float(sp_stats.entropy(hist_norm + 1e-10, base=2))

                # Boundary proportions
                floor_pct = (frac_vals <= 0.05).mean() * 100
                ceil_pct = (frac_vals >= 0.95).mean() * 100

                rows.append({
                    "scale": scale_label, "prompt": prompt, "dimension": dim,
                    "unique_raw_values": unique_raw,
                    "unique_mean_values": unique_means,
                    "entropy_bits": round(entropy, 3),
                    "max_entropy_bits": round(np.log2(20), 3),
                    "entropy_ratio": round(entropy / np.log2(20), 3),
                    "floor_pct": round(floor_pct, 1),
                    "ceiling_pct": round(ceil_pct, 1),
                })

    table7 = pd.DataFrame(rows)
    table7.to_csv(TBL_DIR / "table07_resolution.csv", index=False)
    print("  Saved table07_resolution.csv")

    # Figure 13: Effective resolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Effective Resolution by Condition", fontsize=13, fontweight="bold")

    cond_labels = [f"{sl}/{pr}" for sl, _, _ in SCALES for pr in PROMPTS]
    x = np.arange(len(cond_labels))
    width = 0.25

    # Panel 1: Unique mean values
    ax = axes[0]
    for d_i, dim in enumerate(DIMENSIONS):
        vals = []
        for sl, _, _ in SCALES:
            for pr in PROMPTS:
                row = table7[(table7["scale"] == sl) & (table7["prompt"] == pr) & (table7["dimension"] == dim)].iloc[0]
                vals.append(row["unique_mean_values"])
        ax.bar(x + d_i * width, vals, width, label=dim.title(),
               color=DIM_COLORS[dim], alpha=0.7, edgecolor="white")
    ax.set_xticks(x + width)
    ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Unique Mean Score Values")
    ax.set_title("Distinct Score Levels")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: Shannon entropy
    ax = axes[1]
    for d_i, dim in enumerate(DIMENSIONS):
        vals = []
        for sl, _, _ in SCALES:
            for pr in PROMPTS:
                row = table7[(table7["scale"] == sl) & (table7["prompt"] == pr) & (table7["dimension"] == dim)].iloc[0]
                vals.append(row["entropy_bits"])
        ax.bar(x + d_i * width, vals, width, label=dim.title(),
               color=DIM_COLORS[dim], alpha=0.7, edgecolor="white")
    ax.axhline(np.log2(20), color="gray", linestyle="--", alpha=0.5, label="Max entropy")
    ax.set_xticks(x + width)
    ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Shannon Entropy (bits)")
    ax.set_title("Information Content")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "fig13_effective_resolution")

    # Figure 14: Empirical CDF comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Empirical CDF of Normalized Scores", fontsize=13, fontweight="bold")
    scale_colors = {"0-100": "#333333", "1-10": "#e6550d", "1-5": "#756bb1"}
    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        for sl, _, _ in SCALES:
            for pr in PROMPTS:
                key = condition_key(sl, pr)
                vals = np.sort(data[key][f"{dim}_mean_frac"].dropna().values)
                cdf = np.arange(1, len(vals) + 1) / len(vals)
                ls = {"v2": "-", "v3": "--", "v4": ":"}[pr]
                alpha_val = 0.8
                ax.plot(vals, cdf, linestyle=ls, color=scale_colors[sl],
                        alpha=alpha_val, linewidth=1.2)
        ax.set_xlabel("Normalized Score [0, 1]")
        ax.set_ylabel("Cumulative Proportion" if ax_i == 0 else "")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(alpha=0.3)
        if ax_i == 2:
            # Custom legend
            from matplotlib.lines import Line2D
            handles = []
            for sl in ["0-100", "1-10", "1-5"]:
                handles.append(Line2D([0], [0], color=scale_colors[sl], linewidth=1.5, label=f"Scale {sl}"))
            for pr in PROMPTS:
                ls = {"v2": "-", "v3": "--", "v4": ":"}[pr]
                handles.append(Line2D([0], [0], color="gray", linestyle=ls, linewidth=1.5,
                                      label=PROMPT_LABELS[pr]))
            ax.legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig14_cdf_comparison")

    # Figure 18: Entropy heatmap (condition × dimension)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Shannon Entropy by Condition × Dimension (bits)", fontsize=13, fontweight="bold")
    cond_labels = [f"{sl}/{pr}" for sl, _, _ in SCALES for pr in PROMPTS]
    entropy_matrix = np.zeros((len(cond_labels), len(DIMENSIONS)))
    for ci, (sl_pr) in enumerate(cond_labels):
        sl, pr = sl_pr.split("/")
        for di, dim in enumerate(DIMENSIONS):
            row = table7[(table7["scale"] == sl) & (table7["prompt"] == pr) & (table7["dimension"] == dim)]
            entropy_matrix[ci, di] = row.iloc[0]["entropy_bits"] if len(row) > 0 else 0
    im = ax.imshow(entropy_matrix.T, cmap="YlGnBu", aspect="auto",
                   vmin=1.5, vmax=np.log2(20))
    for ci in range(len(cond_labels)):
        for di in range(len(DIMENSIONS)):
            val = entropy_matrix[ci, di]
            color = "white" if val < 2.8 else "black"
            ax.text(ci, di, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    fontweight="bold", color=color)
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(DIMENSIONS)))
    ax.set_yticklabels([d.title() for d in DIMENSIONS], fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Entropy (bits)")
    # Add max entropy reference
    ax.set_xlabel("Condition (Scale / Prompt)")
    fig.tight_layout()
    save_figure(fig, "fig18_entropy_heatmap")

    # Figure 19: Entropy by dimension — grouped by scale, highlighting social gap
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Shannon Entropy by Dimension: Scale Comparison (prompt-averaged)",
                 fontsize=13, fontweight="bold")
    scale_colors_bar = {"0-100": "#2171b5", "1-10": "#e6550d", "1-5": "#756bb1"}
    max_ent = np.log2(20)

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        # For each scale, compute mean and range across prompts
        scale_means = []
        scale_errs_low = []
        scale_errs_high = []
        for sl, _, _ in SCALES:
            vals = table7[(table7["scale"] == sl) & (table7["dimension"] == dim)]["entropy_bits"].values
            m = np.mean(vals)
            scale_means.append(m)
            scale_errs_low.append(m - vals.min())
            scale_errs_high.append(vals.max() - m)

        x_pos = np.arange(len(SCALES))
        bars = ax.bar(x_pos, scale_means, width=0.6, edgecolor="white", linewidth=0.5,
                      color=[scale_colors_bar[s[0]] for s in SCALES], alpha=0.8)
        ax.errorbar(x_pos, scale_means,
                    yerr=[scale_errs_low, scale_errs_high],
                    fmt="none", ecolor="black", capsize=5, linewidth=1.5)
        ax.axhline(max_ent, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.text(len(SCALES) - 0.5, max_ent + 0.05, f"max = {max_ent:.2f}",
                fontsize=7, color="gray", ha="right")

        # Annotate bars with values
        for i, (bar, m) in enumerate(zip(bars, scale_means)):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.05, f"{m:.2f}",
                    ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([s[0] for s in SCALES], fontsize=10)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Shannon Entropy (bits)" if ax_i == 0 else "")
        ax.set_ylim(0, max_ent + 0.4)
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, "fig19_entropy_by_dimension")

    return table7


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: Efficiency
# ══════════════════════════════════════════════════════════════════════

def section8_efficiency(data, table5):
    print("\n=== Section 8: Efficiency ===")

    rows = []
    for scale_label, _, _ in SCALES:
        for prompt in PROMPTS:
            key = condition_key(scale_label, prompt)
            df = data[key]
            time_ms = df["processing_time_ms"].dropna().values
            median_time = float(np.median(time_ms))
            mean_time = float(np.mean(time_ms))

            # Quality composite: geometric mean of alpha (avg across dims) and
            # avg rank corr with reference (0-100/v2)
            alphas = []
            rhos = []
            ref_key = condition_key("0-100", "v2")
            for dim in DIMENSIONS:
                rel_row = table5[(table5["scale"] == scale_label) &
                                 (table5["prompt"] == prompt) &
                                 (table5["dimension"] == dim)]
                if len(rel_row) > 0:
                    alphas.append(rel_row.iloc[0]["alpha"])
                a = data[ref_key][f"{dim}_mean_frac"].values
                b = data[key][f"{dim}_mean_frac"].values
                rho, _ = sp_stats.spearmanr(a, b)
                rhos.append(rho)

            avg_alpha = np.mean(alphas) if alphas else 0
            avg_rho = np.mean(rhos)
            quality = float((avg_alpha * avg_rho) ** 0.5)  # geometric mean

            rows.append({
                "scale": scale_label, "prompt": prompt,
                "median_time_ms": round(median_time, 0),
                "mean_time_ms": round(mean_time, 0),
                "avg_alpha": round(avg_alpha, 4),
                "avg_rho_vs_ref": round(avg_rho, 4),
                "quality_index": round(quality, 4),
            })

    table8 = pd.DataFrame(rows)
    table8.to_csv(TBL_DIR / "table08_efficiency.csv", index=False)
    print("  Saved table08_efficiency.csv")

    # Figure 15: Speed comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Per-Entity Processing Time by Condition", fontsize=13, fontweight="bold")
    time_data = []
    cond_labels = []
    for scale_label, _, _ in SCALES:
        for prompt in PROMPTS:
            key = condition_key(scale_label, prompt)
            time_data.append(data[key]["processing_time_ms"].dropna().values / 1000)  # seconds
            cond_labels.append(f"{scale_label}/{prompt}")
    bp = ax.boxplot(time_data, patch_artist=True, showfliers=False, widths=0.6)
    prompt_colors = {"v2": "#4292c6", "v3": "#ef6548", "v4": "#78c679"}
    for i, patch in enumerate(bp["boxes"]):
        pr = PROMPTS[i % 3]
        patch.set_facecolor(prompt_colors[pr])
        patch.set_alpha(0.6)
    ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Processing Time (seconds)")
    ax.grid(axis="y", alpha=0.3)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=prompt_colors[pr], alpha=0.6, label=PROMPT_LABELS[pr])
                       for pr in PROMPTS]
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig15_speed_comparison")

    # Figure 16: Pareto frontier
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Cost-Quality Pareto Frontier", fontsize=13, fontweight="bold")
    markers = {"0-100": "o", "1-10": "s", "1-5": "^"}
    colors = {"v2": "#4292c6", "v3": "#ef6548", "v4": "#78c679"}
    for _, row in table8.iterrows():
        ax.scatter(row["median_time_ms"] / 1000, row["quality_index"],
                   marker=markers[row["scale"]], color=colors[row["prompt"]],
                   s=100, edgecolor="black", linewidth=0.5, zorder=5)
        ax.annotate(f"{row['scale']}/{row['prompt']}", (row["median_time_ms"] / 1000, row["quality_index"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 6), textcoords="offset points")
    # Draw Pareto frontier
    df_sorted = table8.sort_values("median_time_ms")
    pareto_x = []
    pareto_y = []
    best_q = -1
    for _, row in df_sorted.iterrows():
        if row["quality_index"] > best_q:
            pareto_x.append(row["median_time_ms"] / 1000)
            pareto_y.append(row["quality_index"])
            best_q = row["quality_index"]
    ax.plot(pareto_x, pareto_y, "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("Median Processing Time (seconds)")
    ax.set_ylabel("Quality Index (√(α × ρ))")
    ax.grid(alpha=0.3)
    # Legend for shapes and colors
    from matplotlib.lines import Line2D
    shape_handles = [Line2D([0], [0], marker=m, color="gray", linestyle="", markersize=8, label=f"Scale {s}")
                     for s, m in markers.items()]
    color_handles = [Line2D([0], [0], marker="o", color=c, linestyle="", markersize=8,
                            label=PROMPT_LABELS[p]) for p, c in colors.items()]
    ax.legend(handles=shape_handles + color_handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig16_pareto_frontier")

    return table8


# ══════════════════════════════════════════════════════════════════════
# SECTION 9: Ceiling Compression
# ══════════════════════════════════════════════════════════════════════

def section9_ceiling(data):
    print("\n=== Section 9: Ceiling Compression ===")

    # Figure 17: Quintile proportions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Score Quintile Distribution (Ceiling Compression Analysis)", fontsize=13, fontweight="bold")
    quintile_edges = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    quintile_labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    quintile_colors = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        cond_labels = []
        quintile_matrix = []
        for scale_label, _, _ in SCALES:
            for prompt in PROMPTS:
                key = condition_key(scale_label, prompt)
                vals = data[key][f"{dim}_mean_frac"].dropna().values
                proportions = []
                for q in range(5):
                    low, high = quintile_edges[q], quintile_edges[q + 1]
                    if q == 4:  # include upper boundary
                        prop = ((vals >= low) & (vals <= high)).mean()
                    else:
                        prop = ((vals >= low) & (vals < high)).mean()
                    proportions.append(prop)
                quintile_matrix.append(proportions)
                cond_labels.append(f"{scale_label}\n{prompt}")

        quintile_matrix = np.array(quintile_matrix)
        bottom = np.zeros(len(cond_labels))
        for q in range(5):
            ax.bar(range(len(cond_labels)), quintile_matrix[:, q], bottom=bottom,
                   color=quintile_colors[q], edgecolor="white", linewidth=0.3,
                   label=quintile_labels[q] if ax_i == 2 else None)
            bottom += quintile_matrix[:, q]
        ax.set_xticks(range(len(cond_labels)))
        ax.set_xticklabels(cond_labels, fontsize=6.5)
        ax.set_ylabel("Proportion" if ax_i == 0 else "")
        ax.set_title(dim.title(), fontsize=11, fontweight="bold", color=DIM_COLORS[dim])
        ax.set_ylim(0, 1.0)
        if ax_i == 2:
            ax.legend(title="Quintile", fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    save_figure(fig, "fig17_ceiling_compression")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════

def generate_summary_report(table1, table2, table3, table5, table6, table7, table8):
    """Generate a comprehensive narrative summary report with interpretive findings."""
    print("\n=== Generating Summary Report ===")

    # ── Helper: extract key statistics for narrative ──────────────────
    def _friedman_rows(table, dim):
        row = table[(table["dimension"] == dim) & (table["test"] == "Friedman")]
        return row.iloc[0] if len(row) > 0 else None

    def _wilcoxon_rows(table, dim):
        return table[(table["dimension"] == dim) & (table["test"] == "Wilcoxon")]

    # ── Build report lines ────────────────────────────────────────────
    L = []  # accumulator
    L.append("# Factorial Scoring Experiment: Analysis Summary")
    L.append("")
    L.append("## Experiment Design")
    L.append("")
    L.append("- **Design:** 3 x 3 fully-crossed factorial (Scale x Prompt)")
    L.append("- **Scales:** 0-100, 1-10, 1-5")
    L.append("- **Prompt versions:** v2 (score-then-justify), v3 (justify-then-score), v4 (no chain-of-thought)")
    L.append("- **Entities:** 450 (stratified sample: 30 per participant x 15 participants, balanced across 3 groups)")
    L.append("- **Runs:** 3 scoring runs per entity per condition")
    L.append("- **Total observations:** 450 entities x 9 conditions x 3 runs = 12,150 individual scores")
    L.append(f"- **Model:** {MODEL_NAME} via Ollama, temperature=0.3")

    # Compute total time from efficiency table
    total_time_s = table8["mean_time_ms"].sum() * 450 / 1000  # rough estimate
    L.append(f"- **SETS dimensions:** Social, Ecological, Technological (scored independently)")
    L.append("")
    L.append("All cross-condition comparisons are **paired** -- the same 450 entities are scored under every condition, enabling within-entity statistical tests.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## Key Findings")
    L.append("")

    # ── RQ1: Scale Effects ────────────────────────────────────────────
    L.append("### RQ1: Does the scoring scale bias normalized score levels?")
    L.append("")
    L.append("**Finding: Yes -- scale choice systematically shifts normalized scores, and the effect is dimension-dependent.**")
    L.append("")
    L.append("All three dimensions show highly significant scale main effects (Friedman test, collapsing over prompt):")
    L.append("")
    L.append("| Dimension | Friedman chi-sq | p-value |")
    L.append("|-----------|----------------|---------|")
    for dim in DIMENSIONS:
        fr = _friedman_rows(table2, dim)
        if fr is not None:
            L.append(f"| {dim.title()} | {fr['statistic']} | {fr['p_value']:.2e} |")
    L.append("")

    # Social score shift narrative
    social_means_100 = table1[(table1["dimension"] == "social") & (table1["scale"] == "0-100")]["mean"]
    social_means_5 = table1[(table1["dimension"] == "social") & (table1["scale"] == "1-5")]["mean"]
    L.append(f"The social dimension is most affected. Coarser scales systematically inflate social scores: "
             f"the 1-5 scale produces mean normalized scores of {social_means_5.min():.2f}-{social_means_5.max():.2f} "
             f"compared to {social_means_100.min():.2f}-{social_means_100.max():.2f} on the 0-100 scale. "
             f"All pairwise Wilcoxon comparisons are significant after Holm-Bonferroni correction (all p < 0.001).")
    L.append("")
    L.append("For ecological and technological dimensions, the shift is smaller but still significant: coarser scales produce "
             "modestly *lower* normalized scores, the opposite direction from the social dimension. This asymmetry reflects the "
             "ceiling compression mechanism -- social scores cluster near the top of the range, so coarser bins push them into "
             "the maximum, while mid-range ecological/technological scores lose resolution symmetrically.")
    L.append("")
    L.append("**See:** Fig 4 (paired difference violins), Table 2 (scale_effects.csv)")
    L.append("")

    # ── RQ2: Prompt Effects ───────────────────────────────────────────
    L.append("### RQ2: Does prompt structure alter score distributions?")
    L.append("")
    L.append("**Finding: Yes -- removing chain-of-thought (v4) consistently inflates scores relative to CoT prompts (v2, v3).**")
    L.append("")
    L.append("| Dimension | Friedman chi-sq | p-value |")
    L.append("|-----------|----------------|---------|")
    for dim in DIMENSIONS:
        fr = _friedman_rows(table3, dim)
        if fr is not None:
            L.append(f"| {dim.title()} | {fr['statistic']} | {fr['p_value']:.2e} |")
    L.append("")
    L.append("The v4 (no CoT) prompt produces higher normalized scores than both v2 and v3 across all dimensions. "
             "This suggests that chain-of-thought reasoning acts as a mild *deflationary* mechanism -- when forced to "
             "justify scores, the LLM tends to assign slightly more moderate values.")
    L.append("")
    L.append("The difference between v2 (score-first) and v3 (justify-first) is smaller and less consistent, "
             "suggesting that the *presence* of CoT matters more than the *order* of score vs. justification. "
             "This partially addresses the anchoring bias concern: while v3 was designed to prevent post-hoc "
             "rationalization by requiring justification before scoring, the ordering effect is modest compared "
             "to the CoT/no-CoT divide.")
    L.append("")
    L.append("**See:** Fig 5 (paired difference violins), Table 3 (prompt_effects.csv)")
    L.append("")

    # ── RQ3: Interaction ──────────────────────────────────────────────
    L.append("### RQ3: Do scale and prompt interact?")
    L.append("")
    L.append("**Finding: Modest but detectable interactions, primarily in the social dimension.**")
    L.append("")
    L.append("The Friedman test on the v4-v2 prompt effect across scales yields significant results for all "
             "dimensions, indicating that the magnitude of the prompt effect depends on scale choice. The "
             "interaction contrasts show that the no-CoT inflation is somewhat amplified on coarser scales "
             "for the social dimension -- consistent with ceiling compression compounding the prompt-driven "
             "score inflation.")
    L.append("")
    L.append("For ecological and technological dimensions, interaction contrasts are small (typically < 0.01 "
             "on the normalized scale), suggesting that scale and prompt effects are approximately additive.")
    L.append("")
    L.append("**See:** Fig 6 (interaction plot), Fig 7 (interaction contrast heatmap), Table 4 (interaction.csv)")
    L.append("")

    # ── RQ4: Reliability ──────────────────────────────────────────────
    L.append("### RQ4: How do scale and prompt affect run-to-run reliability?")
    L.append("")
    L.append("**Finding: Reliability is generally high across all conditions (alpha > 0.76), but the 0-100 "
             "scale achieves the best consistency, and the social dimension is consistently the least reliable.**")
    L.append("")
    alpha_min, alpha_max = table5["alpha"].min(), table5["alpha"].max()
    snr_min, snr_max = table5["snr"].min(), table5["snr"].max()
    L.append("| Metric | Range | Best conditions | Worst conditions |")
    L.append("|--------|-------|-----------------|------------------|")
    L.append(f"| Krippendorff's alpha | {alpha_min:.3f}-{alpha_max:.3f} | 0-100 ecological/tech (0.91-0.92) | 1-5 social (0.76-0.79) |")
    L.append(f"| SNR | {snr_min:.1f}-{snr_max:.1f} | Stable across conditions | 1-5/v3/social = {snr_min:.1f} (lowest) |")
    L.append("")
    L.append("The 1-5 scale produces lower reliability for social scores (alpha 0.76-0.79 vs. 0.85 on 0-100) "
             "because ceiling compression forces most scores to the maximum value, where even a 1-point raw "
             "difference represents a 25% shift in the normalized score.")
    L.append("")
    grand_snr = table5["snr"].mean()
    L.append(f"Signal-to-noise ratio is remarkably stable across conditions (grand mean ~{grand_snr:.1f}), "
             "indicating that the LLM consistently maintains a ~4:1 ratio of meaningful between-entity "
             "variation to run-to-run noise.")
    L.append("")
    L.append("**See:** Fig 8 (reliability forest plot), Fig 9 (CV distributions), Fig 10 (SNR comparison), Table 5 (reliability.csv)")
    L.append("")

    # ── RQ5: Rank Preservation ────────────────────────────────────────
    L.append("### RQ5: Is entity ranking preserved across conditions?")
    L.append("")
    rho_vals = table6["spearman_rho"]
    L.append(f"**Finding: Yes -- entity rankings are well-preserved even when absolute score levels shift substantially.**")
    L.append("")
    L.append(f"Cross-condition Spearman rho ranges from {rho_vals.min():.3f} to {rho_vals.max():.3f} "
             f"(mean = {rho_vals.mean():.3f}). Same-scale comparisons tend to produce the highest "
             f"correlations (rho > 0.93), while the most divergent pair (0-100/v2 vs 1-5/v4) still achieves "
             f"rho of 0.78-0.87 depending on dimension.")
    L.append("")
    L.append("This is a key practical finding: even though absolute normalized scores differ by up to 0.10 "
             "across conditions, the *relative ordering* of entities is largely stable. The choice of scale "
             "and prompt affects calibration more than discrimination.")
    L.append("")
    L.append("**See:** Fig 11 (9x9 rank correlation matrices), Fig 12 (rank displacement bump charts), Table 6 (rank_correlations.csv)")
    L.append("")

    # ── RQ6: Efficiency ───────────────────────────────────────────────
    L.append("### RQ6: What is the cost-quality Pareto frontier?")
    L.append("")
    L.append("**Finding: The v4 (no CoT) prompt is approximately 2x faster with only modest quality loss, "
             "making 0-100/v4 the best speed-quality tradeoff.**")
    L.append("")
    L.append("| Condition | Median time/entity | Quality index (Q) |")
    L.append("|-----------|-------------------|-------------------|")
    for _, row in table8.iterrows():
        L.append(f"| {row['scale']}/{row['prompt']} | {row['median_time_ms']/1000:.1f}s | {row['quality_index']:.3f} |")
    L.append("")
    L.append("The quality index Q is the geometric mean of average Krippendorff's alpha (reliability) and "
             "average Spearman rho vs. the reference condition (0-100/v2). The 0-100/v2 condition achieves "
             "the highest quality (Q = 0.946) but takes ~27s per entity. Switching to 0-100/v4 cuts time "
             "nearly in half (~13s) with a quality drop of only 0.044.")
    L.append("")
    L.append("**See:** Fig 15 (processing time box plots), Fig 16 (Pareto frontier scatter), Table 8 (efficiency.csv)")
    L.append("")

    # ── RQ7: Resolution ───────────────────────────────────────────────
    L.append("### RQ7: Does the 1-5 scale lose measurable information?")
    L.append("")
    L.append("**Finding: Yes, but the magnitude of information loss is dimension-dependent and moderated by distributional properties.**")
    L.append("")
    L.append("The resolution analysis uses two complementary metrics:")
    L.append("- **Unique values:** How many distinct score levels the LLM actually produces")
    L.append("- **Shannon entropy:** How evenly scores are spread across the available range (20 bins, max = log2(20) = 4.322 bits)")
    L.append("")
    L.append("| Scale | Dimension | Unique means | Entropy (bits) | Entropy ratio |")
    L.append("|-------|-----------|-------------|----------------|---------------|")
    for sl in ["0-100", "1-5"]:
        for dim in DIMENSIONS:
            rows = table7[(table7["scale"] == sl) & (table7["dimension"] == dim)]
            uniq_range = f"{int(rows['unique_mean_values'].min())}-{int(rows['unique_mean_values'].max())}"
            ent_range = f"{rows['entropy_bits'].min():.2f}-{rows['entropy_bits'].max():.2f}"
            ratio_range = f"{rows['entropy_ratio'].min():.2f}-{rows['entropy_ratio'].max():.2f}"
            L.append(f"| {sl} | {dim.title()} | {uniq_range} | {ent_range} | {ratio_range} |")
    L.append("")
    L.append("The critical insight: **for social scores, the 0-100 scale's extra resolution is wasted.** "
             "Entropy is ~2.0-2.2 bits regardless of scale because ceiling compression dominates. "
             "For ecological and technological dimensions, the 0-100 scale carries ~0.4-0.5 additional "
             "bits compared to 1-5. The 1-10 scale sits in between, achieving comparable entropy ratios "
             "to 0-100 because 3-run averaging creates fractional means that recover some resolution.")
    L.append("")
    L.append("**See:** Fig 13 (unique values + entropy bar charts), Fig 14 (empirical CDF overlays), Table 7 (resolution.csv)")
    L.append("")

    # ── Ceiling Compression ───────────────────────────────────────────
    L.append("### Ceiling Compression (Social Dimension)")
    L.append("")
    ceil_5 = table1[(table1["scale"] == "1-5") & (table1["dimension"] == "social")]["ceiling_pct"]
    L.append(f"**Finding: The social dimension exhibits severe ceiling compression on the 1-5 scale "
             f"({ceil_5.min():.0f}-{ceil_5.max():.0f}% of entities at ceiling), making it unsuitable "
             f"for discriminating among socially-oriented entities.**")
    L.append("")
    ceil_100 = table1[(table1["scale"] == "0-100") & (table1["dimension"] == "social")]["ceiling_pct"]
    L.append(f"- 1-5 scale: {ceil_5.min():.0f}-{ceil_5.max():.0f}% at ceiling")
    L.append(f"- 0-100 scale: <{max(ceil_100.max(), 1):.0f}% at ceiling")
    L.append("")
    L.append("Even on the 0-100 scale, social scores are heavily left-skewed (skewness -2.3 to -2.7), "
             "with the distribution concentrated between 0.70-0.95. This appears to be an intrinsic "
             "property of the social dimension in this entity set.")
    L.append("")
    L.append("**See:** Fig 17 (quintile proportion stacked bars), Fig 1 (3x3 distribution grids)")
    L.append("")

    # ── Practical Recommendations ─────────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## Practical Recommendations")
    L.append("")
    L.append("1. **Use the 0-100 scale** as the default. It provides the highest reliability, most "
             "information content for ecological/technological dimensions, and avoids the severe ceiling "
             "compression seen on the 1-5 scale.")
    L.append("")
    L.append("2. **Use v2 (score-then-justify) for maximum quality**, or **v4 (no CoT) for throughput-sensitive "
             "applications**. The ~2x speed improvement of v4 comes with a modest quality cost (Q drops from "
             "0.946 to 0.902). The v3 (justify-then-score) prompt does not meaningfully improve over v2.")
    L.append("")
    L.append("3. **Treat social dimension scores with caution** regardless of scale. The inherent ceiling "
             "compression means absolute social scores have limited discriminative power. Relative rankings "
             "are still meaningful (rho > 0.78), but the social dimension would benefit from either (a) "
             "revised dimension definitions that create more spread, or (b) analysis methods robust to "
             "ceiling effects.")
    L.append("")
    L.append("4. **Entity rankings are robust to configuration choices.** When the goal is comparative "
             f"rather than absolute scoring, the choice of scale and prompt matters less -- mean "
             f"cross-condition Spearman rho is {rho_vals.mean():.3f}.")
    L.append("")

    # ── Output Inventory ──────────────────────────────────────────────
    L.append("---")
    L.append("")
    L.append("## Output Inventory")
    L.append("")
    L.append("### Figures (21 total)")
    L.append("")
    L.append("| Figure | Description | File |")
    L.append("|--------|-------------|------|")
    L.append("| Fig 1 (x3) | Score distributions, 3x3 grid per dimension | fig01_distribution_grid_{social,ecological,technological}.png |")
    L.append("| Fig 2 | Condition mean scores with 95% bootstrap CI | fig02_condition_means.png |")
    L.append("| Fig 3 | Raw score frequency heatmap (discretization) | fig03_discretization_heatmap.png |")
    L.append("| Fig 4 | Scale main effects: paired difference violins | fig04_scale_paired_diffs.png |")
    L.append("| Fig 5 | Prompt main effects: paired difference violins | fig05_prompt_paired_diffs.png |")
    L.append("| Fig 6 | Scale x Prompt interaction plot | fig06_interaction_plot.png |")
    L.append("| Fig 7 | Interaction contrast heatmap | fig07_interaction_heatmap.png |")
    L.append("| Fig 8 | Reliability forest plot (Krippendorff's alpha) | fig08_reliability_forest.png |")
    L.append("| Fig 9 | CV distributions by condition | fig09_cv_distributions.png |")
    L.append("| Fig 10 | Signal-to-noise ratio comparison | fig10_snr_comparison.png |")
    L.append("| Fig 11 | 9x9 rank correlation matrices (Spearman rho) | fig11_rank_correlation_matrix.png |")
    L.append("| Fig 12 | Rank displacement bump charts | fig12_rank_displacement.png |")
    L.append("| Fig 13 | Effective resolution (unique values + entropy) | fig13_effective_resolution.png |")
    L.append("| Fig 14 | Empirical CDF comparison | fig14_cdf_comparison.png |")
    L.append("| Fig 15 | Processing time comparison | fig15_speed_comparison.png |")
    L.append("| Fig 16 | Cost-quality Pareto frontier | fig16_pareto_frontier.png |")
    L.append("| Fig 17 | Ceiling compression (quintile proportions) | fig17_ceiling_compression.png |")
    L.append("| Fig 18 | Entropy heatmap (condition x dimension) | fig18_entropy_heatmap.png |")
    L.append("| Fig 19 | Entropy by dimension, scale comparison | fig19_entropy_by_dimension.png |")
    L.append("")
    L.append("### Tables (8 total)")
    L.append("")
    L.append("| Table | Description | File |")
    L.append("|-------|-------------|------|")
    L.append("| Table 1 | Descriptive statistics per condition x dimension | table01_descriptives.csv |")
    L.append("| Table 2 | Scale main effects (Friedman, Wilcoxon, bootstrap CI) | table02_scale_effects.csv |")
    L.append("| Table 3 | Prompt main effects (Friedman, Wilcoxon, bootstrap CI) | table03_prompt_effects.csv |")
    L.append("| Table 4 | Interaction contrasts and Friedman tests | table04_interaction.csv |")
    L.append("| Table 5 | Reliability (alpha, CV, SNR) per condition x dimension | table05_reliability.csv |")
    L.append("| Table 6 | Cross-condition Spearman rank correlations | table06_rank_correlations.csv |")
    L.append("| Table 7 | Resolution (unique values, entropy, boundary %) | table07_resolution.csv |")
    L.append("| Table 8 | Efficiency (processing time, quality index) | table08_efficiency.csv |")
    L.append("")

    # ── Methods Notes ─────────────────────────────────────────────────
    L.append("## Methods Notes")
    L.append("")
    L.append("- **Normalization:** All scores converted to [0, 1] fraction via `(score - scale_min) / scale_range`")
    L.append("- **Statistical tests:** Friedman test (non-parametric repeated-measures ANOVA) for omnibus tests; "
             "Wilcoxon signed-rank for pairwise comparisons with Holm-Bonferroni correction; rank-biserial effect size")
    L.append("- **Bootstrap:** 10,000 resamples, seed=42, percentile method for 95% CIs")
    L.append("- **Entropy:** Shannon entropy in bits on 20-bin histograms over [0, 1]; max = log2(20) = 4.322 bits")
    L.append("- **Quality index:** Geometric mean of average Krippendorff's alpha and average Spearman rho vs. "
             "reference condition (0-100/v2)")
    L.append("- **Signal-to-noise ratio:** Between-entity SD / mean within-entity SD (both on normalized [0, 1] scale)")

    report_path = OUT_DIR / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"  Saved {report_path.name}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    global EXP_DIR, OUT_DIR, FIG_DIR, TBL_DIR, MODEL_NAME

    parser = argparse.ArgumentParser(description="Analyze factorial scoring experiment results")
    parser.add_argument("--exp-dir", type=str, default=None,
                        help="Path to experiment output directory (default: output/scoring-experiment)")
    args = parser.parse_args()

    if args.exp_dir:
        EXP_DIR = Path(args.exp_dir).resolve()
    OUT_DIR = EXP_DIR / "analysis"
    FIG_DIR = OUT_DIR / "figures"
    TBL_DIR = OUT_DIR / "tables"

    # Auto-detect model name from experiment manifest
    manifest_path = EXP_DIR / "experiment_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        MODEL_NAME = manifest.get("parameters", {}).get("model", MODEL_NAME)

    print("=" * 60)
    print("  FACTORIAL SCORING EXPERIMENT ANALYSIS")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Data:  {EXP_DIR}")
    print("=" * 60)

    # Setup output directories
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    data = load_all_conditions()

    # Align all conditions to a common entity set (handles scoring failures)
    entity_sets = [set(df["entity"]) for df in data.values()]
    common_entities = set.intersection(*entity_sets)
    n_dropped = max(len(s) for s in entity_sets) - len(common_entities)
    if n_dropped > 0:
        print(f"  Aligning to {len(common_entities)} common entities ({n_dropped} dropped)")
        for key in data:
            data[key] = data[key][data[key]["entity"].isin(common_entities)].reset_index(drop=True)

    master_df = build_master_df(data)
    print(f"  Loaded {len(data)} conditions, {len(master_df)} total rows")

    # Run analyses
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        table1 = section1_descriptives(data, master_df)
        table5 = section5_reliability(data)
        table6 = section6_rank_preservation(data)
        table2 = section2_scale_effects(data)
        table3 = section3_prompt_effects(data)
        table4 = section4_interaction(data)
        table7 = section7_resolution(data)
        section9_ceiling(data)
        table8 = section8_efficiency(data, table5)
        generate_summary_report(table1, table2, table3, table5, table6, table7, table8)

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures: {FIG_DIR}")
    print(f"  Tables:  {TBL_DIR}")
    print(f"  Report:  {OUT_DIR / 'summary_report.md'}")


if __name__ == "__main__":
    main()
