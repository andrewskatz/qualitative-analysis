#!/usr/bin/env python3
"""
Analysis of temperature sensitivity experiment.

Compares reliability metrics (Krippendorff's alpha, CV, rank stability)
across temperatures for each model, producing publication-ready figures
and tables that address the reviewer question: "How much does sampling
temperature affect scoring reliability?"

Usage:
    python scripts/analyze_temperature_experiment.py
    python scripts/analyze_temperature_experiment.py --exp-dir output/temperature-experiment
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Configuration ───────────────────────────────────────────────────
DIMENSIONS = ["social", "ecological", "technological"]
DIM_COLORS = {
    "social": "#1f77b4",
    "ecological": "#2ca02c",
    "technological": "#d62728",
}
DPI = 150
N_BOOT = 10000
RNG_SEED = 42


def save_figure(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Bootstrap helpers ───────────────────────────────────────────────

def bootstrap_ci(arr, n_boot=N_BOOT, seed=RNG_SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr)
    n = len(arr)
    boot = np.array([np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(boot, alpha * 100)), float(np.percentile(boot, (1 - alpha) * 100))


def krippendorff_alpha(runs_matrix):
    """Compute Krippendorff's alpha for interval data from an N×R matrix (N entities, R runs)."""
    D = np.asarray(runs_matrix, dtype=float)
    mask = ~np.isnan(D)
    n_items, n_raters = D.shape

    # Observed disagreement
    Do = 0.0
    count_o = 0
    for i in range(n_items):
        valid = D[i, mask[i]]
        m = len(valid)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                Do += (valid[a] - valid[b]) ** 2
                count_o += 1
    if count_o == 0:
        return np.nan
    Do /= count_o

    # Expected disagreement
    all_vals = D[mask]
    n_total = len(all_vals)
    De = 0.0
    count_e = 0
    # For efficiency, use variance-based formula: De = var(all_vals) * (n_total) / (n_total - 1)
    De = float(np.var(all_vals, ddof=0) * n_total / (n_total - 1))

    if De == 0:
        return 1.0
    return 1.0 - Do / De


def bootstrap_alpha_ci(runs_matrix, n_boot=2000, seed=RNG_SEED, ci=0.95):
    """Bootstrap CI for Krippendorff's alpha."""
    rng = np.random.default_rng(seed)
    D = np.asarray(runs_matrix, dtype=float)
    n_items = D.shape[0]
    alphas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        alphas.append(krippendorff_alpha(D[idx]))
    alphas = np.array(alphas)
    a = (1 - ci) / 2
    return float(np.percentile(alphas, a * 100)), float(np.percentile(alphas, (1 - a) * 100))


# ── Data Loading ────────────────────────────────────────────────────

def load_experiment(exp_dir: Path):
    """Load all condition CSVs and metadata."""
    manifest_path = exp_dir / "experiment_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    models = manifest["design"]["models"]
    temperatures = manifest["design"]["temperatures"]

    data = {}  # key: (model_short, temp) -> DataFrame
    model_names = {}  # model_short -> full_name

    for cond_dir in sorted(exp_dir.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name == "analysis":
            continue
        csv_files = list(cond_dir.glob("*_scores.csv"))
        if not csv_files:
            continue

        # Parse model and temperature from directory name
        # Format: model-short_temp-X.X
        parts = cond_dir.name.rsplit("_temp-", 1)
        if len(parts) != 2:
            continue
        model_short = parts[0]
        temp = float(parts[1])

        df = pd.read_csv(csv_files[0])
        # Normalize scores to [0, 1] (scale is always 0-100 in this experiment)
        for dim in DIMENSIONS:
            run_cols = [c for c in df.columns if c.startswith(f"{dim}_run")]
            for col in run_cols:
                df[f"{col}_frac"] = df[col] / 100.0
            if f"{dim}_mean" in df.columns:
                df[f"{dim}_mean_frac"] = df[f"{dim}_mean"] / 100.0
            if f"{dim}_std" in df.columns:
                df[f"{dim}_std_frac"] = df[f"{dim}_std"] / 100.0

        data[(model_short, temp)] = df
        # Try to map short name back to full model name
        for full_name in models:
            if model_short in full_name.replace(":", "-").replace("/", "-"):
                model_names[model_short] = full_name

    model_shorts = sorted(set(k[0] for k in data.keys()))
    temps = sorted(set(k[1] for k in data.keys()))

    return data, model_shorts, temps, manifest


# ── Analysis Sections ──────────────────────────────────────────────

def compute_reliability_table(data, model_shorts, temps):
    """Compute alpha, CV, SNR for each model × temperature × dimension."""
    rows = []
    for model in model_shorts:
        for temp in temps:
            key = (model, temp)
            if key not in data:
                continue
            df = data[key]
            for dim in DIMENSIONS:
                run_cols = [c for c in df.columns if c.startswith(f"{dim}_run") and not c.endswith("_frac")]
                if not run_cols:
                    continue
                runs_matrix = df[run_cols].values

                # Krippendorff's alpha
                alpha = krippendorff_alpha(runs_matrix)
                ci_low, ci_high = bootstrap_alpha_ci(runs_matrix)

                # CV
                cvs = df[f"{dim}_cv"].dropna().values
                avg_cv = float(np.mean(cvs))
                median_cv = float(np.median(cvs))

                # SNR
                mean_frac = df[f"{dim}_mean_frac"].dropna().values
                std_frac = df[f"{dim}_std_frac"].dropna().values if f"{dim}_std_frac" in df.columns else np.zeros_like(mean_frac)
                between_sd = float(np.std(mean_frac, ddof=1))
                within_sd = float(np.mean(std_frac))
                snr = between_sd / within_sd if within_sd > 0 else np.inf

                rows.append({
                    "model": model,
                    "temperature": temp,
                    "dimension": dim,
                    "alpha": round(alpha, 4),
                    "alpha_ci_low": round(ci_low, 4),
                    "alpha_ci_high": round(ci_high, 4),
                    "avg_cv": round(avg_cv, 4),
                    "median_cv": round(median_cv, 4),
                    "between_sd": round(between_sd, 4),
                    "within_sd": round(within_sd, 4),
                    "snr": round(snr, 2),
                })

    return pd.DataFrame(rows)


def compute_rank_stability(data, model_shorts, temps):
    """Spearman rank correlation between temperature pairs within each model."""
    rows = []
    for model in model_shorts:
        for dim in DIMENSIONS:
            for i, t1 in enumerate(temps):
                for t2 in temps[i+1:]:
                    k1 = (model, t1)
                    k2 = (model, t2)
                    if k1 not in data or k2 not in data:
                        continue
                    # Align on entity
                    df1 = data[k1][["entity", f"{dim}_mean_frac"]].dropna()
                    df2 = data[k2][["entity", f"{dim}_mean_frac"]].dropna()
                    # Dedup
                    df1 = df1.groupby("entity", as_index=False).mean()
                    df2 = df2.groupby("entity", as_index=False).mean()
                    merged = df1.merge(df2, on="entity", suffixes=("_a", "_b"))
                    if len(merged) < 10:
                        continue
                    rho, p = sp_stats.spearmanr(merged[f"{dim}_mean_frac_a"], merged[f"{dim}_mean_frac_b"])
                    rows.append({
                        "model": model,
                        "dimension": dim,
                        "temp_a": t1,
                        "temp_b": t2,
                        "n_entities": len(merged),
                        "spearman_rho": round(rho, 4),
                        "p_value": p,
                    })
    return pd.DataFrame(rows)


# ── Figures ─────────────────────────────────────────────────────────

def fig_alpha_by_temperature(table, model_shorts, temps, fig_dir):
    """Line plot: alpha vs temperature for each model, paneled by dimension."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Krippendorff's Alpha vs. Sampling Temperature (0-100 / v2)",
                 fontsize=14, fontweight="bold")

    markers = ["o", "s", "^", "D", "v"]
    linestyles = ["-", "--", ":", "-.", "-"]

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        for mi, model in enumerate(model_shorts):
            sub = table[(table["model"] == model) & (table["dimension"] == dim)]
            if sub.empty:
                continue
            sub = sub.sort_values("temperature")
            ax.errorbar(
                sub["temperature"], sub["alpha"],
                yerr=[sub["alpha"] - sub["alpha_ci_low"], sub["alpha_ci_high"] - sub["alpha"]],
                marker=markers[mi % len(markers)],
                linestyle=linestyles[mi % len(linestyles)],
                capsize=4, markersize=8, linewidth=1.5,
                label=model,
            )

        ax.set_xlabel("Temperature", fontsize=11)
        ax.set_ylabel("Krippendorff's Alpha" if ax_i == 0 else "", fontsize=11)
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.set_xticks(temps)
        ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(0.5, 1.02)
        ax.grid(alpha=0.3)
        if ax_i == 2:
            ax.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_temp01_alpha_vs_temperature.png")


def fig_cv_by_temperature(table, model_shorts, temps, fig_dir):
    """Line plot: mean CV vs temperature."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Mean CV vs. Sampling Temperature (0-100 / v2)",
                 fontsize=14, fontweight="bold")

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", ":"]

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        for mi, model in enumerate(model_shorts):
            sub = table[(table["model"] == model) & (table["dimension"] == dim)].sort_values("temperature")
            if sub.empty:
                continue
            ax.plot(sub["temperature"], sub["avg_cv"],
                    marker=markers[mi % len(markers)],
                    linestyle=linestyles[mi % len(linestyles)],
                    markersize=8, linewidth=1.5, label=model)

        ax.set_xlabel("Temperature", fontsize=11)
        ax.set_ylabel("Mean CV" if ax_i == 0 else "", fontsize=11)
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.set_xticks(temps)
        ax.grid(alpha=0.3)
        if ax_i == 2:
            ax.legend(fontsize=9)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_temp02_cv_vs_temperature.png")


def fig_cv_distributions(data, model_shorts, temps, fig_dir):
    """Violin plots: per-entity CV distributions at each temperature."""
    fig, axes = plt.subplots(len(model_shorts), 3,
                             figsize=(18, 5 * len(model_shorts)), squeeze=False)
    fig.suptitle("Per-Entity CV Distributions by Temperature", fontsize=14, fontweight="bold", y=1.01)

    for mi, model in enumerate(model_shorts):
        for ax_i, dim in enumerate(DIMENSIONS):
            ax = axes[mi, ax_i]
            cv_data = []
            labels = []
            for temp in temps:
                key = (model, temp)
                if key not in data:
                    continue
                cvs = data[key][f"{dim}_cv"].dropna().values
                cv_data.append(cvs)
                labels.append(f"T={temp}")

            if cv_data:
                parts = ax.violinplot(cv_data, positions=range(len(cv_data)),
                                      showmeans=True, showmedians=True, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(DIM_COLORS[dim])
                    pc.set_alpha(0.5)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=9)

            if mi == 0:
                ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
            if ax_i == 0:
                ax.set_ylabel(f"{model}\nCV", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_temp03_cv_distributions.png")


def fig_rank_stability_heatmap(rank_table, model_shorts, fig_dir):
    """Heatmap: Spearman rho between temperature pairs."""
    temp_pairs = rank_table[["temp_a", "temp_b"]].drop_duplicates().values.tolist()
    pair_labels = [f"T={a} vs T={b}" for a, b in temp_pairs]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4, len(model_shorts) * 1.2 + 1)))
    fig.suptitle("Rank Stability Across Temperatures (Spearman ρ)", fontsize=14, fontweight="bold")

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        matrix = np.full((len(model_shorts), len(temp_pairs)), np.nan)
        for mi, model in enumerate(model_shorts):
            for pi, (ta, tb) in enumerate(temp_pairs):
                row = rank_table[(rank_table["model"] == model) &
                                 (rank_table["dimension"] == dim) &
                                 (rank_table["temp_a"] == ta) &
                                 (rank_table["temp_b"] == tb)]
                if len(row) > 0:
                    matrix[mi, pi] = row.iloc[0]["spearman_rho"]

        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")
        for mi in range(len(model_shorts)):
            for pi in range(len(temp_pairs)):
                val = matrix[mi, pi]
                if not np.isnan(val):
                    color = "white" if val < 0.85 else "black"
                    ax.text(pi, mi, f"{val:.3f}", ha="center", va="center", fontsize=9, color=color)
        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(model_shorts)))
        ax.set_yticklabels(model_shorts if ax_i == 0 else [], fontsize=10)
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        plt.colorbar(im, ax=ax, shrink=0.8, label="ρ")

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_temp04_rank_stability_heatmap.png")


def fig_snr_by_temperature(table, model_shorts, temps, fig_dir):
    """Bar chart: SNR at each temperature."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Signal-to-Noise Ratio vs. Temperature", fontsize=14, fontweight="bold")

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", ":"]

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        for mi, model in enumerate(model_shorts):
            sub = table[(table["model"] == model) & (table["dimension"] == dim)].sort_values("temperature")
            if sub.empty:
                continue
            ax.plot(sub["temperature"], sub["snr"],
                    marker=markers[mi % len(markers)],
                    linestyle=linestyles[mi % len(linestyles)],
                    markersize=8, linewidth=1.5, label=model)

        ax.set_xlabel("Temperature", fontsize=11)
        ax.set_ylabel("SNR (between-entity SD / within-entity SD)" if ax_i == 0 else "", fontsize=11)
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.set_xticks(temps)
        ax.grid(alpha=0.3)
        if ax_i == 2:
            ax.legend(fontsize=9)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_temp05_snr_vs_temperature.png")


def fig_score_shift(data, model_shorts, temps, fig_dir):
    """Do mean scores shift with temperature? Plot mean normalized score by temp."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle("Mean Normalized Score vs. Temperature (score level shift check)",
                 fontsize=14, fontweight="bold")

    markers = ["o", "s", "^"]
    linestyles = ["-", "--", ":"]

    for ax_i, dim in enumerate(DIMENSIONS):
        ax = axes[ax_i]
        for mi, model in enumerate(model_shorts):
            means = []
            cis = []
            for temp in temps:
                key = (model, temp)
                if key not in data:
                    continue
                vals = data[key][f"{dim}_mean_frac"].dropna().values
                m, cl, ch = bootstrap_ci(vals)
                means.append(m)
                cis.append((cl, ch))
            if means:
                cl_arr = [m - ci[0] for m, ci in zip(means, cis)]
                ch_arr = [ci[1] - m for m, ci in zip(means, cis)]
                ax.errorbar(temps[:len(means)], means,
                            yerr=[cl_arr, ch_arr],
                            marker=markers[mi % len(markers)],
                            linestyle=linestyles[mi % len(linestyles)],
                            capsize=4, markersize=8, linewidth=1.5, label=model)

        ax.set_xlabel("Temperature", fontsize=11)
        ax.set_ylabel("Mean Normalized Score [0,1]" if ax_i == 0 else "", fontsize=11)
        ax.set_title(dim.title(), fontsize=12, fontweight="bold", color=DIM_COLORS[dim])
        ax.set_xticks(temps)
        ax.grid(alpha=0.3)
        if ax_i == 2:
            ax.legend(fontsize=9)

    fig.tight_layout()
    save_figure(fig, fig_dir / "fig_temp06_score_level_shift.png")


# ── Summary Report ─────────────────────────────────────────────────

def write_summary(table, rank_table, model_shorts, temps, out_dir):
    lines = []
    lines.append("# Temperature Sensitivity Analysis\n")
    lines.append(f"**Models:** {', '.join(model_shorts)}\n")
    lines.append(f"**Temperatures:** {', '.join(str(t) for t in temps)}\n")
    lines.append(f"**Fixed condition:** 0-100 scale, v2 (Score→Justify), 5 runs/entity\n")
    lines.append("")

    lines.append("## Key Findings\n")

    # Alpha degradation
    lines.append("### Reliability Degradation (α)\n")
    lines.append("| Model | Dimension | α (T=0.3) | α (T=0.5) | α (T=0.7) | Δ (0.3→0.7) |")
    lines.append("|-------|-----------|-----------|-----------|-----------|-------------|")
    for model in model_shorts:
        for dim in DIMENSIONS:
            alphas = []
            for temp in temps:
                row = table[(table["model"] == model) & (table["temperature"] == temp) & (table["dimension"] == dim)]
                alphas.append(row.iloc[0]["alpha"] if len(row) > 0 else np.nan)
            delta = alphas[-1] - alphas[0] if len(alphas) >= 2 else np.nan
            vals = " | ".join(f"{a:.3f}" if not np.isnan(a) else "—" for a in alphas)
            lines.append(f"| {model} | {dim} | {vals} | {delta:+.3f} |")
    lines.append("")

    # CV increase
    lines.append("### CV Increase\n")
    lines.append("| Model | Dimension | CV (T=0.3) | CV (T=0.5) | CV (T=0.7) | Ratio (0.7/0.3) |")
    lines.append("|-------|-----------|------------|------------|------------|-----------------|")
    for model in model_shorts:
        for dim in DIMENSIONS:
            cvs = []
            for temp in temps:
                row = table[(table["model"] == model) & (table["temperature"] == temp) & (table["dimension"] == dim)]
                cvs.append(row.iloc[0]["avg_cv"] if len(row) > 0 else np.nan)
            ratio = cvs[-1] / cvs[0] if len(cvs) >= 2 and cvs[0] > 0 else np.nan
            vals = " | ".join(f"{c:.4f}" if not np.isnan(c) else "—" for c in cvs)
            lines.append(f"| {model} | {dim} | {vals} | {ratio:.1f}x |")
    lines.append("")

    # Rank stability
    if not rank_table.empty:
        lines.append("### Rank Stability Across Temperatures\n")
        lines.append("| Model | Dimension | T=0.3↔0.5 | T=0.3↔0.7 | T=0.5↔0.7 |")
        lines.append("|-------|-----------|-----------|-----------|-----------|")
        for model in model_shorts:
            for dim in DIMENSIONS:
                rhos = []
                for ta, tb in [(0.3, 0.5), (0.3, 0.7), (0.5, 0.7)]:
                    row = rank_table[(rank_table["model"] == model) &
                                     (rank_table["dimension"] == dim) &
                                     (rank_table["temp_a"] == ta) &
                                     (rank_table["temp_b"] == tb)]
                    rhos.append(f"{row.iloc[0]['spearman_rho']:.3f}" if len(row) > 0 else "—")
                lines.append(f"| {model} | {dim} | {' | '.join(rhos)} |")
        lines.append("")

    report_path = out_dir / "summary_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved summary_report.md")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze temperature sensitivity experiment.")
    parser.add_argument("--exp-dir", default=None,
                        help="Experiment directory (default: output/temperature-experiment).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    exp_dir = Path(args.exp_dir).resolve() if args.exp_dir else script_dir / "output" / "temperature-experiment"

    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    out_dir = exp_dir / "analysis"
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  TEMPERATURE SENSITIVITY ANALYSIS")
    print(f"  Data: {exp_dir}")
    print(f"{'='*60}\n")

    # Load
    print("Loading data...")
    data, model_shorts, temps, manifest = load_experiment(exp_dir)
    print(f"  {len(model_shorts)} models × {len(temps)} temperatures = {len(data)} conditions\n")

    # Compute tables
    print("=== Computing Reliability Metrics ===")
    rel_table = compute_reliability_table(data, model_shorts, temps)
    rel_table.to_csv(tbl_dir / "temperature_reliability.csv", index=False)
    print("  Saved temperature_reliability.csv")

    print("\n=== Computing Rank Stability ===")
    rank_table = compute_rank_stability(data, model_shorts, temps)
    rank_table.to_csv(tbl_dir / "temperature_rank_stability.csv", index=False)
    print("  Saved temperature_rank_stability.csv")

    # Figures
    print("\n=== Generating Figures ===")
    fig_alpha_by_temperature(rel_table, model_shorts, temps, fig_dir)
    fig_cv_by_temperature(rel_table, model_shorts, temps, fig_dir)
    fig_cv_distributions(data, model_shorts, temps, fig_dir)
    fig_rank_stability_heatmap(rank_table, model_shorts, fig_dir)
    fig_snr_by_temperature(rel_table, model_shorts, temps, fig_dir)
    fig_score_shift(data, model_shorts, temps, fig_dir)

    # Summary report
    print("\n=== Generating Summary Report ===")
    write_summary(rel_table, rank_table, model_shorts, temps, out_dir)

    print(f"\n{'='*60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"  Figures: {fig_dir}")
    print(f"  Tables:  {tbl_dir}")
    print(f"  Report:  {out_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
