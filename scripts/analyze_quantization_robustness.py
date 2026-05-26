#!/usr/bin/env python3
"""
Analyze the quantization robustness check.

Two analyses:

1. Quantization comparison (primary):
   Compare Q4_K_M, Q8_0, FP16 (all from the robustness experiment, same
   num_runs) pairwise: alpha, CV, mean-score delta, Spearman rho, KS test.

2. Runs-per-entity sanity check (optional):
   Compare the new Q4_K_M (5 runs) against the factorial Q4_K_M (3 runs)
   to confirm that any quantization differences aren't driven by the
   change in num_runs.

Usage:
    # Primary: pairwise comparison among quantization variants
    python scripts/analyze_quantization_robustness.py \\
        --variants \\
            output/quantization-robustness-check/qwen3-30b-a3b-instruct-2507-q4_K_M \\
            output/quantization-robustness-check/qwen3-30b-a3b-instruct-2507-q8_0 \\
            output/quantization-robustness-check/qwen3-30b-a3b-instruct-2507-fp16

    # Optional: sanity-check num_runs effect
    python scripts/analyze_quantization_robustness.py \\
        --variants output/quantization-robustness-check/qwen3-30b-a3b-instruct-2507-q4_K_M \\
        --runs-baseline output/scoring-experiment-qwen3-30b/scale-0-100_prompt-v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

DIMENSIONS = ["social", "ecological", "technological"]


def krippendorff_alpha(D):
    D = np.asarray(D, dtype=float)
    n_items, n_raters = D.shape
    Do, count = 0.0, 0
    for i in range(n_items):
        v = D[i][~np.isnan(D[i])]
        m = len(v)
        if m < 2: continue
        for x in range(m):
            for y in range(x+1, m):
                Do += (v[x] - v[y])**2
                count += 1
    if count == 0: return np.nan
    Do /= count
    flat = D[~np.isnan(D)]
    n = len(flat)
    De = float(np.var(flat, ddof=0) * n / (n - 1))
    return 1 - Do/De if De > 0 else 1.0


def load_scores(cond_dir: Path) -> pd.DataFrame:
    csvs = list(cond_dir.glob("*_scores.csv"))
    if not csvs:
        raise FileNotFoundError(f"No *_scores.csv in {cond_dir}")
    return pd.read_csv(csvs[0])


def dedup(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("entity", as_index=False)[cols].mean()


def per_variant_metrics(df: pd.DataFrame, dim: str) -> dict:
    run_cols = [c for c in df.columns if c.startswith(f"{dim}_run") and not c.endswith("_frac")]
    runs = df[run_cols].values
    return {
        "n_entities": len(df),
        "n_runs": len(run_cols),
        "alpha": float(krippendorff_alpha(runs)),
        "avg_cv": float(df[f"{dim}_cv"].dropna().mean()),
        "mean_score": float(df[f"{dim}_mean"].dropna().mean()),
    }


def pairwise_metrics(baseline: pd.DataFrame, variant: pd.DataFrame, dim: str) -> dict:
    cols = [f"{dim}_mean"]
    a = dedup(baseline, cols).set_index("entity")
    b = dedup(variant,  cols).set_index("entity")
    common = a.index.intersection(b.index)
    a = a.loc[common, f"{dim}_mean"].values
    b = b.loc[common, f"{dim}_mean"].values

    delta = b - a
    rho, _ = sp_stats.spearmanr(a, b)
    pearson, _ = sp_stats.pearsonr(a, b)
    ks_stat, ks_p = sp_stats.ks_2samp(a, b)

    return {
        "n_common_entities": len(common),
        "mean_score_delta_variant_minus_baseline": float(np.mean(delta)),
        "median_score_delta": float(np.median(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "spearman_rho": float(rho),
        "pearson_r": float(pearson),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
    }


def variant_label(path: Path) -> str:
    """Extract a short label from a variant directory name."""
    name = path.name
    for tag in ["q4_K_M", "q8_0", "fp16", "bf16"]:
        if tag.lower() in name.lower():
            return tag
    return name


def print_pairwise_table(label_a: str, label_b: str,
                         metrics_a: dict, metrics_b: dict, pairwise: dict):
    print(f"\n  {label_a} vs {label_b}:")
    print(f"  {'Dimension':>14s}  {'α(a)':>7s}  {'α(b)':>7s}  {'Δα':>7s}  "
          f"{'ρ':>6s}  {'MAE':>6s}  {'mean Δ':>8s}  {'KS p':>8s}")
    for dim in DIMENSIONS:
        a = metrics_a[dim]; b = metrics_b[dim]; p = pairwise[dim]
        print(f"  {dim:>14s}  {a['alpha']:7.4f}  {b['alpha']:7.4f}  "
              f"{b['alpha']-a['alpha']:+7.4f}  {p['spearman_rho']:6.3f}  "
              f"{p['mae']:6.2f}  {p['mean_score_delta_variant_minus_baseline']:+8.3f}  "
              f"{p['ks_p_value']:8.4g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Two or more variant directories to compare pairwise. "
                             "Order them by precision (Q4 → Q8 → FP16) for cleanest reporting.")
    parser.add_argument("--runs-baseline", default=None,
                        help="Optional: factorial Q4 directory (3 runs/entity) "
                             "to sanity-check the runs-per-entity effect.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: ../quantization_robustness_report.json).")
    args = parser.parse_args()

    variant_dirs = [Path(v).resolve() for v in args.variants]
    if len(variant_dirs) < 2:
        sys.exit("Need at least 2 variants for pairwise comparison.")

    # Load all variants
    print("Loading variants:")
    variants_data = {}
    for vdir in variant_dirs:
        label = variant_label(vdir)
        df = load_scores(vdir)
        variants_data[label] = {
            "path": str(vdir),
            "df": df,
            "n_rows": len(df),
            "n_unique_entities": int(df["entity"].nunique()),
            "metrics": {dim: per_variant_metrics(df, dim) for dim in DIMENSIONS},
        }
        print(f"  {label:8s}: N={len(df)}, unique={df['entity'].nunique()}")

    labels = list(variants_data.keys())

    # Pairwise comparisons
    report = {
        "variants": {l: {k: v for k, v in d.items() if k != "df"}
                     for l, d in variants_data.items()},
        "pairwise_comparisons": {},
    }

    print("\n" + "=" * 78)
    print("PAIRWISE COMPARISONS (all combinations)")
    print("=" * 78)
    for i, la in enumerate(labels):
        for lb in labels[i+1:]:
            a_data = variants_data[la]
            b_data = variants_data[lb]
            pw = {dim: pairwise_metrics(a_data["df"], b_data["df"], dim) for dim in DIMENSIONS}
            report["pairwise_comparisons"][f"{la}_vs_{lb}"] = pw
            print_pairwise_table(la, lb, a_data["metrics"], b_data["metrics"], pw)

    # Optional runs-per-entity sanity check
    if args.runs_baseline:
        print("\n" + "=" * 78)
        print("RUNS-PER-ENTITY SANITY CHECK")
        print("=" * 78)
        runs_baseline_dir = Path(args.runs_baseline).resolve()
        baseline_df = load_scores(runs_baseline_dir)
        baseline_metrics = {dim: per_variant_metrics(baseline_df, dim) for dim in DIMENSIONS}
        report["runs_baseline"] = {
            "path": str(runs_baseline_dir),
            "n_rows": len(baseline_df),
            "n_unique_entities": int(baseline_df["entity"].nunique()),
            "metrics": baseline_metrics,
        }
        report["runs_per_entity_check"] = {}
        print(f"  Baseline (3 runs): {runs_baseline_dir.name}, N={len(baseline_df)}")

        # Compare each variant to the runs baseline
        for la in labels:
            v_data = variants_data[la]
            pw = {dim: pairwise_metrics(baseline_df, v_data["df"], dim) for dim in DIMENSIONS}
            report["runs_per_entity_check"][f"3runs_vs_{la}_5runs"] = pw
            print_pairwise_table("3-run baseline", f"{la} (5 runs)",
                                 baseline_metrics, v_data["metrics"], pw)

    # Write JSON report
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = variant_dirs[0].parent / "quantization_robustness_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nJSON report written to: {out_path}")

    # Write CSVs for the table-generator pipeline
    tables_dir = out_path.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: per-variant metrics (long format)
    per_variant_rows = []
    for label, v in variants_data.items():
        for dim, m in v["metrics"].items():
            per_variant_rows.append({
                "variant": label,
                "dimension": dim,
                "n_entities": m["n_entities"],
                "n_runs": m["n_runs"],
                "alpha": round(m["alpha"], 4),
                "avg_cv": round(m["avg_cv"], 4),
                "mean_score": round(m["mean_score"], 2),
            })
    pd.DataFrame(per_variant_rows).to_csv(
        tables_dir / "per_variant_metrics.csv", index=False)
    print(f"CSV:  {tables_dir / 'per_variant_metrics.csv'}")

    # Table 2: pairwise comparisons (long format)
    pairwise_rows = []
    for pair_label, dims in report["pairwise_comparisons"].items():
        a, b = pair_label.split("_vs_")
        for dim, p in dims.items():
            pairwise_rows.append({
                "variant_a": a,
                "variant_b": b,
                "dimension": dim,
                "n_common_entities": p["n_common_entities"],
                "mean_delta": round(p["mean_score_delta_variant_minus_baseline"], 3),
                "mae": round(p["mae"], 3),
                "rmse": round(p["rmse"], 3),
                "spearman_rho": round(p["spearman_rho"], 4),
                "pearson_r": round(p["pearson_r"], 4),
                "ks_statistic": round(p["ks_statistic"], 4),
                "ks_p_value": p["ks_p_value"],
            })
    pd.DataFrame(pairwise_rows).to_csv(
        tables_dir / "pairwise_comparisons.csv", index=False)
    print(f"CSV:  {tables_dir / 'pairwise_comparisons.csv'}")

    # Table 3: runs-per-entity sanity check (if present)
    if "runs_per_entity_check" in report:
        runs_rows = []
        for pair_label, dims in report["runs_per_entity_check"].items():
            for dim, p in dims.items():
                runs_rows.append({
                    "comparison": pair_label,
                    "dimension": dim,
                    "mean_delta": round(p["mean_score_delta_variant_minus_baseline"], 3),
                    "mae": round(p["mae"], 3),
                    "spearman_rho": round(p["spearman_rho"], 4),
                    "ks_p_value": p["ks_p_value"],
                })
        pd.DataFrame(runs_rows).to_csv(
            tables_dir / "runs_per_entity_sanity_check.csv", index=False)
        print(f"CSV:  {tables_dir / 'runs_per_entity_sanity_check.csv'}")


if __name__ == "__main__":
    main()
