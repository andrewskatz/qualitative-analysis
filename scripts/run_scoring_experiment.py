#!/usr/bin/env python3
"""
Factorial scoring experiment: scale × prompt version.

Runs `qa entity score` across all combinations of scoring scales and
prompt versions, writing results to per-condition subdirectories with
full metadata for reproducibility.

Usage:
    python scripts/run_scoring_experiment.py \
        --input qualitative-analysis/output/abc-sim-platform-detect-test/entities_sample_450.csv \
        --output-dir qualitative-analysis/output/scoring-experiment \
        --model gpt-oss:120b \
        --num-runs 3

    # With full prompt/response logging:
    python scripts/run_scoring_experiment.py \
        --input entities_sample_450.csv \
        --output-dir scoring-experiment \
        --verbose

    # Dry run (print commands without executing):
    python scripts/run_scoring_experiment.py \
        --input entities_sample_450.csv \
        --output-dir scoring-experiment \
        --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path

# ── Factorial design ─────────────────────────────────────────────────
SCALES = [
    (0, 100),
    (1, 10),
    (1, 5),
]

PROMPT_VERSIONS = ["v2", "v3", "v4"]
# ─────────────────────────────────────────────────────────────────────


def condition_label(scale_min: int, scale_max: int, prompt_version: str) -> str:
    """Human-readable directory name for one experimental condition."""
    return f"scale-{scale_min}-{scale_max}_prompt-{prompt_version}"


def build_command(
    input_csv: Path,
    output_dir: Path,
    scale_min: int,
    scale_max: int,
    prompt_version: str,
    model: str,
    provider: str,
    base_url: str,
    temperature: float,
    num_runs: int,
    verbose: bool,
    extra_args: list[str],
) -> list[str]:
    """Build the `qa entity score` CLI command for one condition."""
    cmd = [
        "qa", "entity", "score",
        str(input_csv),
        "--scale-min", str(scale_min),
        "--scale-max", str(scale_max),
        "--prompt-version", prompt_version,
        "--model", model,
        "--provider", provider,
        "--base-url", base_url,
        "--temperature", str(temperature),
        "--num-runs", str(num_runs),
        "--output-dir", str(output_dir),
    ]
    if verbose:
        cmd.append("--verbose")
    cmd.extend(extra_args)
    return cmd


def run_condition(
    cmd: list[str],
    label: str,
    log_path: Path,
) -> dict:
    """Execute one condition, streaming output to both terminal and log file."""
    print(f"\n{'='*60}")
    print(f"  CONDITION: {label}")
    print(f"  Command:   {' '.join(cmd)}")
    print(f"  Log:       {log_path}")
    print(f"{'='*60}\n")

    start = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log_f:
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write(f"Started: {datetime.now().isoformat()}\n")
            log_f.write(f"{'─'*60}\n")
            log_f.flush()

            # Stream stdout+stderr merged, line by line, to both terminal and log
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffered
            )

            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()

            proc.wait()
            elapsed = time.time() - start

            log_f.write(f"{'─'*60}\n")
            log_f.write(f"Exit code: {proc.returncode}\n")
            log_f.write(f"Duration: {elapsed:.1f}s\n")

        # Print summary
        if proc.returncode == 0:
            print(f"\n  >> {label} completed in {elapsed:.1f}s")
        else:
            print(f"\n  >> {label} FAILED (exit {proc.returncode}) after {elapsed:.1f}s")

        return {
            "label": label,
            "status": "success" if proc.returncode == 0 else "failed",
            "exit_code": proc.returncode,
            "duration_seconds": round(elapsed, 1),
            "log_file": str(log_path),
        }

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  >> {label} ERROR: {e}")
        return {
            "label": label,
            "status": "error",
            "error": str(e),
            "duration_seconds": round(elapsed, 1),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run factorial scoring experiment (scale × prompt version).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to sampled entities CSV.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Parent directory for experiment output (subdirs created per condition).",
    )
    parser.add_argument(
        "--model", default="gpt-oss:120b",
        help="LLM model name (default: gpt-oss:120b).",
    )
    parser.add_argument(
        "--provider", default="ollama",
        choices=["ollama", "mlx", "openai", "anthropic"],
        help="LLM provider (default: ollama).",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:11434",
        help="Base URL for Ollama provider (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="LLM temperature (default: 0.3).",
    )
    parser.add_argument(
        "--num-runs", type=int, default=3,
        help="Scoring runs per entity (default: 3).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Pass --verbose to qa entity score (logs full prompts and LLM responses).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip conditions whose output directory already contains scored results.",
    )
    parser.add_argument(
        "extra_args", nargs="*",
        help="Additional arguments passed through to `qa entity score`.",
    )

    args = parser.parse_args()

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        sys.exit(f"Input file not found: {input_csv}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Build the full factorial design ──────────────────────────────
    conditions = list(product(SCALES, PROMPT_VERSIONS))
    n_conditions = len(conditions)
    print(f"Factorial experiment: {len(SCALES)} scales x {len(PROMPT_VERSIONS)} prompts = {n_conditions} conditions")
    print(f"Input:   {input_csv}")
    print(f"Output:  {output_root}")
    print(f"Verbose: {args.verbose}")
    print()

    # ── Experiment-level metadata ────────────────────────────────────
    experiment_meta = {
        "experiment": "factorial_scoring",
        "started_at": datetime.now().isoformat(),
        "input_file": str(input_csv),
        "output_root": str(output_root),
        "design": {
            "scales": [{"min": s[0], "max": s[1]} for s in SCALES],
            "prompt_versions": PROMPT_VERSIONS,
            "n_conditions": n_conditions,
        },
        "parameters": {
            "model": args.model,
            "provider": args.provider,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "num_runs": args.num_runs,
            "verbose": args.verbose,
        },
        "conditions": [],
    }

    # ── Iterate through conditions ───────────────────────────────────
    results = []
    skipped = 0

    for i, ((scale_min, scale_max), prompt_version) in enumerate(conditions, 1):
        label = condition_label(scale_min, scale_max, prompt_version)
        cond_dir = output_root / label
        log_path = cond_dir / "run.log"

        print(f"\n[{i}/{n_conditions}] {label}")

        # Resume support: skip if already completed
        if args.resume and cond_dir.exists():
            meta_path = cond_dir / "score_metadata.json"
            if meta_path.exists():
                print(f"  Already completed, skipping (--resume)")
                skipped += 1
                results.append({
                    "label": label,
                    "status": "skipped (resume)",
                    "duration_seconds": 0,
                })
                continue

        cmd = build_command(
            input_csv=input_csv,
            output_dir=cond_dir,
            scale_min=scale_min,
            scale_max=scale_max,
            prompt_version=prompt_version,
            model=args.model,
            provider=args.provider,
            base_url=args.base_url,
            temperature=args.temperature,
            num_runs=args.num_runs,
            verbose=args.verbose,
            extra_args=args.extra_args,
        )

        if args.dry_run:
            print(f"  [DRY RUN] {' '.join(cmd)}")
            results.append({"label": label, "status": "dry_run"})
            continue

        cond_dir.mkdir(parents=True, exist_ok=True)
        result = run_condition(cmd, label, log_path)
        results.append(result)

    # ── Write experiment manifest ────────────────────────────────────
    experiment_meta["completed_at"] = datetime.now().isoformat()
    experiment_meta["conditions"] = results
    experiment_meta["summary"] = {
        "total": n_conditions,
        "success": sum(1 for r in results if r.get("status") == "success"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "skipped": skipped,
    }

    if not args.dry_run:
        total_time = sum(r.get("duration_seconds", 0) for r in results)
        experiment_meta["summary"]["total_duration_seconds"] = round(total_time, 1)

    manifest_path = output_root / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(experiment_meta, f, indent=2)

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    s = experiment_meta["summary"]
    print(f"  Conditions: {s['total']} total, {s.get('success', 0)} success, {s.get('failed', 0)} failed, {s.get('skipped', 0)} skipped")
    if not args.dry_run:
        print(f"  Total time: {s.get('total_duration_seconds', 0):.0f}s")
    print(f"  Manifest:   {manifest_path}")


if __name__ == "__main__":
    main()
