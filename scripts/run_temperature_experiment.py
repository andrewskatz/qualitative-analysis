#!/usr/bin/env python3
"""
Temperature sensitivity experiment for entity scoring.

Fixes scale (0-100) and prompt (v2) — the best-performing condition from
the factorial experiment — and varies temperature across multiple models
to quantify how much sampling temperature affects reliability metrics.

Design:
    3 temperatures (0.3, 0.5, 0.7) × N models × 5 scoring runs per entity

Usage:
    # Default: 3 representative models spanning the quality range
    python scripts/run_temperature_experiment.py \
        --input output/abc-sim-platform-detect-test/entities_sample_450.csv \
        --output-dir output/temperature-experiment

    # Custom models
    python scripts/run_temperature_experiment.py \
        --input output/abc-sim-platform-detect-test/entities_sample_450.csv \
        --output-dir output/temperature-experiment \
        --models "qwen3.5:122b-a10b-q4_K_M" "qwen3:30b-a3b-instruct-2507-q4_K_M"

    # Dry run
    python scripts/run_temperature_experiment.py \
        --input output/abc-sim-platform-detect-test/entities_sample_450.csv \
        --output-dir output/temperature-experiment \
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

# ── Experimental design ─────────────────────────────────────────────
TEMPERATURES = [0.3, 0.5, 0.7]

# Fixed at best condition from factorial experiment
SCALE_MIN = 0
SCALE_MAX = 100
PROMPT_VERSION = "v2"

# Representative models spanning quality tiers
DEFAULT_MODELS = [
    "qwen3.5:122b-a10b-q4_K_M",       # Tier 1 (highest quality)
    "qwen3:30b-a3b-instruct-2507-q4_K_M",  # Tier 2 (balanced)
    "qwen3:4b-instruct-2507-q4_K_M",   # Tier 4 (fastest, lowest rank stability)
]
# ─────────────────────────────────────────────────────────────────────


def model_short_name(model: str) -> str:
    """Short directory-safe name for a model."""
    name = model.split(":")[0].replace("/", "-")
    parts = model.split(":")
    if len(parts) > 1:
        tag = parts[1]
        for seg in tag.split("-"):
            if seg.endswith("b") and seg[:-1].replace(".", "").isdigit():
                name = f"{name}-{seg}"
                break
    return name


def condition_label(model: str, temperature: float) -> str:
    return f"{model_short_name(model)}_temp-{temperature}"


def build_command(
    input_csv: Path,
    output_dir: Path,
    model: str,
    temperature: float,
    provider: str,
    base_url: str,
    num_runs: int,
    verbose: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        "qa", "entity", "score",
        str(input_csv),
        "--scale-min", str(SCALE_MIN),
        "--scale-max", str(SCALE_MAX),
        "--prompt-version", PROMPT_VERSION,
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


def run_condition(cmd: list[str], label: str, log_path: Path) -> dict:
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

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
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

        status = "success" if proc.returncode == 0 else "failed"
        print(f"\n  >> {label} {'completed' if status == 'success' else 'FAILED'} in {elapsed:.1f}s")

        return {
            "label": label,
            "status": status,
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
        description="Temperature sensitivity experiment for entity scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, help="Path to sampled entities CSV.")
    parser.add_argument("--output-dir", required=True, help="Parent directory for experiment output.")
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"Models to test (default: {', '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--temperatures", nargs="*", type=float, default=None,
        help=f"Temperatures to test (default: {TEMPERATURES}).",
    )
    parser.add_argument("--provider", default="ollama", choices=["ollama", "mlx"])
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--num-runs", type=int, default=5, help="Scoring runs per entity (default: 5).")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip completed conditions.")
    parser.add_argument("extra_args", nargs="*")

    args = parser.parse_args()

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        sys.exit(f"Input file not found: {input_csv}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    models = args.models or DEFAULT_MODELS
    temperatures = args.temperatures or TEMPERATURES

    conditions = list(product(models, temperatures))
    n_conditions = len(conditions)

    print(f"Temperature sensitivity experiment")
    print(f"  Fixed: scale 0-100, prompt v2")
    print(f"  Models:       {len(models)}")
    print(f"  Temperatures: {temperatures}")
    print(f"  Conditions:   {n_conditions}")
    print(f"  Runs/entity:  {args.num_runs}")
    print(f"  Input:        {input_csv}")
    print(f"  Output:       {output_root}")
    print()

    experiment_meta = {
        "experiment": "temperature_sensitivity",
        "started_at": datetime.now().isoformat(),
        "input_file": str(input_csv),
        "output_root": str(output_root),
        "design": {
            "fixed_scale": f"{SCALE_MIN}-{SCALE_MAX}",
            "fixed_prompt": PROMPT_VERSION,
            "temperatures": temperatures,
            "models": models,
            "n_conditions": n_conditions,
        },
        "parameters": {
            "provider": args.provider,
            "base_url": args.base_url,
            "num_runs": args.num_runs,
            "verbose": args.verbose,
        },
        "conditions": [],
    }

    results = []
    skipped = 0

    for i, (model, temp) in enumerate(conditions, 1):
        label = condition_label(model, temp)
        cond_dir = output_root / label
        log_path = cond_dir / "run.log"

        print(f"\n[{i}/{n_conditions}] {label}")

        if args.resume and cond_dir.exists():
            meta_path = cond_dir / "score_metadata.json"
            if meta_path.exists():
                print(f"  Already completed, skipping (--resume)")
                skipped += 1
                results.append({"label": label, "status": "skipped (resume)", "duration_seconds": 0})
                continue

        cmd = build_command(
            input_csv=input_csv,
            output_dir=cond_dir,
            model=model,
            temperature=temp,
            provider=args.provider,
            base_url=args.base_url,
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
