#!/usr/bin/env python3
"""
Quantization robustness check.

Question: do reliability metrics computed at Q4_K_M generalize to less
aggressively quantized variants of the same model? This addresses the
likely reviewer concern: "Q4_K_M may distort results vs. full precision."

Design:
    Fix scale=0--100, prompt=v2 (the recommended condition).
    Vary {model_id, quantization} for one (or a few) representative
    models, holding everything else constant.
    Compare Krippendorff's alpha, CV, and rank correlations against
    the original Q4_K_M factorial run.

Default comparison: qwen3:30b-a3b-instruct-2507 at three precisions
(Q4_K_M, Q8_0, FP16), all with the same number of scoring runs to avoid
a runs-per-entity confound vs. the factorial baseline (which used 3 runs).

Usage:
    # Default: all three precisions at 5 runs/entity
    python scripts/run_quantization_robustness_check.py \\
        --input output/abc-sim-platform-detect-test/entities_sample_450.csv \\
        --output-dir output/quantization-robustness-check

    # Custom variants
    python scripts/run_quantization_robustness_check.py \\
        --input output/abc-sim-platform-detect-test/entities_sample_450.csv \\
        --output-dir output/quantization-robustness-check \\
        --variants "qwen3:30b-a3b-instruct-2507-q4_K_M" \\
                   "qwen3:30b-a3b-instruct-2507-q8_0"

Pre-flight (one-time pulls):
    ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M  # ~17GB
    ollama pull qwen3:30b-a3b-instruct-2507-q8_0    # ~32GB
    ollama pull qwen3:30b-a3b-instruct-2507-fp16    # ~60GB

Estimated runtime (450 entities, 5 runs each):
    Q4_K_M: ~2 hours
    Q8_0:   ~3 hours (slower per token)
    FP16:   ~5-6 hours
    Total:  ~10-11 hours sequential
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Fixed condition (the recommended config from the factorial study) ──
SCALE_MIN = 0
SCALE_MAX = 100
PROMPT_VERSION = "v2"

# Default robustness-check variants. Three precisions of the production-
# recommended model, all run with the same num_runs to avoid a
# runs-per-entity confound vs. the factorial baseline (3 runs).
DEFAULT_VARIANTS = [
    "qwen3:30b-a3b-instruct-2507-q4_K_M",
    "qwen3:30b-a3b-instruct-2507-q8_0",
    "qwen3:30b-a3b-instruct-2507-fp16",
]


def variant_short_name(variant: str) -> str:
    safe = variant.replace(":", "-").replace("/", "-")
    return safe


def build_command(
    input_csv: Path, output_dir: Path, variant: str,
    base_url: str, num_runs: int, temperature: float, verbose: bool,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        "qa", "entity", "score",
        str(input_csv),
        "--scale-min", str(SCALE_MIN),
        "--scale-max", str(SCALE_MAX),
        "--prompt-version", PROMPT_VERSION,
        "--model", variant,
        "--provider", "ollama",
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
    print(f"\n{'='*60}\n  CONDITION: {label}\n  Command:   {' '.join(cmd)}\n{'='*60}\n")
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        log_f.write(f"Command: {' '.join(cmd)}\n")
        log_f.write(f"Started: {datetime.now().isoformat()}\n")
        log_f.write(f"{'-'*60}\n")
        log_f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            log_f.write(line); log_f.flush()
        proc.wait()
        elapsed = time.time() - start
        log_f.write(f"{'-'*60}\nExit code: {proc.returncode}\nDuration: {elapsed:.1f}s\n")

    status = "success" if proc.returncode == 0 else "failed"
    print(f"\n  >> {label} {'completed' if status == 'success' else 'FAILED'} in {elapsed:.1f}s")
    return {"label": label, "status": status, "exit_code": proc.returncode,
            "duration_seconds": round(elapsed, 1), "log_file": str(log_path)}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="*", default=None,
                        help=f"Model variants to test (default: {DEFAULT_VARIANTS}).")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-runs", type=int, default=5,
                        help="Scoring runs per entity (default: 5 to match the temperature sub-experiment).")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("extra_args", nargs="*")
    args = parser.parse_args()

    input_csv = Path(args.input).resolve()
    if not input_csv.exists():
        sys.exit(f"Input file not found: {input_csv}")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    variants = args.variants or DEFAULT_VARIANTS

    print("Quantization robustness check")
    print(f"  Fixed: scale 0-{SCALE_MAX}, prompt {PROMPT_VERSION}, "
          f"temperature {args.temperature}, {args.num_runs} runs/entity")
    print(f"  Variants:    {len(variants)}")
    print(f"  Input:       {input_csv}")
    print(f"  Output:      {output_root}\n")

    manifest = {
        "experiment": "quantization_robustness",
        "started_at": datetime.now().isoformat(),
        "input_file": str(input_csv),
        "output_root": str(output_root),
        "design": {
            "fixed_scale": f"{SCALE_MIN}-{SCALE_MAX}",
            "fixed_prompt": PROMPT_VERSION,
            "temperature": args.temperature,
            "num_runs": args.num_runs,
            "variants": variants,
        },
        "comparison_baseline": {
            "note": "Compare against Q4_K_M results from the factorial experiment",
            "factorial_path": "output/scoring-experiment-qwen3-30b/scale-0-100_prompt-v2/",
            "caveat": "Factorial baseline used 3 runs/entity; this study uses 5. "
                      "Re-aggregate or note the asymmetry when comparing.",
        },
        "conditions": [],
    }

    results = []
    skipped = 0
    for i, variant in enumerate(variants, 1):
        label = variant_short_name(variant)
        cond_dir = output_root / label
        log_path = cond_dir / "run.log"
        print(f"\n[{i}/{len(variants)}] {label}")

        if args.resume and (cond_dir / "score_metadata.json").exists():
            print(f"  Already completed, skipping (--resume)")
            skipped += 1
            results.append({"label": label, "status": "skipped (resume)", "duration_seconds": 0})
            continue

        cmd = build_command(input_csv, cond_dir, variant, args.base_url,
                            args.num_runs, args.temperature, args.verbose, args.extra_args)

        if args.dry_run:
            print(f"  [DRY RUN] {' '.join(cmd)}")
            results.append({"label": label, "status": "dry_run"})
            continue

        cond_dir.mkdir(parents=True, exist_ok=True)
        results.append(run_condition(cmd, label, log_path))

    manifest["completed_at"] = datetime.now().isoformat()
    manifest["conditions"] = results
    manifest["summary"] = {
        "total": len(variants),
        "success": sum(1 for r in results if r.get("status") == "success"),
        "failed":  sum(1 for r in results if r.get("status") == "failed"),
        "skipped": skipped,
    }
    if not args.dry_run:
        manifest["summary"]["total_duration_seconds"] = round(
            sum(r.get("duration_seconds", 0) for r in results), 1)

    (output_root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*60}\n  EXPERIMENT COMPLETE\n{'='*60}")
    s = manifest["summary"]
    print(f"  Variants: {s['total']} total, {s.get('success', 0)} success, "
          f"{s.get('failed', 0)} failed, {s.get('skipped', 0)} skipped")
    if not args.dry_run:
        print(f"  Total time: {s.get('total_duration_seconds', 0):.0f}s")
    print(f"  Manifest:   {output_root / 'experiment_manifest.json'}")


if __name__ == "__main__":
    main()
