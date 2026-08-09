"""
Run the local pipeline end to end.

    python run_pipeline.py              generate, validate, ingest, train, score
    python run_pipeline.py --no-generate    use whatever is already in the outbox

This is the local path. It does not touch Snowflake, so it works without
credentials. The Airflow DAGs run the same stages as separate tasks against
the warehouse.

Each stage is a subprocess. That keeps the stage boundary honest: a stage
either exits zero or the run stops, and no stage can leave state in another
stage's memory. It is also what Airflow does with BashOperator, so behaviour
here matches behaviour there.

Replaces debug_run.py, which ran three stages and called a summarize_results
function whose entire body was the comment "you already have this, no change
needed".
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_stage(name, script, extra_args=()):
    """Run one stage. Returns its duration. Raises SystemExit on failure."""
    command = [PYTHON, str(PROJECT_ROOT / script), *extra_args]
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    start = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        raise SystemExit(f"\nStage failed: {name} (exit {result.returncode})")

    print(f"\n[{name}: {elapsed:.2f}s]")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run the local pipeline.")
    parser.add_argument("--no-generate", action="store_true",
                        help="skip generation and use the existing outbox")
    parser.add_argument("--users", type=int, default=500)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stages = []
    if not args.no_generate:
        stages.append(("Generate transactions", "etl/generate_transactions.py",
                       ["--users", str(args.users), "--days", str(args.days),
                        "--seed", str(args.seed)]))
    stages += [
        ("Validate", "etl/validate_data.py", []),
        ("Ingest to processed", "etl/ingest_local.py", []),
        ("Train and score", "models/train_local.py", []),
    ]

    timings = []
    overall_start = time.perf_counter()
    for name, script, extra in stages:
        timings.append((name, run_stage(name, script, extra)))
    total = time.perf_counter() - overall_start

    print(f"\n{'=' * 70}\nPipeline completed\n{'=' * 70}")
    for name, seconds in timings:
        print(f"  {name:<28} {seconds:>7.2f}s")
    print(f"  {'total':<28} {total:>7.2f}s")


if __name__ == "__main__":
    main()
