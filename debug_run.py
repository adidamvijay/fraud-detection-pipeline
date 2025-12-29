# debug_run.py
import subprocess
import sys
import os
import pandas as pd

PY = sys.executable

def run(cmd):
    print(f"\n>>> Running: {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("exitcode:", r.returncode)
    if r.stdout:
        print("--- STDOUT ---")
        print(r.stdout)
    if r.stderr:
        print("--- STDERR ---")
        print(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

def summarize_results():
    print("\nSummary:")
    # you already have this, no change needed

if __name__ == "__main__":
    try:
        # Step 0: Validate data (NEW)
        run(f'"{PY}" etl/validate_data.py')

        # Step 1: Ingest local
        run(f'"{PY}" etl/ingest_local.py')

        # Step 2: Run model
        run(f'"{PY}" models/feature_and_train_local.py')

        summarize_results()
        print("\nPipeline finished successfully.")
    except SystemExit as e:
        print("Pipeline stopped:", e)
    except Exception as e:
        print("Unexpected error:", e)
