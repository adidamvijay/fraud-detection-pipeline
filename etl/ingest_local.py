"""
Normalise validated files into the canonical on-disk schema.

Reads data/validated/, writes data/processed/, moves the source to
data/archive/validated/.

This step exists so that everything downstream can assume one column order,
one set of column names and parsed types, regardless of how the upstream file
was produced. The Snowflake loader reads data/processed/ only.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    ARCHIVE_VALIDATED_DIR, PROCESSED_DIR, TRANSACTION_COLUMNS,
    VALIDATED_DIR, ensure_dirs,
)


def normalize_df(df):
    """Lowercase column names, add any missing columns, fix column order."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    for col in TRANSACTION_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", format="mixed")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # Drops any extra columns, including helper columns from validation.
    return df[TRANSACTION_COLUMNS]


def process_validated():
    """Normalise every validated file. Returns the paths written."""
    ensure_dirs()

    files = sorted(VALIDATED_DIR.glob("*.csv"))
    if not files:
        print(f"No files to ingest in {VALIDATED_DIR}")
        return []

    written = []
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            # An unreadable file would otherwise block every later run.
            print(f"{path.name}: unreadable, moving to archive without processing. {exc}")
            path.replace(ARCHIVE_VALIDATED_DIR / path.name)
            continue

        normalized = normalize_df(df)

        # Deterministic output name: re-processing the same input overwrites
        # rather than accumulating near-duplicate files.
        destination = PROCESSED_DIR / path.name.replace("valid_", "")
        normalized.to_csv(destination, index=False)
        path.replace(ARCHIVE_VALIDATED_DIR / path.name)

        written.append(destination)
        print(f"{path.name} -> {destination.name} ({len(normalized)} rows)")

    print(f"\ningested {len(written)} files into {PROCESSED_DIR}")
    return written


if __name__ == "__main__":
    process_validated()
