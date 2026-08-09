"""
Score transactions from Snowflake and write the results back.

Reads RAW_TRANSACTIONS, computes the seven per-user features, scores with the
most recent trained artifact, and MERGEs into FRAUD_SCORES.

What changed
------------
It used to read local CSVs from data/processed and load a hardcoded artifact
at /project/models/artifacts/isoforest_local_v1.joblib, a path that no longer
existed and would not have resolved on Windows anyway. Reading the warehouse
is what makes this the warehouse path rather than a second local pipeline.

It also had a fourth copy of the feature calculation. It now imports
models.features, so training and scoring cannot drift apart.

The threshold comes from the metadata file written next to the artifact by
train_local.py, so the flag written here corresponds to the operating point
that was measured rather than to a number chosen at scoring time.

Window
------
Scores a trailing window, default 30 days, so a scheduled hourly run does not
rescore all history. Features need a user's recent history to be meaningful,
so the window has to be wider than the 24-hour feature window; 30 days is the
same span the generator produces.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    MODEL_DIR, missing_snowflake_vars, snowflake_args,
)
from models.features import FEATURE_COLUMNS, compute_features  # noqa: E402

TARGET_TABLE = "FRAUD_SCORES"
STAGING_TABLE = "FRAUD_SCORES_LOAD"

SCORE_COLUMNS = ["TRANSACTION_ID", "USER_ID", "EVENT_TIME", "SCORE",
                 "FLAGGED", "MODEL_VERSION", "SCORED_AT"]

MERGE_SQL = f"""
MERGE INTO {TARGET_TABLE} tgt
USING {STAGING_TABLE} src
    ON tgt.TRANSACTION_ID = src.TRANSACTION_ID
WHEN MATCHED THEN UPDATE SET
    tgt.USER_ID = src.USER_ID,
    tgt.EVENT_TIME = src.EVENT_TIME,
    tgt.SCORE = src.SCORE,
    tgt.FLAGGED = src.FLAGGED,
    tgt.MODEL_VERSION = src.MODEL_VERSION,
    tgt.SCORED_AT = src.SCORED_AT
WHEN NOT MATCHED THEN INSERT (
    TRANSACTION_ID, USER_ID, EVENT_TIME, SCORE, FLAGGED, MODEL_VERSION, SCORED_AT
) VALUES (
    src.TRANSACTION_ID, src.USER_ID, src.EVENT_TIME, src.SCORE,
    src.FLAGGED, src.MODEL_VERSION, src.SCORED_AT
);
"""


def get_conn():
    import snowflake.connector
    args = snowflake_args()
    if args is None:
        raise SystemExit(
            "Snowflake credentials are not set. Missing: "
            f"{', '.join(missing_snowflake_vars())}.\n"
            "Copy .env.example to .env and fill it in.")
    return snowflake.connector.connect(**args)


def latest_artifact():
    """Newest .joblib in models/artifacts, with its metadata if present."""
    artifacts = sorted(MODEL_DIR.glob("*.joblib"))
    if not artifacts:
        raise SystemExit(
            f"No model artifact in {MODEL_DIR}. Run models/train_local.py first.")

    path = artifacts[-1]
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return path, joblib.load(path), metadata


def read_transactions(conn, days):
    sql = f"""
        SELECT TRANSACTION_ID, USER_ID, EVENT_TIME, AMOUNT
        FROM RAW_TRANSACTIONS
        WHERE EVENT_TIME >= DATEADD('day', -{days}, CURRENT_TIMESTAMP())
        ORDER BY EVENT_TIME
    """
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        df = cursor.fetch_pandas_all()
    finally:
        cursor.close()

    df.columns = [c.lower() for c in df.columns]
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def score(features, model, version, threshold):
    X = features[FEATURE_COLUMNS].fillna(0)
    raw = -model.decision_function(X)

    low, high = float(raw.min()), float(raw.max())
    normalised = (raw - low) / (high - low) if high != low else raw * 0.0

    out = features[["transaction_id", "user_id", "event_time"]].copy()
    out["score"] = normalised
    # The flag is applied to the raw score, not the batch-scaled one: min-max
    # scaling depends on what happens to be in the batch, so a fixed
    # threshold on it would mean something different every run.
    out["flagged"] = raw >= threshold
    out["model_version"] = version
    out["scored_at"] = datetime.now(timezone.utc).replace(tzinfo=None)

    out.columns = [c.upper() for c in out.columns]
    out["EVENT_TIME"] = pd.to_datetime(out["EVENT_TIME"]).dt.tz_localize(None)
    return out[SCORE_COLUMNS]


def write(conn, df):
    from snowflake.connector.pandas_tools import write_pandas

    cursor = conn.cursor()
    try:
        cursor.execute(f"TRUNCATE TABLE IF EXISTS {STAGING_TABLE}")
        success, _, staged, _ = write_pandas(
            conn, df, STAGING_TABLE, use_logical_type=True)
        if not success:
            raise RuntimeError(f"write_pandas into {STAGING_TABLE} failed")

        cursor.execute(MERGE_SQL)
        merged = cursor.fetchone()
        cursor.execute(f"TRUNCATE TABLE {STAGING_TABLE}")
        return staged, merged
    finally:
        cursor.close()


def main():
    parser = argparse.ArgumentParser(description="Score RAW_TRANSACTIONS into FRAUD_SCORES.")
    parser.add_argument("--days", type=int, default=30,
                        help="trailing window of transactions to score")
    args = parser.parse_args()

    path, model, metadata = latest_artifact()
    version = metadata.get("version", path.stem)
    threshold = metadata.get("threshold")
    if threshold is None:
        raise SystemExit(
            f"{path.name} has no threshold in its metadata. Retrain with "
            "models/train_local.py so the operating point is recorded.")

    print(f"model {version}")
    print(f"  threshold {threshold:.5f} by {metadata.get('threshold_rule', 'unknown rule')}")

    conn = get_conn()
    try:
        transactions = read_transactions(conn, args.days)
        print(f"read {len(transactions):,} transactions from RAW_TRANSACTIONS "
              f"(last {args.days} days)")
        if transactions.empty:
            print("nothing to score")
            return

        features = compute_features(transactions)
        print(f"computed features for {len(features):,} transactions")

        scored = score(features, model, version, threshold)
        staged, merged = write(conn, scored)

        print(f"staged {staged:,} rows into {STAGING_TABLE}")
        print(f"MERGE into {TARGET_TABLE}: {merged}")

        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*), SUM(IFF(FLAGGED, 1, 0)) FROM {TARGET_TABLE}")
        total, flagged = cursor.fetchone()
        cursor.close()
        print(f"{TARGET_TABLE} now holds {total:,} rows, {flagged:,} flagged "
              f"({flagged / total:.2%} alert rate)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
