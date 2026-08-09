"""
Load normalised CSVs into Snowflake RAW_TRANSACTIONS.

Reads data/processed/, writes RAW_TRANSACTIONS, archives the input.

Two things this used to get wrong
--------------------------------
It read data/outbox first and only fell back to data/processed. data/outbox
holds raw, unvalidated files, so the primary path loaded data that had never
been through validate_data.py. It now reads data/processed only, which is
the output of the validate then normalise chain.

It also called write_pandas straight into RAW_TRANSACTIONS, which appends.
Re-running a file, or an Airflow task retry after a partial failure, would
duplicate every row. It now writes to a staging table and MERGEs on
TRANSACTION_ID, so replaying a window updates rather than duplicates.

Paths came from a mix of relative and absolute /project/... literals, which
resolved to C:\\project on Windows. They now come from config.py.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    ARCHIVE_DIR, PROCESSED_DIR, TRANSACTION_COLUMNS, ensure_dirs,
    missing_snowflake_vars, snowflake_args,
)

TARGET_TABLE = "RAW_TRANSACTIONS"
STAGING_TABLE = "RAW_TRANSACTIONS_LOAD"

ARCHIVE_PROCESSED_DIR = ARCHIVE_DIR / "processed"

MERGE_SQL = f"""
MERGE INTO {TARGET_TABLE} tgt
USING {STAGING_TABLE} src
    ON tgt.TRANSACTION_ID = src.TRANSACTION_ID
WHEN MATCHED THEN UPDATE SET
    tgt.USER_ID = src.USER_ID,
    tgt.EVENT_TIME = src.EVENT_TIME,
    tgt.AMOUNT = src.AMOUNT,
    tgt.MERCHANT_ID = src.MERCHANT_ID,
    tgt.MERCHANT_COUNTRY = src.MERCHANT_COUNTRY,
    tgt.TXN_TYPE = src.TXN_TYPE,
    tgt.DEVICE_ID = src.DEVICE_ID,
    tgt.IP_ADDRESS = src.IP_ADDRESS,
    tgt.LABEL = src.LABEL
WHEN NOT MATCHED THEN INSERT (
    TRANSACTION_ID, USER_ID, EVENT_TIME, AMOUNT, MERCHANT_ID,
    MERCHANT_COUNTRY, TXN_TYPE, DEVICE_ID, IP_ADDRESS, LABEL, LOADED_AT
) VALUES (
    src.TRANSACTION_ID, src.USER_ID, src.EVENT_TIME, src.AMOUNT,
    src.MERCHANT_ID, src.MERCHANT_COUNTRY, src.TXN_TYPE, src.DEVICE_ID,
    src.IP_ADDRESS, src.LABEL, CURRENT_TIMESTAMP()
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


def prepare(df):
    """Coerce to the warehouse schema and uppercase the column names."""
    for column in TRANSACTION_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[TRANSACTION_COLUMNS].copy()

    # TIMESTAMP_NTZ holds UTC; strip any offset at this boundary so the
    # connector does not send a TIMESTAMP_TZ into an NTZ column.
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df["event_time"] = df["event_time"].dt.tz_localize(None)

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").round(2)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    for column in ("transaction_id", "user_id", "merchant_id",
                   "merchant_country", "txn_type", "device_id", "ip_address"):
        df[column] = df[column].astype(str).str.strip()

    # NOT NULL columns: a row that lost its timestamp or amount in transit
    # cannot be loaded, and should not silently become zero.
    before = len(df)
    df = df.dropna(subset=["event_time", "amount"])
    if len(df) < before:
        print(f"  dropped {before - len(df)} rows with a null event_time or amount")

    df.columns = [c.upper() for c in df.columns]
    return df


def load(conn, df):
    """Stage, merge, truncate. Returns (staged, rows_after_merge)."""
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
    ensure_dirs()
    ARCHIVE_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(PROCESSED_DIR.glob("*.csv"))
    if not files:
        print(f"No files to load in {PROCESSED_DIR}")
        return 0

    conn = get_conn()
    total_rows = 0
    try:
        frames = [pd.read_csv(path) for path in files]
        combined = pd.concat(frames, ignore_index=True)
        print(f"read {len(files)} files, {len(combined):,} rows from {PROCESSED_DIR}")

        prepared = prepare(combined)
        staged, merged = load(conn, prepared)
        total_rows = staged

        print(f"staged {staged:,} rows into {STAGING_TABLE}")
        print(f"MERGE into {TARGET_TABLE}: {merged}")

        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TARGET_TABLE}")
        print(f"{TARGET_TABLE} now holds {cursor.fetchone()[0]:,} rows")
        cursor.close()

        # Archive only after the merge has committed.
        for path in files:
            path.replace(ARCHIVE_PROCESSED_DIR / path.name)
        print(f"archived {len(files)} files to {ARCHIVE_PROCESSED_DIR}")
    finally:
        conn.close()

    return total_rows


if __name__ == "__main__":
    main()
