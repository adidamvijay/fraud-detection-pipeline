# models/update_feature_store.py
"""
Populate FEATURE_STORE with per-user 24-hour aggregates.

NOT PART OF THE WORKING PIPELINE. Nothing currently reads FEATURE_STORE:
models/score_batch.py computes features from RAW_TRANSACTIONS directly, and
this script was removed from the daily DAG because scheduling it would
schedule work no other task consumes.

It is kept because a per-user aggregate table is what a request-time scoring
API would need, and this is the sketch of it. It has never been run against a
live warehouse.

Its feature calculation is also the old nested loop rather than
models/features.compute_features, so it would need reworking before use.
"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import missing_snowflake_vars, snowflake_args  # noqa: E402


def get_conn():
    """Return a Snowflake connection built from .env via config.py."""
    import snowflake.connector
    args = snowflake_args()
    if args is None:
        raise SystemExit(
            "Snowflake credentials are not set. Missing: "
            f"{', '.join(missing_snowflake_vars())}.\n"
            "Copy .env.example to .env and fill it in.")
    return snowflake.connector.connect(**args)


def query_recent_transactions():
    """Fetch last 24 hours of user transactions."""
    sql = """
    SELECT USER_ID, EVENT_TIME, AMOUNT
    FROM RAW_TRANSACTIONS
    WHERE EVENT_TIME >= DATEADD('hour', -24, CURRENT_TIMESTAMP())
    ORDER BY USER_ID, EVENT_TIME;
    """

    conn = get_conn()
    df = pd.read_sql(sql, conn)
    conn.close()

    if df.empty:
        return df

    df["EVENT_TIME"] = pd.to_datetime(df["EVENT_TIME"], errors="coerce")
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0.0)

    return df


def compute_user_features(df):
    """Compute 24-hour rolling features per user."""
    records = []

    for user, group in df.groupby("USER_ID"):
        group = group.sort_values("EVENT_TIME")

        for i, row in group.iterrows():
            ts = row["EVENT_TIME"]
            window = group[(group["EVENT_TIME"] >= ts - timedelta(hours=24)) &
                           (group["EVENT_TIME"] <= ts)]

            total = window["AMOUNT"].sum()
            count = len(window)
            avg = window["AMOUNT"].mean() if count > 0 else 0.0

            records.append({
                "USER_ID": user,
                "LAST_24H_TOTAL": float(total),
                "LAST_24H_TXN_COUNT": int(count),
                "LAST_24H_AVG_AMOUNT": float(avg)
            })

    return pd.DataFrame(records)


def upsert_into_feature_store(df):
    """Insert or update the FEATURE_STORE table."""
    conn = get_conn()
    cur = conn.cursor()

    # Create temporary table with same structure
    temp_table = "FEATURE_STORE_STAGE"

    cur.execute(f"""
        CREATE OR REPLACE TEMP TABLE {temp_table} (
            USER_ID STRING,
            LAST_24H_TOTAL FLOAT,
            LAST_24H_TXN_COUNT INT,
            LAST_24H_AVG_AMOUNT FLOAT
        );
    """)

    # Upload dataframe into temp table
    from snowflake.connector.pandas_tools import write_pandas
    success, chunks, rows, _ = write_pandas(conn, df, temp_table)
    print(f"Uploaded {rows} rows into stage table.")

    merge_sql = f"""
    MERGE INTO FEATURE_STORE tgt
    USING {temp_table} src
      ON tgt.USER_ID = src.USER_ID
    WHEN MATCHED THEN UPDATE SET
      LAST_24H_TOTAL = src.LAST_24H_TOTAL,
      LAST_24H_TXN_COUNT = src.LAST_24H_TXN_COUNT,
      LAST_24H_AVG_AMOUNT = src.LAST_24H_AVG_AMOUNT,
      UPDATED_AT = CURRENT_TIMESTAMP()
    WHEN NOT MATCHED THEN INSERT (
      USER_ID,
      LAST_24H_TOTAL,
      LAST_24H_TXN_COUNT,
      LAST_24H_AVG_AMOUNT,
      UPDATED_AT
    ) VALUES (
      src.USER_ID,
      src.LAST_24H_TOTAL,
      src.LAST_24H_TXN_COUNT,
      src.LAST_24H_AVG_AMOUNT,
      CURRENT_TIMESTAMP()
    );
    """

    cur.execute(merge_sql)
    print("Feature Store updated successfully.")

    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    print("Fetching last 24 hours transactions...")
    df_tx = query_recent_transactions()

    if df_tx.empty:
        print("No recent transactions found. Nothing to update.")
        exit()

    print("Computing user features...")
    df_features = compute_user_features(df_tx)

    print("Upserting Feature Store...")
    upsert_into_feature_store(df_features)

    print("Feature Store update completed.")
