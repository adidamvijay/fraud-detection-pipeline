# models/update_feature_store.py

import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

load_dotenv("/opt/airflow/.env")

SNOW_USER = os.getenv("SNOW_USER")
SNOW_PWD = os.getenv("SNOW_PWD")
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_DATABASE = os.getenv("SNOW_DATABASE", "FRAUD_DB")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA", "PUBLIC")
SNOW_WAREHOUSE = os.getenv("SNOW_WAREHOUSE", "COMPUTE_WH")
SNOW_ROLE = os.getenv("SNOW_ROLE", None)


def get_conn():
    """Return a Snowflake connection."""
    return snowflake.connector.connect(
        user=SNOW_USER,
        password=SNOW_PWD,
        account=SNOW_ACCOUNT,
        warehouse=SNOW_WAREHOUSE,
        database=SNOW_DATABASE,
        schema=SNOW_SCHEMA,
        role=SNOW_ROLE
    )


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
