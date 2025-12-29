# etl/ingest_to_snowflake.py
"""
CSV -> Snowflake loader. If data/outbox is empty, will also attempt processing
files that already sit in data/processed (useful if you already ran ingest_local).
"""

import os, glob, sys, traceback
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from snowflake.connector import connect
from snowflake.connector.pandas_tools import write_pandas

load_dotenv()

SNOW_USER = os.getenv("SNOW_USER")
SNOW_PWD  = os.getenv("SNOW_PWD")
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_DATABASE = os.getenv("SNOW_DATABASE", "FRAUD_DB")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA", "PUBLIC")
SNOW_WAREHOUSE = os.getenv("SNOW_WAREHOUSE", "COMPUTE_WH")
SNOW_ROLE = os.getenv("SNOW_ROLE", None)

OUTBOX_DIR = Path("data/outbox")
PROCESSED_DIR = Path("/project/data/processed")
ARCHIVE_DIR = Path("/project/data/outbox/archived")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

REQ_COLS = ["transaction_id","user_id","event_time","amount",
            "merchant_id","merchant_country","txn_type","device_id","ip_address","label"]

def validate_env():
    missing = [n for n,v in (("SNOW_USER",SNOW_USER),("SNOW_PWD",SNOW_PWD),("SNOW_ACCOUNT",SNOW_ACCOUNT)) if not v]
    if missing:
        raise SystemExit(f"Missing Snowflake env vars: {', '.join(missing)}")

def get_conn():
    conn_args = {
        "user": SNOW_USER,
        "password": SNOW_PWD,
        "account": SNOW_ACCOUNT,
        "warehouse": SNOW_WAREHOUSE,
        "database": SNOW_DATABASE,
        "schema": SNOW_SCHEMA
    }
    if SNOW_ROLE:
        conn_args["role"] = SNOW_ROLE
    return connect(**conn_args)

def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQ_COLS:
        if c not in df.columns:
            df[c] = None
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce', utc=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    # trim strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip().replace({'None': None, 'nan': None})
    df.columns = [c.upper() for c in df.columns]
    return df

def write_df_to_snowflake(df: pd.DataFrame, table_name="RAW_TRANSACTIONS"):
    conn = None
    try:
        conn = get_conn()
    except Exception as e:
        raise RuntimeError(f"Snowflake connection failed: {e}")
    try:
        success, nchunks, nrows, _ = write_pandas(conn, df, table_name, use_logical_type=True)
        return success, nchunks, nrows
    finally:
        if conn:
            conn.close()

def process_file(fpath: str, came_from_outbox: bool=True):
    print("Loading:", fpath)
    try:
        df = pd.read_csv(fpath, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as e:
        print("Failed to read CSV:", e)
        return 0
    df = _prepare_df(df)
    try:
        success, nchunks, nrows = write_df_to_snowflake(df)
        print(f"write_pandas => success={success}, nrows={nrows}, nchunks={nchunks}")
    except Exception as e:
        print("Error writing to Snowflake:", e)
        print(traceback.format_exc())
        return 0

    # If file came from outbox, move it to processed (atomic)
    dest = PROCESSED_DIR.joinpath(Path(fpath).name)
    try:
        if came_from_outbox and Path(fpath).exists():
            os.replace(fpath, str(dest))
            print(f"Moved {fpath} -> {dest} (rows={nrows})")
        else:
            print(f"Processed (from processed dir): {fpath} (rows={nrows})")
    except Exception as e:
        print("Failed to move file after processing:", e)

    return nrows

def main():
    try:
        validate_env()
    except Exception as e:
        print("ENV ERROR:", e)
        sys.exit(2)

    # Prefer processing outbox first
    outbox_files = sorted(glob.glob(str(OUTBOX_DIR.joinpath("*.csv"))))
    if outbox_files:
        for f in outbox_files:
            process_file(f, came_from_outbox=True)
        return

    # If outbox empty, fallback to processed dir (useful if ingest_local already ran)
    processed_files = sorted(glob.glob(str(PROCESSED_DIR.joinpath("*.csv"))))
    if processed_files:
        for f in processed_files:
            # We won't move these (they are already in processed)
            process_file(f, came_from_outbox=False)
        return

    print("No files to process in:", OUTBOX_DIR)

if __name__ == "__main__":
    main()
