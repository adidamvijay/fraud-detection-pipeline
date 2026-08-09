import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# -----------------------------------------------------
# ENV + PATH SETUP (CRITICAL FOR AIRFLOW)
# -----------------------------------------------------
load_dotenv("/opt/airflow/.env")

BASE_DIR = Path("/project")
MODEL_PATH = BASE_DIR / "models" / "artifacts" / "isoforest_local_v2.joblib"

SNOW_USER = os.getenv("SNOW_USER")
SNOW_PWD = os.getenv("SNOW_PWD")
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_DATABASE = os.getenv("SNOW_DATABASE")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA")
SNOW_WAREHOUSE = os.getenv("SNOW_WAREHOUSE")
SNOW_ROLE = os.getenv("SNOW_ROLE")

# -----------------------------------------------------
# SNOWFLAKE CONNECTION
# -----------------------------------------------------
def get_conn():
    return snowflake.connector.connect(
        user=SNOW_USER,
        password=SNOW_PWD,
        account=SNOW_ACCOUNT,
        warehouse=SNOW_WAREHOUSE,
        database=SNOW_DATABASE,
        schema=SNOW_SCHEMA,
        role=SNOW_ROLE,
    )

# -----------------------------------------------------
# STEP 1: LOAD ONLY NEW TRANSACTIONS
# -----------------------------------------------------
def load_new_transactions(hours_back=48):
    sql = f"""
    WITH recent AS (
        SELECT *
        FROM RAW_TRANSACTIONS
        WHERE EVENT_TIME >= DATEADD(hour, -{hours_back}, CURRENT_TIMESTAMP())
    )
    SELECT r.*
    FROM recent r
    LEFT JOIN FRAUD_SCORES fs
        ON r.TRANSACTION_ID = fs.TRANSACTION_ID
    WHERE fs.TRANSACTION_ID IS NULL
    ORDER BY r.EVENT_TIME;
    """

    conn = get_conn()
    df = pd.read_sql(sql, conn)
    conn.close()

    if df.empty:
        return df

    df.columns = df.columns.str.upper()
    df["EVENT_TIME"] = pd.to_datetime(df["EVENT_TIME"], errors="coerce")
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0.0)

    return df

# -----------------------------------------------------
# STEP 2: LOAD FEATURE STORE
# -----------------------------------------------------
def load_feature_store():
    sql = "SELECT * FROM FEATURE_STORE"
    conn = get_conn()
    df = pd.read_sql(sql, conn)
    conn.close()

    if df.empty:
        return df

    df.columns = df.columns.str.upper()
    df["USER_ID"] = df["USER_ID"].astype(str)
    return df

# -----------------------------------------------------
# STEP 3: MANUAL FEATURE COMPUTATION (BACKUP)
# -----------------------------------------------------
def compute_features(df):
    records = []

    for user, grp in df.groupby("USER_ID"):
        grp = grp.sort_values("EVENT_TIME").reset_index(drop=True)

        for i, row in grp.iterrows():
            ts = row["EVENT_TIME"]
            window = grp[
                (grp["EVENT_TIME"] >= ts - timedelta(hours=24))
                & (grp["EVENT_TIME"] <= ts)
            ]

            records.append({
                "TRANSACTION_ID": row["TRANSACTION_ID"],
                "USER_ID": user,
                "EVENT_TIME": ts,
                "TOTAL_24H": window["AMOUNT"].sum(),
                "COUNT_24H": len(window),
                "AVG_24H": window["AMOUNT"].mean() if len(window) > 0 else 0.0,
                "HOURS_SINCE_LAST": (
                    (ts - grp.loc[i - 1, "EVENT_TIME"]).total_seconds() / 3600
                    if i > 0 else -1.0
                ),
            })

    return pd.DataFrame(records)

# -----------------------------------------------------
# STEP 4: MERGE FEATURE STORE + MANUAL FEATURES
# -----------------------------------------------------
def merge_features(txns, fs):
    txns["USER_ID"] = txns["USER_ID"].astype(str)

    if fs.empty:
        return compute_features(txns)

    merged = txns.merge(fs, on="USER_ID", how="left")

    manual = compute_features(txns)

    merged = merged.merge(
        manual,
        on=["TRANSACTION_ID", "USER_ID"],
        how="left",
        suffixes=("", "_MANUAL"),
    )

    merged["TOTAL_24H"] = merged["LAST_24H_TOTAL"].fillna(merged["TOTAL_24H"])
    merged["COUNT_24H"] = merged["LAST_24H_TXN_COUNT"].fillna(merged["COUNT_24H"])
    merged["AVG_24H"] = merged["LAST_24H_AVG_AMOUNT"].fillna(merged["AVG_24H"])

    return merged[
        ["TRANSACTION_ID", "USER_ID", "EVENT_TIME",
         "TOTAL_24H", "COUNT_24H", "AVG_24H", "HOURS_SINCE_LAST"]
    ]

# -----------------------------------------------------
# STEP 5: LOAD MODEL
# -----------------------------------------------------
def load_model():
    clf = joblib.load(MODEL_PATH)
    version = MODEL_PATH.stem
    return clf, version

# -----------------------------------------------------
# STEP 6: SCORE TRANSACTIONS
# -----------------------------------------------------
def score_transactions(feats, clf, version):
    X = feats[["TOTAL_24H", "COUNT_24H", "AVG_24H", "HOURS_SINCE_LAST"]].fillna(0)

    raw = -clf.decision_function(X)
    norm = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() != raw.min() else raw

    feats["SCORE"] = norm
    feats["FLAGGED"] = feats["SCORE"] >= np.quantile(norm, 0.98)
    feats["MODEL_VERSION"] = version
    feats["SCORED_AT"] = datetime.now(timezone.utc)

    # TIMESTAMP_NTZ holds UTC; drop the offset at this boundary.
    for col in ("EVENT_TIME", "SCORED_AT"):
        feats[col] = pd.to_datetime(feats[col], utc=True).dt.tz_localize(None)

    return feats[["TRANSACTION_ID", "USER_ID", "EVENT_TIME",
                  "SCORE", "FLAGGED", "MODEL_VERSION", "SCORED_AT"]]

# -----------------------------------------------------
# STEP 7: WRITE TO STAGING TABLE
# -----------------------------------------------------
def write_scores(df):
    conn = get_conn()
    df.columns = df.columns.str.upper()

    success, _, rows, _ = write_pandas(
        conn, df, "FRAUD_SCORES_LOAD", use_logical_type=True
    )
    conn.close()

    print(f"Loaded {rows} rows into FRAUD_SCORES_LOAD")

# -----------------------------------------------------
# STEP 8: MERGE INTO FINAL TABLE
# -----------------------------------------------------
def merge_scores():
    # Idempotency: keyed on TRANSACTION_ID, so re-scoring a window
    # updates existing rows instead of inserting duplicates.
    sql = """
    MERGE INTO FRAUD_SCORES tgt
    USING FRAUD_SCORES_LOAD src
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
        src.TRANSACTION_ID,
        src.USER_ID,
        src.EVENT_TIME,
        src.SCORE,
        src.FLAGGED,
        src.MODEL_VERSION,
        src.SCORED_AT
    );
    """

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()
    conn.close()

# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
if __name__ == "__main__":
    txns = load_new_transactions(hours_back=48)

    if txns.empty:
        print("No new transactions. Exiting.")
        exit(0)

    fs = load_feature_store()
    feats = merge_features(txns, fs)

    clf, version = load_model()
    scored = score_transactions(feats, clf, version)

    write_scores(scored)
    merge_scores()

    print("✅ Feature scoring pipeline completed successfully.")
