# models/score_realtime.py
"""
Real-time scoring script.

- Loads joblib model from models/artifacts/*.joblib
- Reads CSVs from data/processed/*.csv (all files)
- Computes windowed features per user consistent with training
- Scores using IsolationForest (model stored as joblib)
- Writes scores to Snowflake FRAUD_SCORES_LOAD (via write_pandas) and MERGEs into FRAUD_SCORES
- Also writes a local data/alerts.csv file for quick inspection
"""

import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from snowflake.connector import connect
from snowflake.connector.pandas_tools import write_pandas

# --- config / env ---
load_dotenv()

SNOW_USER = os.getenv("SNOW_USER")
SNOW_PWD  = os.getenv("SNOW_PWD")
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_DATABASE = os.getenv("SNOW_DATABASE", "FRAUD_DB")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA", "PUBLIC")
SNOW_WAREHOUSE = os.getenv("SNOW_WAREHOUSE", "COMPUTE_WH")
SNOW_ROLE = os.getenv("SNOW_ROLE", None)

PROCESSED_DIR = "data/processed"
MODEL_PATH = "/project/models/artifacts/isoforest_local_v1.joblib"
ALERTS_PATH = "data/alerts.csv"

# threshold quantile to flag anomalies
THRESHOLD_PCT = float(os.getenv("SCORE_THRESHOLD_PCT", 0.98))

# --- helpers ---
def get_conn():
    args = {
        "user": SNOW_USER,
        "password": SNOW_PWD,
        "account": SNOW_ACCOUNT,
        "warehouse": SNOW_WAREHOUSE,
        "database": SNOW_DATABASE,
        "schema": SNOW_SCHEMA,
    }
    if SNOW_ROLE:
        args["role"] = SNOW_ROLE
    return connect(**args)

def load_model(model_path=MODEL_PATH):
    o = joblib.load(model_path)
    # backward compat: if you stored dict/object, adapt accordingly
    if isinstance(o, dict) and 'model' in o:
        return o['model'], o.get('features', None), o.get('model_version', None)
    # else assume it's a raw model
    return o, None, os.path.basename(model_path)

def read_processed_csvs(limit_files=None):
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
    if not files:
        print("No processed CSVs found in", PROCESSED_DIR)
        return pd.DataFrame()
    if limit_files:
        files = files[-limit_files:]
    dfs = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""])
            d.columns = [c.lower() for c in d.columns]
            # ensure event_time exists and parse
            if 'event_time' in d.columns:
                d['event_time'] = pd.to_datetime(d['event_time'], errors='coerce', utc=True)
            else:
                d['event_time'] = pd.NaT
            # coerce amount
            if 'amount' in d.columns:
                d['amount'] = pd.to_numeric(d['amount'], errors='coerce').fillna(0.0)
            else:
                d['amount'] = 0.0
            dfs.append(d)
        except Exception as e:
            print("Failed reading", f, e)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # drop rows without transaction_id or user_id
    df = df[df['transaction_id'].notna() & df['user_id'].notna()].copy()
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the same features as the training:
      - total_amount_24h
      - txn_count_24h
      - avg_amount_24h
      - hours_since_last_txn
    """
    if df.empty:
        return df

    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce', utc=True)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)

    records = []
    # ensure stable ordering
    df = df.sort_values(['user_id','event_time']).reset_index(drop=True)

    for user, group in df.groupby('user_id'):
        group = group.sort_values('event_time').reset_index(drop=True)
        times = group['event_time']
        amounts = group['amount'].astype(float)
        for i, row in group.iterrows():
            ts = row['event_time']
            if pd.isna(ts):
                # skip if no event_time
                continue
            window_start = ts - pd.Timedelta(hours=24)
            mask = (times >= window_start) & (times <= ts)
            w_amts = amounts[mask]
            total_amt = float(w_amts.sum()) if not w_amts.empty else 0.0
            cnt = int(mask.sum())
            avg_amt = float(w_amts.mean()) if cnt > 0 else 0.0
            if i > 0:
                last_ts = group.loc[i-1, 'event_time']
                hours_since_last = (ts - last_ts).total_seconds() / 3600.0
            else:
                hours_since_last = -1.0
            records.append({
                'transaction_id': row['transaction_id'],
                'user_id': user,
                'event_time': ts,
                'total_amount_24h': total_amt,
                'txn_count_24h': cnt,
                'avg_amount_24h': avg_amt,
                'hours_since_last_txn': hours_since_last
            })
    feats = pd.DataFrame(records)
    return feats

def score_dataframe(feats: pd.DataFrame, model, model_version=None, threshold_pct=THRESHOLD_PCT):
    """
    Returns DataFrame with columns:
      TRANSACTION_ID, SCORE, IS_ANOMALY (bool), MODEL_VERSION, SCORING_TS
    """
    if feats.empty:
        return pd.DataFrame(columns=['TRANSACTION_ID','SCORE','IS_ANOMALY','MODEL_VERSION','SCORING_TS'])

    X = feats[['total_amount_24h','txn_count_24h','avg_amount_24h','hours_since_last_txn']].fillna(0)
    raw_scores = -model.decision_function(X)  # higher -> more anomalous

    # normalize to [0,1]
    mi, ma = float(raw_scores.min()), float(raw_scores.max())
    denom = (ma - mi) if (ma - mi) != 0 else 1.0
    norm = (raw_scores - mi) / denom

    feats = feats.copy()
    feats['score'] = norm
    cutoff = float(np.quantile(norm, threshold_pct))
    feats['is_anomaly'] = feats['score'] >= cutoff

    out = feats[['transaction_id','score','is_anomaly']].copy()
    out['model_version'] = model_version or "unset"
    # SCORING_TS as ISO Zulu string to be Snowflake-friendly
    out['scoring_ts'] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Uppercase columns to match FRAUD_SCORES_LOAD
    out.columns = [c.upper() for c in out.columns]

    # Ensure types
    out['SCORE'] = out['SCORE'].astype(float)
    out['IS_ANOMALY'] = out['IS_ANOMALY'].astype(bool)
    return out

def write_scores_to_snowflake(df_scores: pd.DataFrame):
    if df_scores.empty:
        print("No scores to write.")
        return 0

    conn = get_conn()
    try:
        # write to FRAUD_SCORES_LOAD staging table
        # use_logical_type=True to preserve TIMESTAMP types if any
        success, nchunks, nrows, _ = write_pandas(conn, df_scores, 'FRAUD_SCORES_LOAD', use_logical_type=True)
        print("write_pandas FRAUD_SCORES_LOAD =>", success, nrows, nchunks)
        # MERGE into FRAUD_SCORES
        cs = conn.cursor()
        try:
            merge_sql = """
            MERGE INTO FRAUD_SCORES t
            USING FRAUD_SCORES_LOAD s
            ON t.TXN_ID = s.TXN_ID
            WHEN MATCHED THEN UPDATE SET
              SCORE = s.SCORE,
              IS_ANOMALY = s.IS_ANOMALY,
              MODEL_VERSION = s.MODEL_VERSION,
              FEATURES = NULL,
              SCORING_TS = TRY_TO_TIMESTAMP_NTZ(s.SCORING_TS)
            WHEN NOT MATCHED THEN INSERT (TXN_ID, SCORE, IS_ANOMALY, MODEL_VERSION, FEATURES, SCORING_TS, CREATED_AT)
              VALUES (s.TXN_ID, s.SCORE, s.IS_ANOMALY, s.MODEL_VERSION, NULL, TRY_TO_TIMESTAMP_NTZ(s.SCORING_TS), CURRENT_TIMESTAMP())
            """
            cs.execute(merge_sql)
            print("MERGE into FRAUD_SCORES completed.")
            # Optionally truncate FRAUD_SCORES_LOAD
            cs.execute("TRUNCATE TABLE FRAUD_SCORES_LOAD")
            print("Truncated FRAUD_SCORES_LOAD (cleanup).")
        finally:
            cs.close()
    finally:
        conn.close()
    return nrows

# -----------------------
# Main
# -----------------------
def main():
    print("Loading model:", MODEL_PATH)
    model, feat_cols, model_version = load_model(MODEL_PATH)
    print("Model loaded. model_version:", model_version)

    print("Reading processed CSVs from", PROCESSED_DIR)
    raw = read_processed_csvs()
    if raw.empty:
        print("No processed CSVs found. Exiting.")
        return

    feats = compute_features(raw)
    if feats.empty:
        print("No features computed (no rows with event_time). Exiting.")
        return

    print("Computed features rows:", len(feats))

    scored = score_dataframe(feats, model, model_version=model_version, threshold_pct=THRESHOLD_PCT)
    print("Scored rows:", len(scored))
    print("Sample scored rows:", scored.head(5).to_dict(orient='records'))

    # write local alerts for convenience
    try:
        alerts = scored[scored['IS_ANOMALY'] == True].copy()
        if not alerts.empty:
            alerts.to_csv(ALERTS_PATH, index=False)
            print("Wrote local alerts:", ALERTS_PATH, "count:", len(alerts))
        else:
            print("No alerts flagged in this run.")
    except Exception as e:
        print("Failed to write local alerts:", e)

    # write to Snowflake (FRAUD_SCORES_LOAD -> FRAUD_SCORES)
    try:
        nrows = write_scores_to_snowflake(scored)
        print("Wrote", nrows, "scores to Snowflake (and merged).")
    except Exception as e:
        print("Failed to write scores to Snowflake:", e)

if __name__ == "__main__":
    main()
