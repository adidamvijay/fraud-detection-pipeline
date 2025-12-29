import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from sklearn.ensemble import IsolationForest

load_dotenv()

# -----------------------------------------------------
# Snowflake Credentials
# -----------------------------------------------------
SNOW_USER = os.getenv("SNOW_USER")
SNOW_PWD = os.getenv("SNOW_PWD")
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_DATABASE = os.getenv("SNOW_DATABASE", "FRAUD_DB")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA", "PUBLIC")
SNOW_WAREHOUSE = os.getenv("SNOW_WAREHOUSE", "COMPUTE_WH")
SNOW_ROLE = os.getenv("SNOW_ROLE", None)

MODEL_DIR = "/project/models/artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------------------------------
# Snowflake Connection
# -----------------------------------------------------
def get_conn():
    return snowflake.connector.connect(
        user=SNOW_USER,
        password=SNOW_PWD,
        account=SNOW_ACCOUNT,
        warehouse=SNOW_WAREHOUSE,
        database=SNOW_DATABASE,
        schema=SNOW_SCHEMA,
        role=SNOW_ROLE
    )


# -----------------------------------------------------
# STEP 1: Pull last 30 days transactions
# -----------------------------------------------------
def load_recent_transactions(days=30):
    query = f"""
        SELECT TRANSACTION_ID, USER_ID, EVENT_TIME, AMOUNT
        FROM RAW_TRANSACTIONS
        WHERE EVENT_TIME >= DATEADD('day', -{days}, CURRENT_TIMESTAMP())
        ORDER BY EVENT_TIME ASC;
    """

    conn = get_conn()
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        raise ValueError("No recent transactions found in last 30 days.")

    df["EVENT_TIME"] = pd.to_datetime(df["EVENT_TIME"], errors="coerce")
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0.0)
    return df


# -----------------------------------------------------
# STEP 2: Compute Features (Local Rolling Window)
# -----------------------------------------------------
def compute_features(df):
    records = []
    for user, group in df.groupby("USER_ID"):
        group = group.sort_values("EVENT_TIME").reset_index(drop=True)

        for i, row in group.iterrows():
            ts = row["EVENT_TIME"]
            window_start = ts - timedelta(hours=24)

            window = group[(group["EVENT_TIME"] >= window_start) &
                           (group["EVENT_TIME"] <= ts)]

            total = window["AMOUNT"].sum()
            count = len(window)
            avg = window["AMOUNT"].mean() if count > 0 else 0.0

            # hours since previous transaction
            if i > 0:
                prev_ts = group.loc[i - 1, "EVENT_TIME"]
                hours_since_last = (ts - prev_ts).total_seconds() / 3600.0
            else:
                hours_since_last = -1.0

            records.append({
                "transaction_id": row["TRANSACTION_ID"],
                "user_id": user,
                "event_time": ts,
                "total_amount_24h": float(total),
                "txn_count_24h": int(count),
                "avg_amount_24h": float(avg),
                "hours_since_last_txn": float(hours_since_last)
            })

    return pd.DataFrame(records)


# -----------------------------------------------------
# STEP 3: Train Isolation Forest Model
# -----------------------------------------------------
def train_model(feats):
    X = feats[["total_amount_24h", "txn_count_24h",
               "avg_amount_24h", "hours_since_last_txn"]].fillna(0)

    clf = IsolationForest(contamination=0.02, random_state=42)
    clf.fit(X)

    model_path = os.path.join(MODEL_DIR, "isoforest_local_v2.joblib")
    joblib.dump(clf, model_path)

    return clf, "isoforest_local_v2"


# -----------------------------------------------------
# STEP 4: Score NEW transactions only
# -----------------------------------------------------
def score_new_transactions(feats, model, version):
    # Load previously scored alerts (local)
    alerts_file = "data/alerts.csv"
    if os.path.exists(alerts_file):
        old = pd.read_csv(alerts_file)
        old_tx_ids = set(old["TRANSACTION_ID"].astype(str))
    else:
        old_tx_ids = set()

    # Filter new transactions
    new_feats = feats[~feats["transaction_id"].isin(old_tx_ids)].copy()
    if new_feats.empty:
        print("No new transactions to score.")
        return None

    print(f"Scoring {len(new_feats)} NEW transactions...")

    X = new_feats[["total_amount_24h", "txn_count_24h",
                   "avg_amount_24h", "hours_since_last_txn"]].fillna(0)

    raw_scores = -model.decision_function(X)

    # normalize
    mi, ma = raw_scores.min(), raw_scores.max()
    scores = (raw_scores - mi) / (ma - mi) if ma != mi else raw_scores - mi

    new_feats["SCORE"] = scores
    threshold = np.quantile(scores, 0.98)
    new_feats["FLAGGED"] = new_feats["SCORE"] >= threshold
    new_feats["MODEL_VERSION"] = version
    new_feats["SCORED_AT"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f") + "Z"

    return new_feats.rename(columns=str.upper)


# -----------------------------------------------------
# STEP 5: Save alerts locally
# -----------------------------------------------------
def save_alerts(new_df):
    alerts_file = "data/alerts.csv"

    if os.path.exists(alerts_file):
        old_df = pd.read_csv(alerts_file)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(alerts_file, index=False)
    print("Updated local alerts:", alerts_file)


# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------
if __name__ == "__main__":

    print("Loading last 30 days transactions from Snowflake...")
    df = load_recent_transactions(days=30)
    print("Rows:", len(df))

    print("Computing rolling features...")
    feats = compute_features(df)
    print("Feature rows:", len(feats))

    print("Training model...")
    model, version = train_model(feats)
    print("Model version:", version)

    print("Scoring new transactions...")
    new_scores = score_new_transactions(feats, model, version)

    if new_scores is not None:
        print(new_scores.head())
        save_alerts(new_scores)

    print("Local ML pipeline completed.")