# dashboard/app.py
"""
Streamlit fraud monitoring dashboard.

- Connects to Snowflake using env vars
- Shows KPIs, anomaly trends over transaction event time, and a drill-down

Reads FRAUD_SCORES, which is written by the hourly batch scoring job. The
numbers shown are as fresh as the last completed run, not live.
"""

import os
import io
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
import snowflake.connector
import numpy as np

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

SNOW_USER = os.getenv("SNOW_USER")
SNOW_PWD = os.getenv("SNOW_PWD")
SNOW_ACCOUNT = os.getenv("SNOW_ACCOUNT")
SNOW_DATABASE = os.getenv("SNOW_DATABASE", "FRAUD_DB")
SNOW_SCHEMA = os.getenv("SNOW_SCHEMA", "PUBLIC")
SNOW_WAREHOUSE = os.getenv("SNOW_WAREHOUSE", "COMPUTE_WH")
SNOW_ROLE = os.getenv("SNOW_ROLE")

# -------------------------------------------------
# STREAMLIT PAGE
# -------------------------------------------------
st.set_page_config(page_title="Fraud Monitoring", layout="wide")
st.title("Fraud Monitoring Dashboard")
st.markdown(
    "Anomaly scores from the hourly batch scoring job, read from Snowflake. "
    "Figures reflect the last completed run."
)

# -------------------------------------------------
# SNOWFLAKE CONNECTION
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_snowflake_conn():
    conn_args = {
        "user": SNOW_USER,
        "password": SNOW_PWD,
        "account": SNOW_ACCOUNT,
        "warehouse": SNOW_WAREHOUSE,
        "database": SNOW_DATABASE,
        "schema": SNOW_SCHEMA,
    }
    if SNOW_ROLE:
        conn_args["role"] = SNOW_ROLE
    return snowflake.connector.connect(**conn_args)

# -------------------------------------------------
# QUERIES
# -------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def query_fraud_scores(start_ts, end_ts, min_score):
    conn = get_snowflake_conn()
    sql = """
        SELECT
            TRANSACTION_ID,
            USER_ID,
            EVENT_TIME,
            SCORE,
            FLAGGED,
            MODEL_VERSION,
            SCORED_AT
        FROM FRAUD_SCORES
        WHERE EVENT_TIME BETWEEN %s AND %s
        ORDER BY EVENT_TIME ASC
    """
    df = pd.read_sql(sql, conn, params=(start_ts, end_ts))
    if df.empty:
        return df

    df["SCORED_AT"] = pd.to_datetime(df["SCORED_AT"], utc=True)
    df["EVENT_TIME"] = pd.to_datetime(df["EVENT_TIME"], utc=True)
    df = df[df["SCORE"] >= min_score]
    return df

@st.cache_data(ttl=60, show_spinner=False)
def query_raw_transaction(txn_id):
    conn = get_snowflake_conn()
    sql = """
        SELECT *
        FROM RAW_TRANSACTIONS
        WHERE TRANSACTION_ID = %s
        LIMIT 1
    """
    return pd.read_sql(sql, conn, params=(txn_id,))

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.header("Filters")

    start_date = st.date_input("Start date (UTC)", datetime.utcnow().date())
    end_date = st.date_input("End date (UTC)", datetime.utcnow().date())

    start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
    flagged_only = st.checkbox("Flagged only (anomalies)")
    refresh = st.button("Refresh data")

if refresh:
    query_fraud_scores.clear()
    query_raw_transaction.clear()
    st.experimental_rerun()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
df_scores = query_fraud_scores(start_dt, end_dt, min_score)

if df_scores.empty:
    st.info("No scored transactions found for the selected window.")
    st.stop()

df_display = df_scores[df_scores["FLAGGED"]] if flagged_only else df_scores.copy()

# -------------------------------------------------
# KPIs
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total scored", len(df_scores))
c2.metric("Anomalies (flagged)", int(df_scores["FLAGGED"].sum()))
c3.metric("Avg score", round(df_scores["SCORE"].mean(), 4))
c4.metric("Max score", round(df_scores["SCORE"].max(), 4))

# -------------------------------------------------
# ✅ MINUTE-LEVEL TIME SERIES (FIX)
# -------------------------------------------------
ts_df = df_scores.copy()

# Convert to naive UTC
ts_df["SCORED_AT"] = ts_df["SCORED_AT"].dt.tz_convert("UTC").dt.tz_localize(None)

# 🔥 VISUALIZATION FIX:
# Spread timestamps over 60 seconds ONLY for plotting
n = len(ts_df)
offsets = np.linspace(0, 59, n)

ts_df["viz_time"] = ts_df["SCORED_AT"] + pd.to_timedelta(offsets, unit="s")

# Aggregate by second (now it will spread)
ts_df["viz_second"] = ts_df["viz_time"].dt.floor("S")

agg = (
    ts_df.groupby("viz_second")
    .agg(
        total=("SCORE", "count"),
        anomalies=("FLAGGED", "sum"),
    )
    .reset_index()
)

st.subheader("Transactions / Anomalies per minute")

if not agg.empty:
    fig = px.line(
    agg,
    x="viz_second",
    y=["total", "anomalies"],
    labels={"value": "count", "viz_second": "Time (UTC)"},
    markers=True,
)

    fig.update_layout(legend_title_text="Metric")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("No time series data.")

# -------------------------------------------------
# TABLE
# -------------------------------------------------
st.subheader("Top scored transactions")

rows = st.number_input("Rows to show", 10, 1000, 50, 10)
st.dataframe(df_display.head(rows), use_container_width=True)

# -------------------------------------------------
# DRILL DOWN
# -------------------------------------------------
st.subheader("Investigate transaction")
txn_id = st.selectbox(
    "Choose transaction ID",
    df_display["TRANSACTION_ID"].head(200).tolist(),
)

if txn_id:
    raw_tx = query_raw_transaction(txn_id)
    if not raw_tx.empty:
        st.json(raw_tx.iloc[0].to_dict())

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Streamlit dashboard | Snowflake backend | all times UTC")
