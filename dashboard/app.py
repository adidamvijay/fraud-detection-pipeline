"""
Streamlit fraud monitoring dashboard.

Reads scored transactions and shows volume over time, the score distribution,
and a per-transaction drill-down. The numbers are as fresh as the last
completed scoring run, not live.

Two data sources
----------------
If Snowflake credentials are present it reads FRAUD_SCORES. Otherwise it
falls back to data/scores/scores.csv, which run_pipeline.py produces. The
fallback exists so the dashboard can be run and demonstrated without a
warehouse, and so this file is not the one part of the project that cannot
be verified.

On the chart that used to be here
---------------------------------
The previous version computed offsets = np.linspace(0, 59, n) and added them
to every timestamp before plotting, spreading all rows evenly across sixty
seconds. It was labelled "transactions per minute" and showed a shape that
was not in the data at all. The underlying reason was that the generator
stamped every transaction with the same timestamp, so there was genuinely no
time series to draw. The generator was fixed first; this chart now plots
event_time as it actually is, and where the data is sparse the chart shows
that rather than manufacturing a curve.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SCORES_DIR, load_env, snowflake_args  # noqa: E402

load_env()

st.set_page_config(page_title="Fraud Monitoring", layout="wide")
st.title("Fraud Monitoring Dashboard")


# -------------------------------------------------------------------------
# Data access
# -------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_snowflake_conn():
    import snowflake.connector
    return snowflake.connector.connect(**snowflake_args())


@st.cache_data(ttl=60, show_spinner=False)
def load_scores():
    """
    Return (dataframe, source_description).

    Column names are normalised to lowercase so the rest of the app does not
    care which source it came from.
    """
    if snowflake_args() is not None:
        sql = """
            SELECT TRANSACTION_ID, USER_ID, EVENT_TIME, SCORE,
                   FLAGGED, MODEL_VERSION, SCORED_AT
            FROM FRAUD_SCORES
        """
        # fetch_pandas_all rather than pd.read_sql: pandas warns that it only
        # supports SQLAlchemy connectables and sqlite3, and the connector's
        # own method reads the Arrow result directly instead of going through
        # a row-by-row DBAPI cursor. models/score_batch.py uses the same call.
        cursor = get_snowflake_conn().cursor()
        try:
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
        finally:
            cursor.close()
        source = "Snowflake FRAUD_SCORES"
    else:
        path = SCORES_DIR / "scores.csv"
        if not path.exists():
            return None, str(path)
        df = pd.read_csv(path)
        source = f"local file {path.name}"

    df.columns = [c.lower() for c in df.columns]
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["scored_at"] = pd.to_datetime(df["scored_at"], errors="coerce")
    df["flagged"] = df["flagged"].astype(str).str.lower().isin(["true", "1"])
    return df.dropna(subset=["event_time"]), source


data, source = load_scores()

if data is None:
    st.error(
        f"No scored data found. Expected Snowflake credentials in .env, or a "
        f"local scores file at {source}.\n\nRun `python run_pipeline.py` to "
        f"produce one.")
    st.stop()

st.caption(f"Source: {source}. Figures reflect the last completed scoring run.")


# -------------------------------------------------------------------------
# Filters
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    earliest = data["event_time"].min().date()
    latest = data["event_time"].max().date()

    start_date = st.date_input("Start date (UTC)", earliest,
                               min_value=earliest, max_value=latest)
    end_date = st.date_input("End date (UTC)", latest,
                             min_value=earliest, max_value=latest)

    min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, 0.01)
    flagged_only = st.checkbox("Flagged only")

    if st.button("Refresh data"):
        load_scores.clear()
        st.rerun()

mask = (
    (data["event_time"].dt.date >= start_date)
    & (data["event_time"].dt.date <= end_date)
    & (data["score"] >= min_score)
)
window = data[mask]

if window.empty:
    st.info("No scored transactions in the selected window.")
    st.stop()

display = window[window["flagged"]] if flagged_only else window


# -------------------------------------------------------------------------
# KPIs
# -------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions scored", f"{len(window):,}")
c2.metric("Flagged", f"{int(window['flagged'].sum()):,}")
c3.metric("Alert rate", f"{window['flagged'].mean():.2%}")
c4.metric("Max score", f"{window['score'].max():.4f}")


# -------------------------------------------------------------------------
# Volume over transaction event time
# -------------------------------------------------------------------------
st.subheader("Transactions and flags over time")

span_hours = (window["event_time"].max() - window["event_time"].min()) / pd.Timedelta(hours=1)

# Pick a bucket that yields a readable number of points rather than assuming
# a fixed granularity. With one day selected this gives hourly buckets; with
# thirty days it gives daily ones.
if span_hours <= 3:
    freq, label = "5min", "5-minute buckets"
elif span_hours <= 48:
    freq, label = "1h", "hourly buckets"
else:
    freq, label = "1D", "daily buckets"

buckets = (
    window.set_index("event_time")
    .resample(freq)
    .agg(transactions=("score", "count"), flagged=("flagged", "sum"))
    .reset_index()
)

# Be explicit when there is too little data for the shape to mean anything,
# rather than drawing a smooth line over three points.
if len(buckets) < 5:
    st.warning(
        f"Only {len(buckets)} {label.split()[0]} bucket(s) of data in this "
        f"window. The chart below is shown for completeness but the shape is "
        f"not meaningful at this resolution. Widen the date range.")

fig = px.bar(buckets, x="event_time", y="transactions",
             labels={"event_time": "Event time (UTC)", "transactions": "Transactions"})
fig.add_scatter(x=buckets["event_time"], y=buckets["flagged"],
                mode="lines+markers", name="Flagged", yaxis="y2")
fig.update_layout(
    yaxis2=dict(title="Flagged", overlaying="y", side="right",
                rangemode="tozero", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    bargap=0.1,
)
st.plotly_chart(fig, width="stretch")
st.caption(f"{label}, plotted on transaction event time. "
           f"{len(buckets)} buckets covering {span_hours:.0f} hours.")


# -------------------------------------------------------------------------
# Score distribution
# -------------------------------------------------------------------------
st.subheader("Score distribution")
hist = px.histogram(window, x="score", color="flagged", nbins=60,
                    labels={"score": "Anomaly score", "count": "Transactions"})
hist.update_layout(barmode="overlay", legend=dict(orientation="h",
                                                  yanchor="bottom", y=1.02))
hist.update_traces(opacity=0.75)
st.plotly_chart(hist, width="stretch")
st.caption(
    "Scores are min-max scaled within a scoring run, so the axis is "
    "comparable inside one run but not across runs.")


# -------------------------------------------------------------------------
# Table and drill-down
# -------------------------------------------------------------------------
st.subheader("Highest scoring transactions")

rows = st.number_input("Rows to show", 10, 1000, 50, 10)
ranked = display.sort_values("score", ascending=False)
st.dataframe(
    ranked.head(rows)[["transaction_id", "user_id", "event_time", "score",
                       "flagged", "model_version"]],
    width="stretch", hide_index=True)

st.subheader("Investigate a transaction")

if ranked.empty:
    st.info("No transactions match the current filters.")
else:
    txn_id = st.selectbox("Transaction ID", ranked["transaction_id"].head(200).tolist())
    if txn_id:
        record = ranked[ranked["transaction_id"] == txn_id].iloc[0]
        st.json({k: (str(v) if isinstance(v, (pd.Timestamp, datetime)) else v)
                 for k, v in record.to_dict().items()})

st.markdown("---")
st.caption("Streamlit dashboard | all times UTC")
