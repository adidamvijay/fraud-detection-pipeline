# Fraud Detection Pipeline

A batch pipeline that generates synthetic card transactions, validates them,
loads them into Snowflake, computes per-user rolling features, scores them with
an Isolation Forest, and surfaces the results in a Streamlit dashboard.

Scoring is designed to run **hourly as a batch job orchestrated by Apache
Airflow**, with a separate daily job for retraining. There is no streaming
anywhere in this project: no Kafka, no Kinesis, no event bus. Transactions
arrive as CSV files and are processed in batches.

## Status

This repository is being rebuilt. An audit found that the four components were
individually real but not connected to each other, and that several claims in
the previous README were not true. Work is in progress on the `rebuild` branch.

The table below reflects what has actually been executed on a developer
machine, not what the code appears to do.

| Component | Status |
|---|---|
| Transaction generator (`etl/generate_transactions.py`) | Runs. Output verified, numbers below. |
| Data validation (`etl/validate_data.py`) | Logic is sound but hardcoded absolute paths stop it running on Windows. Not yet fixed. |
| Local ingestion (`etl/ingest_local.py`) | Same path problem. Not yet fixed. |
| Load to Snowflake (`etl/ingest_to_snowflake.py`) | Code exists. Never run against a warehouse. |
| Feature computation | Works, but uses a nested Python loop that is quadratic per user. Being replaced. |
| Model training (`models/feature_and_train_local.py`) | Trains an Isolation Forest. No evaluation of any kind exists yet. |
| Batch scoring (`models/score_batch.py`) | Column-name bug fixed; never run against a warehouse. |
| Dashboard (`dashboard/app.py`) | Real Streamlit app. **Its time-series chart is currently fabricated** and is being removed — see below. |
| Airflow DAGs | Defined, never executed. Written in Airflow 2 syntax; being migrated to Airflow 3. |
| Slack alerting (`models/check_fraud_alerts.py`) | Code exists. Never run. |
| Model evaluation | Not built. |
| Tests | Not built. |

### Known defects not yet fixed

- `dashboard/app.py` spreads all scoring timestamps evenly across 60 seconds
  with `np.linspace` before plotting them, so the "per minute" chart shows a
  shape that is not in the data. This is being deleted, not patched.
- The alert threshold is the 98th percentile of whatever batch is being
  scored, so exactly 2% of any input is flagged regardless of content.
- `contamination=0.02` in the model is undocumented and unjustified.
- The five wrapper scripts in `airflow/scripts/` import functions that do not
  exist in their target modules and raise `ImportError` on execution.
- `requirements.txt` does not match what the code imports.

## What runs today

```bash
python etl/generate_transactions.py
```

500 users over 30 days, seed 42. Measured output:

```
rows                  29,178
users                 500
distinct event_time   28,968  (99.3% of rows)
fraud rows            175
fraud rate            0.5998%

transactions per user      min 1  median 49  mean 58.4  max 308
txns in preceding 24h      median 3  mean 3.93  max 23
```

The same seed produces byte-identical files, verified across repeated runs.

## Architecture as designed

```
etl/generate_transactions.py      synthetic transactions -> data/outbox/
        |
etl/validate_data.py              schema, type and format checks
        |                         splits into validated/ and bad_records/
etl/ingest_to_snowflake.py        -> RAW_TRANSACTIONS
        |
models/                           per-user 24-hour rolling features
        |
models/score_batch.py             Isolation Forest -> FRAUD_SCORES
        |
dashboard/app.py                  Streamlit, reads FRAUD_SCORES
models/check_fraud_alerts.py      Slack webhook on flagged transactions
```

Orchestration is two Airflow DAGs: a daily pipeline covering validation
through retraining, and an hourly scoring job. Neither has been run yet.

## Data model

Defined in [`sql/schema.sql`](sql/schema.sql). Three layers:

- `RAW_TRANSACTIONS` — validated transactions as loaded, keyed on
  `TRANSACTION_ID`, with both `EVENT_TIME` (when it happened) and `LOADED_AT`
  (when it reached the warehouse).
- `FRAUD_SCORES` — one scored row per transaction. The serving table for the
  dashboard and the alerting job.
- `FRAUD_SCORES_LOAD` — transient staging table. The scoring job writes here,
  MERGEs into `FRAUD_SCORES` on `TRANSACTION_ID`, then truncates. That MERGE
  is what makes re-running a window safe rather than duplicating rows.

Amounts are `NUMBER(12,2)` rather than `FLOAT` so money stays exact. All
timestamps are `TIMESTAMP_NTZ` holding UTC.

## Synthetic data and what it does not prove

The transaction data is generated, not real. Fraud is injected as episodes
matching three typologies: card-testing bursts, account-takeover high-value
transactions, and a deliberately subtle pattern that sits inside the normal
spending distribution.

Two of those three typologies are, by construction, visible in the feature
space the model uses. Any evaluation on this data therefore measures whether
the model recovers a signal that was deliberately planted. It is a sanity
check on the pipeline. It is not evidence of real-world fraud detection
performance, and it should not be read as such.

33.1% of fraud rows fall inside the 5th-95th percentile band of legitimate
amounts, so the problem is not trivially separable.

## Repository layout

```
airflow/      DAGs, wrapper scripts, Docker Compose for local Airflow
dashboard/    Streamlit app
data/         generated and processed data (gitignored)
etl/          generation, validation, ingestion
models/       feature engineering, training, scoring, alerting
sql/          Snowflake DDL
```

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is currently wrong — it lists packages the code does not
import and omits ones it does. Fixing it is a tracked task.

Snowflake credentials go in a `.env` file in the repository root, which is
gitignored and must never be committed:

```
SNOW_USER=
SNOW_PWD=
SNOW_ACCOUNT=
SNOW_DATABASE=FRAUD_DB
SNOW_SCHEMA=PUBLIC
SNOW_WAREHOUSE=COMPUTE_WH
SNOW_ROLE=
```

Create the tables once with `sql/schema.sql`.

## Not built yet

These are not implemented. They are listed so the gap is explicit, not as a
roadmap commitment.

- Model evaluation: precision, recall, F1, PR-AUC, a threshold chosen on
  evidence, and comparison against a trivial baseline.
- A test suite.
- Any verification that the Airflow DAGs run.
- Any verification that the Snowflake read and write paths work end to end.

## Not planned

Deliberately out of scope: streaming ingestion, Spark, dbt, MLflow,
Kubernetes, model-serving APIs, and cloud services beyond Snowflake. This is a
portfolio project sized so that every line in it can be explained.
