# Fraud Detection Pipeline

> **Status — 9 August 2026. Actively being rebuilt, in the open.**
> An audit found the components worked individually but were not connected to
> each other, and that the previous README claimed more than the code did.
> **Current phase:** the local pipeline runs end to end and the model is now
> measured against held-out labels — PR-AUC 0.1719 against 0.1477 for the
> trivial "sort by amount" baseline, so the model earns its place narrowly.
> Done so far: the audit, one canonical Snowflake schema, a rewritten
> generator, the local path connected, and a real evaluation with a
> before/after comparison in [DESIGN.md](DESIGN.md). Next: the Snowflake
> path and the fabricated dashboard chart. This line is updated as each phase
> completes; the component table below is kept current.

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
| Data validation (`etl/validate_data.py`) | Runs. Rules verified against deliberately corrupt input. |
| Local ingestion (`etl/ingest_local.py`) | Runs. |
| Feature computation (`models/features.py`) | Runs. Vectorised, 344x faster, verified identical to the old implementation on all four features. |
| Model training (`models/train_local.py`) | Runs from local CSVs, no warehouse needed. |
| Local pipeline (`run_pipeline.py`) | Runs end to end in 10s. |
| Model evaluation (`models/evaluate.py`) | Runs. Held-out temporal split, PR-AUC, per-typology recall, two trivial baselines. Numbers in [DESIGN.md](DESIGN.md). |
| Tests (`tests/`) | 11 tests covering feature causality. Verified by mutation. |
| Load to Snowflake (`etl/ingest_to_snowflake.py`) | Code exists, still has absolute paths. Never run against a warehouse. |
| Batch scoring to Snowflake (`models/score_batch.py`) | Column-name bug fixed, still has absolute paths. Never run against a warehouse. |
| Dashboard (`dashboard/app.py`) | Real Streamlit app. **Its time-series chart is still fabricated** — see below. |
| Airflow DAGs | Defined, never executed. Airflow 2 syntax; migration to Airflow 3 pending. |
| Slack alerting (`models/check_fraud_alerts.py`) | Code exists. Never run. |

## Model results

Held-out temporal split, 8,897 test rows containing 50 fraud. Full protocol
and the before/after comparison are in [DESIGN.md](DESIGN.md).

| Ranking | PR-AUC |
|---|---|
| random ordering | 0.0054 |
| absolute features only (the original 4) | 0.0232 |
| rank by transaction amount, no model | 0.1477 |
| absolute + user-relative (7 features) | **0.1719** |

Adding three user-relative features took PR-AUC up 7.4x and precision from
4% to 32% at the same alert volume. The model still only beats sorting by
amount by 16%, and it catches 0% of card-testing fraud. Both facts are
documented rather than buried.

### Known defects not yet fixed

- `dashboard/app.py` spreads all scoring timestamps evenly across 60 seconds
  with `np.linspace` before plotting them, so the "per minute" chart shows a
  shape that is not in the data. To be deleted, not patched.
- The model catches 0% of card-testing fraud. Isolation Forest splits on
  single axes and card testing is a conjunction of moderate deviations. A
  supervised classifier is the fix; see limitations in DESIGN.md.
- `models/train_local.py` still flags using `contamination=0.02`. The
  evidence-based threshold exists in `models/evaluate.py` but has not been
  wired back into the scoring path.
- Three warehouse scripts still carry absolute `/project/...` paths.
- The five wrapper scripts in `airflow/scripts/` import functions that do not
  exist in their target modules and raise `ImportError` on execution.
- `requirements.txt` does not match what the code imports.

## What runs today

The whole local path, on one command, with no warehouse connection:

```bash
python run_pipeline.py
```

Measured on a Windows machine, Python 3.11:

```
Generate transactions              2.88s
Validate                           2.61s
Ingest to processed                2.29s
Train and score                    6.40s
total                             14.19s
```

29,178 transactions generated, 29,178 validated with 0 rejected, 29,178
scored. Feature computation for all of them takes 0.07s.

### Generator output

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
