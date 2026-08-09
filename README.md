# Fraud Detection Pipeline

> **Status — 9 August 2026. Rebuilt; the warehouse path is verified.**
> An audit found the components worked individually but were not connected to
> each other, and that the previous README claimed more than the code did.
> **The full path now runs end to end and has been verified against a live
> Snowflake account on 9 August 2026:** generated CSVs → validation →
> `RAW_TRANSACTIONS` → features → scoring → `FRAUD_SCORES` → dashboard, with
> 29,178 rows through every stage and both loaders proven idempotent.
> Evidence in [docs/verified-snowflake.md](docs/verified-snowflake.md).
> **Still not run:** the Airflow DAGs and Slack alerting. This line is updated
> as each phase completes; the component table below is kept current.

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
| Feature computation (`models/features.py`) | Runs. Vectorised, over 100x faster (the ratio is noisy — see DESIGN.md), verified identical to the old implementation on all four features. |
| Model training (`models/train_local.py`) | Runs from local CSVs, no warehouse needed. |
| Local pipeline (`run_pipeline.py`) | Runs end to end in 10s. |
| Model evaluation (`models/evaluate.py`) | Runs. Held-out temporal split, four models, per-typology recall, alert-budget table, two trivial baselines. |
| Tests (`tests/`) | 51 tests covering validation rules, feature causality and the scoring path. All 11 mutations caught by `tests/mutation_check.py`. |
| Dashboard (`dashboard/app.py`) | Runs. **Verified reading `FRAUD_SCORES` from Snowflake**, and falls back to local scores when credentials are absent. No console or server errors in either mode. |
| Snowflake schema (`sql/schema.sql`) | **Applied to a live account.** All five tables created. |
| Load to Snowflake (`etl/ingest_to_snowflake.py`) | **Verified against a live account.** 29,178 rows into `RAW_TRANSACTIONS`; replaying the same files inserts 0 and updates 29,178. |
| Batch scoring to Snowflake (`models/score_batch.py`) | **Verified against a live account.** Reads `RAW_TRANSACTIONS`, writes 29,178 rows to `FRAUD_SCORES`, idempotent on replay. |
| Airflow DAGs | Airflow 3 syntax. Files compile and the Compose stack is structurally valid and diffed against the official Airflow 3.0.3 reference, but **never executed**. Procedure to run it: [airflow/RUNBOOK.md](airflow/RUNBOOK.md). |
| Slack alerting (`models/check_fraud_alerts.py`) | Code exists. Never run. |

## Model results

Held-out temporal split: trained on the first 21 days, measured on the last
9. Test set is 8,897 transactions containing 50 fraud. Full protocol,
before/after comparison and limitations are in [DESIGN.md](DESIGN.md).

### At a fixed analyst workload

A fraud team's real constraint is how many alerts a person can review, so
that is how these are stated. At **25 alerts per 8,897 transactions**
(a 0.28% alert rate):

| Model | PR-AUC | Precision | Recall | Fraud caught |
|---|---|---|---|---|
| random ordering | 0.0054 | — | — | — |
| Isolation Forest, original 4 features | 0.0232 | 0.040 | 0.020 | 1 of 50 |
| rank by amount, no model at all | 0.1477 | — | — | — |
| **Isolation Forest, 7 features** (shipped) | **0.1719** | **0.320** | **0.160** | **8 of 50** |
| Logistic regression, 7 features | 0.5188 | 0.680 | 0.340 | 17 of 50 |
| Random forest, 7 features | 0.7221 | 0.960 | 0.480 | 24 of 50 |

Two things this table is meant to make unavoidable.

**Adding user-relative features worked.** PR-AUC up 7.4x, precision from 4%
to 32% at the same workload.

**The supervised models are far better, and the shipped one is not.** A
random forest on identical features is 4.2x the shipped model's PR-AUC and
catches 85.7% of card-testing fraud that the Isolation Forest misses
completely. The unsupervised model ships because chargeback labels arrive 30
to 90 days late, are biased toward disputed transactions, and cannot exist
for a pattern nobody has labelled yet — not because it performs better. The
cost of that choice is measured rather than assumed.

### Recall by fraud typology

| Typology | n | Isolation Forest | Logistic regression | Random forest |
|---|---|---|---|---|
| account takeover | 26 | 30.8% | 76.9% | 92.3% |
| card testing | 14 | 0.0% | 7.1% | 85.7% |
| subtle | 10 | 0.0% | 0.0% | 10.0% |

Card testing is a conjunction of moderate deviations — somewhat elevated
count, unusually small amounts, very short gaps — with no single extreme
value. Isolation Forest splits on one axis at a time and cannot express a
conjunction; a tree ensemble with labels learns one directly. Logistic
regression, being linear, also cannot, which is why it reaches only 7.1%.

Reproduce with `python models/evaluate.py`.

### Known defects not yet fixed

- The shipped model catches 0% of card-testing fraud, and is 4.2x worse than
  a random forest on the same features. Both measured; see DESIGN.md
  section 4 for why the unsupervised model ships anyway and what it costs.
- Neither Airflow DAG has been executed, and `airflow/docker-compose.yaml`
  has never been started. Airflow does not run natively on Windows, so this
  needs WSL or Docker.
- `models/check_fraud_alerts.py` (Slack) has never been run.
- `models/update_feature_store.py` has never been run and nothing reads
  `FEATURE_STORE`. It is kept as a sketch of what request-time scoring would
  need, and is deliberately not in the DAG.

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

Requires Python 3.11 or later. No database, no Docker, no credentials.

Windows:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS and Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Then, from the repository root:

```bash
.venv\Scripts\python.exe run_pipeline.py
.venv\Scripts\python.exe models\evaluate.py
.venv\Scripts\python.exe -m unittest discover -s tests
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The dashboard opens at http://localhost:8501 and reads
`data/scores/scores.csv`, which `run_pipeline.py` produces. Nothing above
touches a network.

This sequence was verified from an empty virtual environment on Windows; the
transcript is in [docs/verified-install.md](docs/verified-install.md).

### Optional: the Snowflake path

Not required for anything above. **Verified against a live Snowflake account
on 9 August 2026** — see [docs/verified-snowflake.md](docs/verified-snowflake.md)
for row counts, table contents and query history captured at the time.

Install the extra packages together with the base ones so pip resolves them
in one pass:

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-snowflake.txt
```

Copy `.env.example` to `.env` in the repository root and fill it in. `.env` is
gitignored and must never be committed; `.env.example` documents the keys
without containing any values.

Then create the tables and run the warehouse path:

```bash
# in a Snowsight worksheet, or via snowsql:
#   CREATE DATABASE IF NOT EXISTS FRAUD_DB;
# then paste sql/schema.sql

.venv\Scripts\python.exe etl\ingest_to_snowflake.py    # data/processed -> RAW_TRANSACTIONS
.venv\Scripts\python.exe models\score_batch.py         # RAW_TRANSACTIONS -> FRAUD_SCORES
.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

Both loaders stage into a transient table and `MERGE` on `TRANSACTION_ID`, so
re-running a window updates rather than duplicates. When credentials are
present the dashboard reads `FRAUD_SCORES` from Snowflake instead of the
local file, and says which source it used.

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
