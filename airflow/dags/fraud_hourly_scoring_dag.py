"""
Hourly scoring: score new transactions, then alert on what was flagged.

Airflow 3 syntax. Not verified against a running scheduler.

Separate from the daily DAG because scoring runs on a different cadence to
loading and retraining. The model artifact this reads is produced by the
daily DAG's train_model task.
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

PROJECT = "/project"

default_args = {
    "owner": "fraud_system",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_hourly_scoring_pipeline",
    description="Hourly batch scoring using the latest trained model",
    default_args=default_args,
    schedule="0 * * * *",
    start_date=datetime(2026, 1, 1),
    # catchup=False so restarting the scheduler does not silently trigger a
    # run for every hour since start_date. A backfill should be a deliberate
    # act; the MERGE on TRANSACTION_ID makes re-running a window safe when it
    # is intended.
    catchup=False,
    # max_active_runs=1 because two scoring runs overlapping would both read
    # and write the same staging table. The MERGE itself is safe under
    # concurrency, but FRAUD_SCORES_LOAD is truncated after each merge, so a
    # second run could truncate rows the first had not merged yet.
    max_active_runs=1,
    tags=["fraud", "ml", "scoring", "batch"],
) as dag:

    # Scores data/processed with the latest artifact, writes
    # FRAUD_SCORES_LOAD, MERGEs into FRAUD_SCORES on TRANSACTION_ID, then
    # truncates the staging table. The MERGE is what makes re-running a
    # window update rather than duplicate.
    score_transactions = BashOperator(
        task_id="score_new_transactions",
        bash_command=f"python {PROJECT}/models/score_batch.py",
    )

    # Queries FRAUD_SCORES for recent flagged transactions and posts to a
    # Slack webhook if the count crosses FRAUD_ALERT_THRESHOLD.
    check_alerts = BashOperator(
        task_id="check_fraud_alerts",
        bash_command=f"python {PROJECT}/models/check_fraud_alerts.py",
    )

    score_transactions >> check_alerts

# ---------------------------------------------------------------------------
# Status
#
# Never executed. Both scripts it calls require Snowflake credentials and
# have never been run against a real warehouse; check_fraud_alerts.py also
# needs a Slack webhook. See the README status table.
# ---------------------------------------------------------------------------
