"""
Daily pipeline: validate, normalise, load to Snowflake, retrain.

Airflow 3 syntax. Not verified against a running scheduler - see the note at
the bottom of this file.

Scoring is a separate DAG (fraud_hourly_scoring_dag.py) because it runs on a
different cadence. This one is about getting new data into the warehouse and
refreshing the model.
"""

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.utils.trigger_rule import TriggerRule

PROJECT = "/project"

default_args = {
    "owner": "fraud_system",
    "depends_on_past": False,
    "email_on_failure": False,
    # The likely failures are transient warehouse connection errors, which a
    # retry fixes. Retrying is only safe because every task is idempotent:
    # validation and ingestion archive their input after writing output, and
    # the warehouse writes MERGE on TRANSACTION_ID. With the old
    # append-to-CSV write, a retry after a partial write would have
    # duplicated rows.
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_detection_daily_pipeline",
    description="Daily batch fraud pipeline: validate, load to Snowflake, retrain",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["fraud", "ml", "snowflake", "batch"],
) as dag:

    # Reads data/outbox, writes data/validated and data/bad_records, then
    # archives the input. Rejected rows are kept with a failure_reason rather
    # than dropped.
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command=f"python {PROJECT}/etl/validate_data.py",
    )

    # Normalises into the canonical column order and types, writing
    # data/processed. This task did not exist in the previous version of this
    # DAG, which is why nothing ever populated data/processed and the
    # warehouse loader downstream had no input.
    ingest_local = BashOperator(
        task_id="normalise_to_processed",
        bash_command=f"python {PROJECT}/etl/ingest_local.py",
    )

    # data/processed -> RAW_TRANSACTIONS.
    ingest_to_snowflake = BashOperator(
        task_id="load_raw_transactions",
        bash_command=f"python {PROJECT}/etl/ingest_to_snowflake.py",
    )

    # Fits a new Isolation Forest and writes a versioned artifact plus its
    # metadata. Previously this task was called train_model but ran
    # feature_and_train.py, which loads a model and scores with it rather
    # than training anything.
    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"python {PROJECT}/models/train_local.py",
    )

    # ALL_DONE so the run is marked complete even when an upstream task
    # failed, which keeps the failure visible on the task rather than
    # cascading into a confusing DAG-level state.
    finalize = BashOperator(
        task_id="finalize_pipeline",
        bash_command="echo 'daily pipeline finished'",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    validate_data >> ingest_local >> ingest_to_snowflake >> train_model >> finalize

# ---------------------------------------------------------------------------
# Status
#
# This DAG has never been executed. Airflow does not run natively on Windows,
# which is the development machine for this project, so verifying it needs
# WSL or the Docker Compose setup in this directory, and that has not been
# done either.
#
# What that means concretely: the syntax is Airflow 3 and the task graph is
# what it claims to be, but nobody has watched it run. Two of the scripts it
# calls (ingest_to_snowflake.py and, downstream, score_batch.py) also still
# carry absolute /project paths from the original codebase and have never
# been run against a real warehouse.
#
# models/update_feature_store.py is deliberately not in this DAG. Nothing in
# the current scoring path reads FEATURE_STORE, so scheduling a task to
# populate it would be scheduling work that no other task consumes.
# ---------------------------------------------------------------------------
