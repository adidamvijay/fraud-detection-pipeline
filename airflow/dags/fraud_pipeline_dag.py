# airflow/dags/fraud_pipeline_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

# ----------------------------------------------------
# Default arguments
# ----------------------------------------------------
default_args = {
    "owner": "fraud_system",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ----------------------------------------------------
# DAG definition (DAILY PIPELINE)
# ----------------------------------------------------
with DAG(
    dag_id="fraud_detection_daily_pipeline",
    description="End-to-end Fraud Detection Pipeline using Snowflake + ML",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fraud", "ml", "snowflake", "resume"],
) as dag:

    # 1️⃣ Validate incoming data
    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="python /project/etl/validate_data.py",
    )

    # 2️⃣ Ingest validated data to Snowflake
    ingest_to_snowflake = BashOperator(
        task_id="ingest_to_snowflake",
        bash_command="python /project/etl/ingest_to_snowflake.py",
    )

    # 3️⃣ Update Feature Store
    update_feature_store = BashOperator(
        task_id="update_feature_store",
        bash_command="python /project/models/update_feature_store.py",
    )

    # 4️⃣ Train / refresh ML model
    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /project/models/feature_and_train.py",
    )

    # 5️⃣ Score transactions
    score_transactions = BashOperator(
        task_id="score_transactions",
        bash_command="python /project/models/score_realtime.py",
    )

    alert_task = BashOperator(
        task_id="slack_fraud_alerts",
        bash_command="python /opt/airflow/scripts/run_check_fraud_alerts.py",
    )
    


    # 6️⃣ Alerts (always runs)
    finalize = BashOperator(
        task_id="finalize_pipeline",
        bash_command="echo 'Fraud pipeline completed successfully'",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    

    # ----------------------------------------------------
    # Task dependencies
    # ----------------------------------------------------
    (
        validate_data
        >> ingest_to_snowflake
        >> update_feature_store
        >> train_model
        >> score_transactions
        >> alert_task
        >> finalize
    )
