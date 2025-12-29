# airflow/dags/fraud_hourly_scoring_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "fraud_system",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fraud_hourly_scoring_pipeline",
    description="Hourly fraud scoring using latest ML model",
    default_args=default_args,
    schedule_interval="0 * * * *",  # every hour
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["fraud", "ml", "scoring", "realtime"],
) as dag:

    score_transactions = BashOperator(
        task_id="score_new_transactions",
        bash_command="python /project/models/score_realtime.py",
    )

    check_alerts = BashOperator(
        task_id="check_fraud_alerts",
        bash_command="python /project/models/check_fraud_alerts.py",
    )

    score_transactions >> check_alerts
