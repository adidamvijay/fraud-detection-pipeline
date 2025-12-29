import os
import requests
import snowflake.connector
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
ALERT_THRESHOLD = int(os.getenv("FRAUD_ALERT_THRESHOLD", 1))

def get_snowflake_conn():
    return snowflake.connector.connect(
        user=os.getenv("SNOW_USER"),
        password=os.getenv("SNOW_PWD"),
        account=os.getenv("SNOW_ACCOUNT"),
        warehouse=os.getenv("SNOW_WAREHOUSE"),
        database=os.getenv("SNOW_DATABASE"),
        schema=os.getenv("SNOW_SCHEMA"),
        role=os.getenv("SNOW_ROLE"),
    )

def send_slack_alert(rows):
    blocks = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "🚨 *Fraud Alert Triggered*"
            }
        }
    ]

    for r in rows:
        blocks.append({
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Transaction ID:*\n{r[0]}"},
                {"type": "mrkdwn", "text": f"*Score:*\n{r[1]:.4f}"},
                {"type": "mrkdwn", "text": f"*Model:*\n{r[2]}"},
                {"type": "mrkdwn", "text": f"*Time (UTC):*\n{r[3]}"}
            ]
        })

    payload = {"blocks": blocks}
    requests.post(SLACK_WEBHOOK_URL, json=payload)

def main():
    conn = get_snowflake_conn()
    cur = conn.cursor()

    sql = """
    SELECT TRANSACTION_ID, SCORE, MODEL_VERSION, SCORED_AT
    FROM FRAUD_SCORES
    WHERE FLAGGED = TRUE
      AND SCORED_AT >= DATEADD(hour, -24, CURRENT_TIMESTAMP())
    ORDER BY SCORE DESC
    LIMIT 5
    """

    cur.execute(sql)
    rows = cur.fetchall()

    print(f"Flagged anomalies in last hour: {len(rows)}")

    if len(rows) >= ALERT_THRESHOLD:
        send_slack_alert(rows)
        print("Slack alert sent.")
    else:
        print("No alert triggered.")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
