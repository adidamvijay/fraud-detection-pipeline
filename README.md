# Real-Time Fraud Detection System

An end-to-end real-time fraud detection pipeline built using Snowflake, Apache Airflow, Python, ML, and Streamlit.

## 🚀 Features
- Real-time transaction scoring using ML
- Daily model training & feature updates via Airflow
- Hourly fraud scoring pipeline
- Real-time dashboard built with Streamlit
- Automated Slack alerts for fraud detection

## 🛠 Tech Stack
- Snowflake
- Apache Airflow
- Python
- Streamlit
- Machine Learning
- Slack Webhooks

## 📊 Architecture
1. Data ingestion & validation
2. Feature engineering & model training
3. Fraud scoring pipeline
4. Alerting & monitoring

## ▶️ How to Run
```bash
docker-compose up -d
streamlit run dashboard/app.py
