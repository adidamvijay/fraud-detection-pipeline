# 🚀 Real-Time Fraud Detection System

An **end-to-end real-time fraud detection pipeline** built using **Snowflake, Apache Airflow, Python, Machine Learning, Streamlit, and Slack**.

This project simulates how modern organizations design **production-grade data pipelines** to detect fraudulent transactions in real time.

---

## 🔍 Project Overview

The system continuously ingests transaction data, engineers features, trains an anomaly detection model, scores new transactions, and triggers alerts when suspicious activity is detected.

It is designed to closely resemble **real-world fraud detection architectures** used in fintech and banking systems.

---

## 🧱 Architecture (High-Level)

Transaction Generator  
↓  
Data Validation & Ingestion (Airflow)  
↓  
Snowflake (RAW_TRANSACTIONS)  
↓  
Feature Store (Snowflake)  
↓  
ML Model (Isolation Forest)  
↓  
Fraud Scores (Snowflake)  
↓  
Streamlit Dashboard + Slack Alerts  

---

## ⚙️ Key Features

- 🔄 Continuous transaction ingestion  
- 🧪 Data validation before ingestion  
- 🧠 Feature engineering & feature store  
- 🤖 ML-based anomaly detection (Isolation Forest)  
- ⏱️ Hourly scoring & daily retraining via Airflow  
- 📊 Real-time fraud monitoring dashboard (Streamlit)  
- 🚨 Automated Slack alerts when fraud exceeds thresholds  

---

## 🛠 Tech Stack

- Snowflake – Cloud data warehouse  
- Apache Airflow – Pipeline orchestration  
- Python – ETL, feature engineering, ML, alerting  
- scikit-learn – Anomaly detection model  
- Streamlit – Real-time dashboard  
- Slack Webhooks – Alert notifications  
- Docker & Docker Compose – Airflow setup  

---

## 📁 Repository Structure

├── airflow/                # Airflow DAGs & Docker setup  
├── dashboard/              # Streamlit dashboard  
├── data/                   # Generated & processed data  
├── etl/                    # Validation & ingestion scripts  
├── models/                 # Training, scoring & alert logic  
├── requirements.txt  
└── README.md  

---

## ▶️ How to Run (High-Level)

### 1️⃣ Clone the repository

git clone https://github.com/adidamvijay/real-time-fraud-detection.git  
cd real-time-fraud-detection  

---

### 2️⃣ Set environment variables

Create a `.env` file with:
- Snowflake credentials  
- Slack webhook URL  

⚠️ The `.env` file is excluded from Git for security reasons.

---

### 3️⃣ Start Airflow

docker compose up -d  

Access Airflow UI:  
http://localhost:8080  

---

### 4️⃣ Run Streamlit Dashboard

streamlit run dashboard/app.py  

---

## 📊 Dashboard

The Streamlit dashboard displays:
- Total transactions  
- Fraud / anomaly count  
- Score distribution  
- Time-based fraud trends  

---

## 🚨 Alerting

- Slack alerts are triggered when fraud count crosses a defined threshold  
- Alerts are evaluated on recent scoring windows  

---

## 🔮 Future Improvements

- Real-time streaming ingestion (Kafka / Kinesis)  
- Model performance monitoring (ROC, Precision-Recall)  
- Role-based dashboard access  
- CI/CD for pipeline deployments  

---

## 📌 Notes

- This is a **portfolio project** built for learning and demonstration  
- Design choices favor clarity and production realism  

---

## 🙌 Feedback

I’d love feedback from data engineers and analytics professionals.  
Suggestions for scaling or improving the system are always welcome!
