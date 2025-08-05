
# 🛡️ Fraud Detection System — Real-Time Scoring & Visualization

This module implements a real-time fraud detection pipeline featuring live transaction streaming, predictive scoring, and interpretability — all wrapped in an interactive dashboard and REST API service. Built using FastAPI, Streamlit, Kafka, and Docker.


## 🎯 Key Features

- **Real-time Processing**: Apache Kafka streaming with sub-10ms latency
- **High Accuracy**: XGBoost model achieving 84% F1-score, 78% precision, 91% recall
- **Scalable Architecture**: Containerized microservices with Docker
- **Interactive Dashboard**: Real-time monitoring with Streamlit
- **Model Interpretability**: SHAP explainer for fraud prediction insights
- **Production Ready**: Complete CI/CD pipeline with monitoring

## 🏗️ System Architecture

#### Fraud Feeder → Kafka Producer → Kafka Broker → Consumer → ML Pipeline → FastAPI → Dashboard

## 📊 Performance Metrics\

- **Precision**: 0.78
- **Recall**: 0.91
- **F1-Score**: 0.84
- **Fraud Detection Rate**: 255/97,600 transactions (0.26%)
- **Processing Speed**: 10K+ transactions/second
- **Latency**: <10ms average inference time

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Streaming** | Apache Kafka |
| **ML Framework** | XGBoost, SHAP |
| **API** | FastAPI |
| **Dashboard** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Data Processing** | Pandas, NumPy |
| **Language** | Python 3.12+ |


##  Quickstart Instructions

1. Option 1: Complete System (Recommended for Demo)

```
git clone https://github.com/RaviSharma1901/ml-experiments-hub
cd fraud_detection_module/fraud_detection_system 

  Build and start the containers  
`docker-compose up --build -d`

  Access dashboard
`http://localhost:8501`

2. Option 2: Individual Services (Advanced Users)

##### Note: Requires external Kafka cluster and proper networking

- docker pull docker4mlpl/fraud_dashboard_app:latest
- docker pull docker4mlpl/fraud_fastapi:latest  
- docker pull docker4mlpl/fraud_producer:latest
- docker pull docker4mlpl/fraud_consumer:latest
- docker pull docker4mlpl/fraud_feeder:latest

##### See deployment guide for full infrastructure setup

Dependencies Required

- Docker & Docker Compose
- Apache Kafka (included in docker-compose.yml)
- Zookeeper (included in docker-compose.yml)
- Python 3.9+ dependencies (containerized)

## 🐳 Docker Deployment

All services are containerized and available on Docker Hub:

| Service | Docker Image | Size | Description |
|---------|--------------|------|-------------|
| **Dashboard** | `docker4mlpl/fraud_dashboard_app:latest` | 190.37 MB | Streamlit monitoring interface |
| **API Service** | `docker4mlpl/fraud_fastapi:latest` | 792.51 MB | ML inference API with XGBoost |
| **Producer** | `docker4mlpl/fraud_producer:latest` | 888.32 MB | Kafka event producer |
| **Consumer** | `docker4mlpl/fraud_consumer:latest` | 888.32 MB | Stream processing service |
| **Data Feeder** | `docker4mlpl/fraud_feeder:latest` | - | Transaction data generator |

## ⚙️ Components Overview

| Directory/File        | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `dashboard_app/`       | Streamlit dashboard for live fraud scoring, metrics, and session summaries |
| `data/`                | Reference datasets for testing and simulation                               |
| `fastapi_app/`         | FastAPI service exposing REST endpoint for ML inference                     |
| `fraud_feeder/`        | Kafka producer for batch-wise transaction injection                         |
| `images/`              | Visual assets: architecture diagrams, SHAP charts, UI snapshots             |
| `kafka_stream/`        | Kafka consumer & stream processor for transaction scoring                   |
| `notebooks/`           | Exploratory analysis and model interpretability using SHAP                  |
| `static_snapshots/`    | Archived dashboard outputs and session stats                                |
| `README.md`            | Module documentation                                                        |
| `docker-compose.yml`   | Service orchestration for dashboard, API, Kafka, and data feeder            |


