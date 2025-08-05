
# 🛡️ Fraud Detection System — Real-Time Scoring & Visualization

This module implements a real-time fraud detection pipeline featuring live transaction streaming, predictive scoring, and interpretability — all wrapped in an interactive dashboard and REST API service. Built using FastAPI, Streamlit, Kafka, and Docker.

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

## 📊 Key Features

# Real-Time Fraud Detection Pipeline

A comprehensive, production-ready fraud detection system built with microservices architecture, featuring real-time transaction processing, machine learning inference, and interactive monitoring dashboard.

## 🎯 Key Features

- **Real-time Processing**: Apache Kafka streaming with sub-10ms latency
- **High Accuracy**: XGBoost model achieving 84% F1-score, 78% precision, 91% recall
- **Scalable Architecture**: Containerized microservices with Docker
- **Interactive Dashboard**: Real-time monitoring with Streamlit
- **Model Interpretability**: SHAP explainer for fraud prediction insights
- **Production Ready**: Complete CI/CD pipeline with monitoring

## 🏗️ System Architecture


## 📊 Performance Metrics\

- **Precision**: 0.78
- **Recall**: 0.91
- **F1-Score**: 0.84
- **Fraud Detection Rate**: 255/97,600 transactions (0.26%)
- **Processing Speed**: 10K+ transactions/second
- **Latency**: <10ms average inference time




##  Quickstart Instructions

1. Clone the repository and navigate to the module:

```bash
git clone https://github.com/RaviSharma1901/ml-experiments-hub
cd fraud_detection_module/fraud_detection_system
docker-compose up --build
