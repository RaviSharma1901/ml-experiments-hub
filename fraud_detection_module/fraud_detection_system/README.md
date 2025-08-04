
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

- 🔄 **Live Fraud Scoring Dashboard**: Streamlit-powered UI for real-time transaction analysis.
- 🧠 **Interpretability**: SHAP-based insights to explain prediction behavior.
- 📈 **Session Metrics**: Track fraud volume, timestamped summaries, and threshold impact.
- 🔌 **REST API**: FastAPI endpoint for external system integration and batch testing.
- 🛰️ **Kafka Integration**: Simulated transaction flow for scoring and log streaming.

## 🚀 Quickstart Instructions

1. Clone the repository and navigate to the module:

```bash
git clone https://github.com/RaviSharma1901/ml-experiments-hub
cd fraud_detection_module/fraud_detection_system
