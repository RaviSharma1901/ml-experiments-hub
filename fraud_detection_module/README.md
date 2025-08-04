
# 🧠 Fraud Detection Module — Real-Time Monitoring & Scoring Suite

This module serves as the foundation for a real-time fraud detection framework, combining predictive scoring and intuitive visualization across two core services:

## 📦 Submodule Overview

| Submodule                    | Description                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| [`fraud_detection_system`](./fraud_detection_system) | FastAPI-based model inference API + Streamlit-powered fraud monitoring dashboard |
| [`fraud_dashboard_app`](./dashboard_app)               | Lightweight dashboard module (in progress) for rapid metric tracking and system snapshots |

##  Features at a Glance

- 🎯 Real-time fraud scoring with threshold tuning and SHAP interpretability  
- 📊 Session metrics, confusion matrices, and predictive insights  
- 📈 Live dashboards with dynamic updates and risk indicators  
- 📦 Docker-based service orchestration with `docker-compose`

## 📎 Quickstart

```bash
git clone https://github.com/RaviSharma1901/ml-experiments-hub
cd fraud_detection_module
docker-compose up --build
git clone https://github.com/RaviSharma1901/ml-experiments-hub
cd fraud_detection_module
docker-compose up --build
```
Streamlit Dashboard → http://localhost:8501

FastAPI Docs → http://localhost:8000/docs
