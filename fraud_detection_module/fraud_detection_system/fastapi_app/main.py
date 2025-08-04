from fastapi import FastAPI
import pandas as pd
import joblib
from datetime import datetime
import csv
import os
from typing import List
import json
from kafka import KafkaProducer

# Load model and SHAP explainer from fastapi_app/models/
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
model = joblib.load(os.path.join(MODEL_DIR, "xgb_fraud_model.pkl"))
explainer = joblib.load(os.path.join(MODEL_DIR, "explainer.pkl"))

# Kafka setup
producer = KafkaProducer(
    bootstrap_servers="kafka:9092",  # or "localhost:9092" if running outside Docker
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)
# Initialize FastAPI app
app = FastAPI()

# Health check route
@app.get("/")
def home():
    return {"message": "Fraud detection API is live!"}

# Logging predictions to CSV
def log_prediction(score, top_features, prediction_type="single"):
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "prediction_logs.csv") # Path to log file
    # if log file doesn't exist, create it with headers
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as file: # Open file in write mode
            writer = csv.writer(file) # Create CSV writer
            writer.writerow([
                "timestamp", "prediction_type", "fraud_score",
                "feature_1", "shap_value_1", 
                "feature_2", "shap_value_2",
                "feature_3", "shap_value_3"
            ])

    with open(log_path, "a", newline="") as file: # Open file in append mode
        writer = csv.writer(file) # Create CSV writer
        # Write the prediction data
        writer.writerow([
            datetime.now().isoformat(),
            prediction_type,
            round(float(score), 3),
            top_features[0][0], top_features[0][1],
            top_features[1][0], top_features[1][1],
            top_features[2][0], top_features[2][1]
        ])

# Send prediction to Kafka
def publish_to_kafka(score, top_features, prediction_type):
    event = {
        "timestamp": datetime.now().isoformat(),
        "prediction_type": prediction_type,
        "fraud_score": round(float(score), 3),
        "top_features": top_features
    }
    producer.send("fraud_predictions", value=event)

# Single prediction
@app.post("/predict") # Endpoint for single prediction
def predict(data: dict): # Input data as a dictionary
    try:
        df = pd.DataFrame([data]) # Convert input data to DataFrame
        score = model.predict_proba(df)[0][1] # Get fraud probability
        shap_values = explainer.shap_values(df) # Get SHAP values for explanation
        # Get top 3 features with SHAP values
        top_features = sorted(
            zip(df.columns.tolist(), shap_values[0].tolist()), # Pair features with SHAP values
            key=lambda x: abs(x[1]), # Sort by absolute SHAP value
            reverse=True # Sort in descending order
        )[:3] # Get top 3 features
        # Round SHAP values for better readability
        rounded_features = [(name, round(value, 4)) for name, value in top_features]

        log_prediction(score, rounded_features, "single") # Log prediction to CSV
        publish_to_kafka(score, rounded_features, "single") # Publish prediction to Kafka

        return {
            "fraud_score": round(float(score), 3), # Round fraud score
            "top_features": rounded_features # Return top features with SHAP values
        }

    except Exception as e:
        return {"error": str(e)}

# Batch prediction
@app.post("/predict_batch") # Endpoint for batch prediction
def predict_batch(payload: List[dict]): # Input data as a list of dictionaries
    try:
        df = pd.DataFrame(payload) # Convert input data to DataFrame
        scores = model.predict_proba(df)[:, 1] # Get fraud probabilities for all transactions
        shap_values = explainer.shap_values(df) # Get SHAP values for all transactions

        responses = [] # List to store responses
        for i in range(len(df)): # Iterate through each transaction
            top_features = sorted( 
                zip(df.columns.tolist(), shap_values[i].tolist()), # Pair features with SHAP values
                key=lambda x: abs(x[1]), # Sort by absolute SHAP value
                reverse=True # Sort in descending order
            )[:3] # Get top 3 features

            rounded_features = [(name, round(value, 4)) for name, value in top_features] # Round SHAP values for better readability
            score = scores[i] # Get fraud score for the transaction

            log_prediction(score, rounded_features, "batch") # Log batch prediction to CSV
            publish_to_kafka(score, rounded_features, "batch") # Publish batch prediction to Kafka
            # Append response for each transaction
            responses.append({
                "fraud_score": round(float(score), 3), # Round fraud score
                "top_features": rounded_features # Return top features with SHAP values
            })

        return {"batch_scores": responses}

    except Exception as e:
        return {"error": str(e)}
