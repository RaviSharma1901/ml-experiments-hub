from kafka import KafkaConsumer
import json
import joblib
import pandas as pd
from datetime import datetime
import os
import shap

#  Load model & explainer
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models") # Path to models directory
model = joblib.load(os.path.join(MODEL_DIR, "xgb_fraud_model.pkl")) # Load the XGBoost model
explainer = shap.Explainer(model) # Load SHAP explainer

#  Feature list
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount", "hour_sin", "hour_cos"]

#  Kafka consumer
consumer = KafkaConsumer(
    "dataset_stream", # Topic to consume from
    bootstrap_servers='kafka:9092', # Kafka server address
    auto_offset_reset='latest', # Start consuming from the latest message
    value_deserializer=lambda m: json.loads(m.decode('utf-8')) # Deserialize JSON messages
)

#  Shared logging folder
LOG_DIR = "/app/kafka_stream" # Path to save logs
os.makedirs(LOG_DIR, exist_ok=True) # Ensure log directory exists

#  Logging variables
logs = [] # List to store transaction logs
tp = fp = tn = fn = 0 # True Positives, False Positives, True Negatives, False Negatives

#  Start
print("📡 Listening for transactions...")

# Process each message from the Kafka stream
for msg in consumer:
    #  Deserialize message
    txn = msg.value
    # Check if all required features are present
    if all(f in txn for f in FEATURES):
        df = pd.DataFrame([txn])[FEATURES] # Create DataFrame with required features
        #  Predict fraud probability
        try:
            proba = model.predict_proba(df)[0][1] # Get probability of fraud
            prediction = int(proba >= 0.86) # Threshold for fraud detection
        except Exception as e: 
            print(f"⚠️ Prediction error: {e}") 
            continue
        #  Log transaction details
        txn["predicted_label"] = prediction # Add predicted label to transaction
        txn["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Add current timestamp

        # If true label is available
        if "Class" in txn: #
            true_label = int(txn.pop("Class")) # Extract true label
            txn["true_label"] = true_label # Add true label to transaction
            txn["is_correct"] = int(prediction) == true_label # Check if prediction is correct
            print(f"Prediction: {prediction} | Actual: {true_label} | Match: {txn['is_correct']}") # Print prediction status
            # Update confusion matrix counts
            if prediction == 1 and true_label == 1: # True Positive
                tp += 1 # Increment True Positive count
            elif prediction == 1 and true_label == 0: # False Positive
                fp += 1 # Increment False Positive count
            elif prediction == 0 and true_label == 0: # True Negative
                tn += 1 # Increment True Negative count
            elif prediction == 0 and true_label == 1:
                fn += 1 # Increment False Negative count
        else: # If true label is not available
            print(f"Dataset Prediction: {prediction} | Amount: {txn['Amount']} | V1: {txn['V1']}")

        #  SHAP Top Features
        shap_values = explainer(df) # Get SHAP values for the transaction
        # Sort features by absolute SHAP value
        top_features = sorted(
            zip(df.columns, shap_values.values[0]), # Pair features with SHAP values
            key=lambda x: abs(x[1]), # Sort by absolute SHAP value
            reverse=True # Sort in descending order
        )[:3] # Get top 3 features

        #  Log transaction with SHAP values
        txn["shap_top_features"] = ", ".join([f"{k}:{round(v, 2)}" for k, v in top_features])
        logs.append(txn) # # Append transaction to logs

        #  Save predictions every 100 transactions
        if len(logs) >= 100: # If we have 100 logs
            # Save to CSV
            prediction_path = os.path.join(LOG_DIR, "dataset_prediction_logs.csv")
            # Ensure directory exists
            expected_fields = ["timestamp", "Amount", "predicted_label", "true_label", "is_correct"]
            
            # Add expected fields to each log
            for log in logs: # Iterate through each log
                for field in expected_fields: # Ensure each expected field exists
                    log.setdefault(field, "") # Set default value if field is missing
                log.setdefault("shap_top_features", "") # Ensure SHAP features are included

            header_needed = not os.path.exists(prediction_path) # Check if file exists for header
            # Save logs to CSV
            pd.DataFrame(logs)[expected_fields + ["shap_top_features"]].to_csv( 
                prediction_path, 
                mode='a', 
                index=False,
                header=header_needed
            )
            # Reset logs
            logs = []
            print("Saved 100 predictions to CSV.")

        #  Save Confusion Matrix
        cm_path = os.path.join(LOG_DIR, "dataset_confusion_log.csv") # Path to confusion matrix log
        os.makedirs(os.path.dirname(cm_path), exist_ok=True) # Ensure directory exists

        # Create or append to confusion matrix log
        cm_df = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        }])
        # Append to confusion matrix log
        cm_df.to_csv(cm_path, mode='a', index=False, header=not os.path.exists(cm_path))

        #  Calculate performance
        precision = tp / (tp + fp) if (tp + fp) else 0 # Calculate precision
        recall = tp / (tp + fn) if (tp + fn) else 0 # Calculate recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0 # Calculate F1 score

        print(f"[Metrics] Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}")

        #  Save metrics
        metrics_path = os.path.join(LOG_DIR, "dataset_metrics_log.csv")
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        # Create or append to metrics log
        metrics_df = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }])
        # Append to metrics log
        metrics_df.to_csv(metrics_path, mode='a', index=False, header=not os.path.exists(metrics_path))
    else:
        print(f"⛔ Skipped invalid transaction: missing features → {txn}")
