import requests
import pandas as pd
import time
from datetime import datetime

#  Your FastAPI batch endpoint (local or cloud)
API_URL = "http://fastapi:8000/predict_batch"

# Path to your full-feature CSV file (include V1–V29, Amount, hour_sin, hour_cos)
#CSV_PATH = "data/sample_transactions_full.csv"
CSV_PATH = "/fraud_feeder/data/sample_transactions_full.csv"


# Load CSV
try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"[Error] Could not read CSV: {e}")
    exit()

#  Send data in chunks (e.g. 10 transactions at a time)
chunk_size = 10

print(f"Starting feeder from: {CSV_PATH}")
for i in range(0, len(df), chunk_size): # Iterate through DataFrame in chunks
    batch = df.iloc[i:i+chunk_size].to_dict(orient="records") # Convert chunk to list of dictionaries
    try:
        response = requests.post(API_URL, json=batch) # Send batch to FastAPI endpoint
        results = response.json().get("results", []) # Check if results are in response
        for item in results: # Iterate through each item in results
            index = item.get("index", "N/A") # Get index of the item
            score = item.get("fraud_score", "N/A") # Get fraud score
            top_feats = item.get("top_features", []) # Get top features
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch Item {index} → Score: {score}, SHAP: {top_feats}")
    except Exception as e:
        print(f"[Error] Failed to send batch at index {i}: {e}")
    
    time.sleep(5)  # Wait 5 seconds before next batch
