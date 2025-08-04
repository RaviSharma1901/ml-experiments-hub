from kafka import KafkaProducer  # Kafka client for sending messages to stream
import json                     # Serialize data to JSON format
import time                     # Simulate delay between messages
import math                     # Encode cyclical time features
import pandas as pd             # Load and manipulate tabular dataset
from datetime import datetime   # Timestamp each streamed message
import os

# Load the credit card fraud dataset
#csv_path = os.path.join("..", "kafka_stream", "creditcard.csv")
df = pd.read_csv("creditcard.csv")

# Select only the required columns for the model
require_col = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class", "Time"]
df = df[require_col]

# Convert the 'Time' column in seconds to cyclical hourly features
def encode_hour_from_seconds(second):
    hour = (second // 3600) % 24  # Extract hour of day from seconds
    radians = 2 * math.pi * (hour / 24)  # Map hour to circular position
    return round(math.sin(radians), 4), round(math.cos(radians), 4)

# Initialize Kafka producer with JSON encoder
producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Stream each transaction as a message to Kafka
try:
    for i, row in df.iterrows():
        trans = row.to_dict()  # Convert row to dictionary

        # Add cyclical features for time and remove raw 'Time' column
        trans["hour_sin"], trans["hour_cos"] = encode_hour_from_seconds(trans["Time"])
        trans.pop("Time")

        # Add metadata for source identification and audit logging
        trans["source_type"] = "dataset"
        trans["streamed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Send the message to the 'dataset_stream' Kafka topic
        producer.send("dataset_stream", value=trans)

        # Print status after each message (simple confirmation)
        print(f"[{i + 1}] Sent → Label: {trans['Class']} | Amount: {trans['Amount']}")

        # Pause briefly to simulate real-time streaming
        time.sleep(0.01)

# Handle any unexpected errors gracefully
except Exception as e:
    print(f"Error at row {i}: {e}")