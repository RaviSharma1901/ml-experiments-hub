import streamlit as st                      # Streamlit for dashboard
import time                                 # Time management for refresh logic 
import pandas as pd
import matplotlib.pyplot as plt
import os                                   # File system operations
from kafka import KafkaConsumer             # Kafka consumer to read live data
import json                                 # JSON for data serialization
from datetime import datetime               # Date and time handling

# Set up Streamlit page configuration
st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# refresh logic
REFRESH_INTERVAL = 30 #  # Refresh interval in seconds
now = time.time() # Current time in seconds since epoch

# Initialize session state
if "last_refresh" not in st.session_state: # if last refresh time not set
    st.session_state["last_refresh"] = now # Initialize last refresh time to current time

if "live_mode_prev" not in st.session_state: # if previous live mode state not set
    st.session_state["live_mode_prev"] = False # Initialize previous live mode state to False

if "prevent_double_rerun" not in st.session_state: # if prevent double rerun state not set
    st.session_state["prevent_double_rerun"] = False # Initialize prevent double rerun state to False

# Sidebar toggle
live_mode = st.sidebar.toggle("Live Mode", value=False) # toggle for live mode

# Check if refresh button was clicked
refresh_clicked = st.button("🔄 Refresh Dashboard")

# Prevent double rerun on toggle change
if st.session_state["prevent_double_rerun"]: # if prevent double rerun is True
    st.session_state["prevent_double_rerun"] = False # Reset to allow future reruns
else:
    # Check for mode change (only trigger rerun if this is the first detection)
    if st.session_state.get("live_mode_prev") != live_mode: # if previous live mode state is different from current
        st.session_state["live_mode_prev"] = live_mode # Update previous live mode state
        st.session_state["last_refresh"] = now # Update last refresh time
        st.session_state["prevent_double_rerun"] = True # Prevent double rerun
        st.rerun() # Rerun the app to apply mode change

    # Check for manual refresh
    if refresh_clicked: # if refresh button was clicked
        st.session_state["last_refresh"] = now # Update last refresh time
        st.rerun() # Rerun the app to refresh data

    # Check for time-based refresh (only in live mode)
    if live_mode and st.session_state.get("last_refresh", 0) < now - REFRESH_INTERVAL: # if last refresh time is older than refresh interval
        st.session_state["last_refresh"] = now # Update last refresh time
        st.rerun()  # Rerun the app to refresh data

# Base directories
BASE_DIR = "/app/kafka_stream"    # Base directory for Kafka stream data
STATIC_DIR = "/app/static_snapshots" # Static directory for non-live data

# Load data based on mode
if live_mode: # if live mode is enabled
    @st.cache_data(ttl=30) # Cache data for 30 seconds
    # Function to load prediction data
    def load_prediction_data():
        # Read live prediction logs from Kafka consumer
        return pd.read_csv(os.path.join(BASE_DIR, "dataset_prediction_logs.csv"), on_bad_lines='skip') # Read CSV with error handling

    @st.cache_data(ttl=30) # Cache data for 30 seconds
    # Function to load metrics data
    def load_metrics_data():
        return pd.read_csv(os.path.join(BASE_DIR, "dataset_metrics_log.csv")) # Read CSV for metrics

    @st.cache_data(ttl=30) # Cache data for 30 seconds
    # Function to load confusion matrix data
    def load_confusion_data(): 
        return pd.read_csv(os.path.join(BASE_DIR, "dataset_confusion_log.csv")) # Read CSV for confusion matrix
else: # if live mode is disabled
    # Static data loading
    def load_prediction_data():
        return pd.read_csv(os.path.join(STATIC_DIR, "static_prediction_logs.csv")) # Load static prediction logs

    def load_metrics_data():
        return pd.read_csv(os.path.join(STATIC_DIR, "static_metrics_log.csv")) # Load static metrics logs

    def load_confusion_data():
        return pd.read_csv(os.path.join(STATIC_DIR, "static_confusion_log.csv")) # Load static confusion matrix logs

# Load CSVs
pred_df = load_prediction_data() # Load prediction data
metrics_df = load_metrics_data() # Load metrics data
conf_df = load_confusion_data() # Load confusion matrix data

# Tab layout

# Create tabs for different sections of the dashboard
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Metrics Overview", 
    "Performance",
    "Recent Transactions + SHAP Insights",
    "Confusion Matrix",
    "Time Patterns",
    "Live Stream"
])

# Tab 1: Metrics Overview
with tab1:
    st.header("Model Performance Metrics")
    
    # Display overall metrics
    if not metrics_df.empty: # if metrics data is available
        latest = metrics_df.iloc[-1] # Get the latest metrics
        col1, col2, col3 = st.columns(3) # Display metrics in three columns
        col1.metric("Precision", round(latest["precision"], 2)) # Round precision to 2 decimal places
        col2.metric("Recall", round(latest["recall"], 2)) # Round recall to 2 decimal places
        col3.metric("F1 Score", round(latest["f1_score"], 2)) # Round F1 score to 2 decimal places
    else: # if metrics data is empty
        st.warning("Metrics log is empty.") # Display warning if no metrics data

    st.subheader("Fraud Summary")
    fraud_total = pred_df["predicted_label"].sum()
    total_txns = len(pred_df)
    ratio = round((fraud_total / total_txns) * 100, 2)
    st.metric("Fraud Count", f"{fraud_total} / {total_txns} ({ratio}%)")

# Tab 2: Performance Over Time
with tab2:
    st.header("Metrics Over Time")
    try: # Convert timestamp to datetime for plotting
        metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"], errors="coerce")
        metrics_df[["precision", "recall", "f1_score"]] = metrics_df[["precision", "recall", "f1_score"]].apply(pd.to_numeric, errors="coerce")
        metrics_df.dropna(inplace=True) # Drop rows with NaN values in metrics columns
        sampled_df = metrics_df.iloc[::10] # Sample every 10th row for performance

        fig, ax = plt.subplots(figsize=(6, 4)) # Create a figure for plotting
        ax.plot(sampled_df["timestamp"], sampled_df["precision"], label="Precision", marker='o') # Plot precision
        ax.plot(sampled_df["timestamp"], sampled_df["recall"], label="Recall", marker='x') # Plot recall
        ax.plot(sampled_df["timestamp"], sampled_df["f1_score"], label="F1 Score", marker='s') # Plot F1 score
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Score")
        ax.legend() # Show legend for the plot
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability
        fig.autofmt_xdate() # Automatically format x-axis dates
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Error loading performance graph: {e}")

# Tab 3: Recent Predictions + SHAP
with tab3:
    st.header("Recent Transactions")
    try:
        pred_df["Amount"] = pred_df["Amount"].round(2) # Round Amount to 2 decimal places
        selected_cols = ["timestamp", "Amount", "predicted_label", "true_label", "is_correct", "shap_top_features"]
        st.dataframe(pred_df[selected_cols].tail(100), use_container_width=True) # Display last 100 transactions
    except:
        st.dataframe(pred_df.tail(100), use_container_width=True) # Display last 100 transactions without SHAP features

    st.subheader("SHAP Feature Viewer")
    # Display SHAP features if available
    if "predicted_label" in pred_df.columns and "shap_top_features" in pred_df.columns:
        fraud_cases = pred_df[pred_df["predicted_label"] == 1] # Filter fraud cases
        if not fraud_cases.empty: # if there are fraud cases
            selected_row = st.selectbox("Choose fraud transaction:", fraud_cases.index) # Select a fraud transaction
            st.markdown("**Top SHAP features for selected case:**") # Display SHAP features for selected case
            st.text(fraud_cases.loc[selected_row, "shap_top_features"]) # Display SHAP features as text
        else:
            st.info("No fraud detected in current stream.")
    else:
        st.info("SHAP features not yet available.")

# Tab 4: Confusion Matrix
with tab4:
    st.header("Confusion Trends")
    try:
        conf_df["timestamp"] = pd.to_datetime(conf_df["timestamp"], errors="coerce") # Convert timestamp to datetime
        sampled_cm = conf_df.iloc[::50] # Sample every 50th row for performance

        fig, ax = plt.subplots(figsize=(6, 4)) # Create a figure for plotting confusion matrix
        ax.plot(sampled_cm["timestamp"], sampled_cm["TP"], label="TP", marker='o') # Plot True Positives
        ax.plot(sampled_cm["timestamp"], sampled_cm["FP"], label="FP", marker='x') # Plot False Positives
        ax.plot(sampled_cm["timestamp"], sampled_cm["TN"], label="TN", marker='^') # Plot True Negatives
        ax.plot(sampled_cm["timestamp"], sampled_cm["FN"], label="FN", marker='s') # Plot False Negatives
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Count")
        ax.legend()
        plt.xticks(rotation=45) # Rotate x-axis labels for better readability
        fig.autofmt_xdate() # Automatically format x-axis dates
        st.pyplot(fig) # Display confusion matrix chart
    except Exception as e:
        st.warning(f"Error rendering confusion matrix: {e}")

# Tab 5: Time Pattern Clustering
with tab5: # Time Patterns
    st.header("Fraud Time Patterns")
    try: # Convert timestamp to datetime and extract hour and day
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
        pred_df["hour"] = pred_df["timestamp"].dt.hour # Extract hour from timestamp
        pred_df["day"] = pred_df["timestamp"].dt.day_name() # Extract day name from timestamp
        fraud_df = pred_df[pred_df["predicted_label"] == 1] # Filter fraud cases
        # Plot hourly and daily fraud distribution
        if not fraud_df.empty:
            st.markdown("**Hourly Fraud Distribution**")
            hourly = fraud_df["hour"].value_counts().sort_index() # Count fraud cases by hour
            fig1, ax1 = plt.subplots(figsize=(6, 3)) # Create figure for hourly distribution
            ax1.bar(hourly.index, hourly.values, color='salmon') # Bar chart for hourly distribution
            ax1.set_xlabel("Hour") # Set x-axis label
            ax1.set_ylabel("Count") # Set y-axis label
            st.pyplot(fig1) # Display the hourly distribution chart

            st.markdown("**Weekly Fraud Distribution**")
            daily = fraud_df["day"].value_counts() # Count fraud cases by day
            fig2, ax2 = plt.subplots(figsize=(6, 3)) # Create figure for daily distribution
            ax2.bar(daily.index, daily.values, color='goldenrod') # Bar chart for daily distribution
            ax2.set_xlabel("Day") # Set x-axis label
            ax2.set_ylabel("Count") # Set y-axis label
            st.pyplot(fig2) # Display the daily distribution chart
        else: # if no fraud cases found
            st.info("No fraud data available yet.")
    except Exception as e:
        st.warning(f"Error rendering time patterns: {e}")

# Tab 6: Live Stream
with tab6:
    st.header("Live Fraud Stream (Kafka)")
    try:
        # Kafka consumer to read live predictions
        consumer = KafkaConsumer(
            "fraud_predictions",    # Topic to consume from
            bootstrap_servers="kafka:9092",  # Kafka server address
            auto_offset_reset="latest", # Start consuming from the latest message
            value_deserializer=lambda m: json.loads(m.decode("utf-8")) # Deserialize JSON messages
        )
        placeholder = st.empty() # Placeholder for live updates
        total_count, high_risk_count, batch_count, single_count = 0, 0, 0, 0 # Initialize counters
        # Process each message from the Kafka stream
        for msg in consumer:
            pred = msg.value # Deserialize message
            fraud_score = pred["fraud_score"] # Extract fraud score
            timestamp = pred["timestamp"] # Extract timestamp
            prediction_type = pred.get("prediction_type", "unknown") # Get prediction type (single/batch)`
            top_features = pred["top_features"] # Extract top features with SHAP values
            
            total_count += 1 # Increment total prediction count
            if fraud_score > 0.7: high_risk_count += 1 # Increment high risk count
            if prediction_type == "batch": batch_count += 1 # Increment batch count
            if prediction_type == "single": single_count += 1 # Increment single prediction count

            with placeholder.container(): # Display live updates
                col1, col2 = st.columns([2, 1]) # Create two columns for layout

                with col1: # Display prediction details
                    st.subheader("Live Prediction")
                    formatted_time = datetime.fromisoformat(timestamp).strftime("%H:%M:%S") # Format timestamp
                    st.write(f"**Timestamp:** `{formatted_time}`") # Display formatted timestamp
                    st.write(f"**Fraud Score:** `{fraud_score:.3f}`") # Display fraud score with 3 decimal places
                    st.write(f"**Type:** `{prediction_type}`") # Display prediction type (single/batch)

                    st.subheader("SHAP Contributions") # Display SHAP contributions
                    for feature, value in top_features: # Display each feature and its SHAP value
                        if value > 0: # if SHAP value is positive
                            st.write(f"**{feature}**: +{value:.4f} (increases risk)") # Display positive contribution
                        else: # if SHAP value is negative
                            st.write(f"**{feature}**: {value:.4f} (reduces risk)") # Display negative contribution

                with col2: # Display risk summary and visual indicator
                    # 🚦 Risk Summary
                    if fraud_score > 0.7: # High risk alert
                        st.error(f"HIGH RISK • Score: {fraud_score:.3f}")
                    elif fraud_score > 0.3: # Medium risk alert
                        st.warning(f"MEDIUM RISK • Score: {fraud_score:.3f}")
                    else: # Low risk alert
                        st.success(f"LOW RISK • Score: {fraud_score:.3f}")

                    # 📊 Visual Indicator
                    st.progress(min(fraud_score, 1.0))

                    # Live Session Stats
                    st.subheader("Live Session Stats") # Display live session statistics
                    c1, c2, c3 = st.columns(3) # Display three columns for stats
                    c1.metric("Total Predictions", total_count) # Total predictions made
                    c2.metric("High Risk Alerts", high_risk_count) # High risk alerts triggered
                    c3.metric("Batch vs Single", f"{batch_count}/{single_count}") # Batch vs Single predictions

    except Exception as e:
        st.error(f"Kafka consumer error: {e}")
