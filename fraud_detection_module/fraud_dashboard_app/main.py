import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for server environments
import matplotlib.pyplot as plt
import os
from pathlib import Path # <--- IMPORT PATHLIB

# ──────────────────────────────
# Page configuration
# ──────────────────────────────
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")

# ──────────────────────────────
# Paths (Corrected for Streamlit Cloud)
# ──────────────────────────────
# Get the directory of the current script
SCRIPT_DIR = Path(__file__).parent
# Define the static directory relative to the script's location
STATIC_DIR = SCRIPT_DIR / "static_snapshots"

FILE_PATHS = {
    "predictions": STATIC_DIR / "static_prediction_logs.csv",
    "metrics": STATIC_DIR / "static_metrics_log.csv",
    "confusion": STATIC_DIR / "static_confusion_log.csv",
}

# ──────────────────────────────
# Cached file-loaders
# ──────────────────────────────
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return pd.DataFrame()
    return pd.read_csv(file_path, on_bad_lines='skip')

pred_df = load_data(FILE_PATHS["predictions"])
metrics_df = load_data(FILE_PATHS["metrics"])
conf_df = load_data(FILE_PATHS["confusion"])

# ──────────────────────────────
# Diagnostic Expander (Optional but helpful)
# ──────────────────────────────
with st.expander("🔬 Click to see Debug Information"):
    st.subheader("File & Data Validation")
    for name, path in FILE_PATHS.items():
        st.markdown(f"**Checking for `{path}`...**")
        if os.path.exists(path):
            st.success("File found!")
        else:
            st.error("File NOT found! Check path and file structure in your repo.")

# ──────────────────────────────
# Tabs
# ──────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Metrics Overview", "Performance", "Recent Tx + SHAP", "Confusion Matrix", "Time Patterns"
])

# ... (The rest of your tab code remains exactly the same) ...
# (The code for Tab 1, Tab 2, Tab 3, Tab 4, and Tab 5 follows here, unchanged)

# ── Tab 1: Metrics Overview ─────────────────────
with tab1:
    st.header("Model Performance Metrics")

    if not metrics_df.empty and all(col in metrics_df.columns for col in ['precision', 'recall', 'f1_score']):
        latest = metrics_df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{latest.get('precision', 0):.2f}")
        c2.metric("Recall", f"{latest.get('recall', 0):.2f}")
        c3.metric("F1 Score", f"{latest.get('f1_score', 0):.2f}")
    else:
        st.info("Metrics data not available to display.")

    st.subheader("Fraud Summary")
    if not pred_df.empty and "predicted_label" in pred_df.columns:
        fraud_total = pred_df["predicted_label"].sum()
        total_txns = len(pred_df)
        ratio = round((fraud_total / total_txns) * 100, 2) if total_txns > 0 else 0
        st.metric("Fraud Count", f"{int(fraud_total)} / {total_txns} ({ratio}%)")
    else:
        st.info("Prediction data not available for summary.")

# ── Tab 2: Performance Over Time ─────────────────────
with tab2:
    st.header("Metrics Over Time")
    if not metrics_df.empty:
        try:
            df_plot = metrics_df.copy()
            df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')
            for col in ["precision", "recall", "f1_score"]:
                df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
            
            df_plot.dropna(subset=['timestamp', 'precision', 'recall', 'f1_score'], inplace=True)

            if not df_plot.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df_plot["timestamp"], df_plot["precision"], label="Precision", marker="o")
                ax.plot(df_plot["timestamp"], df_plot["recall"], label="Recall", marker="x")
                ax.plot(df_plot["timestamp"], df_plot["f1_score"], label="F1 Score", marker="s")
                ax.set_xlabel("Timestamp"); ax.set_ylabel("Score"); ax.legend(); ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45); fig.autofmt_xdate(); ax.set_ylim(0, 1.1)
                st.pyplot(fig)
            else:
                st.info("No valid metric data to plot after cleaning. Check the Debug Expander at the top of the page.")
        except Exception as e:
            st.error(f"Could not render performance graph: {e}")
    else:
        st.info("Metrics data file is empty or could not be loaded.")

# ── Tab 3: Recent Transactions + SHAP ─────────────────────
with tab3:
    st.header("Recent Transactions")
    if not pred_df.empty:
        display_df = pred_df.copy()
        if "Amount" in display_df.columns:
            display_df["Amount"] = display_df["Amount"].round(2)
        
        display_cols = [col for col in ["timestamp", "Amount", "predicted_label", "true_label", "is_correct", "shap_top_features"] if col in display_df.columns]
        st.dataframe(display_df[display_cols].tail(100), use_container_width=True)

        st.subheader("SHAP Feature Viewer")
        if {"predicted_label", "shap_top_features"}.issubset(display_df.columns):
            fraud_cases = display_df[display_df["predicted_label"] == 1]
            if not fraud_cases.empty:
                row_idx = st.selectbox("Select a fraud transaction:", fraud_cases.index)
                st.markdown("**Top SHAP features for this transaction:**")
                st.text(fraud_cases.loc[row_idx, "shap_top_features"])
            else:
                st.info("No fraudulent transactions found in this snapshot.")
        else:
            st.info("SHAP data not available.")
    else:
        st.info("Transaction data not available.")

# ── Tab 4: Confusion Matrix Trends ─────────────────────
with tab4:
    st.header("Confusion Matrix Trends")
    if not conf_df.empty:
        try:
            df_plot = conf_df.copy()
            df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')
            df_plot.dropna(subset=['timestamp'], inplace=True)
            
            plot_cols = [col for col in ["TP", "FP", "TN", "FN"] if col in df_plot.columns]
            if not df_plot.empty and plot_cols:
                fig, ax = plt.subplots(figsize=(10, 5))
                for col, marker in zip(plot_cols, "o^xs"):
                    ax.plot(df_plot["timestamp"], df_plot[col], label=col, marker=marker)
                ax.set_xlabel("Timestamp"); ax.set_ylabel("Count"); ax.legend(); ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45); fig.autofmt_xdate()
                st.pyplot(fig)
            else:
                st.info("No valid confusion matrix data to plot.")
        except Exception as e:
            st.error(f"Could not render confusion matrix graph: {e}")
    else:
        st.info("Confusion matrix data not available.")

# ── Tab 5: Time Patterns ─────────────────────
with tab5:
    st.header("Fraud Time Patterns")
    if not pred_df.empty and "predicted_label" in pred_df.columns and "timestamp" in pred_df.columns:
        try:
            df_plot = pred_df.copy()
            df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')
            fraud_df = df_plot[df_plot["predicted_label"] == 1].copy()

            if not fraud_df.empty:
                fraud_df["hour"] = fraud_df["timestamp"].dt.hour
                fraud_df["day"]  = fraud_df["timestamp"].dt.day_name()

                st.markdown("**Hourly Fraud Distribution**")
                hourly = fraud_df["hour"].value_counts().sort_index()
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.bar(hourly.index, hourly.values, color="salmon")
                ax1.set_xlabel("Hour of Day"); ax1.set_ylabel("Fraud Count"); ax1.grid(True, alpha=0.2)
                ax1.set_xticks(range(24))
                st.pyplot(fig1)

                st.markdown("**Weekly Fraud Distribution**")
                days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                daily = fraud_df["day"].value_counts().reindex(days_order).fillna(0)
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.bar(daily.index, daily.values, color="goldenrod")
                ax2.set_xlabel("Day of Week"); ax2.set_ylabel("Fraud Count"); ax2.grid(True, alpha=0.2)
                plt.xticks(rotation=45)
                st.pyplot(fig2)
            else:
                st.info("No fraudulent transactions found to analyze time patterns.")
        except Exception as e:
            st.error(f"Could not render time patterns: {e}")
    else:
        st.info("Prediction or timestamp data not available for time analysis.")
