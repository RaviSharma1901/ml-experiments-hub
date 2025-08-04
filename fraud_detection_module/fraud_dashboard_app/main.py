import streamlit as st          # UI framework
import pandas as pd             # Data handling
import matplotlib.pyplot as plt # Plotting
import os                       # File paths
from datetime import datetime

# ──────────────────────────────
# Page configuration
# ──────────────────────────────
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# ──────────────────────────────
# Paths
# ──────────────────────────────
STATIC_DIR = "static_snapshots"           # <── adjust if you place the CSVs elsewhere
PRED_LOG  = os.path.join(STATIC_DIR, "static_prediction_logs.csv")
METRIC_LOG = os.path.join(STATIC_DIR, "static_metrics_log.csv")
CONF_LOG   = os.path.join(STATIC_DIR, "static_confusion_log.csv")

# ──────────────────────────────
# Cached file-loaders
# ──────────────────────────────
@st.cache_data
def load_prediction_data(): return pd.read_csv(PRED_LOG, on_bad_lines="skip")

@st.cache_data
def load_metrics_data():    return pd.read_csv(METRIC_LOG)

@st.cache_data
def load_confusion_data():  return pd.read_csv(CONF_LOG)

pred_df   = load_prediction_data()
metrics_df = load_metrics_data()
conf_df    = load_confusion_data()

# ──────────────────────────────
# Tabs
# ──────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Metrics Overview",
    "Performance",
    "Recent Tx + SHAP",
    "Confusion Matrix",
    "Time Patterns"
])

# ── Tab 1 ─────────────────────
with tab1:
    st.header("Model Performance Metrics")

    if not metrics_df.empty:
        latest = metrics_df.iloc[-1]
        c1, c2, c3 = st.columns(3)

        c1.metric("Precision", f"{latest['precision']:.2f}")
        c2.metric("Recall",    f"{latest['recall']:.2f}")
        c3.metric("F1 Score",  f"{latest['f1_score']:.2f}")
    else:
        st.warning("Metrics log is empty.")

    st.subheader("Fraud Summary")
    fraud_total = pred_df["predicted_label"].sum()
    total_txns  = len(pred_df)
    ratio       = round((fraud_total / total_txns) * 100, 2)
    st.metric("Fraud Count", f"{fraud_total} / {total_txns} ({ratio} %)")

# ── Tab 2 ─────────────────────
with tab2:
    st.header("Metrics Over Time")

    try:
        metrics_df["timestamp"] = pd.to_datetime(metrics_df["timestamp"], errors="coerce")
        for col in ["precision", "recall", "f1_score"]:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
        metrics_df.dropna(inplace=True)

        sampled = metrics_df.iloc[::10]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sampled["timestamp"], sampled["precision"], label="Precision", marker="o")
        ax.plot(sampled["timestamp"], sampled["recall"],    label="Recall",    marker="x")
        ax.plot(sampled["timestamp"], sampled["f1_score"],  label="F1 Score",  marker="s")
        ax.set_xlabel("Timestamp"); ax.set_ylabel("Score"); ax.legend()
        plt.xticks(rotation=45); fig.autofmt_xdate()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Performance graph error: {e}")

# ── Tab 3 ─────────────────────
with tab3:
    st.header("Recent Transactions")
    pred_df["Amount"] = pred_df["Amount"].round(2)
    cols = ["timestamp", "Amount", "predicted_label", "true_label",
            "is_correct", "shap_top_features"]
    st.dataframe(pred_df[cols].tail(100), use_container_width=True)

    st.subheader("SHAP Feature Viewer")
    if {"predicted_label", "shap_top_features"}.issubset(pred_df.columns):
        fraud_cases = pred_df[pred_df["predicted_label"] == 1]
        if not fraud_cases.empty:
            row_idx = st.selectbox("Select fraud transaction:", fraud_cases.index)
            st.markdown("**Top SHAP features:**")
            st.text(fraud_cases.loc[row_idx, "shap_top_features"])
        else:
            st.info("No fraud cases in snapshot.")
    else:
        st.info("SHAP features not available.")

# ── Tab 4 ─────────────────────
with tab4:
    st.header("Confusion Matrix Trends")

    try:
        conf_df["timestamp"] = pd.to_datetime(conf_df["timestamp"], errors="coerce")
        sampled = conf_df.iloc[::50]
        fig, ax = plt.subplots(figsize=(6, 4))
        for col, marker in zip(["TP", "FP", "TN", "FN"], "o^xs"):
            ax.plot(sampled["timestamp"], sampled[col], label=col, marker=marker)
        ax.set_xlabel("Timestamp"); ax.set_ylabel("Count"); ax.legend()
        plt.xticks(rotation=45); fig.autofmt_xdate()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Confusion-matrix error: {e}")

# ── Tab 5 ─────────────────────
with tab5:
    st.header("Fraud Time Patterns")

    try:
        pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
        fraud_df = pred_df[pred_df["predicted_label"] == 1]

        if not fraud_df.empty:
            fraud_df["hour"] = fraud_df["timestamp"].dt.hour
            fraud_df["day"]  = fraud_df["timestamp"].dt.day_name()

            st.markdown("**Hourly Fraud Distribution**")
            hourly = fraud_df["hour"].value_counts().sort_index()
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            ax1.bar(hourly.index, hourly.values, color="salmon")
            ax1.set_xlabel("Hour"); ax1.set_ylabel("Count")
            st.pyplot(fig1)

            st.markdown("**Weekly Fraud Distribution**")
            daily = fraud_df["day"].value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.bar(daily.index, daily.values, color="goldenrod")
            ax2.set_xlabel("Day"); ax2.set_ylabel("Count")
            st.pyplot(fig2)
        else:
            st.info("No fraud data in snapshot.")
    except Exception as e:
        st.warning(f"Time-pattern error: {e}")
