"""
drift_dashboard.py
------------------
Interactive Streamlit dashboard for monitoring feature drift between
training and test datasets using the KS-test.
"""

import streamlit as st # type: ignore
import pandas as pd
import plotly.express as px # type: ignore
from data import load_data, preprocess_data, split_data
from data_drift import detect_data_drift

st.set_page_config(page_title="Data Drift Dashboard", layout="wide")

st.title("📊 Data Drift Monitoring Dashboard")
st.markdown("""
Use this dashboard to compare the **training** and **test** distributions
of your engineered features and identify drifted variables over time.
""")

# --- Sidebar configuration ---
st.sidebar.header("Configuration")
data_path = st.sidebar.text_input("Data CSV path:", "data/prices.csv")
pval_threshold = st.sidebar.slider("P-value threshold for drift detection", 0.01, 0.10, 0.05, 0.01)

# --- Load data ---
try:
    df = load_data(data_path)
    df = preprocess_data(df)
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()

# --- Split into train/val/test (reuse project logic) ---
train_df, val_df, test_df = split_data(df)

st.sidebar.write(f"✅ Train size: {len(train_df)}, Test size: {len(test_df)}")

# --- Compute drift statistics ---
st.subheader("📈 KS-Test Feature Drift Results")

drift_df = detect_data_drift(train_df, test_df)
drift_df["drift_detected"] = drift_df["p_value"] < pval_threshold

# --- Sort by p-value ---
drift_df_sorted = drift_df.sort_values("p_value").reset_index(drop=True)

# --- Display summary metrics ---
n_drifted = drift_df_sorted["drift_detected"].sum()
total_features = len(drift_df_sorted)
st.metric(label="Number of Drifted Features", value=f"{n_drifted}/{total_features}")
st.dataframe(drift_df_sorted.style.background_gradient(cmap="coolwarm_r", subset=["p_value"]))

# --- Plot Top Drifted Features ---
st.subheader("🔍 Top Drifted Features")

top_features = drift_df_sorted.query("drift_detected == True").head(5)["feature"].tolist()
if not top_features:
    st.info("No significant drift detected at current p-value threshold.")
else:
    selected_feature = st.selectbox("Select a feature to visualize:", top_features)
    fig = px.histogram(
        pd.concat(
            [
                pd.DataFrame({"value": train_df[selected_feature], "dataset": "Train"}),
                pd.DataFrame({"value": test_df[selected_feature], "dataset": "Test"})
            ]
        ),
        x="value",
        color="dataset",
        nbins=50,
        barmode="overlay",
        title=f"Distribution Comparison for '{selected_feature}'"
    )
    fig.update_traces(opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Powered by KS-test statistical drift detection — retrain your model if key features drift significantly.")
