import streamlit as st
import numpy as np
import joblib
import os

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="KMeans Clustering Deployment",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Global Development Clustering")
st.write("Enter country-level indicators to predict the cluster")

# ---------------------------------------------------
# Load Model & Scaler
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists("scaler.pkl"):
        st.error("❌ scaler.pkl not found")
        st.stop()

    if not os.path.exists("kmeans.pkl"):
        st.error("❌ kmeans.pkl not found")
        st.stop()

    scaler = joblib.load("scaler.pkl")
    kmeans = joblib.load("kmeans.pkl")
    return scaler, kmeans


scaler, kmeans = load_artifacts()

# ---------------------------------------------------
# Feature Names (22 columns used in training)
# ---------------------------------------------------
feature_names = [
    "Birth Rate",
    "Business Tax",
    "CO2 Emissions",
    "Days to Start Business",
    "Ease of Business",
    "Energy Usage",
    "GDP",
    "Health Expenditure (% GDP)",
    "Health Expenditure (Per Capita)",
    "Hours to Do Tax",
    "Infant Mortality",
    "Internet Usage",
    "Lending Rate",
    "Life Expectancy (Female)",
    "Life Expectancy (Male)",
    "Mobile Phone Usage",
    "Population (0–14)",
    "Population (15–64)",
    "Population (65+)",
    "Urban Population",
    "Unemployment Rate",
    "Service Sector (% GDP)"
]

st.subheader("🔢 Input Features")
st.caption(f"Model expects *{scaler.n_features_in_} features*")

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------
inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button("🔍 Predict Cluster"):
    try:
        X = np.array(inputs).reshape(1, -1)
        X_scaled = scaler.transform(X)
        cluster = kmeans.predict(X_scaled)[0]

        st.success(f"✅ Predicted Cluster: *Cluster {cluster}*")

        # Optional interpretation
        cluster_meaning = {
            0: "Low Development Countries",
            1: "Medium Development Countries",
            2: "High Development Countries"
        }

        if cluster in cluster_meaning:
            st.info(f"📌 Cluster Interpretation: *{cluster_meaning[cluster]}*")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
