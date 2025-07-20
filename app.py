import streamlit as st
from outputs import load_model, load_scaler
from utils import load_bearing_data
import pandas as pd
import numpy as np
import time

# Page setup
st.set_page_config(page_title="Bearing Fault Diagnosis", page_icon="🔧")
st.title('🔧 Bearing Fault Diagnosis App')

# File uploader
uploaded_file = st.file_uploader("📂 Upload a `.mat` file (bearing vibration data)", type=["mat"])

# Main app logic
if uploaded_file is not None:
    # Features Extraction Section
    st.header("📊 Extracted Signal Features (for every 0.2 seconds)")
    features = load_bearing_data(uploaded_file)
    st.dataframe(features.style.background_gradient(cmap='Blues'), use_container_width=True)

    # Load Model and Scaler
    scaler = load_scaler()
    model = load_model()

    # Scale features and Predict
    scaled_features = scaler.transform(features)
    predictions = model.predict(scaled_features)

    # Status + Progress Bar
    st.header("🔍 Prediction Status (Live Update)")
    progress_bar = st.progress(0)
    placeholder = st.empty()

    results = []
    total_segments = len(predictions)

    for i in range(total_segments):
        pred = predictions[i]
        results.append(pred)

        progress_percentage = (i + 1) / total_segments
        progress_bar.progress(progress_percentage)

        placeholder.markdown(f"**Segment {i+1}/{total_segments}** → Prediction: `{pred}`")
        time.sleep(0.01)

    progress_bar.empty()
    st.success("✅ All segments processed successfully!")

    # Final Results Section
    st.header("📋 Prediction Summary Table")
    df_results = pd.DataFrame({
        "Segment": np.arange(1, total_segments + 1),
        "Time (s)": np.round(np.arange(0.2, 0.2 * total_segments + 0.1, 0.2), 2),
        "Predicted Condition": results
    })
    st.dataframe(df_results.style.background_gradient(cmap='Greens'), use_container_width=True)

    st.caption("ℹ️ *Each 'Segment' represents 0.2 seconds of vibration data. Predictions are either 'Normal' or 'Fault'.*")

    # Distribution Plot
    st.header("📈 Fault Occurrence (Across All Segments)")
    fault_counts = df_results["Predicted Condition"].value_counts().reset_index()
    fault_counts.columns = ["Condition", "Count"]
    st.bar_chart(fault_counts.set_index("Condition"))
