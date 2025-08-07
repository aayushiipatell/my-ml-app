import streamlit as st
import numpy as np
import joblib
import os

# App title and setup
st.set_page_config(page_title="Ridge Regression Predictor")
st.title("🔮 Ridge Regression Predictor")
st.markdown("Enter values for the selected **15 input features**:")

# Model file
MODEL_FILE = "ridge_pipeline.pkl"

# Load model and feature names
if os.path.exists(MODEL_FILE):
    try:
        model, feature_names = joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
else:
    st.error(f"❌ Model file '{MODEL_FILE}' not found.")
    st.stop()

# Manually define feature names (if not loaded with model)
# Uncomment if you're not storing feature names inside the model
# feature_names = [
#     'bank_id_AXIS', 'bank_id_KOTAK', 'quarter_Q1_2021', 'quarter_Q1_2023',
#     'quarter_Q1_2024', 'quarter_Q1_2025', 'quarter_Q2_2023', 'quarter_Q2_2024',
#     'quarter_Q2_2025', 'quarter_Q3_2023', 'quarter_Q3_2024', 'quarter_Q3_2025',
#     'quarter_Q4_2023', 'quarter_Q4_2024', 'quarter_Q4_2025'
# ]

# Collect input for all 15 features
st.markdown("### Input Features")
feature_inputs = []

for name in feature_names:
    value = st.number_input(
        label=name,
        key=name,
        value=0.0,
        min_value=0.0,
        max_value=1.0,
        step=1.0,
        format="%.0f"
    )
    feature_inputs.append(value)

# Predict button
if st.button("Predict"):
    try:
        input_array = np.array([feature_inputs])  # Shape: (1, 15)
        prediction = model.predict(input_array)
        st.success(f"🎯 Predicted Value: **{prediction[0]:.4f}**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
