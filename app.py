import streamlit as st
import numpy as np
import joblib
import os

st.title("Ridge Regression Predictor")

MODEL_FILE = "ridge_pipeline.pkl"

# Load trained pipeline (RFE + Ridge)
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error("Trained model not found.")
    st.stop()

# Collect input for all 15 features
feature_inputs = []

for i in range(1, 16):  # assuming features are f1 to f15
    value = st.number_input(f"Feature {i}", key=f"f{i}")
    feature_inputs.append(value)

# Make prediction
if st.button("Predict"):
    try:
        features = np.array([feature_inputs])
        prediction = model.predict(features)
        st.success(f"Predicted value: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

