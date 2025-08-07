import streamlit as st
import numpy as np
import joblib
import os

# Set up Streamlit page
st.set_page_config(page_title="Ridge Regression Predictor")
st.title("🔮 Ridge Regression Predictor")
st.markdown("Enter values for the selected **15 input features** to predict the target value.")

# Path to model file
MODEL_FILE = "ridge_model.pkl"

# Load the model and feature names
if os.path.exists(MODEL_FILE):
    try:
        model, feature_names = joblib.load(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model file: {str(e)}")
        st.stop()
else:
    st.error(f"Model file '{MODEL_FILE}' not found. Please upload it to the app directory.")
    st.stop()

# Input section
st.markdown("### Input Features")
features = []

for name in feature_names:
    value = st.number_input(name, value=0.0, format="%.4f")
    features.append(value)

# Predict button
if st.button("Predict"):
    try:
        input_array = np.array([features])  # shape: (1, 15)
        prediction = model.predict(input_array)
        st.success(f"🎯 Predicted Value: **{prediction[0]:.4f}**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")

