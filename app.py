import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Ridge Regression Predictor")
st.title("🔮 Ridge Regression Predictor")
st.markdown("Enter values for **15 input features** to predict the target value.")

# Model file path
MODEL_FILE = "ridge_model.pkl"

# Load the model or show error
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error(f"Model file '{MODEL_FILE}' not found. Please upload it to the app directory.")
    st.stop()

# Create input fields for 15 features
st.markdown("### Input Features")
features = []
for i in range(1, 16):
    value = st.number_input(f"Feature {i}", value=0.0)
    features.append(value)

# Prediction logic
if st.button("Predict"):
    try:
        input_array = np.array([features])  # shape: (1, 15)
        prediction = model.predict(input_array)
        st.success(f"🎯 Predicted Value: **{prediction[0]:.4f}**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
