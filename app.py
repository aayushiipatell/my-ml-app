import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Ridge Regression Predictor")
st.title("🔮 Ridge Regression Predictor")
st.markdown("Enter values for the selected features to predict the target value.")

MODEL_FILE = "ridge_model.pkl"

# Load the model & feature names
if os.path.exists(MODEL_FILE):
    data = joblib.load(MODEL_FILE)
    model = data["model"]
    feature_names = data["features"]
else:
    st.error(f"Model file '{MODEL_FILE}' not found. Please upload it to the app directory.")
    st.stop()

# Create number inputs dynamically
st.markdown("### Input Features")
features = []
for name in feature_names:
    value = st.number_input(f"{name}", value=0.0)
    features.append(value)

# Predict button
if st.button("Predict"):
    try:
        input_array = np.array([features])
        prediction = model.predict(input_array)
        st.success(f"🎯 Predicted Value: **{prediction[0]:.4f}**")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
