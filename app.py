import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Ridge Regression Predictor")
st.title("🔮 Ridge Regression Predictor")

MODEL_FILE = "ridge_model.pkl"
FEATURES_FILE = "selected_feature_names.pkl"

# Load model & features
if os.path.exists(MODEL_FILE) and os.path.exists(FEATURES_FILE):
    model = joblib.load(MODEL_FILE)
    selected_feature_names = joblib.load(FEATURES_FILE)
else:
    st.error("Model or feature names file not found. Please upload them to the app directory.")
    st.stop()

# Input form
st.header("Enter feature values:")
features = []
for name in selected_feature_names:
    val = st.number_input(f"{name}", value=0.0)
    features.append(val)

# Predict
if st.button("Predict"):
    try:
        input_array = np.array([features])
        prediction = model.predict(input_array)
        st.success(f"🎯 Predicted value: **{prediction[0]:.4f}**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")


