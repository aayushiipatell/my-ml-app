import streamlit as st
import numpy as np
import joblib
import os

st.title("Ridge Regression Predictor")
st.markdown("Enter values for the selected features to predict the target value.")

# Check if the model file exists
MODEL_FILE = "ridge_model.pkl"

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    st.error(f"Model file '{MODEL_FILE}' not found. Please upload it to the app directory.")
    st.stop()

# Input fields — adjust number and names according to your selected features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")
feature5 = st.number_input("Feature 5")

# Predict button
if st.button("Predict"):
    try:
        # Arrange inputs as a 2D array
        features = np.array([[feature1, feature2, feature3, feature4, feature5]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display result
        st.success(f"Predicted Value: {prediction[0]:.4f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
