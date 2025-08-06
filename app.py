import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('linear_model.pkl')

# Streamlit App UI
st.title("Linear Regression Predictor")

st.write("Enter feature values below:")

# Replace these with actual feature names and count
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")
feature5 = st.number_input("Feature 5")

# Prediction button
if st.button("Predict"):
    features = np.array([[feature1, feature2, feature3, feature4, feature5]])
    prediction = model.predict(features)
    st.success(f"Predicted Value: {prediction[0]}")
