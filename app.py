import streamlit as st
import joblib
import numpy as np

# ✅ Load model
model = joblib.load('heart_model.pkl')  # <-- This must be the model, not a NumPy array

st.title("Heart Disease Prediction")

# Example inputs
age = st.number_input("Age", 1, 120)
chol = st.number_input("Cholesterol", 0, 600)
trestbps = st.number_input("Resting BP", 0, 200)
thalach = st.number_input("Max heart rate", 0, 250)

# Build input data array
input_data = np.array([[age, chol, trestbps, thalach]])

# ✅ Predict only if model is correct
if st.button("Predict"):
    prediction = model.predict(input_data)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.success(f"Prediction: {result}")
