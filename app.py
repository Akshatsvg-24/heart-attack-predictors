
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Heart Attack Risk Predictor")
st.title("🫀 Heart Attack Risk Predictor")
st.write("Enter your basic health info to predict the risk.")

# Input form
age = st.number_input("Age", min_value=1, max_value=120, value=30)
cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=500, value=200)
bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)

# Predict
if st.button("Predict Risk"):
    input_data = scaler.transform([[age, cholesterol, bp, max_hr]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][int(pred)]

    if pred == 1:
        st.error(f"⚠️ High risk of heart attack ({prob:.2%} confidence)")
    else:
        st.success(f"✅ Low risk of heart attack ({prob:.2%} confidence)")
