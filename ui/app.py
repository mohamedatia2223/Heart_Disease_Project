import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pipeline = joblib.load("models/heart_disease_pipeline_svm.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("Heart Disease Prediction App")
st.write("Enter patient health details to predict heart disease risk.")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
chol = st.number_input("Cholesterol (chol)", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalch = st.number_input("Max Heart Rate (thalach)", 70, 220, 150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

input_data = pd.DataFrame([{
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalch": thalch,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1] if hasattr(pipeline, "predict_proba") else None

    if prediction == 1:
        st.error("High risk of Heart Disease detected!")
    else:
        st.success("Low risk of Heart Disease.")

    if prob is not None:
        st.write(f"**Prediction Confidence:** {prob*100:.2f}%")

