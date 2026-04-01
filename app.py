import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction (Framingham Dataset)")

MODEL_PATH = "logistic_14 (2).pkl"

# ------------------------------------------------
# LOAD MODEL + SCALER
# ------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model, scaler = pickle.load(f)
        return model, scaler
    else:
        st.error("❌ Model file not found! Place logistic_14.pkl in the same folder.")
        return None, None

model, scaler = load_model()

# ------------------------------------------------
# INPUT UI (14 FEATURES)
# ------------------------------------------------
st.header("📌 Patient Information")

col1, col2 = st.columns(2)

with col1:
    male = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 1, 120, 45)
    currentSmoker = st.selectbox("Current Smoker?", [0, 1])
    cigsPerDay = st.number_input("Cigarettes Per Day", 0, 70, 0)
    BPMeds = st.selectbox("On BP Medication?", [0, 1])
    prevalentStroke = st.selectbox("Prevalent Stroke?", [0, 1])
    prevalentHyp = st.selectbox("Hypertension?", [0, 1])

with col2:
    diabetes = st.selectbox("Diabetes?", [0, 1])
    totChol = st.number_input("Total Cholesterol", 100, 700, 200)
    sysBP = st.number_input("Systolic BP", 80, 240, 120)
    diaBP = st.number_input("Diastolic BP", 50, 150, 80)
    BMI = st.number_input("BMI", 10.0, 60.0, 25.0)
    heartRate = st.number_input("Heart Rate", 30, 220, 70)
    glucose = st.number_input("Glucose", 40, 500, 80)

# ------------------------------------------------
# PREPARE INPUT FOR MODEL
# ------------------------------------------------
input_data = np.array([[
    1 if male == "Male" else 0,
    age,
    currentSmoker,
    cigsPerDay,
    BPMeds,
    prevalentStroke,
    prevalentHyp,
    diabetes,
    totChol,
    sysBP,
    diaBP,
    BMI,
    heartRate,
    glucose
]])

# ------------------------------------------------
# PREDICTION BUTTON
# ------------------------------------------------
st.write("")

if st.button("🔍 Predict"):
    if model is None:
        st.error("Model not loaded.")
    else:
        scaled_data = scaler.transform(input_data)
        pred = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0][1]

        if pred == 1:
            st.error(f"⚠️ High Risk of Heart Disease\nProbability: {prob:.2f}")
        else:
            st.success(f"✅ Low Risk of Heart Disease\nProbability: {prob:.2f}")