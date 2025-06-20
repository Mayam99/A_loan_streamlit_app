# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("models/xgb_loan_default_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# App title
st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💸")
st.title("🏦 Loan Default Risk Predictor")
st.markdown("This app predicts the likelihood that a customer will default on a loan based on their profile. Provide the input values below.")

# Input form
with st.form("prediction_form"):
    st.subheader("📋 Enter Applicant Details")

    income = st.number_input("Annual Income (INR)", min_value=0.0, step=1000.0, value=50000.0)
    credit_amount = st.number_input("Loan Amount Requested (INR)", min_value=0.0, step=1000.0, value=200000.0)
    annuity = st.number_input("Monthly Payment (Annuity)", min_value=0.0, step=500.0, value=15000.0)
    age_days = st.number_input("Age (in negative days)", value=-12000)
    employment_days = st.number_input("Days Employed (negative = currently working)", value=-3000)
    ext_score = st.slider("External Credit Score (EXT_SOURCE_2)", min_value=0.0, max_value=1.0, value=0.5)

    submitted = st.form_submit_button("🔍 Predict Risk")

# Predict and display result
if submitted:
    input_data = pd.DataFrame({
        'AMT_INCOME_TOTAL': [income],
        'AMT_CREDIT': [credit_amount],
        'AMT_ANNUITY': [annuity],
        'DAYS_BIRTH': [age_days],
        'DAYS_EMPLOYED': [employment_days],
        'EXT_SOURCE_2': [ext_score]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The applicant is likely to DEFAULT. Risk Score: **{probability:.2%}**")
    else:
        st.success(f"✅ The applicant is NOT likely to default. Risk Score: **{probability:.2%}**")
