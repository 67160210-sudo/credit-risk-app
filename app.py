# ==========================================
# CREDIT RISK APP (STREAMLIT UI)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
model = joblib.load("model_artifacts/model.pkl")

with open("model_artifacts/features.json") as f:
    feature_names = json.load(f)


# -------------------------------
# Function
# -------------------------------
def predict_risk_percent(model, input_df):
    proba = model.predict_proba(input_df)[:, 1]
    risk_percent = proba * 100

    # RULE
    mask = input_df["balance"] > input_df["income"]
    risk_percent[mask] = 100.0

    label = ["เสี่ยง" if r >= 50 else "ไม่เสี่ยง" for r in risk_percent]

    return risk_percent, label


# -------------------------------
# UI
# -------------------------------
st.title("💳 Credit Default Risk Predictor")
st.markdown("ประเมินความเสี่ยงการผิดนัดชำระหนี้ด้วย Machine Learning")

st.divider()

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📥 กรอกข้อมูล")

col1, col2 = st.columns(2)

with col1:
    student = st.selectbox("เป็นนักเรียนหรือไม่", ["No", "Yes"])
    student_val = 1 if student == "Yes" else 0

with col2:
    balance = st.number_input("ยอดหนี้ (Balance)", min_value=0.0, value=1000.0)

income = st.number_input("รายได้ (Income)", min_value=0.0, value=3000.0)


# -------------------------------
# Predict Button
# -------------------------------
if st.button("🔮 ทำนายความเสี่ยง"):

    input_data = pd.DataFrame([{
        "student": student_val,
        "balance": balance,
        "income": income
    }])

    # Feature Engineering
    input_data["debt_income_ratio"] = input_data["balance"] / (input_data["income"] + 1e-6)
    input_data["is_debt_gt_income"] = (input_data["balance"] > input_data["income"]).astype(int)

    # เรียง feature
    input_data = input_data[feature_names]

    # Predict
    risk_percent, risk_label = predict_risk_percent(model, input_data)

    risk = risk_percent[0]
    label = risk_label[0]

    st.divider()

    # -------------------------------
    # Result Display
    # -------------------------------
    st.subheader("📊 ผลลัพธ์")

    # Progress bar
    st.progress(int(risk))

    # สีตามความเสี่ยง
    if risk >= 70:
        st.error(f"🔴 ความเสี่ยงสูง: {risk:.2f}%")
    elif risk >= 40:
        st.warning(f"🟠 ความเสี่ยงปานกลาง: {risk:.2f}%")
    else:
        st.success(f"🟢 ความเสี่ยงต่ำ: {risk:.2f}%")

    st.markdown(f"### 📌 สรุป: **{label}**")

    # Rule explanation
    if balance > income:
        st.info("⚠️ หนี้มากกว่ารายได้ → ระบบปรับเป็นเสี่ยงทันที (Business Rule)")


# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("Model: RandomForest | Built with ❤️ using Streamlit")