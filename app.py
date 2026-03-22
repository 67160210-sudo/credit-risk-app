# ==========================================
# CREDIT RISK APP (FIX PATH VERSION)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# -------------------------------
# Paths (🔥 แก้ตรงนี้)
# -------------------------------
BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "features.json")
DATA_PATH = os.path.join(BASE_DIR, "credit_default_risk.csv")


# -------------------------------
# Load or Train
# -------------------------------
@st.cache_resource
def load_or_train():

    # ✅ ถ้ามี model อยู่แล้ว → โหลดเลย
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_PATH) as f:
            features = json.load(f)

        return model, features

    # ❗ ถ้าไม่มี → train ใหม่
    st.warning("⚠️ ไม่พบโมเดล → กำลัง train ใหม่...")

    df = pd.read_csv(DATA_PATH)

    # preprocess
    df["student"] = df["student"].map({"Yes": 1, "No": 0})
    df["debt_income_ratio"] = df["balance"] / (df["income"] + 1e-6)
    df["is_debt_gt_income"] = (df["balance"] > df["income"]).astype(int)

    X = df.drop("default", axis=1)
    y = df["default"]

    features = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    # save (🔥 จะ save ไว้ root ตามไฟล์คุณ)
    joblib.dump(model, MODEL_PATH)

    with open(FEATURE_PATH, "w") as f:
        json.dump(features, f)

    return model, features


model, feature_names = load_or_train()


# -------------------------------
# Predict Function
# -------------------------------
def predict_risk(model, input_df):
    proba = model.predict_proba(input_df)[:, 1]
    risk = proba * 100

    # Business Rule
    mask = input_df["balance"] > input_df["income"]
    risk[mask] = 100.0

    label = ["เสี่ยง" if r >= 50 else "ไม่เสี่ยง" for r in risk]

    return risk, label


# -------------------------------
# UI
# -------------------------------
st.title("💳 Credit Default Risk Predictor")
st.markdown("ระบบประเมินความเสี่ยงด้วย Machine Learning + Business Rule")

st.divider()

# Input
st.subheader("📥 กรอกข้อมูล")

col1, col2 = st.columns(2)

with col1:
    student = st.selectbox("เป็นนักเรียนหรือไม่", ["No", "Yes"])
    student_val = 1 if student == "Yes" else 0

with col2:
    balance = st.number_input("ยอดหนี้ (Balance)", min_value=0.0, value=1000.0)

income = st.number_input("รายได้ (Income)", min_value=0.0, value=3000.0)


# -------------------------------
# Predict
# -------------------------------
if st.button("🔮 ทำนายความเสี่ยง"):

    input_df = pd.DataFrame([{
        "student": student_val,
        "balance": balance,
        "income": income
    }])

    # feature engineering
    input_df["debt_income_ratio"] = input_df["balance"] / (input_df["income"] + 1e-6)
    input_df["is_debt_gt_income"] = (input_df["balance"] > input_df["income"]).astype(int)

    input_df = input_df[feature_names]

    risk, label = predict_risk(model, input_df)

    risk_val = risk[0]
    label_val = label[0]

    st.divider()
    st.subheader("📊 ผลลัพธ์")

    st.progress(int(risk_val))

    if risk_val >= 70:
        st.error(f"🔴 ความเสี่ยงสูง: {risk_val:.2f}%")
    elif risk_val >= 40:
        st.warning(f"🟠 ความเสี่ยงปานกลาง: {risk_val:.2f}%")
    else:
        st.success(f"🟢 ความเสี่ยงต่ำ: {risk_val:.2f}%")

    st.markdown(f"### 📌 สรุป: **{label_val}**")

    if balance > income:
        st.info("⚠️ หนี้มากกว่ารายได้ → ระบบปรับเป็นเสี่ยงทันที")


# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("🔥 Fixed Path Version | Ready to Deploy")
