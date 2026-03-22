# ==========================================
# TRAIN MODEL (SAVE FOR STREAMLIT APP)
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

import joblib
import os
import json


# ===============================
# 1. Load Data
# ===============================
df = pd.read_csv("credit_default_risk.csv")

print("🔍 Shape:", df.shape)


# ===============================
# 2. Preprocessing
# ===============================
target_col = "default"
debt_col = "balance"
income_col = "income"

# แปลง student เป็นตัวเลข
df["student"] = df["student"].map({"Yes": 1, "No": 0})

# Feature Engineering
df["debt_income_ratio"] = df[debt_col] / (df[income_col] + 1e-6)
df["is_debt_gt_income"] = (df[debt_col] > df[income_col]).astype(int)

# แยก X, y
X = df.drop(target_col, axis=1)
y = df[target_col]

feature_names = X.columns.tolist()

print("✅ Features:", feature_names)


# ===============================
# 3. Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ===============================
# 4. Pipeline
# ===============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ))
])


# ===============================
# 5. Train
# ===============================
print("\n🚀 Training...")
pipeline.fit(X_train, y_train)
print("✅ Train เสร็จแล้ว!")


# ===============================
# 6. Evaluate
# ===============================
y_pred = pipeline.predict(X_test)

# Business Rule
rule_mask = X_test[debt_col] > X_test[income_col]
y_pred[rule_mask] = 1

acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Accuracy: {acc:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))


# ===============================
# 7. Save Model
# ===============================
os.makedirs("model_artifacts", exist_ok=True)

joblib.dump(pipeline, "model_artifacts/model.pkl")

with open("model_artifacts/features.json", "w") as f:
    json.dump(feature_names, f)

metadata = {
    "model": "RandomForest",
    "accuracy": float(acc),
    "features": feature_names,
    "target": target_col,
    "rule": "balance > income => default = 1"
}

with open("model_artifacts/meta.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n💾 บันทึก model_artifacts เรียบร้อย!")