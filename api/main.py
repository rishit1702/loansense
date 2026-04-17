from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load model and features
model = joblib.load('/Users/rishitgambhir17/loansense/models/xgboost_v1.joblib')

with open('/Users/rishitgambhir17/loansense/models/feature_names.txt') as f:
    feature_names = [line.strip() for line in f.readlines()]

app = FastAPI(title="LoanSense API")


@app.get("/")
def root():
    return {"status": "LoanSense API is running"}


@app.post("/predict")
def predict(data: dict):
    # Create dataframe from input
    df = pd.DataFrame([data])

    # Add missing columns with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Keep only model features in correct order
    df = df[feature_names]

    # Predict
    risk_score = float(model.predict_proba(df)[:, 1][0])

    # Risk level
    if risk_score >= 0.6:
        level = "HIGH"
    elif risk_score >= 0.3:
        level = "MEDIUM"
    else:
        level = "LOW"

    # SHAP explanation
    explainer = joblib.load('/Users/rishitgambhir17/loansense/models/shap_explainer.joblib')
    shap_values = explainer.shap_values(df)
    shap_series = pd.Series(shap_values[0], index=feature_names)
    top_reasons = shap_series.abs().sort_values(ascending=False).head(5)

    reasons = []
    for feat in top_reasons.index:
        direction = "increases risk" if shap_series[feat] > 0 else "decreases risk"
        reasons.append({
            "feature": feat,
            "value": float(df[feat].iloc[0]),
            "impact": direction,
            "shap_value": round(float(shap_series[feat]), 4)
        })

    return {
        "risk_score": round(risk_score, 4),
        "risk_level": level,
        "reasons": reasons
    }