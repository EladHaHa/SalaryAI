import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# הגדרת כותרת ועיצוב כללי
st.set_page_config(
    page_title="חיזוי שכר לפי שנות ניסיון", page_icon="💼", layout="wide"
)

st.title("💼 מערכת לחיזוי שכר (Salary Prediction)")
st.write(
    "ברוכים הבאים! אפליקציה זו משתמשת במודל רגרסיה ליניארית כדי לחזות שכר של עובד בהתבסס על שנות הניסיון שלו."
)

# ---------------------------------------------------------
# טעינת הנתונים ואימון המודל (שימוש ב-cache למניעת הורדה חוזרת)
# ---------------------------------------------------------


@st.cache_resource
def train_model():
    KAGGLE_DATASET = "rkiattisak/salaly-prediction-for-beginer"
    FEATURE_COLUMN = "Years of Experience"
    TARGET_COLUMN = "Salary"

    # הורדת הדאטה-סט מ-Kaggle
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_path = os.path.join(path, "Salary Data.csv")
    data = pd.read_csv(csv_path)

    # ניקוי הנתונים
    data[FEATURE_COLUMN] = pd.to_numeric(
        data[FEATURE_COLUMN], errors="coerce"
    )
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    data = data.dropna(subset=[FEATURE_COLUMN, TARGET_COLUMN])

    # הכנת המשתנים
    X = data[FEATURE_COLUMN].to_numpy()
    y = data[TARGET_COLUMN].to_numpy()

    # חישוב Baseline Loss (MAE)
    baseline_prediction = np.mean(y)
    baseline_loss = np.mean(np.abs(y - baseline_prediction))

    # אימון מודל רגרסיה ליניארית
    X_reshaped = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X_reshaped, y)

    # חישוב Model Loss (MAE)
    y_hat = model.predict(X_reshaped)
    model_loss = np.mean(np.abs(y - y_hat))

    return model, baseline_loss, model_loss, data, FEATURE_COLUMN, TARGET_COLUMN


# הרצת פונקציית האימון
with st.spinner("טוען נתונים ומאמן את המודל..."):
    model, baseline_loss, model_loss, data, FEATURE_COLUMN, TARGET_COLUMN = (
        train_model()
    )

st.divider()

# ---------------------------------------------------------
# סעיף א': הסבר על המודל והנתונים
# ---------------------------------------------------------
st.header("1. אודות המודל והנתונים")
st.write(
    "המודל שנבנה הוא מודל **Linear Regression** (רגרסיה ליניארית).\n"
    f"* **משתנה חיזוי (Target):
