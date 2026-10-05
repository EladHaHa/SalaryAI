import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. טעינת הנתונים ואימון המודל (הקוד המקורי)
# ==========================================
KAGGLE_DATASET = "rkiattisak/salaly-prediction-for-beginer"
FEATURE_COLUMN = "Years of Experience"
TARGET_COLUMN = "Salary"

# הורדת הנתונים וקריאת הקובץ
path = kagglehub.dataset_download(KAGGLE_DATASET)
csv_path = os.path.join(path, "Salary Data.csv")
data = pd.read_csv(csv_path)

# ניקוי ערכים לא מספריים
data[FEATURE_COLUMN] = pd.to_numeric(
    data[FEATURE_COLUMN], errors="coerce"
)
data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
data = data.dropna(subset=[FEATURE_COLUMN, TARGET_COLUMN])

# הכנת מערכים
X = data[FEATURE_COLUMN].to_numpy()
y = data[TARGET_COLUMN].to_numpy()

# חישוב Baseline Loss
baseline_prediction = np.mean(y)
baseline_loss = np.mean(np.abs(y - baseline_prediction))

# אימון המודל
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, y)

# חישוב Model Loss
y_hat = model.predict(X_reshaped)
model_loss = np.mean(np.abs(y - y_hat))


# ==========================================
# 2. ממשק Streamlit פשוט
# ==========================================

# כותרת והסבר על המודל
st.title("מערכת חיזוי שכר")

st.header("הסבר על המודל")
st.write(
    "האפליקציה משתמשת במודל רגרסיה ליניארית כדי לחזות את השכר (Salary) בהתבסס על מספר שנות הניסיון (Years of Experience)."
)

# קלט ופלט לקבלת תחזית
st.header("קבלת תחזית")
years_input = st.number_input("הכנס שנות ניסיון:", value=5.0, step=0.5)

if st.button("חשב תחזית"):
    pred = model.predict([[years_input]])[0]
    st.write(f"השכר המשוער עבור {years_input} שנות ניסיון הוא: {pred:.2f}")

# הסבר על תהליך האימון וה-Loss
st.header("הסבר על תהליך האימון וה-Loss")
st.write(f"Baseline Loss (MAE): {baseline_loss:.2f}")
st.write(f"Model Loss (MAE): {model_loss:.2f}")

st.write(
    "ה-Baseline מייצג תחזית פשוטה לפי ממוצע השכר בלבד. ה-Model Loss נמוך יותר, מה שמעיד על כך ששימוש בשנות הניסיון משפר את דיוק החיזוי."
)
