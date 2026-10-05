import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. טעינת הנתונים ואימון המודל
# ==========================================
KAGGLE_DATASET = "rkiattisak/salaly-prediction-for-beginer"
FEATURE_COLUMN = "Years of Experience"
TARGET_COLUMN = "Salary"

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
# 2. ממשק Streamlit
# ==========================================

# תיקון הצמדה פיזית לימין של כל העמוד והתוכן
st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stBlock"] {
        float: right;
    }
    input {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("מערכת חיזוי שכר")

# ------------------------------------------
# א. הסבר על המודל
# ------------------------------------------
st.header("1. הסבר על המודל והנתונים")
st.write(
    "האפליקציה משתמשת במודל **רגרסיה ליניארית (Linear Regression)** כדי לחזות את השכר (Salary) בהתבסס על מספר שנות הניסיון (Years of Experience)."
)
st.write(
    "המודל מוצא קשר ישר (קו ישר) בין שנות הניסיון לשכר, כך שעבור כל שנת ניסיון נוספת נחזית תוספת שכר קבועה."
)

if st.checkbox("הצג את טבלת הנתונים"):
    st.dataframe(data)

# ------------------------------------------
# ב. קבלת תחזית מהמשתמש
# ------------------------------------------
st.header("2. קבלת תחזית שכר")
years_input = st.number_input("הכנס שנות ניסיון:", value=5.0, step=0.5)

if st.button("חשב תחזית"):
    pred = model.predict([[years_input]])[0]
    st.write(
        f"**השכר המשוער עבור {years_input} שנות ניסיון הוא: ${pred:,.2f}**"
    )

# ------------------------------------------
# ג. הסבר מפורט על תהליך האימון וה-Loss
# ------------------------------------------
st.header("3. תהליך אימון המודל והערכת ה-Loss")

st.write("### שלבי תהליך האימון:")
st.write(
    "1. **ניקוי והכנת הנתונים:** טענו את קובץ ה-CSV, המרנו את עמודות הניסיון והשכר למספרים והשמטנו ערכים חסרים או לא תקינים."
)
st.write(
    "2. **בניית מודל הבסיס (Baseline):** נבחר מודל פשוט שאינו מתחשב בשנות הניסיון אלא פשוט מנבא את השכר הממוצע של כל העובדים."
)
st.write(
    "3. **אימון רגרסיה ליניארית:** המודל חיושב באמצעות שיטת ריבועים פחותים (Ordinary Least Squares) למציאת הקו הישר שעובר בצורה המדויקת ביותר דרך נקודות הנתונים."
)
st.write(
    "4. **חישוב השגיאה (MAE - Mean Absolute Error):** מדדנו את המרחק הממוצע בדולרים בין התחזית לשכר האמיתי."
)

st.write("### תוצאות ההערכה:")
st.write(f"* **Baseline Loss (שגיאת הבסיס):** `${baseline_loss:,.2f}`")
st.write(f"* **Model Loss (שגיאת המודל):** `${model_loss:,.2f}`")

st.write(
    f"**מסקנה:** שגיאת המודל נמוכה ב-**${(baseline_loss - model_loss):,.2f}** משגיאת הבסיס. זה מוכיח כי שימוש בשנות הניסיון כמשתנה מנבא משפר משמעותית את דיוק הניבוי לעומת ניחוי ממוצע פשוט."
)
