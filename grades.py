import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


# =========================
# הגדרות עיצוב - עברית ויישור לימין
# =========================

st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
        text-align: right;
    }

    h1, h2, h3, h4, h5, h6, p, div, label {
        text-align: right;
        direction: rtl;
    }

    input {
        direction: rtl;
        text-align: right;
    }

    [data-testid="stDataFrame"] {
        direction: rtl;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# טעינת הנתונים
# =========================

KAGGLE_DATASET = "rkiattisak/salaly-prediction-for-beginer"

FEATURE_COLUMN = "Years of Experience"
TARGET_COLUMN = "Salary"

path = kagglehub.dataset_download(KAGGLE_DATASET)

csv_path = os.path.join(path, "Salary Data.csv")

data = pd.read_csv(csv_path)


# המרת העמודות לערכים מספריים
data[FEATURE_COLUMN] = pd.to_numeric(
    data[FEATURE_COLUMN],
    errors="coerce"
)

data[TARGET_COLUMN] = pd.to_numeric(
    data[TARGET_COLUMN],
    errors="coerce"
)


# הסרת שורות שבהן חסרים ערכים
data = data.dropna(
    subset=[FEATURE_COLUMN, TARGET_COLUMN]
)


# =========================
# הכנת הנתונים למודל
# =========================

X = data[FEATURE_COLUMN].to_numpy()
y = data[TARGET_COLUMN].to_numpy()


# =========================
# חישוב Baseline
# =========================

# ה-Baseline מנבא תמיד את ממוצע השכר
baseline_prediction = np.mean(y)

# חישוב MAE של ה-Baseline
baseline_loss = np.mean(
    np.abs(y - baseline_prediction)
)


# =========================
# אימון מודל Linear Regression
# =========================

X_reshaped = X.reshape(-1, 1)

model = LinearRegression()

model.fit(X_reshaped, y)


# תחזיות המודל על נתוני האימון
y_hat = model.predict(X_reshaped)


# חישוב MAE של המודל
model_loss = np.mean(
    np.abs(y - y_hat)
)


# =========================
# ממשק המשתמש
# =========================

st.title("מערכת חיזוי שכר")


# =========================
# 1. הסבר על המודל והנתונים
# =========================

st.header("1. הסבר על המודל והנתונים")

st.write(
    "האפליקציה משתמשת במודל **רגרסיה ליניארית (Linear Regression)** "
    "כדי לחזות את השכר (Salary) של עובד בתחום מדעי הנתונים "
    "בהתבסס על מספר שנות הניסיון שלו (Years of Experience)."
)

st.write(
    "רגרסיה ליניארית מנסה למצוא קשר ישר בין שנות הניסיון לבין השכר. "
    "המודל מתאים קו ישר לנתונים, כאשר ציר אחד מייצג את שנות הניסיון "
    "והציר השני מייצג את השכר."
)


# הצגת טבלת הנתונים
if st.checkbox("הצג את טבלת הנתונים"):
    st.dataframe(data)


# =========================
# 2. קבלת תחזית שכר
# =========================

st.header("2. קבלת תחזית שכר")

years_input = st.number_input(
    "הכנס שנות ניסיון:",
    value=5.0,
    step=0.5
)


if st.button("חשב תחזית"):

    pred = model.predict([[years_input]])[0]

    st.write(
        f"**השכר המשוער עבור {years_input} שנות ניסיון "
        f"הוא: ${pred:,.2f}**"
    )


# =========================
# 3. תהליך אימון המודל והערכת ה-Loss
# =========================

st.header("3. תהליך אימון המודל והערכת ה-Loss")

st.write("### שלבי תהליך האימון:")


st.write(
    "1. **טעינת הנתונים:** "
    "בחרתי מערך נתונים מ-Kaggle וטענתי אותו לאפליקציה. "
    "לאחר מכן המרתי את העמודות של שנות הניסיון והשכר לערכים מספריים "
    "והסרתי שורות שבהן היו ערכים חסרים או לא תקינים."
)


st.write(
    "2. **יצירת Baseline:** "
    "חישבתי נקודת ייחוס (Baseline) באמצעות ממוצע השכר של כלל העובדים. "
    "ה-Baseline מניח שאין לנו מידע נוסף על העובד, ולכן בכל מקרה "
    "הוא חוזה שהשכר יהיה שווה לממוצע השכר של כל העובדים."
)


st.write(
    "3. **אימון המודל:** "
    "השתמשתי בספריות Scikit-learn ו-NumPy כדי לאמן מודל "
    "של רגרסיה ליניארית. המודל מקבל את שנות הניסיון כקלט "
    "ואת השכר כיעד. במהלך האימון המודל מוצא את הפרמטרים "
    "**w** ו-**b**, כך שהוא יוצר משוואת קו ישר מהצורה "
    "**y = wx + b**."
)


st.write(
    "4. **חישוב ה-Loss והשוואה ל-Baseline:** "
    "לאחר שהמודל אומן, חישבתי את התחזיות שלו והשוויתי אותן "
    "לשכר האמיתי. לצורך ההערכה השתמשתי ב-MAE "
    "(Mean Absolute Error), שמחשב את ממוצע המרחק המוחלט "
    "בין התחזית לבין הערך האמיתי."
)


st.write(
    "כדי לבדוק האם המודל הליניארי טוב יותר מה-Baseline, "
    "משווים בין ה-Model Loss לבין ה-Baseline Loss. "
    "אם ה-Model Loss נמוך יותר, המשמעות היא שהמודל הליניארי "
    "מבצע תחזיות מדויקות יותר מה-Baseline על הנתונים."
)


# =========================
# תוצאות ההערכה
# =========================

st.write("### תוצאות ההערכה:")

st.write(
    f"* **Baseline Loss (שגיאת הבסיס):** "
    f"${baseline_loss:,.2f}"
)

st.write(
    f"* **Model Loss (שגיאת המודל):** "
    f"${model_loss:,.2f}"
)

