
import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


# =========================
# עיצוב האפליקציה - עברית
# =========================

st.set_page_config(page_title="מערכת חיזוי שכר", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
        text-align: right;
    }

    h1, h2, h3, h4, h5, h6, p, label {
        text-align: right;
        direction: rtl;
    }

    input {
        direction: ltr;
        text-align: right;
    }

    [data-testid="stDataFrame"] {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# טעינת הנתונים
# =========================

KAGGLE_DATASET = "rkiattisak/salaly-prediction-for-beginer"

FEATURE_COLUMNS = ["Years of Experience", "Age"]
TARGET_COLUMN = "Salary"


@st.cache_data
def load_data():
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_path = os.path.join(path, "Salary Data.csv")
    df = pd.read_csv(csv_path)

    # המרת הערכים בעמודות לערכים מספריים
    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # הסרת שורות עם ערכים חסרים
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return df


data = load_data()


# =========================
# הכנת הנתונים למודל
# =========================

X = data[FEATURE_COLUMNS].to_numpy()
y = data[TARGET_COLUMN].to_numpy()


# =========================
# Baseline
# =========================

# ה-Baseline חוזה תמיד את ממוצע השכר
baseline_prediction = np.mean(y)

# חישוב MSE של ה-Baseline
baseline_loss = np.mean(
    (y - baseline_prediction) ** 2
)


# =========================
# אימון מודל Linear Regression
# עם שני מאפיינים
# =========================

model = LinearRegression()

model.fit(X, y)


# =========================
# תחזיות המודל
# =========================

y_hat = model.predict(X)


# =========================
# חישוב Model Loss
# =========================

model_loss = np.mean(
    (y - y_hat) ** 2
)


# =========================
# אימון מודל עם מאפיין אחד
# לצורך השוואה
# =========================

model_one_feature = LinearRegression()

X_one_feature = data[["Years of Experience"]].to_numpy()

model_one_feature.fit(X_one_feature, y)

y_hat_one = model_one_feature.predict(X_one_feature)

model_loss_one = np.mean(
    (y - y_hat_one) ** 2
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
    "בהתבסס על שני מאפיינים: מספר שנות הניסיון שלו (Years of Experience) "
    "והגיל שלו (Age)."
)

st.write(
    "רגרסיה ליניארית מנסה למצוא קשר בין מאפייני הקלט לבין השכר. "
    "המודל מתאים משוואה ליניארית לנתונים, כאשר שנות הניסיון והגיל "
    "הם משתני הקלט והשכר הוא משתנה המטרה."
)

st.latex(
    r"Salary = w_1 \cdot Experience + w_2 \cdot Age + b"
)

st.write(
    "במשוואה, w₁ מייצג את המשקל של שנות הניסיון, "
    "w₂ מייצג את המשקל של הגיל, ו-b מייצג את נקודת החיתוך "
    "של המודל עם ציר ה-Y."
)


# =========================
# הצגת טבלת הנתונים
# =========================

if st.checkbox("הצג את טבלת הנתונים"):

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )


# =========================
# 2. קבלת תחזית שכר
# =========================

st.header("2. קבלת תחזית שכר")

st.write(
    "הזינו את גיל העובד ואת מספר שנות הניסיון שלו "
    "כדי לקבל תחזית של השכר."
)

col1, col2 = st.columns(2)

with col1:
    years_input = st.number_input(
        "הכנס שנות ניסיון:",
        value=5.0,
        step=0.5,
        min_value=0.0
    )

with col2:
    age_input = st.number_input(
        "הכנס גיל:",
        value=30,
        step=1,
        min_value=0
    )


if st.button("חשב תחזית"):

    features = np.array([[years_input, age_input]])

    pred = model.predict(features)[0]

    st.write(
        f"**השכר המשוער עבור עובד בן {age_input} "
        f"עם {years_input} שנות ניסיון הוא: ${pred:,.2f}**"
    )


# =========================
# 3. תהליך אימון המודל והערכת ה-Loss
# =========================

st.header("3. תהליך אימון המודל והערכת ה-Loss")

st.write("### שלבי תהליך האימון:")


st.write(
    "1. **טעינת הנתונים וניקוים:** "
    "בחרתי מערך נתונים מ-Kaggle וטענתי אותו לאפליקציה. "
    "השתמשתי בשלוש עמודות: שנות ניסיון, גיל ושכר. "
    "המרתי את הערכים בעמודות לערכים מספריים והסרתי שורות "
    "שבהן היה חסר ערך באחד משלושת המשתנים."
)


st.write(
    "2. **הכנת הנתונים:** "
    "הגדרתי את שנות הניסיון ואת הגיל כמאפייני הקלט (X), "
    "ואת השכר כמשתנה המטרה (y). כך כל דוגמה מכילה שני נתוני קלט "
    "שבהתבסס עליהם המודל מנסה לחזות את השכר."
)


st.write(
    "3. **יצירת Baseline:** "
    "חישבתי נקודת ייחוס (Baseline) באמצעות ממוצע השכר "
    "של כלל העובדים. ה-Baseline מניח שאין לנו מידע נוסף "
    "על העובד, ולכן עבור כל עובד הוא חוזה את אותו ערך: "
    "ממוצע השכר של כל העובדים."
)


st.write(
    "4. **אימון המודל:** "
    "השתמשתי בספריית sklearn כדי לאמן מודל של רגרסיה ליניארית. "
    "המודל מקבל שני מאפייני קלט: שנות ניסיון וגיל, ומחשב "
    "את המשקולות w₁ ו-w₂ ואת נקודת החיתוך b. "
    "המטרה היא למצוא את המקדמים של המשוואה כך שסכום ריבועי "
    "השגיאות בין התחזיות לבין השכר האמיתי יהיה קטן ככל האפשר. "
    "כל משקל מייצג את התרומה של אחד ממאפייני הקלט לחיזוי השכר."
)


st.write(
    "5. **חישוב ה-Loss:** "
    "לאחר אימון המודל חישבתי את התחזיות שלו והשוויתי אותן "
    "לשכר האמיתי. לצורך ההערכה השתמשתי במדד MSE "
    "(Mean Squared Error), שמחשב את ממוצע ריבועי ההפרשים "
    "בין התחזית לבין הערך האמיתי. ככל שה-MSE נמוך יותר, "
    "כך השגיאה של המודל קטנה יותר לפי מדד זה."
)

st.latex(
    r"MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2"
)


st.write(
    "6. **השוואה בין המודלים:** "
    "השוויתי בין שלוש אפשרויות: ה-Baseline שמנבא את ממוצע השכר, "
    "מודל רגרסיה ליניארית שמשתמש רק בשנות הניסיון, "
    "ומודל רגרסיה ליניארית שמשתמש בשני מאפיינים – "
    "שנות ניסיון וגיל. כך ניתן לבדוק האם הוספת הגיל משפרת "
    "את יכולת החיזוי של המודל."
)


# =========================
# הצגת המשקולות
# =========================

st.write("### הפרמטרים שחושבו במהלך האימון:")

st.write(
    f"**משקל שנות הניסיון (w₁):** {model.coef_[0]:,.2f}"
)

st.write(
    f"**משקל הגיל (w₂):** {model.coef_[1]:,.2f}"
)

st.write(
    f"**נקודת החיתוך (b):** {model.intercept_:,.2f}"
)


# =========================
# תוצאות ההערכה
# =========================

st.write("### תוצאות ההערכה:")

st.write(
    f"**Baseline Loss:** "
    f"${baseline_loss:,.2f}"
)

st.write(
    f"**One Feature Loss:** "
    f"${model_loss_one:,.2f}"
)

st.write(
    f"**Two Features Loss:** "
    f"${model_loss:,.2f}"
)


st.write(
    "אם ה-Loss של המודל עם שני המאפיינים נמוך יותר "
    "מה-Loss של המודל עם מאפיין אחד ושל ה-Baseline, "
    "המשמעות היא שהמודל עם שני המאפיינים מתאים טוב יותר "
    "לנתונים שעליהם הוא אומן."
)

st.caption(
    "הערה: האימון והערכת ה-Loss מתבצעים על אותו מערך נתונים. "
    "לכן התוצאות מתארות את ההתאמה לנתוני האימון ואינן בהכרח "
    "מעידות על רמת הדיוק של המודל כאשר הוא מקבל נתונים חדשים."
)
