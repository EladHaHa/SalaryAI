
import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


# =========================
# עיצוב האפליקציה - עברית
# =========================

st.set_page_config(
    page_title="מערכת חיזוי שכר",
    page_icon="💼",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* יישור כל האפליקציה לימין */
    .stApp {
        direction: rtl;
        text-align: right;
    }

    /* יישור כותרות וטקסט */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown li,
    .stText,
    label {
        direction: rtl !important;
        text-align: right !important;
    }

    /* יישור תיבות הקלט */
    div[data-testid="stNumberInput"] label {
        direction: rtl !important;
        text-align: right !important;
    }

    /* יישור תיבות הקלט */
    div[data-testid="stNumberInput"] input {
        text-align: right !important;
    }

    /* יישור תיבות סימון */
    div[data-testid="stCheckbox"] {
        direction: rtl;
        text-align: right;
    }

    /* יישור הטבלה */
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
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # הסרת שורות עם ערכים חסרים
    df = df.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

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
# אימון מודל עם שני מאפיינים
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

X_one_feature = data[
    ["Years of Experience"]
].to_numpy()

model_one_feature.fit(X_one_feature, y)

y_hat_one = model_one_feature.predict(
    X_one_feature
)

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
    "האפליקציה משתמשת במודל רגרסיה ליניארית "
    "(Linear Regression) כדי לחזות את השכר של עובד "
    "בהתבסס על שני מאפיינים: מספר שנות הניסיון שלו "
    "(Years of Experience) והגיל שלו (Age)."
)

st.write(
    "רגרסיה ליניארית היא שיטה שמנסה למצוא קשר בין "
    "מאפייני הקלט לבין השכר. במקרה שלנו, המודל משתמש "
    "בשנות הניסיון ובגיל של העובד כדי להעריך מה יהיה שכרו."
)

st.write(
    "במהלך האימון המודל לומד את מידת ההשפעה של כל אחד "
    "משני המאפיינים על השכר, ומנסה להתאים את התחזיות "
    "שלו לערכי השכר האמיתיים שבמערך הנתונים."
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

    features = np.array([
        [years_input, age_input]
    ])

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
    "המרתי את הערכים בעמודות לערכים מספריים והסרתי "
    "שורות שבהן היה חסר ערך באחד משלושת המשתנים."
)


st.write(
    "2. **הכנת הנתונים:** "
    "הגדרתי את שנות הניסיון ואת הגיל כמאפייני הקלט "
    "(X), ואת השכר כמשתנה המטרה (y). "
    "כלומר, המודל מקבל שני נתונים על העובד ומשתמש בהם "
    "כדי לנסות לחזות את השכר שלו."
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
    "השתמשתי בספריית sklearn כדי לאמן מודל של רגרסיה "
    "ליניארית. המודל מקבל שני מאפייני קלט: שנות ניסיון "
    "וגיל. במהלך האימון הוא לומד את המשקל של כל אחד "
    "מהמאפיינים ואת נקודת החיתוך. המטרה היא למצוא את "
    "הערכים שיאפשרו למודל לבצע תחזיות שקרובות ככל האפשר "
    "לשכר האמיתי של העובדים."
)


st.write(
    "5. **חישוב ה-Loss:** "
    "לאחר אימון המודל חישבתי את התחזיות שלו והשוויתי "
    "אותן לשכר האמיתי. לצורך ההערכה השתמשתי במדד MSE "
    "(Mean Squared Error), שמחשב את ממוצע ריבועי "
    "ההפרשים בין התחזיות לבין הערכים האמיתיים. "
    "ככל שערך ה-MSE נמוך יותר, כך השגיאה של המודל קטנה יותר."
)


st.write(
    "6. **השוואה בין המודלים:** "
    "השוויתי בין שלוש אפשרויות: ה-Baseline שחוזה את "
    "ממוצע השכר, מודל רגרסיה ליניארית שמשתמש רק בשנות "
    "הניסיון, ומודל רגרסיה ליניארית שמשתמש בשני מאפיינים: "
    "שנות ניסיון וגיל. באמצעות ההשוואה ניתן לבדוק האם "
    "הוספת הגיל משפרת את יכולת החיזוי של המודל."
)


# =========================
# הצגת המשקולות
# =========================

st.write("### הפרמטרים שחושבו במהלך האימון:")

st.write(
    f"**משקל שנות הניסיון:** {model.coef_[0]:,.2f}"
)

st.write(
    f"**משקל הגיל:** {model.coef_[1]:,.2f}"
)

st.write(
    f"**נקודת החיתוך:** {model.intercept_:,.2f}"
)


# =========================
# תוצאות ההערכה
# =========================

st.write("### תוצאות ההערכה:")

st.write(
    f"**Baseline Loss:** ${baseline_loss:,.2f}"
)

st.write(
    f"**One Feature Loss:** ${model_loss_one:,.2f}"
)

st.write(
    f"**Two Features Loss:** ${model_loss:,.2f}"
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
