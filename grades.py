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

# ---------------------------------------------------------
# הגדרת יישור לימין (RTL) עבור עברית
# ---------------------------------------------------------
st.markdown(
    """
    
    """,
    unsafe_allow_html=True,
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

st.write("המודל שנבנה הוא מודל **Linear Regression** (רגרסיה ליניארית).")
st.write(f"* **משתנה חיזוי (Target):** `{TARGET_COLUMN}` - השכר הצפוי.")
st.write(
    f"* **משתנה מנבא (Feature):** `{FEATURE_COLUMN}` - מספר שנות הניסיון של העובד."
)
st.write(
    "המודל לומד את הקשר הקווי בין שנות הניסיון לבין גובה השכר במאגר הנתונים, ומחשב משוואה מהצורה:"
)

# הצגת הנוסחה
st.latex(r"\text{Salary} = m \times \text{Years of Experience} + b")

# הצגת טבלה לדוגמה מתוך הדאטה-סט
with st.expander("לחץ לצפייה בדוגמה מתוך מאגר הנתונים (Data Sample)"):
    st.dataframe(data.head(10))

st.divider()

# ---------------------------------------------------------
# סעיף ב': חיזוי אינטראקטיבי לפי קלט המשתמש
# ---------------------------------------------------------
st.header("2. קבלת תחזית שכר אישית")
st.write("הכנס את מספר שנות הניסיון כדי לקבל תחזית שכר מוערכת:")

col1, col2 = st.columns([1, 2])

with col1:
    user_experience = st.number_input(
        "שנות ניסיון:",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="הכנס מספר שנות ניסיון (למשל: 2.5, 5, 10)",
    )

    if st.button("חשב תחזית שכר 🚀"):
        prediction = model.predict([[user_experience]])[0]
        st.success(f"**השכר המשוער עבור {user_experience} שנות ניסיון הוא:**")
        st.metric(label="תחזית שכר מוערכת", value=f"${prediction:,.2f}")

st.divider()

# ---------------------------------------------------------
# סעיף ג': הסבר על תהליך האימון והשוואת ה-Loss
# ---------------------------------------------------------
st.header("3. תהליך האימון והערכת איכות המודל")

st.write(
    "כדי לבדוק עד כמה המודל שלנו מוצלח, אנו משווים אותו ל-**Baseline Model** (מודל בסיס פשוט שמנבא תמיד את השכר הממוצע, ללא תלות בשנות הניסיון)."
)
st.write(
    "מדד השגיאה שנבדק הוא **MAE (Mean Absolute Error)** - המרחק הממוצע בדולרים בין התחזית לשכר האמיתי."
)

col_metric1, col_metric2, col_metric3 = st.columns(3)

with col_metric1:
    st.metric(
        label="Baseline Loss (שגיאת בסיס)",
        value=f"${baseline_loss:,.2f}",
        help="השגיאה הממוצעת אם היינו פשוט מנבאים את השכר הממוצע לכולם.",
    )

with col_metric2:
    st.metric(
        label="Model Loss (שגיאת המודל)",
        value=f"${model_loss:,.2f}",
        delta=f"-${baseline_loss - model_loss:,.2f}",
        delta_color="normal",
        help="השגיאה הממוצעת של מודל הרגרסיה הליניארית שלנו.",
    )

with col_metric3:
    improvement = ((baseline_loss - model_loss) / baseline_loss) * 100
    st.metric(
        label="אחוז שיפור לעומת ה-Baseline", value=f"{improvement:.1f}%"
    )

st.info(
    f"**מסקנה:** ה-Loss של המודל (\({model_loss:,.2f}) נמוך משמעותית מ-Loss ה-Baseline (\){baseline_loss:,.2f}). פירוש הדבר הוא ששימוש בשנות הניסיון כמשתנה מנבא מוריד את השגיאה הממוצעת בחיזוי בכ-{improvement:.1f}%, ומציג שיפור ניכר בחיזוי השכר."
)
