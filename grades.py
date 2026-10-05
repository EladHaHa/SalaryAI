import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

KAGGLE_DATASET = "rkiattisak/salaly-prediction-for-beginer"
FEATURE_COLUMN = "Years of Experience"
TARGET_COLUMN = "Salary"

path = kagglehub.dataset_download(KAGGLE_DATASET)
csv_path = os.path.join(path, "Salary Data.csv")
data = pd.read_csv(csv_path)

data[FEATURE_COLUMN] = pd.to_numeric(
    data[FEATURE_COLUMN], errors="coerce"
)
data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
data = data.dropna(subset=[FEATURE_COLUMN, TARGET_COLUMN])

X = data[FEATURE_COLUMN].to_numpy()
y = data[TARGET_COLUMN].to_numpy()

baseline_prediction = np.mean(y)
baseline_loss = np.mean(np.abs(y - baseline_prediction))

X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, y)

y_hat = model.predict(X_reshaped)
model_loss = np.mean(np.abs(y - y_hat))

st.title("מערכת חיזוי שכר")

st.header("1. הסבר על המודל והנתונים")
st.write(
    "האפליקציה משתמשת במודל **רגרסיה ליניארית (Linear Regression)** כדי לחזות את השכר (Salary) בהתבסס על מספר שנות הניסיון (Years of Experience)."
)
st.write(
    "המודל מוצא קשר ישר בין שנות הניסיון לשכר, כמו קו ישר בגרף."
)

if st.checkbox("הצג את טבלת הנתונים"):
    st.dataframe(data)

st.header("2. קבלת תחזית שכר")
years_input = st.number_input("הכנס שנות ניסיון:", value=5.0, step=0.5)

if st.button("חשב תחזית"):
    pred = model.predict([[years_input]])[0]
    st.write(
        f"**השכר המשוער עבור {years_input} שנות ניסיון הוא: ${pred:,.2f}**"
    )

st.header("3. תהליך אימון המודל והערכת ה-Loss")

st.write("### שלבי תהליך האימון:")
st.write(
    "1. בחרתי קובץ נתונים בkaggle וטענתי אותו, ואז הורדתי את כל הערכים הלא מספריים כדי שלא יהיו בעיות."
)
st.write(
    "2. מחשבים את הbaseline, כלומר את  הנקודת ייחוס שהיא הממוצע שכר של כלל העובדים, ביחס אליה מחשבים את הloss של המודל, כלומר ככל שהמודל יותר רחוק מהממוצע סימן שהוא יותר מדוייק"
)
st.write(
    "3. השתמשתי בספרייה sklearn ובnumpy כדי לקחת את העמודה של שנות הנסיון והעמודה של השכר ואימנתי את המודל עליהן, והוא חישב את הw והb כלומר יצר את הקו הישר (חישב את היחס הישר) בין שכר לשנות נסיון"
)
st.write(
    "4. מחשבים את הloss של המודל ביחס לbaseline, מכיוון שהbaseline הוא פשוט ממוצע של השכר לא ניתן ליצור מודל יותר גרוע ממנו, וככל שהמודל שלנו יותר רחוק ממנו סימן שהוא יותר טוב"
)

st.write("### תוצאות ההערכה:")
st.write(f"* **Baseline Loss (שגיאת הבסיס):** `${baseline_loss:,.2f}`")
st.write(f"* **Model Loss (שגיאת המודל):** `${model_loss:,.2f}`")
