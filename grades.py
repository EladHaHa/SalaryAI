
import os
import kagglehub
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# =====================================================
# הגדרות ראשוניות
# =====================================================

st.set_page_config(
    page_title="מנהל חברת ההייטק",
    page_icon="💼",
    layout="wide"
)

# עיצוב האתר ויישור לעברית
st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
        text-align: right;
    }

    h1, h2, h3, h4, h5, h6 {
        direction: rtl !important;
        text-align: right !important;
        width: 100%;
    }

    p, label, li {
        direction: rtl;
        text-align: right;
    }

    div[data-testid="stMetric"] {
        direction: rtl;
        text-align: right;
    }

    div[data-testid="stNumberInput"] input {
        direction: ltr;
        text-align: right;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# טעינת הנתונים
# =====================================================

FEATURE_COLUMNS = ["Years of Experience", "Age"]
TARGET_COLUMN = "Salary"

@st.cache_data
def load_data():

    path = kagglehub.dataset_download(
        "rkiattisak/salaly-prediction-for-beginer"
    )

    csv_path = os.path.join(path, "Salary Data.csv")

    data = pd.read_csv(csv_path)

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    return data


# =====================================================
# אימון המודל
# =====================================================

@st.cache_resource
def train_model():

    data = load_data()

    train_data, test_data = train_test_split(
        data,
        test_size=0.25,
        random_state=42
    )

    X_train = train_data[FEATURE_COLUMNS].to_numpy()
    y_train = train_data[TARGET_COLUMN].to_numpy()

    X_test = test_data[FEATURE_COLUMNS].to_numpy()
    y_test = test_data[TARGET_COLUMN].to_numpy()

    # מודל המשתמש בשני משתנים
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_mse = mean_squared_error(
        y_test,
        predictions
    )

    # מודל המשתמש רק בניסיון
    one_feature_model = LinearRegression()

    one_feature_model.fit(
        train_data[["Years of Experience"]],
        y_train
    )

    one_feature_predictions = one_feature_model.predict(
        test_data[["Years of Experience"]]
    )

    one_feature_mse = mean_squared_error(
        y_test,
        one_feature_predictions
    )

    # מודל בסיסי שמנבא תמיד את ממוצע השכר
    baseline_prediction = np.mean(y_train)

    baseline_mse = mean_squared_error(
        y_test,
        np.full(len(y_test), baseline_prediction)
    )

    return (
        model,
        train_data,
        test_data,
        model_mse,
        one_feature_mse,
        baseline_mse
    )


try:
    (
        model,
        train_data,
        test_data,
        model_mse,
        one_feature_mse,
        baseline_mse
    ) = train_model()

except Exception as e:
    st.error("אירעה שגיאה בטעינת הנתונים או באימון המודל.")
    st.exception(e)
    st.stop()


# =====================================================
# אתחול מצב המשחק
# =====================================================

def initialize_game():

    defaults = {
        "game_started": False,
        "round_number": 0,
        "user_score": 0,
        "model_score": 0,
        "rounds": [],
        "history": [],
        "answered": False,
        "last_result": None
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_game()


# =====================================================
# יצירת משחק חדש
# =====================================================

def start_game():

    if len(test_data) < 15:
        st.error("אין מספיק נתונים כדי להתחיל משחק.")
        return

    # בחירת 15 עובדים אמיתיים מתוך נתוני הבדיקה
    selected_data = test_data.sample(
        n=15
    ).reset_index(drop=True)

    rounds = []

    # תקציבי המשחק מבוססים על התפלגות השכר בנתוני האימון
    budgets = [
        float(train_data[TARGET_COLUMN].quantile(0.40)),
        float(train_data[TARGET_COLUMN].quantile(0.50)),
        float(train_data[TARGET_COLUMN].quantile(0.60))
    ]

    rng = np.random.default_rng()

    for round_index in range(5):

        candidates = []

        for candidate_index in range(3):

            row = selected_data.iloc[
                round_index * 3 + candidate_index
            ]

            features = np.array([[
                row["Years of Experience"],
                row["Age"]
            ]])

            prediction = float(
                model.predict(features)[0]
            )

            candidates.append({

                "name": f"מועמד {chr(65 + candidate_index)}",

                "age": int(row["Age"]),

                "experience": float(
                    row["Years of Experience"]
                ),

                "prediction": prediction,

                "actual_salary": float(
                    row[TARGET_COLUMN]
                )
            })

        budget = float(rng.choice(budgets))

        rounds.append({
            "candidates": candidates,
            "budget": budget
        })

    # שמירת המשחק ב-session_state
    st.session_state.game_started = True
    st.session_state.round_number = 0
    st.session_state.user_score = 0
    st.session_state.model_score = 0
    st.session_state.rounds = rounds
    st.session_state.history = []
    st.session_state.answered = False
    st.session_state.last_result = None


# =====================================================
# בחירת המועמד של המודל
# =====================================================

def model_choose_candidate(candidates, budget):

    # המודל בודק אילו מועמדים עומדים בתקציב לפי התחזית
    affordable = [
        i for i, candidate in enumerate(candidates)
        if candidate["prediction"] <= budget
    ]

    if affordable:

        # בחירה במועמד בעל הניסיון הרב ביותר
        # מבין אלה שהמודל מעריך שיעמדו בתקציב
        return max(
            affordable,
            key=lambda i: (
                candidates[i]["experience"],
                -candidates[i]["prediction"]
            )
        )

    # אם אף מועמד אינו עומד בתקציב לפי התחזית,
    # המודל בוחר את המועמד עם תחזית השכר הנמוכה ביותר
    return min(
        range(len(candidates)),
        key=lambda i: candidates[i]["prediction"]
    )


# =====================================================
# חישוב ניקוד
# =====================================================

def calculate_score(candidate, candidates, budget):

    # אם השכר האמיתי חורג מהתקציב - אין נקודות
    if candidate["actual_salary"] > budget:
        return 0

    score = 10

    # מציאת הניסיון הגבוה ביותר מבין המועמדים
    # שהשכר האמיתי שלהם נמצא במסגרת התקציב
    affordable = [
        c for c in candidates
        if c["actual_salary"] <= budget
    ]

    max_experience = max(
        c["experience"] for c in affordable
    )

    if candidate["experience"] == max_experience:
        score += 5

    return score


# =====================================================
# מסך פתיחה
# =====================================================

def show_start_screen():

    st.title("💼 מנהל חברת ההייטק")

    st.subheader("ברוכים הבאים לחברת ההייטק שלכם!")

    st.write(
        """
        אתם מנהלי חברת הייטק, ועליכם לגייס עובדים חדשים
        תוך שמירה על תקציב מוגבל.

        בכל סיבוב יוצגו בפניכם שלושה מועמדים.
        תצטרכו לבחור את העובד שלדעתכם יהיה המשתלם ביותר.

        אבל יש לכם מתחרה: מודל הבינה המלאכותית!

        גם הוא יבחר מועמד על סמך תחזיות השכר שלו.
        מי יצליח לנהל את החברה בצורה טובה יותר?
        """
    )

    st.divider()

    st.header("📖 איך משחקים?")

    st.markdown(
        """
        **1.** המשחק כולל חמישה סיבובים.

        **2.** בכל סיבוב יוצגו שלושה מועמדים עם גיל,
        שנות ניסיון ותחזית שכר של המודל.

        **3.** יוצג לכם תקציב הגיוס של החברה.

        **4.** עליכם לבחור מועמד אחד להעסקה.
        במקביל, המודל יבחר מועמד משלו.

        **5.** לאחר הבחירה ייחשף השכר האמיתי של כל מועמד.

        **6. שיטת הניקוד:**

        - 10 נקודות על בחירת מועמד שהשכר האמיתי שלו
          עומד בתקציב.
        - 5 נקודות נוספות אם בחרתם גם במועמד בעל
          הניסיון הרב ביותר מבין המועמדים העומדים בתקציב.
        - אם השכר האמיתי חורג מהתקציב, לא תקבלו נקודות.

        **7.** בסוף חמשת הסיבובים נשווה את הניקוד שלכם
        לניקוד של המודל.
        """
    )

    st.info(
        "המטרה: לקבל החלטות חכמות, לשמור על התקציב "
        "ולנצח את מודל הבינה המלאכותית!"
    )

    st.divider()

    if st.button(
        "🚀 התחל משחק",
        use_container_width=True
    ):
        start_game()
        st.rerun()


# =====================================================
# מסך המשחק
# =====================================================

def show_round():

    round_number = st.session_state.round_number

    current_round = st.session_state.rounds[
        round_number
    ]

    candidates = current_round["candidates"]
    budget = current_round["budget"]

    st.title("💼 מנהל חברת ההייטק")

    # הצגת הניקוד וההתקדמות
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "הניקוד שלך",
        st.session_state.user_score
    )

    col2.metric(
        "ניקוד המודל",
        st.session_state.model_score
    )

    col3.metric(
        "סיבוב",
        f"{round_number + 1} מתוך 5"
    )

    st.progress(
        (round_number + 1) / 5
    )

    st.divider()

    st.header(
        f"סיבוב {round_number + 1}"
    )

    st.subheader(
        f"💰 תקציב הגיוס: ${budget:,.0f}"
    )

    st.write(
        "בחרו את המועמד שלדעתכם יהיה המשתלם ביותר לחברה."
    )

    # הצגת שלושת המועמדים
    columns = st.columns(3)

    for i, candidate in enumerate(candidates):

        with columns[i]:

            st.markdown(
                f"### {candidate['name']}"
            )

            st.write(
                f"**גיל:** {candidate['age']}"
            )

            st.write(
                f"**שנות ניסיון:** {candidate['experience']:.1f}"
            )

            st.metric(
                "תחזית השכר של המודל",
                f"${candidate['prediction']:,.0f}"
            )

    st.divider()

    # בחירת המשתמש
    if not st.session_state.answered:

        with st.form(
            key=f"round_form_{round_number}"
        ):

            choice = st.radio(
                "איזה מועמד תרצו להעסיק?",
                options=[0, 1, 2],
                format_func=lambda i: candidates[i]["name"],
                horizontal=True
            )

            submitted = st.form_submit_button(
                "אשר בחירה",
                use_container_width=True
            )

        if submitted:

            model_choice = model_choose_candidate(
                candidates,
                budget
            )

            user_points = calculate_score(
                candidates[choice],
                candidates,
                budget
            )

            model_points = calculate_score(
                candidates[model_choice],
                candidates,
                budget
            )

            result = {
                "user_choice": choice,
                "model_choice": model_choice,
                "user_points": user_points,
                "model_points": model_points
            }

            st.session_state.user_score += user_points
            st.session_state.model_score += model_points

            st.session_state.last_result = result

            st.session_state.history.append(
                result
            )

            st.session_state.answered = True

            st.rerun()

    # =================================================
    # תוצאות הסיבוב
    # =================================================

    else:

        result = st.session_state.last_result

        user_choice = result["user_choice"]
        model_choice = result["model_choice"]

        st.divider()

        st.header("📊 תוצאות הסיבוב")

        st.write(
            f"**הבחירה שלך:** "
            f"{candidates[user_choice]['name']}"
        )

        st.write(
            f"**בחירת המודל:** "
            f"{candidates[model_choice]['name']}"
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "הנקודות שקיבלת",
            result["user_points"]
        )

        col2.metric(
            "הנקודות שקיבל המודל",
            result["model_points"]
        )

        if result["user_points"] > result["model_points"]:

            st.success(
                "🎉 ניצחת בסיבוב הזה!"
            )

        elif result["user_points"] < result["model_points"]:

            st.warning(
                "המודל ניצח בסיבוב הזה."
            )

        else:

            st.info(
                "תיקו! שניכם קיבלתם את אותו הניקוד."
            )

        # הצגת השכר האמיתי
        st.subheader(
            "השכר האמיתי של המועמדים"
        )

        results_table = []

        for i, candidate in enumerate(candidates):

            results_table.append({

                "מועמד": candidate["name"],

                "גיל": candidate["age"],

                "שנות ניסיון": candidate["experience"],

                "תחזית המודל": (
                    f"${candidate['prediction']:,.0f}"
                ),

                "שכר אמיתי": (
                    f"${candidate['actual_salary']:,.0f}"
                ),

                "הבחירה שלך": (
                    "כן" if i == user_choice else "לא"
                ),

                "בחירת המודל": (
                    "כן" if i == model_choice else "לא"
                )
            })

        st.dataframe(
            pd.DataFrame(results_table),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # מעבר לסיבוב הבא
        if round_number < 4:

            if st.button(
                "לסיבוב הבא ➜",
                use_container_width=True
            ):

                st.session_state.round_number += 1

                st.session_state.answered = False

                st.session_state.last_result = None

                st.rerun()

        else:

            if st.button(
                "לסיכום המשחק 🏆",
                use_container_width=True
            ):

                st.session_state.round_number = 5

                st.rerun()


# =====================================================
# מסך סיום
# =====================================================

def show_end_screen():

    st.title("🏆 סיכום המשחק")

    st.subheader(
        "המשחק הסתיים!"
    )

    user_score = st.session_state.user_score
    model_score = st.session_state.model_score

    col1, col2 = st.columns(2)

    col1.metric(
        "הניקוד שלך",
        user_score
    )

    col2.metric(
        "ניקוד המודל",
        model_score
    )

    st.divider()

    if user_score > model_score:

        st.success(
            "🎉 כל הכבוד! ניצחת את מודל הבינה המלאכותית!"
        )

    elif user_score < model_score:

        st.warning(
            "הפעם מודל הבינה המלאכותית הצליח לנהל "
            "את החברה טוב יותר."
        )

    else:

        st.info(
            "המשחק הסתיים בתיקו!"
        )

    st.header(
        "תוצאות לפי סיבוב"
    )

    history_table = []

    for i, result in enumerate(
        st.session_state.history
    ):

        history_table.append({

            "סיבוב": i + 1,

            "הנקודות שלך": result["user_points"],

            "נקודות המודל": result["model_points"]
        })

    st.dataframe(
        pd.DataFrame(history_table),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "תודה ששיחקתם! האם תצליחו לנצח את המודל גם בפעם הבאה?"
    )

    if st.button(
        "🔄 שחק שוב",
        use_container_width=True
    ):

        start_game()
        st.rerun()


# =====================================================
# מידע על המודל
# =====================================================

def show_model_information():

    with st.expander(
        "🧠 מידע על מודל הבינה המלאכותית"
    ):

        st.write(
            """
            המשחק מבוסס על מודל רגרסיה לינארית.

            המודל לומד את הקשר בין גיל העובד,
            שנות הניסיון שלו והשכר שלו.

            לאחר האימון, המודל מסוגל לחזות שכר
            עבור עובדים חדשים על סמך גילם וניסיונם.
            """
        )

        st.subheader(
            "השוואת ביצועי המודלים"
        )

        st.write(
            "המדד המוצג הוא MSE – ממוצע ריבועי שגיאות החיזוי. "
            "ערך נמוך יותר מעיד על תחזיות מדויקות יותר."
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "מודל בסיסי",
            f"{baseline_mse:,.0f}"
        )

        col2.metric(
            "מודל עם משתנה אחד",
            f"{one_feature_mse:,.0f}"
        )

        col3.metric(
            "מודל עם שני משתנים",
            f"{model_mse:,.0f}"
        )

        st.write(
            """
            המודל הבסיסי חוזה תמיד את השכר הממוצע
            בנתוני האימון.

            המודל עם משתנה אחד משתמש בשנות הניסיון בלבד.

            המודל עם שני משתנים משתמש גם בגיל וגם בשנות הניסיון.
            """
        )


# =====================================================
# הפעלת האפליקציה
# =====================================================

if not st.session_state.game_started:

    show_start_screen()

elif st.session_state.round_number < 5:

    show_round()

else:

    show_end_screen()


# מידע על המודל בתחתית האתר
st.divider()

show_model_information()
