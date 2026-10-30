
st.markdown(
    """
    <style>
    /* כיוון כללי של האפליקציה */
    .stApp {
        direction: rtl !important;
        text-align: right !important;
    }

    /* יישור כל הכותרות לימין */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        direction: rtl !important;
        text-align: right !important;
        width: 100% !important;
    }

    /* יישור כל הטקסטים לימין */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp .stMarkdown {
        direction: rtl !important;
        text-align: right !important;
    }

    /* יישור תיבות הקלט */
    div[data-testid="stNumberInput"] label {
        direction: rtl !important;
        text-align: right !important;
    }

    div[data-testid="stNumberInput"] input {
        text-align: right !important;
    }

    /* יישור תיבות סימון */
    div[data-testid="stCheckbox"] {
        direction: rtl !important;
        text-align: right !important;
    }

    /* טבלת הנתונים */
    [data-testid="stDataFrame"] {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)
