
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# --------------------------------------------------
# 2. Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "credit_risk_pipeline.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# --------------------------------------------------
# 3. Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1150px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-size: 2.3rem;
            font-weight: 750;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
            color: #64748b;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 650;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        .result-card {
            padding: 1.2rem;
            border: 1px solid #dbe3ee;
            border-radius: 12px;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }

        .small-note {
            font-size: 0.9rem;
            color: #64748b;
        }

        div.stButton > button {
            width: 100%;
            min-height: 3rem;
            font-weight: 650;
            border-radius: 9px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 4. App header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">💳 Credit Risk Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Assess borrower credit risk using a trained machine learning model.
        Enter the applicant details below to generate a prediction.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# 5. Applicant input form
# --------------------------------------------------

with st.form("credit_risk_form"):

    st.markdown(
        '<div class="section-title">👤 Applicant Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    with col2:
        sex = st.selectbox(
            "Sex",
            ["male", "female"]
        )

    with col3:
        job = st.selectbox(
            "Job category",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 — Unskilled / non-resident",
                1: "1 — Unskilled / resident",
                2: "2 — Skilled",
                3: "3 — Highly skilled"
            }[x]
        )

    st.divider()

    st.markdown(
        '<div class="section-title">🏠 Financial Background</div>',
        unsafe_allow_html=True
    )

    col4, col5 = st.columns(2)

    with col4:
        housing = st.selectbox(
            "Housing",
            ["own", "rent", "free"]
        )

        saving_accounts = st.selectbox(
            "Saving accounts",
            ["little", "moderate", "quite rich", "rich"]
        )

    with col5:
        checking_account = st.selectbox(
            "Checking account",
            ["little", "moderate", "rich"]
        )

    st.divider()

    st.markdown(
        '<div class="section-title">💰 Credit Details</div>',
        unsafe_allow_html=True
    )

    col6, col7 = st.columns(2)

    with col6:
        credit_amount = st.number_input(
            "Credit amount",
            min_value=250,
            max_value=200000,
            value=2500,
            step=250
        )

    with col7:
        duration = st.number_input(
            "Loan duration (months)",
            min_value=1,
            max_value=72,
            value=18,
            step=1
        )

    purpose = st.selectbox(
        "Purpose of credit",
        [
            "radio/TV",
            "education",
            "furniture/equipment",
            "car",
            "business",
            "domestic appliances",
            "repairs",
            "vacation/other"
        ]
    )

    submitted = st.form_submit_button(
        "🔍 Assess Credit Risk"
    )


# --------------------------------------------------
# 6. Prediction
# --------------------------------------------------

if submitted:

    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "Job": [job],
        "Housing": [housing],
        "Saving accounts": [saving_accounts],
        "Checking account": [checking_account],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose]
    })

    try:
        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        probability_dict = dict(zip(classes, probabilities))

        if "bad" not in probability_dict or "good" not in probability_dict:
            st.error("The model does not contain the expected 'bad' and 'good' classes.")
            st.stop()

        bad_probability = float(probability_dict["bad"])
        good_probability = float(probability_dict["good"])

        # Keep the project's existing decision threshold.
        threshold = 0.25

        # Show both the model's default prediction and the
        # project's threshold-based assessment.
        threshold_risk = (
            "BAD" if bad_probability >= threshold else "GOOD"
        )

        st.divider()
        st.subheader("📊 Credit Risk Results")

        metric1, metric2 = st.columns(2)

        with metric1:
            st.metric(
                "BAD-risk probability",
                f"{bad_probability:.1%}"
            )
            st.progress(bad_probability)

        with metric2:
            st.metric(
                "GOOD-risk probability",
                f"{good_probability:.1%}"
            )
            st.progress(good_probability)

        if threshold_risk == "BAD":
            st.error(
                "⚠️ Higher predicted credit risk",
                icon="⚠️"
            )
            st.write(
                "The estimated BAD-risk probability meets or exceeds "
                "the project's 0.25 decision threshold."
            )
        else:
            st.success(
                "✅ Lower predicted credit risk",
                icon="✅"
            )
            st.write(
                "The estimated BAD-risk probability is below "
                "the project's 0.25 decision threshold."
            )

        st.caption(
            f"Model's default prediction: {str(prediction).upper()} "
            f"| Threshold-based assessment: {threshold_risk} "
            f"| Decision threshold: {threshold:.2f}"
        )

        with st.expander("View applicant details"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:
        st.error(
            "Prediction failed. Please check the input values "
            "and ensure they match the model's expected features."
        )
        st.exception(error)


# --------------------------------------------------
# 7. Footer
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="small-note">
        <b>About this project:</b> A machine learning portfolio project
        using the German Credit dataset.<br><br>
        <b>Disclaimer:</b> This tool is intended for educational and
        demonstration purposes only. Its predictions should not be used
        as the sole basis for real-world lending decisions.
    </div>
    """,
    unsafe_allow_html=True
)