import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Load trained model
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "credit_risk_pipeline.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("💳 Credit Risk Prediction App")

st.write(
    "Enter the applicant's information to estimate credit risk."
)


# -----------------------------
# User inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

job = st.selectbox(
    "Job",
    [0, 1, 2, 3]
)

housing = st.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

saving_accounts = st.selectbox(
    "Saving accounts",
    ["little", "moderate", "quite rich", "rich"]
)

checking_account = st.selectbox(
    "Checking account",
    ["little", "moderate", "rich"]
)

credit_amount = st.number_input(
    "Credit amount",
    min_value=250,
    max_value=200000,
    value=2500
)

duration = st.number_input(
    "Duration (months)",
    min_value=1,
    max_value=72,
    value=18
)

purpose = st.selectbox(
    "Purpose",
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


# -----------------------------
# Create input dataframe
# -----------------------------

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


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Credit Risk"):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    classes = model.classes_

    probability_dict = dict(zip(classes, probabilities))

    bad_probability = probability_dict["bad"]
    good_probability = probability_dict["good"]


    # -----------------------------
    # Display result
    # -----------------------------

    st.subheader("Credit Risk Assessment")

    st.write(
        f"Probability of BAD risk: "
        f"{bad_probability:.2%}"
    )

    st.write(
        f"Probability of GOOD risk: "
        f"{good_probability:.2%}"
    )


    if bad_probability >= 0.25:

        st.error("⚠️ Higher Credit Risk")

        st.write(
            "The model classifies this applicant as "
            "BAD risk based on the selected threshold."
        )

    else:

        st.success("✅ Lower Credit Risk")

        st.write(
            "The model classifies this applicant as "
            "GOOD risk based on the selected threshold."
        )