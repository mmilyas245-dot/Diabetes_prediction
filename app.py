import streamlit as st
import pandas as pd
import joblib


# ==============================
# LOAD SAVED MODEL
# ==============================

model_package = joblib.load(r"diabetes_model.pkl")

model = model_package["model"]
numeric = model_package["numeric_columns"]
mean = model_package["mean"]
std = model_package["std"]
feature_columns = model_package["feature_columns"]
threshold = model_package["threshold"]


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)


# ==============================
# TITLE
# ==============================

st.title("🩺 Diabetes Prediction System")

st.write(
    "Enter the following information to get a machine-learning "
    "prediction."
)

st.warning(
    "This application is for educational purposes only and "
    "is not a medical diagnosis."
)


# ==============================
# USER INPUT
# ==============================

st.subheader("Patient Information")

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

smoking_history = st.selectbox(
    "Smoking History",
    [
        "never",
        "No Info",
        "former",
        "current",
        "not current",
        "ever"
    ]
)

bmi = st.number_input(
    "BMI",
    min_value=5.0,
    max_value=100.0,
    value=27.0
)

HbA1c_level = st.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.8
)

blood_glucose_level = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=140
)


# ==============================
# PREDICTION
# ==============================

if st.button("🔍 Predict Diabetes"):

    # Create DataFrame
    data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [HbA1c_level],
        "blood_glucose_level": [blood_glucose_level]
    })

    # One-hot encoding
    data = pd.get_dummies(
        data,
        columns=["gender", "smoking_history"]
    )

    # Match training columns
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Standardization
    data[numeric] = (
        data[numeric] - mean
    ) / std

    # Prediction probability
    probability = model.predict_proba(data)[0, 1]

    # Apply threshold
    prediction = int(probability >= threshold)

    # ==============================
    # DISPLAY RESULT
    # ==============================

    st.subheader("Prediction Result")

    st.metric(
        "Diabetes Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:

        st.error(
            "⚠️ Model Prediction: Diabetes"
        )

    else:

        st.success(
            "✅ Model Prediction: No Diabetes"
        )

    st.info(
        f"Decision threshold: {threshold * 100:.0f}%"
    )
    st.markdown(
    """
    <hr>
    <div style="text-align: center; color: gray;">
        <p>Developed by <b>M. ILYAS</b> | Diabetes Prediction System</p>
    </div>
    """,
    unsafe_allow_html=True
)
