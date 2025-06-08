import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models and encoders
rf_model = joblib.load("random_forest_model.pkl")
rf_label_encoders = joblib.load("rf_label_encoders.pkl")
los_model = joblib.load("los_model.pkl")
los_label_encoders = joblib.load("los_label_encoders.pkl")

# Cached diagnosis dropdown from top frequency diagnoses
@st.cache_data
def get_diagnosis_options():
    df = pd.read_csv("MERGED_DATA_cleaned_final1.csv")
    all_diagnoses = df["DIAGNOSIS"].dropna().astype(str).str.upper()

    # Split and clean
    simplified = all_diagnoses.str.split(r"[;,/]", expand=True).stack().str.strip()

    # Filter: remove short, non-alphabetic, or specific irrelevant diagnoses
    simplified = simplified[simplified.str.len() > 2]
    simplified = simplified[simplified.str.contains(r"[A-Z]{2,}", regex=True)]

    # Exclude specific patterns or terms
    blacklist = [
        "P FALL", "P MOTOR VEHICLE ACCIDENT", "MOTOR VEHICLE ACCIDENT", "FALL", "TRAUMA", "ASSAULT",
        "UNKNOWN", "MULTIPLE", "NONE", "UNSPECIFIED", "OTHER"
    ]
    simplified = simplified[~simplified.isin(blacklist)]

    # Final top options
    top_diagnoses = simplified.value_counts().nlargest(20).index.tolist()
    top_diagnoses.append("Other")
    return sorted(top_diagnoses)

diagnosis_options = get_diagnosis_options()

# Streamlit UI
st.title("🏥 Patient Admission & Length of Stay Prediction")
st.markdown("Enter patient information below:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=50)
gender = st.selectbox("Gender", ["M", "F"])
heartrate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=80)
resprate = st.number_input("Respiratory Rate (rpm)", min_value=5, max_value=60, value=20)
o2sat = st.number_input("Oxygen Saturation (%)", min_value=50, max_value=100, value=95)
temp = st.number_input("Temperature (°F)", min_value=85.0, max_value=110.0, value=98.6)
sbp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
dbp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=150, value=80)
diagnosis = st.selectbox("Diagnosis", diagnosis_options)

# Safe encoding helper
def safe_encode(encoders, col, val):
    try:
        return encoders[col].transform([val])[0]
    except:
        return 0

# Prepare input for ICU admission model
input_dict = {
    "AGE": age,
    "GENDER": safe_encode(rf_label_encoders, "GENDER", gender),
    "HEARTRATE": heartrate,
    "RESPRATE": resprate,
    "O2SATURATION": o2sat,
    "TEMPERATURE": temp,
    "SYSTOLICBP": sbp,
    "DIASTOLICBP": dbp,
    "DIAGNOSIS": safe_encode(rf_label_encoders, "DIAGNOSIS", diagnosis)
}

if st.button("Predict Admission"):
    input_array = np.array([list(input_dict.values())])
    prediction = rf_model.predict(input_array)[0]
    proba = rf_model.predict_proba(input_array)[0][1]

    if prediction == 1:
        st.error(f" Patient is likely to be admitted (Probability: {proba:.2%})")

        # Prepare input for LOS prediction
        los_input = {
            "AGE": age,
            "GENDER": safe_encode(los_label_encoders, "GENDER", gender),
            "HEARTRATE": heartrate,
            "RESPRATE": resprate,
            "O2SATURATION": o2sat,
            "TEMPERATURE": temp,
            "SYSTOLICBP": sbp,
            "DIASTOLICBP": dbp,
            "DIAGNOSIS": safe_encode(los_label_encoders, "DIAGNOSIS", diagnosis)
        }

        los_array = np.array([list(los_input.values())])
        log_pred = los_model.predict(los_array)[0]
        los_days = np.expm1(log_pred)  # Reverse log1p transformation

        st.warning(f"Estimated Length of Stay: **{los_days:.1f} days**")

    else:
        st.success(f"Patient is unlikely to be admitted (Probability: {proba:.2%})")
