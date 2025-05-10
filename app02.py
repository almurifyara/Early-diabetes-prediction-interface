import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# --- Load Model ---
try:
    model_path = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
    diabetes_model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Load Images (from same folder as script) ---
def load_image(filename):
    try:
        return Image.open(os.path.join(os.path.dirname(__file__), filename))
    except Exception as e:
        st.warning(f"⚠️ Could not load {filename} — {e}")
        return None

img_low = load_image("Maintenance Plan.png")
img_moderate = load_image("Improvement Plan.png")
img_high = load_image("Mitigation Plan.png")
img_very_high = load_image("Intervention Plan.png")

# --- App Title ---
st.title("Early Diabetes Prediction")

# --- Section 1: Gender
st.subheader("1. Personal Information")
gender_input = st.selectbox("Gender", ["Male", "Female"])
gender = 0 if gender_input == "Male" else 1

# --- Section 2: Age
age = st.number_input("Age", min_value=1, max_value=120)

# --- Section 3: Medical History
hypertension = st.selectbox("Hypertension", ["Yes", "No"])
hypertension_val = 1 if hypertension == "Yes" else 0

heart_disease = st.selectbox("Heart Disease", ["Yes", "No"])
heart_disease_val = 1 if heart_disease == "Yes" else 0

smoking_input = st.selectbox("Smoking History", [
    "never", "No Info", "former", "current", "ever", "not current"
])
smoking_map = {
    "never": 0, "No Info": 1, "former": 2,
    "current": 3, "ever": 4, "not current": 5
}
smoking = smoking_map[smoking_input]

# --- Section 4: Body Metrics
st.subheader("2. Body Metrics")
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0)

bmi = weight / ((height / 100) ** 2) if height > 0 else np.nan
st.info(f"Calculated BMI: **{bmi:.2f}**")

hba1c = st.number_input("HbA1c Level (%)", min_value=0.0, max_value=15.0)
hba1c_val = hba1c if hba1c > 0 else np.nan

blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=500.0)

# --- Predict ---
if st.button("Predict Diabetes Risk"):
    input_data = np.array([[gender, age, hypertension_val, heart_disease_val,
                            smoking, bmi, hba1c_val, blood_glucose_level]])

    col_means = np.nanmean(input_data, axis=0)
    input_data = np.where(np.isnan(input_data), col_means, input_data)

    try:
        probability = diabetes_model.predict_proba(input_data)[0][1] * 100
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # --- Risk Classification
    if 0 <= probability < 25:
        risk_text = "Low Risk"
        image = img_low
    elif 25 <= probability < 50:
        risk_text = "Moderate Risk"
        image = img_moderate
    elif 50 <= probability < 75:
        risk_text = "High Risk"
        image = img_high
    else:
        risk_text = "Very High Risk"
        image = img_very_high

    # --- Output
    st.markdown(f"### Risk Stage: **{risk_text}**")
    st.markdown(f"**Predicted Diabetes Risk Probability:** {probability:.2f}%")
    st.markdown("Here’s a recommended lifestyle plan based on your result:")

    if image:
        st.image(image, use_container_width=True)
    else:
        st.warning("⚠️ Lifestyle plan image not found.")
