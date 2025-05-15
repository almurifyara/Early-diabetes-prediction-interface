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

# --- Load Images ---
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

# --- Section 1: Personal Info ---
st.subheader("1. Personal Information")
gender_input = st.selectbox("Gender", ["Male", "Female"], index=0)
gender = 0 if gender_input == "Male" else 1

age = st.number_input("Age", min_value=1, max_value=120, value=0)

# --- Section 2: Medical History ---
hypertension = st.selectbox("Hypertension", ["Yes", "No"], index=1)
hypertension_val = 1 if hypertension == "Yes" else 0

heart_disease = st.selectbox("Heart Disease", ["Yes", "No"], index=1)
heart_disease_val = 1 if heart_disease == "Yes" else 0

smoking_input = st.selectbox("Smoking History", [
    "never", "No Info", "former", "current", "ever", "not current"
], index=0)
smoking_map = {
    "never": 0, "No Info": 1, "former": 2,
    "current": 3, "ever": 4, "not current": 5
}
smoking = smoking_map[smoking_input]

# --- Section 3: Body Metrics ---
st.subheader("2. Body Metrics")
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=0.0)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=0.0)

bmi = weight / ((height / 100) ** 2) if height > 0 else np.nan
st.info(f"Calculated BMI: **{bmi:.2f}**" if not np.isnan(bmi) else "BMI: Not available")

hba1c = st.number_input("HbA1c Level (%)", min_value=0.0, max_value=15.0, value=0.0)
blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=50.0, max_value=500.0, value=0.0)

# --- Prediction ---
if st.button("Predict Diabetes Risk"):

    input_data = np.array([[gender, age, hypertension_val, heart_disease_val,
                            smoking, bmi, hba1c, blood_glucose_level]])

    col_means = np.nanmean(input_data, axis=0)
    input_data = np.where(np.isnan(input_data), col_means, input_data)

    try:
        model_probability = diabetes_model.predict_proba(input_data)[0][1] * 100
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # --- Rule-Based Override ---
    if hba1c >= 6.5 or blood_glucose_level >= 140:
        probability = 90.0
        risk_text = "Very High Risk"
        image = img_very_high
    elif 5.7 <= hba1c <= 6.4 or 117 <= blood_glucose_level < 140:
        probability = 55.0
        risk_text = "High Risk"
        image = img_high
    elif hba1c < 5.7 and blood_glucose_level < 117:
        probability = 10.0
        risk_text = "Low Risk"
        image = img_low
    else:
        # Fallback to model probability
        probability = model_probability
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

    # --- Output ---
    st.markdown(f"### Risk Stage: **{risk_text}**")
    st.markdown(f"**Predicted Diabetes Risk Probability:** {probability:.2f}%")
    st.markdown("Here’s a recommended lifestyle plan based on your result:")

    if image:
        st.image(image, use_container_width=True)
    else:
        st.warning("Lifestyle plan image not found.")

