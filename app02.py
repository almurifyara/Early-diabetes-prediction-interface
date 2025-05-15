import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# --- Load Model ---

try:
model\_path = os.path.join(os.path.dirname(**file**), "diabetes\_model.pkl")
diabetes\_model = joblib.load(model\_path)
except Exception as e:
st.error(f"Error loading model: {e}")
st.stop()

# --- Load Lifestyle Plan Images ---

def load\_image(filename):
try:
return Image.open(os.path.join(os.path.dirname(**file**), filename))
except Exception as e:
st.warning(f"Could not load {filename} — {e}")
return None

img\_low = load\_image("Maintenance Plan.png")
img\_moderate = load\_image("Improvement Plan.png")
img\_high = load\_image("Mitigation Plan.png")
img\_very\_high = load\_image("Intervention Plan.png")

# --- App Title ---

st.title("Early Diabetes Prediction")

# --- Section 1: Personal Information ---

st.subheader("1. Personal Information")
gender\_input = st.selectbox("Gender", \["Male", "Female"])
gender = 0 if gender\_input == "Male" else 1

age = st.number\_input("Age", min\_value=0, max\_value=120, value=0)

# --- Section 2: Medical History ---

hypertension = st.selectbox("Hypertension", \["No", "Yes"])
hypertension\_val = 1 if hypertension == "Yes" else 0

heart\_disease = st.selectbox("Heart Disease", \["No", "Yes"])
heart\_disease\_val = 1 if heart\_disease == "Yes" else 0

smoking\_input = st.selectbox("Smoking History", \[
"never", "No Info", "former", "current", "ever", "not current"
])
smoking\_map = {
"never": 0, "No Info": 1, "former": 2,
"current": 3, "ever": 4, "not current": 5
}
smoking = smoking\_map\[smoking\_input]

# --- Section 3: Body Metrics ---

st.subheader("2. Body Metrics")
height = st.number\_input("Height (cm)", min\_value=0.0, max\_value=250.0, value=0.0)
weight = st.number\_input("Weight (kg)", min\_value=0.0, max\_value=200.0, value=0.0)

bmi = weight / ((height / 100) \*\* 2) if height > 0 else np.nan
st.info(f"Calculated BMI: **{bmi:.2f}**")

hba1c = st.number\_input("HbA1c Level (%)", min\_value=0.0, max\_value=15.0, value=0.0)
hba1c\_val = hba1c if hba1c > 0 else np.nan

blood\_glucose\_level = st.number\_input("Blood Glucose Level (mg/dL)", min\_value=0.0, max\_value=500.0, value=0.0)

# --- Prediction Button ---

if st.button("Predict Diabetes Risk"):
input\_data = np.array(\[\[gender, age, hypertension\_val, heart\_disease\_val,
smoking, bmi, hba1c\_val, blood\_glucose\_level]])

```
col_means = np.nanmean(input_data, axis=0)
input_data = np.where(np.isnan(input_data), col_means, input_data)

try:
    probability = diabetes_model.predict_proba(input_data)[0][1] * 100

    # --- Override prediction using clinical rules ---
    if hba1c_val >= 6.5 or blood_glucose_level >= 140:
        probability = 100.0  # Diabetes
    elif 5.7 <= hba1c_val < 6.5 or 117 <= blood_glucose_level < 140:
        probability = max(probability, 60.0)  # Prediabetes

except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --- Risk Classification ---
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
st.markdown(f"### 🩺 Risk Stage: **{risk_text}**")
st.markdown(f"**Predicted Diabetes Risk Probability:** {probability:.2f}%")
st.markdown("Here’s a recommended lifestyle plan based on your result:")
if image:
    st.image(image, use_container_width=True)
else:
    st.warning("Lifestyle plan image not found.")
```
