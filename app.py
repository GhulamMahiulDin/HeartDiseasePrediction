
import pickle
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("heart_disease_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

st.title("Heart Disease Prediction System")
st.write(
    "Enter the information below to see the prediction "
    "made by our machine learning model."
)

st.info(
    "Educational project only. This application is not a "
    "medical diagnostic tool. Consult a qualified healthcare "
    "professional for medical advice."
)

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age", min_value=1, max_value=120, value=45
    )
    sex = st.selectbox(
        "Sex code (0 = female, 1 = male)", [0, 1]
    )
    cp = st.selectbox("Chest pain type", [1, 2, 3, 4])
    trestbps = st.number_input(
        "Resting blood pressure", min_value=50,
        max_value=300, value=130
    )
    chol = st.number_input(
        "Cholesterol", min_value=50,
        max_value=700, value=240
    )
    fbs = st.selectbox(
        "Fasting blood sugar above 120 mg/dl", [0, 1]
    )
    restecg = st.selectbox("Resting ECG code", [0, 1, 2])

with col2:
    thalach = st.number_input(
        "Maximum heart rate achieved", min_value=40,
        max_value=250, value=150
    )
    exang = st.selectbox(
        "Exercise-induced angina (0 = no, 1 = yes)", [0, 1]
    )
    oldpeak = st.number_input(
        "ST depression (oldpeak)", min_value=0.0,
        max_value=10.0, value=1.0, step=0.1
    )
    slope = st.selectbox("Slope code", [1, 2, 3])
    ca = st.selectbox("Major vessels code (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thal code", [3, 6, 7])

if st.button("Predict", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("The model indicates possible heart disease.")
    else:
        st.success("The model indicates no heart disease.")

    st.caption(
        "This is an educational model prediction, not a medical diagnosis."
    )
