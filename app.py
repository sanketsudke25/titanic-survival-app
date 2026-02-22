import streamlit as st
import pickle
import numpy as np

# Page Config (Change browser tab name)
st.set_page_config(page_title="🚢Titanic Survival Predictor")

# Background Gradient
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #74b9ff, #6c5ce7);
    }

    h1, h2, h3, h4, h5, h6, label, p {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
with open("titanic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter Passenger Details Below:")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=8, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=6, value=0)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# Convert categorical inputs
sex_val = 1 if sex == "Male" else 0
embarked_val = {"S": 0, "C": 1, "Q": 2}[embarked]

# Create feature array
features = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])

# Scale features
features_scaled = scaler.transform(features)

# Prediction button
if st.button("Predict Survival"):
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.success(" Passenger Survived")
    else:
        st.error(" Passenger Did Not Survive")
