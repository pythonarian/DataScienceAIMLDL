import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 Boston House Price Predictor")

st.write("Enter property details:")

# Input fields
CRIM = st.number_input("Crime Rate", 0.0)
ZN = st.number_input("Residential Land Zoning", 0.0)
INDUS = st.number_input("Industrial Area", 0.0)
CHAS = st.number_input("Charles River (0 or 1)", 0)
NOX = st.number_input("Nitric Oxide", 0.0)
RM = st.number_input("Rooms", 0.0)
AGE = st.number_input("Age", 0.0)
DIS = st.number_input("Distance", 0.0)
RAD = st.number_input("Accessibility", 0.0)
TAX = st.number_input("Tax Rate", 0.0)
PTRATIO = st.number_input("Pupil-Teacher Ratio", 0.0)
B = st.number_input("Black Index", 0.0)
LSTAT = st.number_input("Lower Status %", 0.0)

if st.button("Predict"):
    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE,
                            DIS, RAD, TAX, PTRATIO, B, LSTAT]])

    input_scaled = scaler.transform(input_data)

    pred_log = model.predict(input_scaled)
    pred = np.exp(pred_log)  # reverse log

    st.success(f"Estimated House Price: ${pred[0]*1000:.2f}")
