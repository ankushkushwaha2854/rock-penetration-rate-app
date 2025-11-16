import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle

# ------------------------------
# Load model
# ------------------------------
model = XGBRegressor()
model.load_model("rop_model.json")

# ------------------------------
# Load scaler
# ------------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("⛏️ Rock Penetration Rate Prediction (ROP)")
st.write("Enter drilling/mechanical & rock properties to predict penetration rate (m/min).")

# ------------------------------
# Input fields
# ------------------------------
rock_drill_power_kw = st.number_input("Rock Drill Power (kW)", value=14.0)
blow_frequency_bpm = st.number_input("Blow Frequency (BPM)", value=2100.0)
pulldown_pressure_bar = st.number_input("Pulldown Pressure (bar)", value=80.0)
blow_pressure_bar = st.number_input("Blow Pressure (bar)", value=6.5)
rotational_pressure_bar = st.number_input("Rotational Pressure (bar)", value=40.0)
ucs_mpa = st.number_input("UCS (MPa)", value=85.0)
tensile_strength_mpa = st.number_input("Tensile Strength (MPa)", value=5.0)
point_load_strength_mpa = st.number_input("Point Load Strength (MPa)", value=3.2)
p_wave = st.number_input("P-wave Velocity (km/s)", value=3.8)
elastic_modulus_mpa = st.number_input("Elastic Modulus (MPa)", value=6000.0)
density = st.number_input("Density (g/cm3)", value=2.7)

# ------------------------------
# CORRECT FEATURE ORDER (VERY IMPORTANT)
# ------------------------------
cols = [
    'rock_drill_power_kw',
    'blow_frequency_bpm',
    'pulldown_pressure_bar',
    'blow_pressure_bar',
    'rotational_pressure_bar',
    'ucs_mpa',
    'tensile_strength_mpa',
    'point_load_strength_mpa',
    'p-wave_velocity_km/s',
    'elastic_modulus_mpa',
    'density_g/cm3'
]

# ------------------------------
# Predict button
# ------------------------------
if st.button("Predict ROP"):
    new_data = pd.DataFrame([[
        rock_drill_power_kw,
        blow_frequency_bpm,
        pulldown_pressure_bar,
        blow_pressure_bar,
        rotational_pressure_bar,
        ucs_mpa,
        tensile_strength_mpa,
        point_load_strength_mpa,
        p_wave,
        elastic_modulus_mpa,
        density
    ]], columns=cols)

    # Scale input
    scaled = scaler.transform(new_data)

    # Predict
    prediction = model.predict(scaled)[0]

    st.success(f"🔥 Predicted Penetration Rate (ROP): **{prediction:.3f} m/min**")
