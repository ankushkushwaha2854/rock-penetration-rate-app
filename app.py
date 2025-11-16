import warnings
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import pickle
import os

# ------------------------------
# SET BACKGROUND IMAGE
# ------------------------------
def set_background_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://plus.unsplash.com/premium_photo-1673002094039-3b4a9e8d1fff?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        
        /* Improve readability of content */
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 3rem;
            border-radius: 15px;
            margin-top: 2rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Style the header */
        h1 {
            color: #1f3d7a;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Style the success message */
        .stSuccess {
            background-color: rgba(209, 231, 221, 0.9);
            border: 1px solid rgba(56, 161, 105, 0.3);
            border-radius: 10px;
            padding: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_background_image()

# ------------------------------
# Load model and scaler safely
# ------------------------------
model = XGBRegressor()
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "rop_model.json")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model.load_model(model_path)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

st.title("⛏️ Rock Penetration Rate Prediction (ROP)")
st.write("Enter drilling/mechanical & rock properties to predict penetration rate (m/min).")

# ------------------------------
# Input fields
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    rock_drill_power_kw = st.number_input("Rock Drill Power (kW)", value=14.0)
    blow_frequency_bpm = st.number_input("Blow Frequency (BPM)", value=2100.0)
    pulldown_pressure_bar = st.number_input("Pulldown Pressure (bar)", value=80.0)
    blow_pressure_bar = st.number_input("Blow Pressure (bar)", value=6.5)
    rotational_pressure_bar = st.number_input("Rotational Pressure (bar)", value=40.0)
    
with col2:
    ucs_mpa = st.number_input("UCS (MPa)", value=85.0)
    tensile_strength_mpa = st.number_input("Tensile Strength (MPa)", value=5.0)
    point_load_strength_mpa = st.number_input("Point Load Strength (MPa)", value=3.2)
    p_wave = st.number_input("P-wave Velocity (km/s)", value=3.8)
    elastic_modulus_mpa = st.number_input("Elastic Modulus (MPa)", value=6000.0)
    density = st.number_input("Density (g/cm3)", value=2.7)

# ------------------------------
# CORRECT FEATURE ORDER
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
st.markdown("---")
if st.button("🚀 Predict ROP", use_container_width=True):
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
    
    # Additional visual feedback
    st.balloons()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Rock Penetration Rate Prediction App ⛏️"
    "</div>",
    unsafe_allow_html=True
)
