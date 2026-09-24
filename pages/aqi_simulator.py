"""
Page 3 — AQI Simulator (Interactive Educational Sandbox)
Dynamic interactive simulator with sliders for AQI, PM2.5, PM10, NO2, O3,
interactive Plotly gauge indicator, EPA category cards, and sensitive-group guidance.
"""

import streamlit as st
import plotly.graph_objects as go
import os
import sys

# Path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.advisory import get_aqi_details, MEDICAL_DISCLAIMER

def render_aqi_simulator():
    st.markdown("""
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🎛️ Interactive AQI Educational Simulator
            </h1>
            <p style="font-size: 1rem; color: #94A3B8; margin-top: 6px;">
                Adjust environmental parameters to observe how ambient pollutant concentrations alter the overall Air Quality Index, health bands, and public exposure advisories.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Preset scenarios
    presets = {
        "Custom Sliders": None,
        "🍃 Pristine Alpine Environment (AQI 22)": {"aqi": 22, "pm25": 5.0, "pm10": 12.0, "no2": 10.0, "o3": 25.0},
        "🟡 Typical Urban Afternoon (AQI 78)": {"aqi": 78, "pm25": 25.0, "pm10": 55.0, "no2": 35.0, "o3": 50.0},
        "🟠 Winter Temperature Inversion (AQI 135)": {"aqi": 135, "pm25": 49.0, "pm10": 110.0, "no2": 65.0, "o3": 30.0},
        "🔴 Heavy Industrial Smog (AQI 185)": {"aqi": 185, "pm25": 120.0, "pm10": 210.0, "no2": 95.0, "o3": 75.0},
        "🟣 Severe Agricultural Stubble Burning (AQI 260)": {"aqi": 260, "pm25": 210.0, "pm10": 340.0, "no2": 120.0, "o3": 60.0},
        "🟤 Hazardous Wildfire Smoke Plume (AQI 380)": {"aqi": 380, "pm25": 330.0, "pm10": 480.0, "no2": 160.0, "o3": 110.0},
    }

    c_preset, c_tag = st.columns([2, 1])
    with c_preset:
        chosen_preset = st.selectbox(
            "Load Scenario Preset:",
            list(presets.keys()),
            index=0
        )
    with c_tag:
        st.write("")
        st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 10px 14px; text-align: center; font-size: 0.82rem; color: #93C5FD;">
                🔬 <strong>Educational Simulation Mode</strong>
            </div>
        """, unsafe_allow_html=True)

    preset_values = presets[chosen_preset]

    # Controls Layout: Sliders on left, Gauge and Guidance on right
    col_sliders, col_display = st.columns([1.1, 1.3])

    with col_sliders:
        st.markdown("### 🎚️ Environmental Sliders")
        st.caption("Adjust AQI and specific ambient pollutant metrics:")

        aqi_val = st.slider(
            "Overall Air Quality Index (AQI):",
            min_value=0,
            max_value=500,
            value=preset_values["aqi"] if preset_values else 145,
            step=5,
            help="EPA Standard AQI scale from 0 to 500."
        )

        st.divider()
        st.markdown("**Individual Pollutant Concentrations:**")

        pm25_val = st.slider(
            "Fine Particulate Matter (PM2.5) [µg/m³]:",
            min_value=0.0,
            max_value=350.0,
            value=preset_values["pm25"] if preset_values else 55.0,
            step=1.0,
            help="Particles <= 2.5 µm that penetrate deep into lung alveoli."
        )

        pm10_val = st.slider(
            "Coarse Particulate Matter (PM10) [µg/m³]:",
            min_value=0.0,
            max_value=500.0,
            value=preset_values["pm10"] if preset_values else 115.0,
            step=5.0,
            help="Inhalable dust, pollen, and mold particles <= 10 µm."
        )

        no2_val = st.slider(
            "Nitrogen Dioxide (NO2) [ppb]:",
            min_value=0.0,
            max_value=250.0,
            value=preset_values["no2"] if preset_values else 45.0,
            step=5.0,
            help="Traffic and power plant combustion emissions."
        )

        o3_val = st.slider(
            "Ground-Level Ozone (O3) [ppb]:",
            min_value=0.0,
            max_value=200.0,
            value=preset_values["o3"] if preset_values else 40.0,
            step=5.0,
            help="Secondary pollutant formed by sunlight reacting with hydrocarbons."
        )

    # Calculate category from current AQI
    aqi_info = get_aqi_details(aqi_val)

    with col_display:
        st.markdown("### 🧭 Live Air Quality Status")

        # Interactive Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Category: {aqi_info['category']}", 'font': {'size': 20, 'color': aqi_info['color']}},
            gauge={
                'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                'bar': {'color': aqi_info['color'], 'thickness': 0.3},
                'bgcolor': "rgba(255,255,255,0.05)",
                'borderwidth': 1,
                'bordercolor': "rgba(255,255,255,0.15)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(16, 185, 129, 0.35)'},
                    {'range': [51, 100], 'color': 'rgba(251, 191, 36, 0.35)'},
                    {'range': [101, 150], 'color': 'rgba(249, 115, 22, 0.35)'},
                    {'range': [151, 200], 'color': 'rgba(239, 68, 68, 0.35)'},
                    {'range': [201, 300], 'color': 'rgba(139, 92, 246, 0.35)'},
                    {'range': [301, 500], 'color': 'rgba(136, 19, 55, 0.45)'}
                ],
            }
        ))
        fig_gauge.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC", family="Plus Jakarta Sans")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Pollutant Quick Summary Cards
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric("PM2.5", f"{pm25_val:.0f} µg/m³")
        with p2:
            st.metric("PM10", f"{pm10_val:.0f} µg/m³")
        with p3:
            st.metric("NO2", f"{no2_val:.0f} ppb")
        with p4:
            st.metric("Ozone", f"{o3_val:.0f} ppb")

    st.write("")
    st.divider()

    # Dynamic Guidance Cards Row
    st.markdown("### 🛡️ Recommended Health & Activity Actions")

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;">👥 General Population</div>
                <div style="font-size: 0.92rem; color: #E2E8F0; margin-top: 8px; line-height: 1.5;">
                    {aqi_info['general_guidance']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #F59E0B; text-transform: uppercase;">⚠️ Sensitive & Vulnerable Groups</div>
                <div style="font-size: 0.92rem; color: #E2E8F0; margin-top: 8px; line-height: 1.5;">
                    {aqi_info['sensitive_guidance']}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with g3:
        mask_status = "😷 Certified N95 / KN95 Mask Recommended Outdoors" if aqi_info['mask_recommended'] else "✅ Masks not generally required for healthy individuals"
        st.markdown(f"""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; color: #10B981; text-transform: uppercase;">🏃 Outdoor Activities & Sports</div>
                <div style="font-size: 0.92rem; color: #E2E8F0; margin-top: 8px; line-height: 1.5;">
                    {aqi_info['outdoor_activity']}
                </div>
                <div style="font-size: 0.8rem; font-weight: 600; color: #FBBF24; margin-top: 10px;">
                    {mask_status}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # Dynamic Anatomical & Particle Deposition Section
    st.markdown("### 🔬 Microscopic Particle Scale & Anatomical Penetration")
    st.caption("Visualizing how inhalable particulate sizes dictate biological infiltration depth in the human respiratory system.")

    respiratory_img_path = os.path.join(parent_dir, "assets", "respiratory_pm.jpg")
    if os.path.exists(respiratory_img_path):
        st.image(
            respiratory_img_path,
            caption="Comparative Scale: PM10 Deposition in Upper Airway vs Deep PM2.5 Penetration into Alveolar Capillary Beds",
            use_container_width=True
        )

    col_bio1, col_bio2 = st.columns([1, 1])
    with col_bio1:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px;">
                <h4 style="color: #38BDF8; margin-top: 0;">Coarse Particulate Matter (PM10)</h4>
                <ul style="font-size: 0.88rem; color: #CBD5E1; padding-left: 18px; line-height: 1.6;">
                    <li><strong>Diameter:</strong> ≤ 10 µm (approx 1/7th the width of a human hair).</li>
                    <li><strong>Origin:</strong> Construction dust, crushed rock, road dust, and pollen.</li>
                    <li><strong>Deposition:</strong> Trapped by nasal cilia, mucus membranes, and upper tracheobronchial passages. Provokes sneezing, coughing, and runny nose.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col_bio2:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px;">
                <h4 style="color: #EF4444; margin-top: 0;">Fine Particulate Matter (PM2.5)</h4>
                <ul style="font-size: 0.88rem; color: #CBD5E1; padding-left: 18px; line-height: 1.6;">
                    <li><strong>Diameter:</strong> ≤ 2.5 µm (approx 1/30th the width of a human hair).</li>
                    <li><strong>Origin:</strong> Vehicle combustion, industrial exhaust, agricultural burning.</li>
                    <li><strong>Deposition:</strong> Bypasses upper airways and deposits directly into microscopic alveoli, translocating into the bloodstream and triggering systemic vascular inflammation.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    with st.expander("🎬 Watch Video: Particulate Pollution and Human Lung Anatomy (3-Min Animation)"):
        st.video("https://www.youtube.com/watch?v=GVBeY1jSG9Y")
        st.caption("Educational animation: How particulate matter alters blood oxygenation and alveoli function.")

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Educational Notice:</strong> This simulator provides educational approximations based on standard environmental guidance models. It is not an instrument for clinical medical diagnosis or a substitute for local emergency agency advisories.
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_aqi_simulator()
