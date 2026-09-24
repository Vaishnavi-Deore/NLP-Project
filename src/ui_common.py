"""
Shared UI Components for AQI-Sense
Provides consistent sidebar branding, system status indicator,
CSS injection, and page configuration across all pages.
"""

import streamlit as st
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
CSS_PATH = os.path.join(ROOT_DIR, "assets", "style.css")

def inject_custom_css():
    """Injects custom stylesheet into current Streamlit view."""
    if os.path.exists(CSS_PATH):
        try:
            with open(CSS_PATH, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception:
            pass

def render_sidebar():
    """Renders consistent branding and status indicators in the sidebar."""
    # Inject CSS
    inject_custom_css()

    st.sidebar.markdown("""
        <div style="padding: 10px 0 16px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2rem;">🍃</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.45rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AQI-Sense</h2>
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">NLP Query Classifier</div>
                </div>
            </div>
            <div style="display: inline-block; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 6px; padding: 2px 8px; font-size: 0.72rem; color: #38BDF8; font-weight: 600; margin-top: 8px;">
                v2.4.0 • Academic Edition
            </div>
        </div>
    """, unsafe_allow_html=True)

    # System Status Card
    st.sidebar.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
            <div style="font-size: 0.72rem; text-transform: uppercase; font-weight: 700; color: #94A3B8; margin-bottom: 6px; letter-spacing: 0.05em;">SYSTEM STATUS</div>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span><strong>NLP Engine:</strong> Online</span>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span><strong>Model:</strong> Loaded (Multinomial/SVM)</span>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span>
                <span><strong>Database:</strong> Connected (SQLite)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Categories Quick Guide Accordion
    with st.sidebar.expander("ℹ️ Categories Quick Guide"):
        st.markdown("""
        - **AQI Information:** Scale & pollutant definitions
        - **Health Effects:** Systemic & chronic biological risks
        - **Symptoms:** Irritations, cough, eye stinging
        - **Precautions:** Air purifiers, window sealing
        - **Masks & Protection:** N95/KN95 respirators
        - **Outdoor Activities:** Jogging, cardio, sports
        - **Vulnerable Groups:** Children, seniors, pregnancy
        - **Respiratory Health:** Asthma, COPD, inhalers
        - **Pollution Exposure:** Dosage, duration, highways
        """)

    st.sidebar.caption("AQI-Sense Project • Academic Demonstration")
