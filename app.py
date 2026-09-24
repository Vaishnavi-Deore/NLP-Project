"""
AQI-Sense — AQI Health Query Classifier
Main Application Entrypoint
An NLP-Based System for Classifying Air Quality and Health-Related Queries
"""

import streamlit as st
import os
import sys

# Anchor root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.ui_common import render_sidebar

# Configure wide layout and favicon
st.set_page_config(
    page_title="AQI-Sense — Intelligent Air Quality Query Classifier",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Organized multi-page navigation hierarchy with icons and grouped sections
pages = {
    "Overview": [
        st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
    ],
    "Intelligent Features": [
        st.Page("pages/analyzer.py", title="Query Analyzer", icon="🔬"),
        st.Page("pages/aqi_simulator.py", title="AQI Simulator", icon="🎛️"),
    ],
    "Model Evaluation": [
        st.Page("pages/analytics.py", title="Analytics", icon="📈"),
        st.Page("pages/model_comparison.py", title="Model Comparison", icon="⚖️"),
    ],
    "Data & Records": [
        st.Page("pages/history.py", title="Query History", icon="📜"),
        st.Page("pages/dataset_explorer.py", title="Dataset Explorer", icon="📁"),
    ],
    "Documentation": [
        st.Page("pages/about.py", title="About Project", icon="ℹ️"),
    ]
}

# Run Streamlit Multi-Page Navigation
pg = st.navigation(pages)

# Render consistent sidebar branding and system status indicator
render_sidebar()

# Execute selected page
pg.run()
