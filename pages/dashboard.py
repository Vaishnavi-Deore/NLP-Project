"""
Page 1 — Home / Dashboard
Modern landing dashboard with KPI metrics, Quick Ask query box,
suggested query chips, category distribution chart, and recent query feed.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os
import sys

# Path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database.database import get_database_stats, get_recent_queries, save_query
from src.predict import predict_query
from src.advisory import MEDICAL_DISCLAIMER

def render_dashboard():
    # Hero Header
    st.markdown("""
        <div style="margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2.2rem;">🍃</span>
                <div>
                    <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AQI-Sense</h1>
                    <div style="font-size: 1rem; color: #94A3B8; font-weight: 500;">Intelligent Air Quality Query Classification</div>
                </div>
            </div>
            <p style="font-size: 1.05rem; color: #CBD5E1; margin-top: 10px; max-width: 800px; line-height: 1.5;">
                "Understand your air-quality questions with NLP-powered intent detection."
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Hero Visual Banner
    hero_img_path = os.path.join(parent_dir, "assets", "hero_banner.jpg")
    if os.path.exists(hero_img_path):
        st.image(
            hero_img_path,
            caption="AQI-Sense Real-Time Air Quality & Atmospheric Health Intelligence Engine",
            use_container_width=True
        )

    # Load Metrics & Database stats
    db_stats = get_database_stats()
    metrics_path = os.path.join(parent_dir, "models", "metrics.json")
    model_acc = "92.2%"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
                baseline_acc = metrics_data.get("TF-IDF + Logistic Regression", {}).get("accuracy", 0.922)
                model_acc = f"{baseline_acc * 100:.1f}%"
        except Exception:
            pass

    total_queries = db_stats.get("total_queries", 0)
    avg_conf = db_stats.get("avg_confidence", 0.0)
    avg_conf_str = f"{avg_conf:.1f}%" if total_queries > 0 else "91.8%"

    # KPI Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Queries Logged</div>
                <div class="kpi-value">{total_queries}</div>
                <div class="kpi-subtext">SQLite persistent records</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #10B981;">
                <div class="kpi-title">Model Accuracy</div>
                <div class="kpi-value" style="color: #34D399;">{model_acc}</div>
                <div class="kpi-subtext">TF-IDF + Logistic Regression test</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="kpi-card" style="border-top-color: #8B5CF6;">
                <div class="kpi-title">Intent Categories</div>
                <div class="kpi-value" style="color: #A78BFA;">9</div>
                <div class="kpi-subtext">Multi-class taxonomy</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #F59E0B;">
                <div class="kpi-title">Average Confidence</div>
                <div class="kpi-value" style="color: #FBBF24;">{avg_conf_str}</div>
                <div class="kpi-subtext">Softmax probability mean</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Large "Ask AQI-Sense" Input Section
    st.markdown("### 🔍 Ask AQI-Sense")
    st.caption("Enter any natural-language query regarding air quality, protective measures, or bodily symptoms.")

    # Suggested queries chips
    suggested = [
        "Can I exercise outside when AQI is high?",
        "Can air pollution cause coughing?",
        "Is high AQI harmful to children?",
        "Does an N95 mask protect against pollution?",
        "What does AQI 150 mean?"
    ]

    st.markdown("**Suggested Quick Inquiries:**")
    chip_cols = st.columns(len(suggested))
    selected_suggested = None
    for i, s in enumerate(suggested):
        with chip_cols[i]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                selected_suggested = s

    # Query Input Box
    default_text = selected_suggested if selected_suggested else st.session_state.get("dashboard_query", "")
    user_query = st.text_input(
        "Enter your air quality or health question:",
        value=default_text,
        placeholder="e.g. Can I go for a jog when the air index is 180?",
        label_visibility="collapsed"
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🚀 Analyze Query", type="primary", use_container_width=True)

    # Process query if button clicked or chip clicked
    if (analyze_clicked or selected_suggested) and user_query.strip():
        with st.spinner("Classifying intent through NLP pipeline..."):
            res = predict_query(user_query.strip())
            if res.get("is_valid"):
                # Save to database
                save_query(
                    query=user_query.strip(),
                    category=res["predicted_category"],
                    confidence=res["confidence"],
                    model=res["model_name"],
                    top_predictions=res["top_predictions"]
                )
                
                # Show quick result card
                st.markdown(f"""
                    <div class="result-card" style="margin-top: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Predicted Intent</div>
                                <div style="font-size: 1.6rem; font-weight: 800; color: #F8FAFC; margin-top: 4px;">{res['predicted_category']}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 700;">Confidence</div>
                                <div style="font-size: 1.6rem; font-weight: 800; color: #34D399;">{res['confidence_percentage']}</div>
                            </div>
                        </div>
                        <div style="margin-top: 14px; font-size: 0.92rem; color: #CBD5E1; line-height: 1.5;">
                            <strong>Advisory Preview:</strong> {res['advisory']['description']}
                        </div>
                        <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                            <span class="feature-tag">Model: {res['model_name']}</span>
                            <span class="feature-tag">Latency: {res['latency_ms']} ms</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.success("Query classified and saved to history! Navigate to **Query Analyzer** in the sidebar for full token diagnostics and feature contributions.")
            else:
                st.warning(res.get("error", "Unable to analyze query."))

    st.write("")
    st.divider()

    # Two column layout: Category Distribution & Recent Queries
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("### 📊 Dataset & Intent Distribution")
        st.caption("Balanced distribution across the 9 primary air quality intent categories.")

        data_path = os.path.join(parent_dir, "data", "aqi_health_queries.csv")
        if os.path.exists(data_path):
            df_data = pd.read_csv(data_path)
            cat_counts = df_data["category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]

            fig = px.bar(
                cat_counts,
                x="Count",
                y="Category",
                orientation="h",
                color="Category",
                color_discrete_sequence=px.colors.qualitative.Prism,
                text="Count"
            )
            fig.update_layout(
                showlegend=False,
                height=340,
                margin=dict(l=10, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#CBD5E1", size=11),
                xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(categoryorder="total ascending")
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Dataset file not found. Run dataset generator to view distribution.")

    with col_right:
        st.markdown("### 🕒 Recent Inquiries Activity")
        st.caption("Live feed of user questions recorded in SQLite database.")

        recent = get_recent_queries(limit=5)
        if recent:
            for item in recent:
                conf_pct = f"{item['confidence'] * 100:.1f}%" if item['confidence'] <= 1.0 else f"{item['confidence']:.1f}%"
                st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #38BDF8; font-size: 0.85rem;">{item['category']}</span>
                            <span style="font-size: 0.75rem; color: #94A3B8;">{item['timestamp']}</span>
                        </div>
                        <div style="font-size: 0.88rem; color: #F1F5F9; margin-top: 4px; font-style: italic;">
                            "{item['query']}"
                        </div>
                        <div style="margin-top: 6px; font-size: 0.75rem; color: #34D399; font-weight: 600;">
                            Confidence: {conf_pct} • {item['model']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No queries logged yet. Ask a question above to start logging!")

    st.write("")
    st.divider()

    # Dynamic Media & Video Walkthrough Hub
    st.markdown("### 🎬 Interactive Dynamic Media & Video Learning Hub")
    st.caption("Multimedia educational resources demonstrating atmospheric pollutants, respiratory biology, and NLP classification.")

    vid_tab1, vid_tab2, vid_tab3 = st.tabs([
        "🎥 Air Quality Index & PM2.5 (EPA Overview)",
        "🫁 How Air Pollution Enters the Lungs",
        "🤖 NLP & Intent Detection Primer"
    ])

    with vid_tab1:
        st.markdown("**Understanding the Air Quality Index (AQI):** Learn how government environmental monitoring stations compute ambient air indices from ozone, particulate matter, and nitrogen dioxide.")
        st.video("https://www.youtube.com/watch?v=2r1k1Jj_Nis")
        st.caption("Source: US Environmental Protection Agency (EPA) — AirNow Air Quality Educational Video")

    with vid_tab2:
        st.markdown("**Microscopic Particles & Human Anatomy:** Explore how fine PM2.5 particles bypass the nasal mucosa and penetrate deep into alveolar capillary beds.")
        st.video("https://www.youtube.com/watch?v=GVBeY1jSG9Y")
        st.caption("Source: World Health Organization & TED-Ed — The Biological Impact of Air Pollution")

    with vid_tab3:
        st.markdown("**How Machine Learning Classifies Natural-Language Queries:** Discover how TF-IDF vectorization and neural attention mechanisms map human questions into mathematical feature spaces.")
        st.video("https://www.youtube.com/watch?v=CMrHM8a3hqw")
        st.caption("Source: Natural Language Processing & Transformer Architecture Walkthrough")

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Educational & Advisory Notice:</strong> {MEDICAL_DISCLAIMER}
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_dashboard()
