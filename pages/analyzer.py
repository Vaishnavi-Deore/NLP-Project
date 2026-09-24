"""
Page 2 — Query Analyzer (Core NLP Feature)
Allows natural-language query input, model selection, full step-by-step pipeline inspection,
predicted intent with visual confidence progress bar, top-3 probabilities,
feature explainability, and retrieved advisory guidance.
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

from src.predict import predict_query
from database.database import save_query
from src.advisory import MEDICAL_DISCLAIMER

CATEGORY_COLOR_MAP = {
    "AQI Information": "#3B82F6",
    "Health Effects": "#EF4444",
    "Symptoms": "#F59E0B",
    "Precautions": "#10B981",
    "Masks and Protection": "#8B5CF6",
    "Outdoor Activities": "#EC4899",
    "Vulnerable Groups": "#06B6D4",
    "Respiratory Health": "#F97316",
    "Pollution Exposure": "#6366F1",
}

def render_analyzer():
    st.markdown("""
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                🔬 Query Analyzer & Intent Classifier
            </h1>
            <p style="font-size: 1rem; color: #94A3B8; margin-top: 6px;">
                Enter any air quality or health-related query to trace the end-to-end NLP preprocessing pipeline, feature extraction, and ML classifier predictions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Model and settings controls
    col_model, col_sample = st.columns([1.5, 2.5])
    with col_model:
        selected_model = st.selectbox(
            "Select Classification Model:",
            [
                "TF-IDF + Logistic Regression (Baseline)",
                "TF-IDF + Linear Support Vector Machine",
                "TF-IDF + Random Forest Classifier",
                "Advanced NLP (Deep Self-Attention / PyTorch)"
            ],
            index=0,
            help="Choose the underlying machine learning or deep learning classifier."
        )

    # Preset query chips
    sample_queries = [
        "Can I exercise outside when the AQI is high?",
        "Does an N95 mask protect against toxic air pollution?",
        "Why do my eyes burn and throat sting from smog?",
        "What does an AQI of 180 indicate for healthy people?",
        "Is severe air pollution harmful to infants and seniors?",
        "Can high PM2.5 trigger a life-threatening asthma attack?",
        "What happens after 8 hours of continuous exposure to toxic air?",
        "Should I keep all windows closed and use an air purifier at home?",
        "How does dirty air impact cardiovascular and blood pressure health?"
    ]

    with col_sample:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #94A3B8; margin-bottom: 6px;'>Or Pick a Benchmark Example:</div>", unsafe_allow_html=True)
        selected_example = st.selectbox(
            "Load Example Query:",
            ["-- Select an Example Query --"] + sample_queries,
            index=0,
            label_visibility="collapsed"
        )

    # Query Input Box
    default_text = ""
    if selected_example != "-- Select an Example Query --":
        default_text = selected_example
    elif "analyzer_query" in st.session_state:
        default_text = st.session_state["analyzer_query"]

    query_input = st.text_area(
        "Natural-Language Query Input:",
        value=default_text,
        placeholder="e.g. Can I run outside when the AQI is 180?",
        height=100
    )

    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
    with btn_col1:
        run_analysis = st.button("⚡ Analyze Query", type="primary", use_container_width=True)
    with btn_col2:
        clear_input = st.button("🗑️ Clear", use_container_width=True)

    if clear_input:
        st.session_state["analyzer_query"] = ""
        st.rerun()

    if run_analysis or default_text:
        query_text = query_input.strip() if query_input else default_text
        if not query_text:
            st.warning("Please enter a query or select an example above before running analysis.")
            return

        with st.spinner("Processing query across NLP pipeline..."):
            res = predict_query(query_text, model_type=selected_model)

        if not res.get("is_valid"):
            st.error(res.get("error", "Error analyzing query."))
            return

        # Save to database
        save_query(
            query=query_text,
            category=res["predicted_category"],
            confidence=res["confidence"],
            model=res["model_name"],
            top_predictions=res["top_predictions"]
        )

        category = res["predicted_category"]
        conf = res["confidence"]
        badge_color = CATEGORY_COLOR_MAP.get(category, "#3B82F6")

        st.write("")
        # Top Result Hero Card
        st.markdown(f"""
            <div class="result-card" style="border-left: 6px solid {badge_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                    <div>
                        <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                            Identified Intent Category
                        </div>
                        <div style="display: flex; align-items: center; gap: 12px; margin-top: 6px;">
                            <span class="badge-category" style="background-color: {badge_color};">
                                {category}
                            </span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">
                            Classification Confidence
                        </div>
                        <div style="font-size: 2rem; font-weight: 800; color: #34D399; margin-top: 2px;">
                            {res['confidence_percentage']}
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Visual Confidence Progress Bar
        st.markdown("**Confidence Level:**")
        conf_clamped = min(max(conf, 0.0), 1.0)
        st.progress(conf_clamped)

        if res["is_low_confidence"]:
            st.warning("⚠️ **Low Confidence Prediction:** The model's confidence is below 35%. This query may contain out-of-domain terms or ambiguous phrasing.")

        st.write("")

        # Two Tabs: 1. Model Predictions & Explainability, 2. NLP Pipeline Inspector
        tab1, tab2, tab3 = st.tabs(["📊 Predictions & Explainability", "🔍 NLP Pipeline Diagnostics", "💡 Advisory Guidance"])

        with tab1:
            col_p1, col_p2 = st.columns([1.2, 1.0])

            with col_p1:
                st.markdown("#### 🎯 Top-3 Predicted Probabilities")
                top_preds = res["top_predictions"][:3]
                top_df = pd.DataFrame(top_preds)

                # Horizontal bar chart for top predictions
                fig_top = px.bar(
                    top_df,
                    x="probability",
                    y="category",
                    orientation="h",
                    text="percentage",
                    color="category",
                    color_discrete_map=CATEGORY_COLOR_MAP
                )
                fig_top.update_layout(
                    showlegend=False,
                    height=200,
                    margin=dict(l=10, r=20, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#CBD5E1", size=12),
                    xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(categoryorder="total ascending")
                )
                fig_top.update_traces(textposition="outside")
                st.plotly_chart(fig_top, use_container_width=True)

                # Table breakdown
                st.dataframe(
                    top_df.rename(columns={"category": "Intent Category", "probability": "Probability", "percentage": "Confidence"}),
                    use_container_width=True,
                    hide_index=True
                )

            with col_p2:
                st.markdown("#### 🧠 Explainability & Term Contributions")
                explain_items = res.get("explainability", [])
                if explain_items:
                    st.caption("Key TF-IDF features that contributed positively toward this prediction:")
                    terms_df = pd.DataFrame(explain_items)
                    fig_terms = px.bar(
                        terms_df,
                        x="contribution_score",
                        y="term",
                        orientation="h",
                        color="contribution_score",
                        color_continuous_scale="Blues",
                        text="contribution_score"
                    )
                    fig_terms.update_layout(
                        showlegend=False,
                        height=240,
                        margin=dict(l=10, r=20, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#CBD5E1", size=11),
                        yaxis=dict(categoryorder="total ascending"),
                        coloraxis_showscale=False
                    )
                    fig_terms.update_traces(textposition="outside")
                    st.plotly_chart(fig_terms, use_container_width=True)
                else:
                    st.info("Feature weight breakdown is available when using linear models (TF-IDF + Logistic Regression). Deep neural and ensemble models use distributed attention / tree representations.")

                # Metadata card
                st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px 16px; margin-top: 10px;">
                        <div style="font-size: 0.8rem; color: #94A3B8;">QUERY METRICS</div>
                        <div style="display: flex; gap: 20px; margin-top: 6px;">
                            <div><strong>Characters:</strong> {len(query_text)}</div>
                            <div><strong>Raw Words:</strong> {len(query_text.split())}</div>
                            <div><strong>Latency:</strong> {res['latency_ms']} ms</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with tab2:
            st.markdown("#### ⚙️ Step-by-Step NLP Preprocessing Pipeline")
            p_info = res["pipeline_info"]

            st.markdown(f"""
                <div class="pipeline-step">
                    <div class="pipeline-step-title">Step 1 — Raw User Input</div>
                    <div class="pipeline-step-content mono">"{p_info['raw_query']}"</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-title">Step 2 — Text Cleaning & Domain Normalization</div>
                    <div class="pipeline-step-content mono">{p_info['cleaned_text']}</div>
                    <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">Lowercased, punctuation stripped, domain tokens (e.g. PM2.5 → pm25, N95 → n95) preserved.</div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-title">Step 3 — Tokenization (NLTK)</div>
                    <div class="pipeline-step-content">
                        {' '.join([f'<span class="feature-tag">{t}</span>' for t in p_info['tokens']])}
                    </div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-title">Step 4 — Stopword Removal</div>
                    <div class="pipeline-step-content">
                        {' '.join([f'<span class="feature-tag" style="border-color:#10B981; color:#6EE7B7;">{t}</span>' for t in p_info['tokens_no_stopwords']])}
                    </div>
                </div>
                <div class="pipeline-step">
                    <div class="pipeline-step-title">Step 5 — Lemmatization (WordNet) & Final Representation</div>
                    <div class="pipeline-step-content mono" style="color: #FCD34D;">"{p_info['processed_text']}"</div>
                </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("#### 📋 Advisory & Public Health Information")
            adv = res["advisory"]

            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
                    <h3 style="margin-top: 0; color: #38BDF8;">{adv['title']}</h3>
                    <p style="font-size: 0.95rem; color: #CBD5E1; line-height: 1.6;">{adv['description']}</p>
                </div>
            """, unsafe_allow_html=True)

            col_adv1, col_adv2 = st.columns(2)
            with col_adv1:
                st.markdown("**📌 Key Scientific Takeaways:**")
                for item in adv["key_takeaways"]:
                    st.markdown(f"- {item}")

                st.markdown("**✅ Recommended Actions:**")
                for item in adv["suggested_actions"]:
                    st.markdown(f"- {item}")

            with col_adv2:
                st.markdown("**👍 Do's:**")
                for d in adv["dos_and_donts"]["dos"]:
                    st.markdown(f"- 🟢 {d}")

                st.write("")
                st.markdown("**🚫 Don'ts:**")
                for d in adv["dos_and_donts"]["donts"]:
                    st.markdown(f"- 🔴 {d}")

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Medical & Advisory Disclaimer:</strong> {MEDICAL_DISCLAIMER}
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_analyzer()
