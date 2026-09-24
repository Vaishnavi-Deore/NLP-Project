"""
Page 5 — Model Comparison
Side-by-side empirical comparison of Baseline (TF-IDF + Logistic Regression),
Linear SVM, Random Forest, and Advanced Deep NLP / BERT models.
Features real performance metrics, comparison charts, and on-demand retraining trigger.
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

from src.train import train_and_evaluate
from src.advisory import MEDICAL_DISCLAIMER

def render_model_comparison():
    st.markdown("""
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ⚖️ Model Architecture Comparison
            </h1>
            <p style="font-size: 1rem; color: #94A3B8; margin-top: 6px;">
                Empirical comparison between the linear baseline (TF-IDF + Logistic Regression), maximum-margin classifier (Linear SVM), ensemble methods, and advanced neural sequence architectures.
            </p>
        </div>
    """, unsafe_allow_html=True)

    metrics_file = os.path.join(parent_dir, "models", "metrics.json")
    if not os.path.exists(metrics_file):
        st.warning("Metrics file not found. Please click 'Retrain Models' below.")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models and calculating test metrics..."):
                train_and_evaluate()
            st.rerun()
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # Retrain button bar
    c_head, c_btn = st.columns([3, 1])
    with c_btn:
        if st.button("🔄 Retrain All Models", type="primary", use_container_width=True):
            with st.spinner("Retraining classifiers and evaluating on stratified test split..."):
                train_and_evaluate()
                from src.train_advanced import train_advanced_model
                train_advanced_model()
            st.success("Retraining complete! Metrics refreshed.")
            st.rerun()

    # Build Comparison Summary Table
    comparison_rows = []
    for model_name, m in metrics_data.items():
        comparison_rows.append({
            "Model Architecture": model_name,
            "Accuracy": m["accuracy"],
            "Precision (W)": m["precision_weighted"],
            "Recall (W)": m["recall_weighted"],
            "F1-Score (W)": m["f1_weighted"],
            "Macro F1": m["f1_macro"],
            "Test Samples": m["total_samples"]
        })

    df_comp = pd.DataFrame(comparison_rows).sort_values("F1-Score (W)", ascending=False)

    st.markdown("### 🏆 Comprehensive Performance Leaderboard")
    st.caption("Models ranked by held-out test set F1-Score (weighted):")

    # Stylized Leaderboard Table
    st.dataframe(
        df_comp.style.format({
            "Accuracy": "{:.2%}",
            "Precision (W)": "{:.2%}",
            "Recall (W)": "{:.2%}",
            "F1-Score (W)": "{:.2%}",
            "Macro F1": "{:.2%}",
            "Test Samples": "{:d}"
        }).highlight_max(subset=["Accuracy", "Precision (W)", "Recall (W)", "F1-Score (W)", "Macro F1"], color="#1E3A8A"),
        use_container_width=True,
        hide_index=True
    )

    st.write("")
    st.divider()

    # Comparison Grouped Bar Chart
    st.markdown("### 📊 Metrics Comparison Across Architectures")
    
    df_chart = df_comp.melt(
        id_vars=["Model Architecture"],
        value_vars=["Accuracy", "Precision (W)", "Recall (W)", "F1-Score (W)"],
        var_name="Evaluation Metric",
        value_name="Score"
    )

    fig_bar = px.bar(
        df_chart,
        x="Model Architecture",
        y="Score",
        color="Evaluation Metric",
        barmode="group",
        text_auto=".1%",
        color_discrete_sequence=["#38BDF8", "#34D399", "#FBBF24", "#A78BFA"]
    )
    fig_bar.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", size=11),
        yaxis=dict(range=[0, 1.1], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(tickangle=-15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.write("")
    st.divider()

    # Per-Class F1 Score Comparison Across Models
    st.markdown("### 🎯 Per-Class F1 Score Comparison")
    st.caption("How each model performs on specific intent categories:")

    per_class_comp_data = []
    labels = list(metrics_data.values())[0]["labels"]

    for label in labels:
        row = {"Category": label}
        for model_name, m in metrics_data.items():
            f1_val = m.get("per_class", {}).get(label, {}).get("f1_score", 0.0)
            row[model_name] = f1_val
        per_class_comp_data.append(row)

    df_pclass = pd.DataFrame(per_class_comp_data)
    
    fig_pclass = px.bar(
        df_pclass.melt(id_vars=["Category"], var_name="Model", value_name="F1 Score"),
        x="Category",
        y="F1 Score",
        color="Model",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_pclass.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1", size=10),
        yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
        xaxis=dict(tickangle=-35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_pclass, use_container_width=True)

    st.write("")
    st.divider()

    # Architectural Trade-offs Analysis
    st.markdown("### 🧠 Architectural Trade-Off Analysis")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <h4 style="color: #38BDF8; margin-top: 0;">Baseline: TF-IDF + Logistic Regression</h4>
                <ul style="font-size: 0.88rem; color: #CBD5E1; padding-left: 18px; line-height: 1.6;">
                    <li><strong>Latency:</strong> Ultra-fast (< 2 ms)</li>
                    <li><strong>Explainability:</strong> High (direct linear feature weights)</li>
                    <li><strong>Footprint:</strong> Extremely lightweight (~1.5 MB)</li>
                    <li><strong>Recommendation:</strong> Ideal production baseline for low-resource deployment.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <h4 style="color: #34D399; margin-top: 0;">Calibrated Linear SVM</h4>
                <ul style="font-size: 0.88rem; color: #CBD5E1; padding-left: 18px; line-height: 1.6;">
                    <li><strong>Latency:</strong> Fast (~3 ms)</li>
                    <li><strong>Calibration:</strong> Platt sigmoid scaling (CalibratedClassifierCV)</li>
                    <li><strong>Margin:</strong> Maximizes geometric margin between query classes</li>
                    <li><strong>Recommendation:</strong> Strongest linear generalization on high-dimensional n-grams.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; height: 100%;">
                <h4 style="color: #A78BFA; margin-top: 0;">Advanced Deep NLP / Transformers</h4>
                <ul style="font-size: 0.88rem; color: #CBD5E1; padding-left: 18px; line-height: 1.6;">
                    <li><strong>Latency:</strong> Moderate (~12-25 ms)</li>
                    <li><strong>Representation:</strong> Learned self-attention and dense embeddings</li>
                    <li><strong>Context:</strong> Captures non-linear term relationships and word order</li>
                    <li><strong>Recommendation:</strong> Best for complex, conversational, and paraphrased queries.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Model Evaluation Notice:</strong> {MEDICAL_DISCLAIMER}
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_model_comparison()
