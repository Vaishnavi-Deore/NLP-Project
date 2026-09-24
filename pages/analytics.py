"""
Page 4 — Analytics (Machine Learning Performance Dashboard)
Calculates and displays genuine metrics computed from the stratified test dataset:
Accuracy, Precision, Recall, F1, Confusion Matrix, Per-Category Breakdowns,
and Confidence Distributions without any hardcoded values.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
import sys

# Path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.advisory import MEDICAL_DISCLAIMER

def render_analytics():
    st.markdown("""
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                📈 ML Evaluation & Analytics Dashboard
            </h1>
            <p style="font-size: 1rem; color: #94A3B8; margin-top: 6px;">
                Empirical evaluation metrics calculated directly on the held-out stratified test dataset (20% split). No hardcoded metrics.
            </p>
        </div>
    """, unsafe_allow_html=True)

    metrics_file = os.path.join(parent_dir, "models", "metrics.json")
    preds_file = os.path.join(parent_dir, "models", "test_predictions.csv")

    if not os.path.exists(metrics_file):
        st.warning("Metrics file not found. Please train models first using `python src/train.py`.")
        return

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    available_models = list(metrics_data.keys())

    # Model selector for viewing specific evaluation details
    c_sel, c_info = st.columns([1.5, 2.5])
    with c_sel:
        active_model_name = st.selectbox(
            "Select Evaluated Model:",
            available_models,
            index=0,
            help="Switch between models to view individual test set confusion matrix and per-class metrics."
        )

    active_metrics = metrics_data[active_model_name]

    # Primary KPI Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #38BDF8;">
                <div class="kpi-title">Accuracy</div>
                <div class="kpi-value" style="color: #38BDF8;">{active_metrics['accuracy'] * 100:.1f}%</div>
                <div class="kpi-subtext">Overall correct ratio</div>
            </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #34D399;">
                <div class="kpi-title">Precision (W)</div>
                <div class="kpi-value" style="color: #34D399;">{active_metrics['precision_weighted'] * 100:.1f}%</div>
                <div class="kpi-subtext">Weighted precision</div>
            </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #FBBF24;">
                <div class="kpi-title">Recall (W)</div>
                <div class="kpi-value" style="color: #FBBF24;">{active_metrics['recall_weighted'] * 100:.1f}%</div>
                <div class="kpi-subtext">Weighted recall</div>
            </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #A78BFA;">
                <div class="kpi-title">F1 Score (W)</div>
                <div class="kpi-value" style="color: #A78BFA;">{active_metrics['f1_weighted'] * 100:.1f}%</div>
                <div class="kpi-subtext">Harmonic mean</div>
            </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #F472B6;">
                <div class="kpi-title">Test Samples</div>
                <div class="kpi-value" style="color: #F472B6;">{active_metrics['total_samples']}</div>
                <div class="kpi-subtext">Stratified split</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # 1. Confusion Matrix Heatmap
    st.markdown(f"### 🔲 Confusion Matrix — {active_model_name}")
    st.caption("Diagonal elements represent correct predictions; off-diagonal elements illustrate confusion between categories.")

    labels = active_metrics["labels"]
    cm = np.array(active_metrics["confusion_matrix"])

    # Create Plotly Heatmap
    fig_cm = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Category", y="True Category", color="Count"),
        text_auto=True
    )
    fig_cm.update_layout(
        height=520,
        margin=dict(l=40, r=40, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0", size=11),
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.write("")
    st.divider()

    # Two columns: Per-category metrics bar chart & Confidence distribution
    col_chart1, col_chart2 = st.columns([1.2, 1.0])

    with col_chart1:
        st.markdown("### 📊 Precision, Recall & F1 by Category")
        st.caption("Class-level performance breakdown across all 9 intent classes.")

        per_class_dict = active_metrics["per_class"]
        per_class_rows = []
        for cat, scores in per_class_dict.items():
            per_class_rows.append({
                "Category": cat,
                "Precision": scores["precision"],
                "Recall": scores["recall"],
                "F1 Score": scores["f1_score"],
                "Support": scores["support"]
            })
        df_per_class = pd.DataFrame(per_class_rows)

        # Melt for grouped bar chart
        df_melted = df_per_class.melt(
            id_vars=["Category", "Support"],
            value_vars=["Precision", "Recall", "F1 Score"],
            var_name="Metric",
            value_name="Score"
        )

        fig_pc = px.bar(
            df_melted,
            x="Category",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#38BDF8", "#34D399", "#A78BFA"]
        )
        fig_pc.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=20, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", size=10),
            yaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(tickangle=-35),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pc, use_container_width=True)

    with col_chart2:
        st.markdown("### 📉 Confidence Distribution")
        st.caption("Distribution of predicted softmax probabilities on test cases.")

        if os.path.exists(preds_file):
            df_preds = pd.read_csv(preds_file)
            conf_cols = [c for c in df_preds.columns if c.startswith("conf_")]
            if conf_cols:
                # Pick corresponding confidence column
                chosen_conf_col = conf_cols[0]
                for col in conf_cols:
                    if "logistic" in active_model_name.lower() and "logistic" in col:
                        chosen_conf_col = col
                    elif "svm" in active_model_name.lower() and "svm" in col:
                        chosen_conf_col = col
                    elif "forest" in active_model_name.lower() and "forest" in col:
                        chosen_conf_col = col
                    elif "neural" in active_model_name.lower() and "deep" in col:
                        chosen_conf_col = col

                fig_hist = px.histogram(
                    df_preds,
                    x=chosen_conf_col,
                    nbins=20,
                    color_discrete_sequence=["#10B981"],
                    opacity=0.8,
                    labels={chosen_conf_col: "Confidence Score"}
                )
                fig_hist.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=20, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#CBD5E1", size=11),
                    xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="rgba(255,255,255,0.08)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.08)")
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Confidence scores column not found in test predictions.")
        else:
            st.info("Test predictions CSV not found.")

    st.write("")
    st.divider()

    # Per-Class Table Breakdown
    st.markdown("### 📋 Detailed Per-Class Classification Report")
    st.dataframe(
        df_per_class.style.format({
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Academic & Performance Notice:</strong> {MEDICAL_DISCLAIMER}
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_analytics()
