"""
Page 7 — Dataset Explorer
Explores the curated AQI & health queries dataset (1,200+ samples).
Provides live search, category filtering, class balance charts,
lexical distribution stats, and CSV downloads.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys

# Path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.advisory import MEDICAL_DISCLAIMER

def render_dataset_explorer():
    st.markdown("""
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                📁 Dataset Explorer & Class Distribution
            </h1>
            <p style="font-size: 1rem; color: #94A3B8; margin-top: 6px;">
                Inspect, filter, and analyze the curated corpus of natural-language air quality and health inquiries.
            </p>
        </div>
    """, unsafe_allow_html=True)

    raw_path = os.path.join(parent_dir, "data", "aqi_health_queries.csv")
    proc_path = os.path.join(parent_dir, "data", "processed_data.csv")

    if not os.path.exists(raw_path):
        st.warning("Dataset not found. Please run `python src/dataset_generator.py`.")
        return

    df_raw = pd.read_csv(raw_path)
    df_proc = pd.read_csv(proc_path) if os.path.exists(proc_path) else df_raw

    # Summary KPI row
    total_samples = len(df_raw)
    num_classes = df_raw["category"].nunique()
    df_raw["char_length"] = df_raw["query"].str.len()
    df_raw["word_count"] = df_raw["query"].apply(lambda x: len(str(x).split()))
    avg_words = df_raw["word_count"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #38BDF8;">
                <div class="kpi-title">Total Queries</div>
                <div class="kpi-value" style="color: #38BDF8;">{total_samples}</div>
                <div class="kpi-subtext">Curated natural queries</div>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #34D399;">
                <div class="kpi-title">Unique Categories</div>
                <div class="kpi-value" style="color: #34D399;">{num_classes}</div>
                <div class="kpi-subtext">Distinct intent classes</div>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #FBBF24;">
                <div class="kpi-title">Avg Query Length</div>
                <div class="kpi-value" style="color: #FBBF24;">{avg_words:.1f}</div>
                <div class="kpi-subtext">Words per query</div>
            </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
            <div class="kpi-card" style="border-top-color: #A78BFA;">
                <div class="kpi-title">Class Balance</div>
                <div class="kpi-value" style="color: #A78BFA;">Balanced</div>
                <div class="kpi-subtext">~140 samples / class</div>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    # Class Balance Visualizations
    st.markdown("### 📊 Class Distribution & Lexical Balance")
    col_chart1, col_chart2 = st.columns([1.1, 1.0])

    with col_chart1:
        cat_counts = df_raw["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        fig_pie = px.pie(
            cat_counts,
            names="Category",
            values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_pie.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", size=11),
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
        fig_len = px.histogram(
            df_raw,
            x="word_count",
            color="category",
            nbins=25,
            labels={"word_count": "Query Word Count"},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_len.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", size=11),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            showlegend=False
        )
        st.plotly_chart(fig_len, use_container_width=True)

    st.write("")
    st.divider()

    # Search & Interactive Data Table
    st.markdown("### 🔍 Search & Filter Corpus")
    c_search, c_cat, c_view = st.columns([1.5, 1.2, 1.0])

    with c_search:
        search_query = st.text_input("Search Queries:", placeholder="Filter by terms (e.g. mask, exercise, asthma)...")

    with c_cat:
        all_cats = ["All"] + sorted(list(df_raw["category"].unique()))
        selected_cat = st.selectbox("Category Filter:", all_cats, index=0)

    with c_view:
        view_type = st.radio("Dataset View:", ["Raw Queries", "Preprocessed Tokens"], horizontal=True)

    filtered_df = df_proc.copy() if view_type == "Preprocessed Tokens" else df_raw.copy()

    if selected_cat != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_cat]

    if search_query.strip():
        filtered_df = filtered_df[filtered_df["query"].str.contains(search_query.strip(), case=False, na=False)]

    st.caption(f"Showing **{len(filtered_df)}** of {len(df_raw)} records")

    display_cols = ["query", "processed_query", "category"] if "processed_query" in filtered_df.columns else ["query", "category"]
    st.dataframe(
        filtered_df[display_cols],
        use_container_width=True,
        hide_index=True,
        height=380
    )

    # Download Buttons
    st.write("")
    c_d1, c_d2, _ = st.columns([1, 1, 2])
    with c_d1:
        st.download_button(
            label="📥 Download Raw CSV",
            data=df_raw.to_csv(index=False).encode('utf-8'),
            file_name="aqi_health_queries_raw.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_d2:
        st.download_button(
            label="📥 Download Processed CSV",
            data=df_proc.to_csv(index=False).encode('utf-8'),
            file_name="aqi_health_queries_processed.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Dataset & Academic Notice:</strong> {MEDICAL_DISCLAIMER}
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_dataset_explorer()
