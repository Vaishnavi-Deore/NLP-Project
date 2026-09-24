"""
Page 6 — Query History
Stores, searches, filters, sorts, and manages historical query classifications from SQLite.
Supports individual deletion, full log clearance, and CSV exports.
"""

import streamlit as st
import pandas as pd
import json
import os
import sys

# Path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from database.database import get_all_queries, delete_query, clear_all_queries
from src.advisory import MEDICAL_DISCLAIMER

CATEGORIES = [
    "All",
    "AQI Information",
    "Health Effects",
    "Symptoms",
    "Precautions",
    "Masks and Protection",
    "Outdoor Activities",
    "Vulnerable Groups",
    "Respiratory Health",
    "Pollution Exposure",
]

def render_history():
    st.markdown("""
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                📜 Query History & Audit Log
            </h1>
            <p style="font-size: 1rem; color: #94A3B8; margin-top: 6px;">
                Persistent audit trail of user inquiries, predicted categories, confidence probabilities, and models used, stored in SQLite (<code>aqi_sense.db</code>).
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Filter & Search Controls Bar
    c_search, c_filter, c_sort, c_order = st.columns([1.5, 1.2, 1.0, 0.8])
    with c_search:
        search_term = st.text_input("🔍 Search Query Text:", placeholder="Filter by keywords...", key="hist_search")
    with c_filter:
        cat_filter = st.selectbox("Category Filter:", CATEGORIES, index=0, key="hist_cat")
    with c_sort:
        sort_by = st.selectbox("Sort By:", ["timestamp", "confidence", "category"], format_func=lambda x: x.capitalize(), key="hist_sort")
    with c_order:
        sort_order = st.selectbox("Order:", ["Descending", "Ascending"], key="hist_order")

    ascending = (sort_order == "Ascending")

    # Fetch records from database
    records = get_all_queries(
        search_term=search_term,
        category_filter=cat_filter,
        sort_by=sort_by,
        ascending=ascending
    )

    # Action Toolbar
    st.write("")
    c_count, c_clear, c_export = st.columns([2, 1, 1])
    with c_count:
        st.markdown(f"**Found {len(records)} query record(s)**")

    with c_export:
        if records:
            export_df = pd.DataFrame(records)
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name="aqi_sense_query_history.csv",
                mime="text/csv",
                use_container_width=True
            )

    with c_clear:
        if records:
            with st.popover("🗑️ Clear All Logs"):
                st.write("Are you sure you want to permanently clear all history records?")
                if st.button("Confirm Clear History", type="primary"):
                    cleared_count = clear_all_queries()
                    st.success(f"Cleared {cleared_count} records.")
                    st.rerun()

    st.write("")

    if not records:
        st.info("No queries found matching the selected filter criteria. Try searching for other terms or run a query in the Query Analyzer.")
        return

    # Render History Cards / Table
    for r in records:
        conf_val = r["confidence"]
        conf_display = f"{conf_val * 100:.1f}%" if conf_val <= 1.0 else f"{conf_val:.1f}%"

        # Card container
        with st.container():
            col_main, col_del = st.columns([5, 0.5])
            with col_main:
                st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 14px 18px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                            <span style="font-weight: 700; color: #38BDF8; font-size: 0.95rem;">{r['category']}</span>
                            <span style="font-size: 0.78rem; color: #94A3B8;">ID #{r['id']} • {r['timestamp']}</span>
                        </div>
                        <div style="font-size: 1rem; color: #F8FAFC; margin-top: 6px; font-weight: 500;">
                            "{r['query']}"
                        </div>
                        <div style="margin-top: 8px; display: flex; gap: 10px; align-items: center; font-size: 0.8rem; color: #CBD5E1;">
                            <span style="background: rgba(16, 185, 129, 0.15); color: #34D399; padding: 2px 8px; border-radius: 4px; font-weight: 600;">
                                Confidence: {conf_display}
                            </span>
                            <span>Model: <strong>{r['model']}</strong></span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col_del:
                st.write("")
                if st.button("❌", key=f"del_{r['id']}", help="Delete this log record"):
                    delete_query(r["id"])
                    st.rerun()

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Data Privacy & Advisory Notice:</strong> {MEDICAL_DISCLAIMER}
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_history()
