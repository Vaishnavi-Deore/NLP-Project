"""
Page 8 — About Page
Academic project documentation detailing:
- Project Title & Problem Statement
- Objectives & Motivation
- Complete End-to-End NLP Architecture
- Mathematical Formulation of ML & Deep Learning Classifiers
- The 9 Classification Categories Taxonomy
- Technology Stack
- Limitations, Ethical Considerations, and Disclaimers
"""

import streamlit as st
import os
import sys

# Path resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.advisory import MEDICAL_DISCLAIMER

def render_about():
    st.markdown("""
        <div>
            <div style="font-size: 0.85rem; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.05em;">
                Academic Mini-Project Documentation
            </div>
            <h1 style="margin: 4px 0 0 0; font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                AQI-Sense — AQI Health Query Classifier
            </h1>
            <div style="font-size: 1.05rem; color: #E2E8F0; font-weight: 600; margin-top: 6px;">
                An NLP-Based System for Classifying Air Quality and Health-Related Queries
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Tabbed Project Sections
    t1, t2, t3, t4, t5 = st.tabs([
        "🎯 Overview & Objectives",
        "⚙️ NLP Pipeline Architecture",
        "🧠 Machine Learning Models",
        "🏷️ Intent Taxonomy (9 Classes)",
        "💻 Tech Stack & Limitations"
    ])

    with t1:
        st.markdown("### 📌 Problem Statement")
        st.write("""
        Severe air pollution episodes—characterized by hazardous concentrations of fine particulate matter ($\text{PM}_{2.5}$), 
        ground-level ozone ($\text{O}_3$), and nitrogen oxides ($\text{NO}_x$)—pose major acute and chronic public health hazards. 
        During heavy smog days, citizens generate thousands of natural-language inquiries across digital channels regarding 
        outdoor safety, respirator efficacy, physical symptoms, and protective protocols.
        
        However, users often struggle to find immediate, context-appropriate guidance or confuse general ambient indices 
        with physiological symptoms. Existing search engines frequently surface uncurated or alarmist medical diagnoses.
        """)

        st.markdown("### 🎯 Project Objectives")
        st.markdown("""
        1. **Intent Classification:** Automatically categorize natural-language user queries into 9 well-defined air quality and health intent classes.
        2. **Transparent Pipeline Inspection:** Trace and visualize the multi-stage NLP transformations (normalization, tokenization, stopword filtering, lemmatization).
        3. **Calibrated Confidence Scoring:** Deliver well-calibrated posterior probabilities ($P(C_k|X)$) and top-3 category rankings.
        4. **Educational Advisory Retrieval:** Couple identified query intent with validated public health recommendations.
        5. **Strict Medical Boundary:** Emphasize educational advisory and intent routing while explicitly disclaiming clinical diagnosis.
        """)

    with t2:
        st.markdown("### 🔄 End-to-End NLP Pipeline Flow")
        
        nlp_img_path = os.path.join(parent_dir, "assets", "nlp_pipeline.jpg")
        if os.path.exists(nlp_img_path):
            st.image(
                nlp_img_path,
                caption="AQI-Sense Natural Language Processing & Neural Query Classification Pipeline",
                use_container_width=True
            )

        st.markdown("""
        The system implements a modular, reproducible NLP preprocessing and classification pipeline:
        """)

        st.markdown("""
        ```
        [User Natural-Language Query]
                     │
                     ▼
        [Text Cleaning & Domain Token Normalization]
          - Lowercasing, regex symbol stripping
          - Air-quality token preservation (PM2.5 → pm25, N95 → n95)
                     │
                     ▼
        [NLTK Word Tokenization]
          - Word boundary splitting
                     │
                     ▼
        [Stopword Filtering]
          - Removal of non-informative syntactic tokens
                     │
                     ▼
        [WordNet Morphological Lemmatization]
          - Normalizing verb and noun inflections to base lemmas
                     │
                     ▼
        [TF-IDF Feature Extraction / Neural Embeddings]
          - Unigram + Bigram feature matrices with sublinear TF scaling
                     │
                     ▼
        [Multi-Class ML / Deep Learning Classifier]
          - Logistic Regression / Calibrated Linear SVM / PyTorch Self-Attention
                     │
                     ▼
        [Intent Prediction + Confidence Score + Top-3 Probabilities]
                     │
                     ▼
        [Educational Advisory Retrieval + SQLite Audit Logging]
        ```
        """)

        with st.expander("🎬 Watch Video Walkthrough: NLP Query Classification & Transformer Embeddings"):
            st.video("https://www.youtube.com/watch?v=CMrHM8a3hqw")
            st.caption("Educational tutorial: How vector embeddings and attention layers process linguistic semantics.")

    with t3:
        st.markdown("### 🧮 Mathematical Formulation of Classifiers")

        st.markdown("#### 1. Baseline: TF-IDF + Multinomial Logistic Regression")
        st.latex(r"""
        \text{TF-IDF}(t, d, D) = (1 + \log \text{tf}(t, d)) \times \log \left(\frac{1 + |D|}{1 + \text{df}(t, D)}\right)
        """)
        st.latex(r"""
        P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{j=1}^{K} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}
        """)
        st.write("""
        Logistic Regression optimizes the multinomial cross-entropy loss with $L_2$ regularization. Feature contributions 
        are computed directly from the dot product $w_{c,i} \cdot x_i$, enabling transparent explainability without surrogate approximations.
        """)

        st.markdown("#### 2. Calibrated Linear Support Vector Machine (Linear SVM)")
        st.latex(r"""
        \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{N} \xi_i \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0
        """)
        st.write("""
        Linear SVM maximizes the geometric margin between intent hyperplanes. Platt sigmoid calibration via 5-fold cross-validation 
        transforms uncalibrated margin distances into well-calibrated posterior probabilities.
        """)

        st.markdown("#### 3. Advanced NLP: Deep Self-Attention Classifier (PyTorch)")
        st.latex(r"""
        \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
        """)
        st.write("""
        A neural sequence classifier with learned word embeddings, Multi-Head Self-Attention layers, global sequence pooling, 
        and fully connected projection layers. This architecture captures contextual term interactions across arbitrary word spans.
        """)

    with t4:
        st.markdown("### 🏷️ The 9 Primary Classification Categories")

        categories_info = [
            ("1. AQI Information", "Numerical scale, EPA index categories, pollutant definitions, calculation methods.", "What does AQI 150 mean?"),
            ("2. Health Effects", "Systemic physiological impact, cardiovascular risks, stroke, long-term biological consequences.", "Can air pollution cause heart disease?"),
            ("3. Symptoms", "Physical irritations, burning eyes, coughing, headaches, dizziness, and throat dryness.", "Why do my eyes burn when smog is high?"),
            ("4. Precautions", "Indoor protection, HEPA air purifiers, sealing windows, ventilation protocols, daily habits.", "Should I keep windows closed during smog?"),
            ("5. Masks and Protection", "N95, KN95, KF94 respirators, facial seals, filtration mechanics, mask reuse.", "Does an N95 mask protect against PM2.5?"),
            ("6. Outdoor Activities", "Running, sports, cycling, children recess, exercise safety thresholds.", "Can I jog outdoors when AQI is 180?"),
            ("7. Vulnerable Groups", "Risks for infants, elderly seniors, pregnant women, immunocompromised individuals.", "Is severe smog dangerous for babies?"),
            ("8. Respiratory Health", "Asthma attacks, COPD exacerbations, bronchitis, bronchial spasms, inhaler adherence.", "Can pollution trigger acute asthma?"),
            ("9. Pollution Exposure", "Dosage, hours of exposure, highway proximity, occupational risks, cumulative toxicity.", "What happens after 8 hours of smog exposure?")
        ]

        for title, desc, eg in categories_info:
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                    <div style="font-weight: 700; color: #38BDF8; font-size: 0.95rem;">{title}</div>
                    <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 4px;">{desc}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px; font-style: italic;">Example: "{eg}"</div>
                </div>
            """, unsafe_allow_html=True)

    with t5:
        st.markdown("### 💻 Technology Stack")

        st.markdown("""
        | Layer | Technology | Purpose |
        |---|---|---|
        | **Language** | Python 3.11+ / 3.13 | Core programming language |
        | **User Interface** | Streamlit 1.51+ | Reactive, component-driven AI dashboard |
        | **Data Manipulation** | Pandas & NumPy | Dataset curation, vectorized array computations |
        | **Classical ML** | Scikit-learn 1.7+ | TF-IDF vectorization, Logistic Regression, Linear SVM, metrics |
        | **Deep Learning** | PyTorch 2.9+ / Transformers 5.16+ | Deep self-attention sequence classifier & transformer pipelines |
        | **NLP Core** | NLTK 3.9+ / WordNet | Tokenization, stopword removal, morphological lemmatization |
        | **Data Visualization** | Plotly Express & Graph Objects | Dynamic confusion matrices, gauge meters, responsive charts |
        | **Database** | SQLite 3 | Embedded persistent query logging and audit history |
        | **Model Serialization** | Joblib & PyTorch `.pt` | Binary model persistence and zero-overhead loading |
        """)

        st.write("")
        st.markdown("### ⚠️ System Limitations & Ethical Scope")
        st.markdown("""
        - **Non-Diagnostic Scope:** This system does not diagnose diseases or prescribe treatment. It routes queries to structured educational guidelines.
        - **Monolingual Focus:** Current implementation is optimized for English queries; multi-lingual cross-lingual transfers are slated for future iterations.
        - **Out-of-Domain Inputs:** Queries unrelated to air quality or bodily health are flagged as low-confidence ($< 35\%$).
        """)

    # Mandatory medical disclaimer
    st.markdown(f"""
        <div class="disclaimer-banner">
            <strong>⚠️ Mandatory Medical Disclaimer:</strong> {MEDICAL_DISCLAIMER}
        </div>
    """, unsafe_allow_html=True)

# Auto-execute when routed to this page
render_about()
