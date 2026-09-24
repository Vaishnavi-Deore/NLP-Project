# AQI-Sense — AQI Health Query Classifier

**Project Title:**  
*AQI Health Query Classifier: An NLP-Based System for Classifying Air Quality and Health-Related Queries*

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7-orange.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic%20Demo-green.svg)]()

AQI-Sense is an academic NLP and Machine Learning application designed to classify natural-language inquiries related to air quality, particulate pollution, and human health into 9 structured intent categories. It couples intent detection with calibrated confidence scores, token-level pipeline diagnostics, linear feature attribution, and public health advisories.

> **⚠️ Non-Diagnostic Advisory Disclaimer:**  
> This system is strictly engineered for query intent classification, information retrieval, and educational advisory routing. It does **not** provide clinical diagnosis, medical triage, or substitute for consultation with qualified healthcare professionals.

---

## 📌 Primary Classification Taxonomy

The system classifies queries into 9 distinct intent categories:

| # | Intent Category | Typical Scope & Inquiries |
|---|---|---|
| 1 | **AQI Information** | AQI scale definition, EPA breakpoint categories, PM2.5/PM10 calculation methods |
| 2 | **Health Effects** | Systemic long-term risks, cardiovascular disease, hypertension, arterial inflammation |
| 3 | **Symptoms** | Acute bodily reactions, eye stinging, cough, dry throat, headaches, nausea, dizziness |
| 4 | **Precautions** | Indoor air filtration, sealing doors/windows, HEPA purifiers, kitchen ventilation |
| 5 | **Masks and Protection** | N95, KN95, KF94 respirators, filtration standards, facial seal checks, reuse guidelines |
| 6 | **Outdoor Activities** | Jogging, morning walks, children recess, marathon training, sports cancellation thresholds |
| 7 | **Vulnerable Groups** | Risks and guidelines for infants, elderly seniors, pregnant women, and immunocompromised patients |
| 8 | **Respiratory Health** | Asthma attacks, COPD exacerbations, bronchitis, bronchial spasms, rescue inhaler protocols |
| 9 | **Pollution Exposure** | Cumulative dosage, duration of exposure, highway proximity, occupational worker risks |

---

## 🏗️ Architecture & NLP Pipeline

```
User Query (e.g. "Can I jog outside when AQI is 180?")
                    │
                    ▼
          [Text Normalization]
          - Lowercased, URLs & symbols stripped
          - Particulate tokens preserved (PM2.5 → pm25, N95 → n95)
                    │
                    ▼
          [NLTK Word Tokenization]
                    │
                    ▼
          [Stopword Removal]
                    │
                    ▼
          [WordNet Lemmatization]
                    │
                    ▼
          [Feature Extraction]
          - TF-IDF Vectorizer (Unigrams + Bigrams, Sublinear TF)
          - Or Neural Word Embeddings
                    │
                    ▼
          [ML / Deep Learning Classifier]
          - Baseline: Multinomial Logistic Regression
          - Calibrated Linear SVM
          - Random Forest Classifier
          - Advanced NLP: Deep Self-Attention Neural Network (PyTorch)
                    │
                    ▼
     ┌──────────────┴──────────────┐
     ▼                             ▼
[Intent & Confidence]      [Advisory Retrieval]
- Predicted Category       - Key scientific takeaways
- Top-3 probabilities      - Recommended actions & Do's/Don'ts
- Feature Explainability   - Mandatory Medical Disclaimer
     │                             │
     └──────────────┬──────────────┘
                    ▼
         [SQLite Persistence]
         - Timestamp, query, category, confidence, model
```

---

## 🚀 Quick Setup Instructions

### 1. Clone or Open Project Directory

```powershell
cd "c:\Users\deore\Downloads\NLP Project"
```

### 2. Create and Activate Virtual Environment (Recommended)

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Generate Dataset & Train Classifiers

Generate the balanced corpus (1,200+ samples across 9 categories):
```powershell
python src/dataset_generator.py
```

Train baseline, SVM, and Random Forest models on stratified test splits:
```powershell
python src/train.py
```

Train the Advanced Deep NLP Self-Attention classifier (PyTorch):
```powershell
python src/train_advanced.py
```

### 5. Launch the Streamlit Web Application

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ Application Features & Page Tour

1. **📊 Home / Dashboard:**
   - Real-time KPI summary cards (Total Queries, Model Accuracy, Intent Categories, Avg Confidence).
   - "Ask AQI-Sense" search bar with 5 interactive suggested query chips.
   - Plotly interactive category balance chart and recent query activity feed.

2. **🔬 Query Analyzer (Core NLP Engine):**
   - Natural-language query input with benchmark presets.
   - Multi-model switcher (Logistic Regression, Linear SVM, Random Forest, Deep Attention).
   - Identified Intent Category badge and visual confidence progress bar.
   - Top-3 predicted categories with horizontal probability bars.
   - **Feature Explainability:** Mathematical attribution of contributing words based on model coefficients ($w_{c,i} \cdot x_i$).
   - **NLP Pipeline Inspector:** Step-by-step visibility of raw text, normalized string, tokens, stopword removal, and lemmas.
   - **Retrieved Educational Advisory:** Scientific context, actions, and Do's & Don'ts.

3. **🎛️ Interactive AQI Simulator:**
   - Educational parameter sandbox with sliders for AQI (0–500), PM2.5, PM10, NO2, and Ozone.
   - Dynamic Plotly gauge meter with EPA color bands.
   - Scenario presets (Alpine air, Urban afternoon, Smog inversion, Wildfire smoke).
   - Dynamic guidance for the general population, vulnerable groups, and outdoor athletic activities.

4. **📈 Analytics Dashboard:**
   - True empirical metrics calculated directly on the held-out 20% test split:
     - Accuracy, Precision (Weighted), Recall (Weighted), F1-Score (Weighted), Sample Count.
   - Interactive Plotly Confusion Matrix with category hover tools.
   - Precision / Recall / F1 grouped bar chart by category.
   - Softmax confidence distribution histogram.

5. **⚖️ Model Comparison:**
   - Side-by-side leaderboard ranking all trained architectures.
   - Grouped metrics bar chart and per-category F1 comparisons.
   - Trade-off analysis covering latency, parameter footprint, and explainability.
   - "Retrain All Models" action trigger.

6. **📜 Query History:**
   - SQLite persistent storage (`database/aqi_sense.db`).
   - Keyword search, category dropdown filter, and sorting controls.
   - Individual record deletion and full database purge options.
   - CSV export for query audits.

7. **📁 Dataset Explorer:**
   - Interactive preview of raw and preprocessed datasets.
   - Category counts and Plotly donut chart.
   - Query length and lexical distribution visualizations.
   - CSV download for external research.

8. **ℹ️ About & Documentation:**
   - Comprehensive problem statement, objectives, mathematical formulations, and limitations.

---

## 📊 Evaluation Summary (Held-Out Test Set)

| Model Architecture | Accuracy | Precision (W) | Recall (W) | F1-Score (W) | Macro F1 |
|---|---|---|---|---|---|
| **Calibrated Linear SVM** | **92.55%** | **92.79%** | **92.55%** | **92.60%** | **92.59%** |
| **TF-IDF + Logistic Regression (Baseline)** | **92.16%** | **92.67%** | **92.16%** | **92.17%** | **92.17%** |
| **Advanced NLP (Deep Self-Attention)** | **86.67%** | **87.21%** | **86.67%** | **86.59%** | **86.62%** |
| **TF-IDF + Random Forest** | **83.14%** | **83.69%** | **83.14%** | **83.11%** | **83.05%** |

*All metrics are automatically recomputed and written to `models/metrics.json` upon retraining.*

---

## 📂 Project Structure

```
NLP Project/
├── app.py                     # Main Streamlit application entrypoint
├── requirements.txt           # Python dependencies
├── README.md                  # Complete documentation and setup guide
├── .gitignore                 # Version control ignore rules
│
├── data/
│   ├── aqi_health_queries.csv # Curated dataset (1,272 queries across 9 classes)
│   └── processed_data.csv     # Preprocessed NLP dataset
│
├── models/
│   ├── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
│   ├── logistic_model.pkl     # Trained Logistic Regression classifier
│   ├── svm_model.pkl          # Trained Calibrated Linear SVM
│   ├── rf_model.pkl           # Trained Random Forest classifier
│   ├── advanced_attention_model.pt # PyTorch Self-Attention neural weights
│   ├── neural_vocab.json      # Vocabulary and label mappings
│   ├── metrics.json           # Real test evaluation metrics and confusion matrices
│   └── test_predictions.csv   # Predictions comparison on held-out test split
│
├── src/
│   ├── dataset_generator.py   # Synthesizes balanced dataset (1,200+ samples)
│   ├── preprocessing.py       # Cleaning, domain tokenization, lemmatization
│   ├── train.py               # Classical ML training and evaluation script
│   ├── train_advanced.py      # PyTorch Deep Learning attention trainer
│   ├── train_bert.py          # Hugging Face Transformer fine-tuning script
│   ├── predict.py             # Inference engine with feature explainability
│   ├── evaluation.py          # Metric calculation and confusion matrix helpers
│   └── advisory.py            # Public health guidelines and medical disclaimers
│
├── database/
│   ├── database.py            # SQLite CRUD operations and connection management
│   └── aqi_sense.db           # Persistent SQLite database
│
├── pages/
│   ├── dashboard.py           # Page 1: Home dashboard
│   ├── analyzer.py            # Page 2: Query Analyzer
│   ├── aqi_simulator.py       # Page 3: AQI Simulator
│   ├── analytics.py           # Page 4: ML Analytics
│   ├── model_comparison.py    # Page 5: Model Comparison
│   ├── history.py             # Page 6: Query History
│   ├── dataset_explorer.py    # Page 7: Dataset Explorer
│   └── about.py               # Page 8: Project About & Documentation
│
└── assets/
    ├── hero_banner.jpg        # Digital atmospheric intelligence hero visual
    ├── nlp_pipeline.jpg       # NLP query classification architecture infographic
    ├── respiratory_pm.jpg     # Microscopic PM2.5 vs PM10 lung penetration infographic
    └── style.css              # Custom AI analytics design stylesheet
```

---

## 🎓 Presentation & Demonstration Walkthrough

For academic project demonstrations:

1. **Start on Dashboard:** Show the live KPI cards and ask one of the preset questions (e.g. *"Can I exercise outside when AQI is 180?"*).
2. **Move to Query Analyzer:**
   - Demonstrate how the system normalizes `AQI 180` and `exercise`.
   - Point out the step-by-step preprocessing diagnostics in the **NLP Pipeline Diagnostics** tab.
   - Highlight the **Explainability** chart showing which specific words shifted the classifier weights.
   - Review the **Advisory Guidance** tab and the explicit non-medical disclaimer.
3. **Showcase AQI Simulator:**
   - Move sliders to simulate an emergency smog event (AQI > 250).
   - Observe the dynamic gauge update and the change in outdoor activity recommendations.
4. **Demonstrate Model Analytics:**
   - Present the real 9x9 Confusion Matrix and per-class Precision/Recall metrics.
   - Explain the trade-offs between Logistic Regression, Linear SVM, and Deep Self-Attention.
5. **Showcase Query History:**
   - Show how the previous query was instantly logged to SQLite with timestamp and top predictions.
