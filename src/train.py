"""
Model Training Script for AQI-Sense
Trains Baseline (TF-IDF + Logistic Regression), Calibrated Linear SVM,
and Random Forest / Advanced NLP classifiers with stratified train/test split.
Evaluates all models on the real test split, exports metrics.json, test_predictions.csv,
and serializes artifacts into models/.
"""

import os
import sys
import json
import joblib
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

# Local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from preprocessing import preprocess_query
from evaluation import evaluate_predictions

CATEGORIES = [
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

def load_and_preprocess_data(csv_path="data/aqi_health_queries.csv"):
    """
    Loads raw CSV and applies NLP preprocessing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Run dataset generator first.")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Drop any nulls
    df = df.dropna(subset=["query", "category"]).copy()

    # Preprocess text
    print("Preprocessing text queries...")
    df["processed_query"] = df["query"].apply(preprocess_query)

    # Save processed data for dataset explorer
    processed_path = "data/processed_data.csv"
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"Saved preprocessed data to {processed_path}")

    return df

def train_and_evaluate(csv_path="data/aqi_health_queries.csv", models_dir="models"):
    """
    Executes training and evaluation pipeline for multiple models.
    """
    os.makedirs(models_dir, exist_ok=True)
    df = load_and_preprocess_data(csv_path)

    X_raw = df["query"].values
    X_proc = df["processed_query"].values
    y = df["category"].values

    # Stratified Train-Test Split (80% train, 20% test)
    X_train_raw, X_test_raw, X_train_proc, X_test_proc, y_train, y_test = train_test_split(
        X_raw, X_proc, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train_proc)}, Testing samples: {len(X_test_proc)}")

    # 1. Feature Extraction: TF-IDF
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
        min_df=1
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_proc)
    X_test_tfidf = vectorizer.transform(X_test_proc)

    # Save Vectorizer
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Saved TF-IDF vectorizer to {vectorizer_path}")

    labels = sorted(list(set(y)))
    metrics_all = {}
    test_preds_df = pd.DataFrame({
        "raw_query": X_test_raw,
        "processed_query": X_test_proc,
        "true_category": y_test
    })

    # ==========================================
    # Model 1: Baseline (Logistic Regression)
    # ==========================================
    print("\n--- Training Baseline: TF-IDF + Logistic Regression ---")
    log_reg = LogisticRegression(
        max_iter=1000,
        C=2.0,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )
    log_reg.fit(X_train_tfidf, y_train)
    y_pred_lr = log_reg.predict(X_test_tfidf)
    y_proba_lr = log_reg.predict_proba(X_test_tfidf)

    lr_metrics = evaluate_predictions(y_test, y_pred_lr, labels=labels)
    metrics_all["TF-IDF + Logistic Regression"] = lr_metrics
    test_preds_df["pred_logistic_regression"] = y_pred_lr
    test_preds_df["conf_logistic_regression"] = np.max(y_proba_lr, axis=1)

    lr_path = os.path.join(models_dir, "logistic_model.pkl")
    joblib.dump(log_reg, lr_path)
    print(f"Logistic Regression Test Accuracy: {lr_metrics['accuracy'] * 100:.2f}% | F1: {lr_metrics['f1_weighted']:.4f}")
    print(f"Saved Logistic Regression model to {lr_path}")

    # ==========================================
    # Model 2: Calibrated Linear SVM
    # ==========================================
    print("\n--- Training Model 2: TF-IDF + Linear Support Vector Machine (Calibrated) ---")
    base_svm = LinearSVC(C=1.0, random_state=42, dual="auto", max_iter=2000)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv=5)
    calibrated_svm.fit(X_train_tfidf, y_train)
    y_pred_svm = calibrated_svm.predict(X_test_tfidf)
    y_proba_svm = calibrated_svm.predict_proba(X_test_tfidf)

    svm_metrics = evaluate_predictions(y_test, y_pred_svm, labels=labels)
    metrics_all["TF-IDF + Linear SVM"] = svm_metrics
    test_preds_df["pred_linear_svm"] = y_pred_svm
    test_preds_df["conf_linear_svm"] = np.max(y_proba_svm, axis=1)

    svm_path = os.path.join(models_dir, "svm_model.pkl")
    joblib.dump(calibrated_svm, svm_path)
    print(f"Linear SVM Test Accuracy: {svm_metrics['accuracy'] * 100:.2f}% | F1: {svm_metrics['f1_weighted']:.4f}")
    print(f"Saved SVM model to {svm_path}")

    # ==========================================
    # Model 3: Random Forest Classifier
    # ==========================================
    print("\n--- Training Model 3: TF-IDF + Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=120, max_depth=25, random_state=42, n_jobs=-1)
    rf.fit(X_train_tfidf, y_train)
    y_pred_rf = rf.predict(X_test_tfidf)
    y_proba_rf = rf.predict_proba(X_test_tfidf)

    rf_metrics = evaluate_predictions(y_test, y_pred_rf, labels=labels)
    metrics_all["TF-IDF + Random Forest"] = rf_metrics
    test_preds_df["pred_random_forest"] = y_pred_rf
    test_preds_df["conf_random_forest"] = np.max(y_proba_rf, axis=1)

    rf_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(rf, rf_path)
    print(f"Random Forest Test Accuracy: {rf_metrics['accuracy'] * 100:.2f}% | F1: {rf_metrics['f1_weighted']:.4f}")
    print(f"Saved Random Forest model to {rf_path}")

    # Save test predictions table
    test_preds_path = os.path.join(models_dir, "test_predictions.csv")
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions comparison to {test_preds_path}")

    # Save metrics JSON
    metrics_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"Saved comprehensive evaluation metrics to {metrics_path}")

    # Also build a lightweight BERT / Transformer checkpoint or fine-tuner
    print("\nTraining complete for all models.")
    return metrics_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AQI-Sense classifiers")
    parser.add_argument("--data", default="data/aqi_health_queries.csv", help="Path to queries CSV")
    parser.add_argument("--models-dir", default="models", help="Directory to save models")
    args = parser.parse_args()

    train_and_evaluate(csv_path=args.data, models_dir=args.models_dir)
