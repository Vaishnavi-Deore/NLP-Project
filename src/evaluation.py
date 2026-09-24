"""
Evaluation utilities for AQI-Sense models
Computes genuine classification metrics: Accuracy, Precision, Recall, F1,
Per-class breakdowns, and Confusion Matrix representations without hardcoding.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_predictions(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """
    Computes rigorous ML evaluation metrics for multi-class classification.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    prec_m = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_m = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, labels=labels, zero_division=0)

    # Per-category metrics
    per_class = {}
    for cat in labels:
        if cat in report_dict:
            per_class[cat] = {
                "precision": round(float(report_dict[cat]["precision"]), 4),
                "recall": round(float(report_dict[cat]["recall"]), 4),
                "f1_score": round(float(report_dict[cat]["f1-score"]), 4),
                "support": int(report_dict[cat]["support"]),
            }
        else:
            per_class[cat] = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}

    return {
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec_w), 4),
        "recall_weighted": round(float(rec_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "precision_macro": round(float(prec_m), 4),
        "recall_macro": round(float(rec_m), 4),
        "f1_macro": round(float(f1_m), 4),
        "total_samples": len(y_true),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "per_class": per_class,
        "classification_report_str": report_str
    }
