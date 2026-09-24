"""
Smoke test to verify that all modules, page functions, and inference pipelines run without errors.
"""

import sys
import os
import json

# Ensure workspace root is in path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

print("1. Testing Preprocessing...")
from src.preprocessing import inspect_pipeline, preprocess_query
sample_res = inspect_pipeline("Can I exercise outside when AQI is 180 and PM2.5 is high?")
assert sample_res["processed_text"] != ""
print("   [OK] Preprocessing passed.")

print("2. Testing Prediction & Inference...")
from src.predict import predict_query
pred_lr = predict_query("Can I jog outside when AQI is 180?", model_type="TF-IDF + Logistic Regression")
assert pred_lr["is_valid"] is True
assert pred_lr["predicted_category"] == "Outdoor Activities"
print(f"   [OK] Baseline prediction: {pred_lr['predicted_category']} ({pred_lr['confidence_percentage']})")

pred_svm = predict_query("Does an N95 mask protect against PM2.5?", model_type="TF-IDF + Linear Support Vector Machine")
assert pred_svm["is_valid"] is True
assert pred_svm["predicted_category"] == "Masks and Protection"
print(f"   [OK] SVM prediction: {pred_svm['predicted_category']} ({pred_svm['confidence_percentage']})")

pred_deep = predict_query("Can air pollution cause heart attacks and stroke?", model_type="Advanced NLP (Deep Self-Attention / PyTorch)")
assert pred_deep["is_valid"] is True
print(f"   [OK] Deep Attention prediction: {pred_deep['predicted_category']} ({pred_deep['confidence_percentage']})")

print("3. Testing Database CRUD...")
from database.database import init_db, save_query, get_all_queries, delete_query, get_database_stats
init_db()
qid = save_query("Test asthma query", "Respiratory Health", 0.95, "Test Model", [{"category": "Respiratory Health", "probability": 0.95}])
assert qid > 0
all_q = get_all_queries(search_term="Test asthma")
assert len(all_q) >= 1
delete_query(qid)
print("   [OK] Database CRUD passed.")

print("4. Testing Advisory Module...")
from src.advisory import get_aqi_details, get_category_advisory
aqi_test = get_aqi_details(180)
assert aqi_test["category"] == "Unhealthy"
adv_test = get_category_advisory("Outdoor Activities")
assert "Outdoor" in adv_test["title"]
print("   [OK] Advisory module passed.")

print("5. Testing Metrics JSON and Predictions...")
with open("models/metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)
assert "TF-IDF + Logistic Regression" in metrics
assert "TF-IDF + Linear SVM" in metrics
assert "Advanced NLP (Deep Self-Attention / PyTorch)" in metrics
print(f"   [OK] {len(metrics)} models verified in metrics.json.")

print("\nAll integration & smoke tests passed successfully! [OK]")
