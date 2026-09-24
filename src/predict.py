"""
Inference & Explainability Engine for AQI-Sense
Handles query classification across all trained models (Logistic Regression, Linear SVM,
Random Forest, Deep Self-Attention), computes top-3 confidence probabilities,
extracts contributing feature weights for explainability, and retrieves educational advisories.
"""

import os
import sys
import time
import json
import joblib
import torch
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from preprocessing import inspect_pipeline, preprocess_query
from advisory import get_category_advisory
from train_advanced import TextAttentionClassifier, text_to_tensor

MODELS_DIR = os.path.join(parent_dir, "models")

# Cache loaded models in memory
_CACHED_OBJECTS = {}

def get_loaded_models():
    """
    Loads and caches TF-IDF vectorizer and ML classifiers.
    """
    if "vectorizer" not in _CACHED_OBJECTS:
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        lr_path = os.path.join(MODELS_DIR, "logistic_model.pkl")
        svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
        rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")

        if not os.path.exists(vec_path) or not os.path.exists(lr_path):
            raise FileNotFoundError("Trained models not found. Please run train.py first.")

        _CACHED_OBJECTS["vectorizer"] = joblib.load(vec_path)
        _CACHED_OBJECTS["logistic"] = joblib.load(lr_path)
        
        if os.path.exists(svm_path):
            _CACHED_OBJECTS["svm"] = joblib.load(svm_path)
        if os.path.exists(rf_path):
            _CACHED_OBJECTS["rf"] = joblib.load(rf_path)

        # Load Neural Model if available
        neural_path = os.path.join(MODELS_DIR, "advanced_attention_model.pt")
        vocab_path = os.path.join(MODELS_DIR, "neural_vocab.json")
        if os.path.exists(neural_path) and os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                neural_meta = json.load(f)
            vocab = neural_meta["vocab"]
            id2label = {int(k): v for k, v in neural_meta["id2label"].items()}
            neural_model = TextAttentionClassifier(vocab_size=len(vocab), num_classes=len(id2label))
            neural_model.load_state_dict(torch.load(neural_path, map_location=torch.device("cpu")))
            neural_model.eval()
            _CACHED_OBJECTS["neural_model"] = neural_model
            _CACHED_OBJECTS["neural_vocab"] = vocab
            _CACHED_OBJECTS["neural_id2label"] = id2label

    return _CACHED_OBJECTS

def get_feature_explainability(query_tfidf, model, predicted_class_idx, vectorizer, top_n=6):
    """
    Computes top contributing TF-IDF terms for the predicted class using model weights.
    Contribution = tfidf_weight * model_coefficient
    """
    try:
        if not hasattr(model, "coef_"):
            return []

        coefs = model.coef_[predicted_class_idx]
        feature_names = vectorizer.get_feature_names_out()

        # Non-zero indices in query vector
        nonzero_indices = query_tfidf.nonzero()[1]
        contributions = []

        for idx in nonzero_indices:
            term = feature_names[idx]
            tfidf_val = query_tfidf[0, idx]
            weight = coefs[idx]
            score = tfidf_val * weight
            contributions.append({
                "term": term,
                "tfidf": round(float(tfidf_val), 4),
                "weight": round(float(weight), 4),
                "contribution_score": round(float(score), 4)
            })

        # Sort descending by contribution score
        contributions = sorted(contributions, key=lambda x: x["contribution_score"], reverse=True)
        return contributions[:top_n]
    except Exception:
        return []

def predict_query(query: str, model_type: str = "TF-IDF + Logistic Regression") -> dict:
    """
    Main prediction pipeline:
    - Preprocesses input query
    - Extracts TF-IDF or tensor features
    - Runs selected classifier
    - Calculates confidence and top-3 probabilities
    - Computes feature explainability
    - Retrieves educational advisory
    """
    t_start = time.perf_counter()

    if not query or not query.strip():
        return {
            "error": "Query cannot be empty. Please enter an air quality or health-related question.",
            "is_valid": False
        }

    pipeline_info = inspect_pipeline(query)
    processed_text = pipeline_info["processed_text"]

    cached = get_loaded_models()
    vectorizer = cached["vectorizer"]
    
    # Check if query resulted in zero tokens after stopword removal
    if not processed_text.strip():
        processed_text = pipeline_info["cleaned_text"]

    # Select model
    if "svm" in model_type.lower() and "svm" in cached:
        active_model = cached["svm"]
        model_display_name = "TF-IDF + Linear Support Vector Machine"
        query_vec = vectorizer.transform([processed_text])
        classes = active_model.classes_
        proba = active_model.predict_proba(query_vec)[0]
        explainability = [] # CalibratedClassifierCV wraps base estimator

    elif "random forest" in model_type.lower() and "rf" in cached:
        active_model = cached["rf"]
        model_display_name = "TF-IDF + Random Forest Classifier"
        query_vec = vectorizer.transform([processed_text])
        classes = active_model.classes_
        proba = active_model.predict_proba(query_vec)[0]
        explainability = []

    elif "neural" in model_type.lower() or "attention" in model_type.lower() or "deep" in model_type.lower():
        if "neural_model" in cached:
            model_display_name = "Advanced NLP (Deep Self-Attention / PyTorch)"
            neural_model = cached["neural_model"]
            vocab = cached["neural_vocab"]
            id2label = cached["neural_id2label"]
            x_tensor = text_to_tensor(processed_text, vocab).unsqueeze(0)
            with torch.no_grad():
                out = neural_model(x_tensor)
                proba = torch.softmax(out, dim=1).squeeze(0).numpy()
            classes = [id2label[i] for i in range(len(id2label))]
            query_vec = None
            explainability = []
        else:
            # Fallback to logistic regression
            active_model = cached["logistic"]
            model_display_name = "TF-IDF + Logistic Regression (Baseline)"
            query_vec = vectorizer.transform([processed_text])
            classes = active_model.classes_
            proba = active_model.predict_proba(query_vec)[0]
            pred_idx = np.argmax(proba)
            explainability = get_feature_explainability(query_vec, active_model, pred_idx, vectorizer)

    else:
        # Default: Baseline Logistic Regression
        active_model = cached["logistic"]
        model_display_name = "TF-IDF + Logistic Regression (Baseline)"
        query_vec = vectorizer.transform([processed_text])
        classes = active_model.classes_
        proba = active_model.predict_proba(query_vec)[0]
        pred_idx = np.argmax(proba)
        explainability = get_feature_explainability(query_vec, active_model, pred_idx, vectorizer)

    # Sort probabilities
    sorted_indices = np.argsort(proba)[::-1]
    best_idx = sorted_indices[0]
    predicted_category = classes[best_idx]
    confidence = float(proba[best_idx])

    top_predictions = []
    for i in sorted_indices[:5]:
        top_predictions.append({
            "category": classes[i],
            "probability": round(float(proba[i]), 4),
            "percentage": f"{proba[i] * 100:.1f}%"
        })

    is_low_confidence = bool(confidence < 0.35)
    advisory = get_category_advisory(predicted_category)
    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "is_valid": True,
        "query": query,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.1f}%",
        "is_low_confidence": is_low_confidence,
        "top_predictions": top_predictions,
        "model_name": model_display_name,
        "pipeline_info": pipeline_info,
        "explainability": explainability,
        "advisory": advisory,
        "latency_ms": latency_ms
    }

if __name__ == "__main__":
    test_queries = [
        "Can I exercise outside when the AQI is 180?",
        "Does an N95 mask protect against toxic air pollution?",
        "Why do my eyes burn when the smog is bad?",
        "What does AQI 250 mean?",
        "Is high pollution dangerous for pregnant women?",
    ]
    print("Testing AQI-Sense Prediction Engine:\n")
    for q in test_queries:
        res = predict_query(q)
        print(f"Query: '{q}'")
        print(f"  -> Predicted: {res['predicted_category']} ({res['confidence_percentage']})")
        print(f"  -> Top 3: {[(p['category'], p['percentage']) for p in res['top_predictions'][:3]]}")
        if res["explainability"]:
            print(f"  -> Top features: {[t['term'] for t in res['explainability'][:3]]}")
        print()
