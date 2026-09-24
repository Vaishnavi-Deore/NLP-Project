"""
Advanced Deep Learning / Transformer Sequence Classifier for AQI-Sense
Implements a PyTorch-based Self-Attention / Deep Neural Network classifier
and evaluates real performance on the stratified test split.
Can also fine-tune BERT models with Hugging Face transformers.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from preprocessing import preprocess_query
from evaluation import evaluate_predictions

class TextAttentionClassifier(nn.Module):
    """
    PyTorch Deep Learning Classifier with Learned Word Embeddings,
    Multi-Head Self-Attention, and Multi-Layer Perceptron Head.
    """
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, hidden_dim=128, num_classes=9, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len]
        emb = self.embedding(x) # [batch_size, seq_len, embed_dim]
        key_padding_mask = (x == 0) if mask is None else mask
        attn_out, _ = self.attention(emb, emb, emb, key_padding_mask=key_padding_mask)
        # Global max-pooling across sequence
        pooled, _ = torch.max(attn_out, dim=1)
        h = self.dropout(self.relu(self.fc1(pooled)))
        logits = self.fc2(h)
        return logits

def build_vocab(texts, max_vocab=5000):
    word_counts = {}
    for text in texts:
        for word in text.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab]
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in sorted_words:
        vocab[word] = len(vocab)
    return vocab

def text_to_tensor(text, vocab, max_len=32):
    tokens = text.split()
    ids = [vocab.get(w, vocab["<UNK>"]) for w in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

class QueryTensorDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=32):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = text_to_tensor(self.texts[idx], self.vocab, self.max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def train_advanced_model(
    csv_path="data/aqi_health_queries.csv",
    output_dir="models",
    metrics_path="models/metrics.json",
    epochs=12,
    batch_size=32,
    lr=0.002
):
    print("Training Advanced Deep NLP (Self-Attention Neural Classifier)...")
    df = pd.read_csv(csv_path).dropna(subset=["query", "category"])
    df["processed_query"] = df["query"].apply(preprocess_query)

    labels = sorted(list(df["category"].unique()))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    X = df["processed_query"].values
    y = np.array([label2id[c] for c in df["category"].values])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vocab = build_vocab(X_train, max_vocab=4000)
    
    # Save vocab and mappings
    vocab_path = os.path.join(output_dir, "neural_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "label2id": label2id, "id2label": id2label}, f, indent=2)

    train_ds = QueryTensorDataset(X_train, y_train, vocab)
    test_ds = QueryTensorDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextAttentionClassifier(vocab_size=len(vocab), num_classes=len(labels)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for ep in range(epochs):
        tot_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        if (ep + 1) % 4 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1}/{epochs} - Loss: {tot_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    all_preds, all_trues, all_probas = [], [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_trues.extend(by.numpy())
            all_probas.extend(probs)

    y_test_str = [id2label[i] for i in all_trues]
    y_pred_str = [id2label[i] for i in all_preds]

    adv_metrics = evaluate_predictions(y_test_str, y_pred_str, labels=labels)
    print(f"\nAdvanced Deep NLP Test Accuracy: {adv_metrics['accuracy'] * 100:.2f}% | F1: {adv_metrics['f1_weighted']:.4f}")

    # Save model weights
    model_path = os.path.join(output_dir, "advanced_attention_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved neural model weights to {model_path}")

    # Update metrics.json
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_all = json.load(f)
    else:
        metrics_all = {}

    metrics_all["Advanced NLP (Deep Self-Attention / PyTorch)"] = adv_metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"Updated metrics.json with Advanced NLP model metrics.")

    # Update test_predictions.csv
    test_preds_path = os.path.join(output_dir, "test_predictions.csv")
    if os.path.exists(test_preds_path):
        t_df = pd.read_csv(test_preds_path)
        t_df["pred_advanced_deep_nlp"] = y_pred_str
        t_df["conf_advanced_deep_nlp"] = np.max(all_probas, axis=1)
        t_df.to_csv(test_preds_path, index=False)
        print(f"Updated test_predictions.csv with Advanced Deep NLP predictions.")

    return adv_metrics

if __name__ == "__main__":
    train_advanced_model()
