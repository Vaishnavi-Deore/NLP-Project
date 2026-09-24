"""
Database module for AQI-Sense
Manages SQLite persistence for query logs, prediction history, confidence scores,
and analytics aggregation.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

# Ensure database path is anchored relative to this file
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "aqi_sense.db")

def get_connection() -> sqlite3.Connection:
    """
    Returns a connection to the SQLite database with row factory enabled.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite database and creates the queries table if it doesn't exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            category TEXT NOT NULL,
            confidence REAL NOT NULL,
            model TEXT NOT NULL,
            top_predictions TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_query(query: str, category: str, confidence: float, model: str, top_predictions: Any) -> int:
    """
    Inserts a newly classified user query into the database.
    top_predictions can be a list of dicts or a JSON string.
    """
    init_db()
    if not isinstance(top_predictions, str):
        top_predictions_json = json.dumps(top_predictions)
    else:
        top_predictions_json = top_predictions

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO queries (timestamp, query, category, confidence, model, top_predictions)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (timestamp, query.strip(), category, round(float(confidence), 4), model, top_predictions_json))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_all_queries(search_term: Optional[str] = None, 
                    category_filter: Optional[str] = None, 
                    sort_by: str = "timestamp", 
                    ascending: bool = False) -> List[Dict[str, Any]]:
    """
    Retrieves filtered and sorted query history records.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    sql = "SELECT id, timestamp, query, category, confidence, model, top_predictions FROM queries WHERE 1=1"
    params = []

    if search_term and search_term.strip():
        sql += " AND query LIKE ?"
        params.append(f"%{search_term.strip()}%")

    if category_filter and category_filter != "All":
        sql += " AND category = ?"
        params.append(category_filter)

    valid_sort_cols = {"timestamp", "confidence", "category", "id"}
    sort_col = sort_by if sort_by in valid_sort_cols else "timestamp"
    direction = "ASC" if ascending else "DESC"

    sql += f" ORDER BY {sort_col} {direction}"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    
    results = []
    for r in rows:
        top_preds = []
        try:
            top_preds = json.loads(r["top_predictions"])
        except Exception:
            top_preds = []
            
        results.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "query": r["query"],
            "category": r["category"],
            "confidence": r["confidence"],
            "model": r["model"],
            "top_predictions": top_preds
        })

    conn.close()
    return results

def get_recent_queries(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent queries.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, query, category, confidence, model, top_predictions 
        FROM queries 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    
    results = []
    for r in rows:
        top_preds = []
        try:
            top_preds = json.loads(r["top_predictions"])
        except Exception:
            top_preds = []
            
        results.append({
            "id": r["id"],
            "timestamp": r["timestamp"],
            "query": r["query"],
            "category": r["category"],
            "confidence": r["confidence"],
            "model": r["model"],
            "top_predictions": top_preds
        })
    conn.close()
    return results

def delete_query(query_id: int) -> bool:
    """
    Deletes a single query record by ID.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM queries WHERE id = ?", (query_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

def clear_all_queries() -> int:
    """
    Clears all records from queries table.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM queries")
    conn.commit()
    count = cursor.rowcount
    conn.close()
    return count

def get_database_stats() -> Dict[str, Any]:
    """
    Returns summary statistics from the queries table.
    """
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*), AVG(confidence) FROM queries")
    row = cursor.fetchone()
    total = row[0] if row else 0
    avg_conf = (row[1] * 100) if row and row[1] is not None else 0.0

    cursor.execute("SELECT category, COUNT(*) as cnt FROM queries GROUP BY category ORDER BY cnt DESC")
    cat_counts = {r["category"]: r["cnt"] for r in cursor.fetchall()}

    conn.close()
    return {
        "total_queries": total,
        "avg_confidence": round(avg_conf, 1),
        "category_counts": cat_counts
    }

if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)
    # Test insert
    test_id = save_query("Can I run outside when AQI is 150?", "Outdoor Activities", 0.942, "TF-IDF + Logistic Regression", [
        {"category": "Outdoor Activities", "probability": 0.942},
        {"category": "Precautions", "probability": 0.038},
        {"category": "AQI Information", "probability": 0.020}
    ])
    print("Inserted test record with ID:", test_id)
    stats = get_database_stats()
    print("Database stats:", stats)
