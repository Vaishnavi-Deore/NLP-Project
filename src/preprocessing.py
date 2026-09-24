"""
NLP Preprocessing Pipeline for AQI-Sense
Reusable text cleaning, tokenization, stopword removal, and lemmatization.
Preserves air-quality domain terms such as PM2.5, PM10, N95, KN95, KF94, FFP2.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Safe download of required NLTK corpora
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
except Exception as e:
    pass

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
        "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", 
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", 
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", 
        "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", 
        "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", 
        "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", 
        "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", 
        "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", 
        "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
    }

# Retain certain question markers or directional words that might be useful, but standard stopwords work well
# Custom domain tokens mapping to preserve compound air-quality terms
DOMAIN_TOKEN_MAP = {
    r"\bpm\s*2[\.\_]?5\b": "pm25",
    r"\bpm\s*10\b": "pm10",
    r"\bn\s*95\b": "n95",
    r"\bkn\s*95\b": "kn95",
    r"\bkf\s*94\b": "kf94",
    r"\bn\s*99\b": "n99",
    r"\bffp\s*2\b": "ffp2",
    r"\baqi\s*[-–]?\s*": "aqi ",
}

def clean_text(text: str) -> str:
    """
    Cleans raw query string:
    - Lowercase
    - Preserves domain particulate terms (PM2.5 -> pm25, N95 -> n95)
    - Strips URLs, HTML, punctuation, extra spaces
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Normalize special air-quality tokens
    for pattern, replacement in DOMAIN_TOKEN_MAP.items():
        text = re.sub(pattern, replacement, text)

    # Remove URLs and HTML
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)

    # Replace punctuation with spaces, keeping alphanumeric
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text into words. Falls back to regex if NLTK tokenizer issues occur.
    """
    if not text:
        return []
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
    return tokens

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes common English stopwords.
    """
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """
    Lemmatizes tokens using WordNetLemmatizer.
    """
    lemmatized = []
    for t in tokens:
        # Check noun/verb lemmatization
        lemma = lemmatizer.lemmatize(t, pos='v')
        if lemma == t:
            lemma = lemmatizer.lemmatize(t, pos='n')
        lemmatized.append(lemma)
    return lemmatized

def preprocess_query(text: str) -> str:
    """
    Complete end-to-end preprocessing string suitable for TF-IDF feature extraction.
    """
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    no_stops = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(no_stops)
    return " ".join(lemmas)

def inspect_pipeline(raw_text: str) -> dict:
    """
    Inspects and returns each intermediate step in the NLP pipeline for UI visualization.
    """
    cleaned = clean_text(raw_text)
    tokens = tokenize_text(cleaned)
    no_stops = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(no_stops)
    final_str = " ".join(lemmas)

    return {
        "raw_query": raw_text,
        "cleaned_text": cleaned,
        "token_count_raw": len(raw_text.split()),
        "tokens": tokens,
        "tokens_no_stopwords": no_stops,
        "lemmatized_tokens": lemmas,
        "processed_text": final_str,
        "filtered_words_count": len(lemmas),
    }

if __name__ == "__main__":
    sample = "Can I exercise outside when AQI is 180 and PM2.5 is high?"
    res = inspect_pipeline(sample)
    print("Sample Pipeline Inspection:")
    for k, v in res.items():
        print(f"  {k}: {v}")
