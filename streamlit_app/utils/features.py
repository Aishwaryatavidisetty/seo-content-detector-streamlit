# streamlit_app/utils/features.py
import re, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def clean_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_corpus_and_vectors(extracted_df: pd.DataFrame):
    texts = extracted_df['body_text'].fillna("").map(clean_text)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
    X = tfidf.fit_transform(texts)

    # Compact embedding via SVD (fast, good enough for demo)
    n_components = min(50, max(2, min(X.shape[1]-1, X.shape[0]-1)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = svd.fit_transform(X)

    return tfidf, svd, X, X_svd

def vectorize_text(text: str, tfidf, svd):
    text_clean = clean_text(text)
    row = tfidf.transform([text_clean])
    vec = svd.transform(row)  # 50-D by default
    return vec
