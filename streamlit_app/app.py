# streamlit_app/app.py
import json
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from utils.parser import fetch_html, html_to_title_and_text
from utils.features import build_corpus_and_vectors, vectorize_text, clean_text
from utils.scorer import predict_label, rule_label, safe_sentence_count
import textstat

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
CORPUS_CSV = DATA_DIR / "extracted_content.csv"

st.set_page_config(page_title="SEO Content Detector", page_icon="🕵️", layout="centered")
st.title("SEO Content Detector (Streamlit)")

# Load corpus
@st.cache_resource(show_spinner=True)
def load_corpus_and_vectors():
    if not CORPUS_CSV.exists():
        # empty corpus fallback
        df = pd.DataFrame(columns=["url","title","body_text","word_count","sentence_count","flesch_reading_ease"])
        return df, None, None, None, None
    df = pd.read_csv(CORPUS_CSV)
    # Fill required columns if missing
    for c in ["url","title","body_text","word_count","sentence_count","flesch_reading_ease"]:
        if c not in df.columns:
            df[c] = "" if c in ["url","title","body_text"] else 0
    tfidf, svd, X, X_svd = build_corpus_and_vectors(df)
    return df, tfidf, svd, X, X_svd

corpus_df, tfidf, svd, X_tfidf, X_svd = load_corpus_and_vectors()

url = st.text_input("Paste a URL to analyze:", placeholder="https://example.com/article")
threshold = st.slider("Duplicate threshold (cosine)", 0.50, 0.99, 0.80, 0.01)
top_k = st.slider("Show top matches", 1, 10, 3)

if st.button("Analyze"):
    if not url.strip():
        st.warning("Please enter a URL.")
        st.stop()

    html = fetch_html(url)
    if not html:
        st.error("Failed to fetch the page (check the URL and internet connectivity).")
        st.stop()

    title, body = html_to_title_and_text(html)
    wc = len(body.split()) if body else 0
    sc = safe_sentence_count(body)
    fre = textstat.flesch_reading_ease(body) if body else 0.0

    label = predict_label(wc, sc, fre, MODELS_DIR)
    is_thin = wc < 500

    # Similarity vs corpus (use SVD embeddings; fallback to tf-idf if corpus small)
    similar_list = []
    if tfidf is not None and svd is not None and X_svd is not None and len(corpus_df) > 0:
        q_vec = vectorize_text(body, tfidf, svd)
        sims = cosine_similarity(q_vec, X_svd)[0]
        order = np.argsort(sims)[::-1]

        for idx in order:
            sim = float(sims[idx])
            if sim >= threshold:
                similar_list.append({"url": corpus_df.iloc[idx]["url"], "similarity": round(sim, 4)})
                if len(similar_list) >= top_k:
                    break

        if not similar_list and len(order) > 0:
            # provide the single closest for context
            best_idx = int(order[0])
            similar_list = [{"url": corpus_df.iloc[best_idx]["url"], "similarity": round(float(sims[best_idx]), 4)}]

    # Output
    st.subheader("Results")
    st.json({
        "url": url,
        "title": title,
        "word_count": wc,
        "readability": round(float(fre), 3),
        "quality_label": label,
        "is_thin": bool(is_thin),
        "similar_to": similar_list
    })

    with st.expander("Details"):
        st.write(f"**Title:** {title or '—'}")
        st.write(f"**Sentence count:** {sc}")
        st.write(f"**Thin page:** {is_thin}")
        if len(similar_list):
            st.write("**Similar pages:**")
            st.dataframe(pd.DataFrame(similar_list))
        else:
            st.write("No similar pages above the threshold.")
