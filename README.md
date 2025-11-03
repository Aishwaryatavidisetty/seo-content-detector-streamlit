SEO Content Detector

🧩 Project Overview
- This project automates SEO content quality analysis for webpages.
- It parses HTML to extract clean text, computes linguistic and readability metrics, detects duplicate or thin pages, and classifies content quality (Low / Medium / High).
- The pipeline supports both offline datasets and real-time URL analysis.

⚙️ Setup Instructions
1. git clone https://github.com/Aishwaryatavidisetty/seo-content-detector-streamlit
2. cd seo-content-detector-streamlit
3. pip install -r requirements.txt
4. jupyter notebook notebooks/seo_pipeline.ipynb

🚀 Quick Start
1. Place your dataset in data/data.csv with headers:
    - url,html_content
2. Open notebooks/seo_pipeline.ipynb and Run All cells.
3. Outputs are generated automatically:
    - data/extracted_content.csv – parsed titles and body text
    - data/features.csv – metrics + embeddings + thin flag
    - data/duplicates.csv – duplicate URL pairs
    - models/quality_model.pkl – trained classifier
4. For real-time testing, run the final notebook cell, enter any URL, and view metrics \& quality score interactively.

🌐 Deployed Streamlit URL 

https://seo-content-detector-app-sizmuqwh2dxmeako7jygrt.streamlit.app/



🔍 Key Decisions
- Libraries: BeautifulSoup for parsing, textstat for readability, scikit-learn for features + models, sentence-transformers for semantic embeddings.
- Parsing: Focused on , , and headings/paragraphs to keep meaningful text while removing boilerplate.
- Similarity Threshold: 0.80 cosine similarity—empirically balances recall vs. false duplicates for mid-sized corpora.
- Model: Compared Logistic Regression and Random Forest; RF chosen for higher accuracy and stable class separation.

📊 Results Summary
- Model: Random Forest
- Accuracy: 0.92 Macro F1: 0.88
- Top Features: flesch_reading_ease (0.45), word\_count (0.28), sentence\_count (0.27)
- Duplicate Pairs: 3 (> 0.80 similarity)
- Thin Pages: 10 % of dataset
- Sample Quality Scores: Low → thin/poorly readable; High → 1500 + words \& readability 50–70.

⚠️ Limitations
- Parsing rules may miss dynamic or script-generated content.
- TF-IDF/embeddings capture semantics only in English text.
- Rule-based labeling is synthetic—true SEO quality may vary with richer signals.

Author: Aishwarya Tavidisetty

