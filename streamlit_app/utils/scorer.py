# streamlit_app/utils/scorer.py
import numpy as np
import textstat
import joblib
from pathlib import Path

def rule_label(word_count: int, fre: float) -> str:
    if (word_count > 1500) and (50 <= fre <= 70):
        return "High"
    if (word_count < 500) or (fre < 30):
        return "Low"
    return "Medium"

def safe_sentence_count(text: str) -> int:
    if not text: return 0
    # leave NLTK out to simplify Streamlit deploy; simple heuristic:
    return max(1, text.count(".") + text.count("!") + text.count("?"))

def predict_label(word_count: int, sentence_count: int, flesch: float, models_dir: Path) -> str:
    # Try loading trained model; fallback to rules
    p = models_dir / "quality_model.pkl"
    if p.exists():
        try:
            model = joblib.load(p)
            x = np.array([[float(word_count), float(sentence_count), float(flesch)]])
            return model.predict(x)[0]
        except Exception:
            pass
    return rule_label(word_count, flesch)
