import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from src.preprocess import clean_text_classic, remove_leakage
from src.features import extract_handcrafted_features

MODELS_DIR = "models"
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LOGREG_PATH = os.path.join(MODELS_DIR, "logreg_model.pkl")
SVM_PATH = os.path.join(MODELS_DIR, "best_classic_model.pkl")

class FakeNewsPredictor:
    """Production predictor for the Fake News Detection pipeline."""

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.vectorizer = None
        self.logreg_model = None
        self.svm_model = None
        self.transformer_tokenizer = None
        self.transformer_model = None
        self._load_models()

    def _load_models(self):
        tfidf_file = os.path.join(self.models_dir, "tfidf_vectorizer.pkl")
        logreg_file = os.path.join(self.models_dir, "logreg_model.pkl")
        svm_file = os.path.join(self.models_dir, "best_classic_model.pkl")
        transformer_dir = os.path.join(self.models_dir, "distilbert-fake-news")

        if os.path.exists(tfidf_file):
            self.vectorizer = joblib.load(tfidf_file)
        if os.path.exists(logreg_file):
            self.logreg_model = joblib.load(logreg_file)
        if os.path.exists(svm_file):
            self.svm_model = joblib.load(svm_file)

        if os.path.exists(transformer_dir):
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.transformer_tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
                self.transformer_model = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
                self.transformer_model.eval()
            except Exception as e:
                print(f"[INFO] Transformer model not loaded: {e}")

    def is_ready(self) -> bool:
        return self.vectorizer is not None and (self.logreg_model is not None or self.svm_model is not None)

    def predict(self, text: str, model_choice: str = "Logistic Regression") -> Dict[str, Any]:
        """
        Classifies an input news article and explains which words drove the decision.
        Returns:
            - verdict: 'REAL' or 'FAKE'
            - confidence: float between 0.0 and 1.0 (calibrated probability)
            - fake_probability: float
            - real_probability: float
            - top_fake_words: list of (word, score)
            - top_real_words: list of (word, score)
            - stylistic_stats: dict of length, caps, and punctuation metrics
        """
        if not self.is_ready():
            self._load_models()
            if not self.is_ready():
                raise RuntimeError("Models have not been trained yet. Run train_ml.py first.")

        # 1. Preprocess input text
        clean = clean_text_classic(text)
        if not clean.strip():
            return {
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "fake_probability": 0.5,
                "real_probability": 0.5,
                "top_fake_words": [],
                "top_real_words": [],
                "stylistic_stats": {"word_count": 0, "uppercase_ratio": 0.0, "exclamations": 0, "questions": 0},
                "explanation": "Text was too short or contained no valid words after stopword cleaning."
            }

        # 2. Extract Stylistic Metrics
        dummy_df = pd.DataFrame([{"content": text}])
        styles = extract_handcrafted_features(dummy_df, text_col="content").iloc[0].to_dict()

        # 3. Vectorize text with pre-fit TF-IDF
        X_vec = self.vectorizer.transform([clean])

        # 4. Select Model & Compute Probabilities
        if "DistilBERT" in model_choice and self.transformer_model is not None and self.transformer_tokenizer is not None:
            import torch
            inputs = self.transformer_tokenizer(
                text,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                logits = self.transformer_model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                prob_real = float(probs[1])
        elif model_choice == "Linear SVM" and self.svm_model is not None:
            active_model = self.svm_model
            # LinearSVC decision function score converted via logistic sigmoid
            df_score = float(active_model.decision_function(X_vec)[0])
            prob_real = float(1.0 / (1.0 + np.exp(-df_score)))
        else:
            active_model = self.logreg_model if self.logreg_model is not None else self.svm_model
            if hasattr(active_model, "predict_proba"):
                probs = active_model.predict_proba(X_vec)[0]
                prob_real = float(probs[1])
            else:
                df_score = float(active_model.decision_function(X_vec)[0])
                prob_real = float(1.0 / (1.0 + np.exp(-df_score)))

        prob_fake = 1.0 - prob_real
        verdict = "REAL" if prob_real >= 0.5 else "FAKE"
        confidence = prob_real if verdict == "REAL" else prob_fake

        # 5. Word-Level Explainability (Linear coefficient attribution)
        # Using Logistic Regression weights to explain token contributions
        expl_model = self.logreg_model if self.logreg_model is not None else active_model
        top_fake_words, top_real_words = self._explain_prediction(clean, X_vec, expl_model)

        return {
            "verdict": verdict,
            "confidence": round(confidence * 100, 2),
            "fake_probability": round(prob_fake * 100, 2),
            "real_probability": round(prob_real * 100, 2),
            "top_fake_words": top_fake_words,
            "top_real_words": top_real_words,
            "stylistic_stats": {
                "word_count": int(styles.get("word_count", 0)),
                "uppercase_ratio": round(float(styles.get("uppercase_ratio", 0.0)) * 100, 1),
                "exclamation_ratio": round(float(styles.get("exclamation_ratio", 0.0)), 2),
                "question_ratio": round(float(styles.get("question_ratio", 0.0)), 2),
                "avg_sentence_len": round(float(styles.get("avg_sentence_len", 0.0)), 1)
            }
        }

    def _explain_prediction(self, clean_text: str, X_vec, model) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """Calculates token contribution by multiplying TF-IDF value by model coefficient."""
        if not hasattr(model, "coef_"):
            return [], []

        feature_names = self.vectorizer.get_feature_names_out()
        coefs = model.coef_[0]

        # Non-zero indices in sparse TF-IDF vector
        nonzero_indices = X_vec.nonzero()[1]
        word_scores = []
        for idx in nonzero_indices:
            word = feature_names[idx]
            tfidf_val = X_vec[0, idx]
            weight = coefs[idx] * tfidf_val
            word_scores.append((word, float(weight)))

        # Positive weight pushes to Real (1), Negative weight pushes to Fake (0)
        fake_words = sorted([(w, abs(s)) for w, s in word_scores if s < 0], key=lambda x: x[1], reverse=True)[:6]
        real_words = sorted([(w, s) for w, s in word_scores if s > 0], key=lambda x: x[1], reverse=True)[:6]

        return fake_words, real_words

# Convenience singleton instance
_predictor = None

def predict_article(text: str, model_choice: str = "Logistic Regression") -> Dict[str, Any]:
    global _predictor
    if _predictor is None:
        _predictor = FakeNewsPredictor()
    return _predictor.predict(text, model_choice=model_choice)
