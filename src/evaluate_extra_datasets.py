"""
Cross-dataset generalization check: scores the classic ML models --
already trained on WELFake TF-IDF features -- against the extra corpora
fetched by `src/load_extra_datasets.py` (LIAR, COVID-19 Fake News,
FakeNewsNet titles), WITHOUT any retraining.

This mirrors the blueprint's own doctrine (Section 6): "always add one
cross-dataset result: that single row is what separates a strong project
from a copied tutorial." A visible accuracy drop here is expected and
correct -- it demonstrates the topic/domain shift documented in
reports/cross_dataset_results.csv, not a bug.

Run after `python -m src.load_extra_datasets --all`:
    python -m src.evaluate_extra_datasets
"""
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

MODELS_DIR = "models"
EXTRA_DIR = os.path.join("data", "processed", "extra")
REPORTS_DIR = "reports"

DATASETS = {
    "LIAR (binary, short political statements)": "liar_binary.csv",
    "COVID-19 Fake News (social media posts)": "covid_fake_news.csv",
    "FakeNewsNet (title-only, cross-domain)": "fakenewsnet_titles.csv",
}


def _score_model(model, X, y_true):
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def main():
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "logreg_model.pkl")),
        "Linear SVM": joblib.load(os.path.join(MODELS_DIR, "best_classic_model.pkl")),
    }

    rows = []
    for label, filename in DATASETS.items():
        path = os.path.join(EXTRA_DIR, filename)
        if not os.path.exists(path):
            print(f"[skip] {path} not found -- run `python -m src.load_extra_datasets --all` first.")
            continue

        df = pd.read_csv(path).dropna(subset=["clean_text", "label"])
        X = vectorizer.transform(df["clean_text"])
        y_true = df["label"].astype(int).values

        print(f"\n=== {label} ({len(df)} rows) ===")
        for model_name, model in models.items():
            m = _score_model(model, X, y_true)
            print(f"  {model_name:<22} acc={m['accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}  roc_auc={m['roc_auc']:.3f}")
            rows.append({"dataset": label, "model": model_name, "n_rows": len(df), **m})

    if rows:
        out_df = pd.DataFrame(rows)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        out_path = os.path.join(REPORTS_DIR, "extra_dataset_results.csv")
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
