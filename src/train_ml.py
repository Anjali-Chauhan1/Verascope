import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from src.features import fit_and_save_tfidf, load_tfidf_vectorizer

# -------------------------------------------------------------------------
# 1. METRICS EVALUATION HELPER
# -------------------------------------------------------------------------
def evaluate_model(model: Any, X: Any, y: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics for a binary classifier."""
    t0 = time.time()
    y_pred = model.predict(X)
    infer_time = time.time() - t0

    # Probability scores for ROC-AUC if supported
    roc_auc = 0.0
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        roc_auc = float(roc_auc_score(y, y_prob))
    elif hasattr(model, "decision_function"):
        # For LinearSVC and PassiveAggressive
        df_scores = model.decision_function(X)
        roc_auc = float(roc_auc_score(y, df_scores))

    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "roc_auc": roc_auc,
        "inference_time_sec": infer_time
    }

# -------------------------------------------------------------------------
# 2. MAIN TRAINING AND GRID-SEARCH PIPELINE
# -------------------------------------------------------------------------
def train_and_compare_classic_models(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports"
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Trains 6 classic ML algorithms on TF-IDF features:
    1. Multinomial Naive Bayes (fast baseline)
    2. Logistic Regression (interpretable, strong baseline)
    3. Passive Aggressive Classifier (online streaming learning)
    4. Linear SVM (LinearSVC) (usually the strongest on high-dim sparse text)
    5. Random Forest (non-linear ensemble)
    6. XGBoost (gradient boosted trees)
    
    Tunes the top 2 models using GridSearchCV and saves the best model to disk.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load data splits
    print("Loading preprocessed datasets...")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Drop any null clean texts if present
    train_df = train_df.dropna(subset=["clean_text", "label"]).copy()
    val_df = val_df.dropna(subset=["clean_text", "label"]).copy()
    test_df = test_df.dropna(subset=["clean_text", "label"]).copy()

    y_train = train_df["label"].values.astype(int)
    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    # 2. Extract or Load TF-IDF (1,2-grams, max 50,000 features)
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    if os.path.exists(tfidf_path):
        print(f"Loading cached TF-IDF vectorizer from {tfidf_path}...")
        vectorizer = load_tfidf_vectorizer(tfidf_path)
    else:
        print("Fitting TF-IDF vectorizer on training data...")
        vectorizer = fit_and_save_tfidf(train_df["clean_text"], save_path=tfidf_path, max_features=50000)

    print("Transforming text to TF-IDF sparse matrices...")
    X_train = vectorizer.transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])
    X_test = vectorizer.transform(test_df["clean_text"])

    print(f"Feature matrix shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 3. Define candidate model family
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=5.0, max_iter=1000, random_state=42),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=50, random_state=42),
        "Linear SVM (LinearSVC)": LinearSVC(C=0.8, random_state=42, max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=50, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric="logloss")
    }

    results = []
    trained_models = {}

    print("\n--- Training Candidate Models on WELFake Splits ---")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t_start
        trained_models[name] = model

        # Evaluate on validation split
        metrics = evaluate_model(model, X_val, y_val)
        metrics["model"] = name
        metrics["train_time_sec"] = train_time
        results.append(metrics)

        print(f" -> Val Accuracy: {metrics['accuracy']:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f} | Time: {train_time:.1f}s")

    results_df = pd.DataFrame(results).sort_values(by="macro_f1", ascending=False).reset_index(drop=True)
    print("\n=== Validation Split Leaderboard ===")
    print(results_df[["model", "accuracy", "macro_f1", "roc_auc", "train_time_sec"]].to_string(index=False))

    # Save candidate comparison table
    results_df.to_csv(os.path.join(reports_dir, "classic_ml_val_comparison.csv"), index=False)

    # 4. Hyperparameter Tuning with GridSearchCV on the Top 2 Models
    # Typically Logistic Regression and Linear SVM
    print("\n--- Phase 5 Tuning: GridSearchCV on Top 2 Models ---")
    
    # 4A. Tune Logistic Regression
    print("Tuning Logistic Regression (C values: [0.5, 1.0, 5.0, 10.0])...")
    lr_grid = {"C": [0.5, 1.0, 5.0, 10.0], "penalty": ["l2"]}
    lr_cv = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    lr_cv.fit(X_train, y_train)
    best_lr = lr_cv.best_estimator_
    print(f" -> Best Logistic Regression: C={lr_cv.best_params_['C']} (CV Macro-F1: {lr_cv.best_score_:.4f})")

    # 4B. Tune Linear SVM
    print("Tuning Linear SVM (C values: [0.1, 0.5, 1.0, 2.0])...")
    svm_grid = {"C": [0.1, 0.5, 1.0, 2.0]}
    svm_cv = GridSearchCV(LinearSVC(random_state=42, max_iter=2000), svm_grid, cv=3, scoring="f1_macro", n_jobs=-1)
    svm_cv.fit(X_train, y_train)
    best_svm = svm_cv.best_estimator_
    print(f" -> Best Linear SVM: C={svm_cv.best_params_['C']} (CV Macro-F1: {svm_cv.best_score_:.4f})")

    # 5. Evaluate Tuned Models on Held-out Test Set
    test_metrics_lr = evaluate_model(best_lr, X_test, y_test)
    test_metrics_svm = evaluate_model(best_svm, X_test, y_test)

    print("\n=== Final Test Split Performance (Held-out 10%) ===")
    print(f"Logistic Regression -> Test Acc: {test_metrics_lr['accuracy']:.4f} | Macro-F1: {test_metrics_lr['macro_f1']:.4f} | ROC-AUC: {test_metrics_lr['roc_auc']:.4f}")
    print(f"Linear SVM          -> Test Acc: {test_metrics_svm['accuracy']:.4f} | Macro-F1: {test_metrics_svm['macro_f1']:.4f} | ROC-AUC: {test_metrics_svm['roc_auc']:.4f}")

    # Determine winning classic model
    if test_metrics_svm["macro_f1"] >= test_metrics_lr["macro_f1"]:
        best_model = best_svm
        best_name = "LinearSVC"
    else:
        best_model = best_lr
        best_name = "LogisticRegression"

    # 6. Save the Best Classic Model and Logistic Regression (for explainability & LIME)
    best_model_path = os.path.join(models_dir, "best_classic_model.pkl")
    logreg_path = os.path.join(models_dir, "logreg_model.pkl")

    joblib.dump(best_model, best_model_path)
    joblib.dump(best_lr, logreg_path)

    print(f"\n Saved winning classic model ({best_name}) to: {best_model_path}")
    print(f" Saved tuned Logistic Regression to: {logreg_path}")

    return results_df, best_model, vectorizer

if __name__ == "__main__":
    train_and_compare_classic_models()
