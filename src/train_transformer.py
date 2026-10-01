import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------------------------------------------------------
# 1. METRICS CALCULATION FOR HUGGING FACE TRAINER
# -------------------------------------------------------------------------
def compute_metrics(eval_pred) -> Dict[str, float]:
    """Computes accuracy, precision, recall, and macro-F1 for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(f1)
    }

# -------------------------------------------------------------------------
# 2. FINE-TUNING PIPELINE
# -------------------------------------------------------------------------
def train_distilbert(
    model_name: str = "distilbert-base-uncased",
    data_dir: str = "data/processed",
    output_dir: str = "models/distilbert-fake-news",
    sample_size: int = 15000,
    max_length: int = 256,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01
):
    """
    Fine-tunes DistilBERT on 'bert_text' (syntax, casing, and punctuation preserved):
    - Uses AutoTokenizer with max_length=256
    - Applies dynamic padding via DataCollatorWithPadding
    - Trains using Hugging Face Trainer with AdamW and linear lr schedule
    - Evaluates each epoch and saves the best model checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = torch.cuda.is_available()
    print(f"--- Training Transformer ({model_name}) | Compute Device: {device.upper()} (fp16={fp16}) ---")

    # 1. Load data
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    print(f"Loading datasets from {data_dir}...")
    train_df = pd.read_csv(train_path).dropna(subset=["bert_text", "label"])
    val_df = pd.read_csv(val_path).dropna(subset=["bert_text", "label"])

    # If sample_size is set and smaller than training size, take a stratified sample
    # (Useful for laptops without high-end GPU or Google Colab free tier)
    if sample_size and len(train_df) > sample_size:
        print(f"Subsampling training set to {sample_size} stratified rows for fast GPU/CPU execution...")
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=sample_size // 2, random_state=42)
        ).reset_index(drop=True)

    print(f"Train samples: {len(train_df)} | Validation samples: {len(val_df)}")

    # 2. Initialize Tokenizer & Model
    print(f"Initializing pre-trained tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Initializing sequence classification head: {model_name} (num_labels=2)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1}
    )

    # 3. Tokenize datasets
    def tokenize_function(batch):
        return tokenizer(
            batch["bert_text"],
            truncation=True,
            max_length=max_length
        )

    # Convert pandas to Hugging Face Datasets
    hf_train = Dataset.from_pandas(train_df[["bert_text", "label"]])
    hf_val = Dataset.from_pandas(val_df[["bert_text", "label"]])

    print("Tokenizing datasets...")
    tokenized_train = hf_train.map(tokenize_function, batched=True, remove_columns=["bert_text"])
    tokenized_val = hf_val.map(tokenize_function, batched=True, remove_columns=["bert_text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=fp16,
        logging_dir="reports/transformer_logs",
        logging_steps=100,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # 5. Train
    print("\nStarting Transformer Fine-Tuning...")
    train_result = trainer.train()

    # 6. Evaluate on Validation Set
    print("\nEvaluating fine-tuned transformer on validation split...")
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation Macro-F1: {eval_results['eval_macro_f1']:.4f}")

    # 7. Save Final Model and Tokenizer
    print(f"\nSaving fine-tuned model and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Transformer fine-tuning completed successfully!")

    return trainer, eval_results

if __name__ == "__main__":
    train_distilbert()
