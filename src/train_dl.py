import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure src modules can be resolved
sys.path.append(os.path.abspath("."))
from src.features import SimpleSequenceTokenizer, load_glove_matrix

# Set fixed random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# 1. PYTORCH DATASET CLASS
# -------------------------------------------------------------------------
class FakeNewsSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# -------------------------------------------------------------------------
# 2. BIDIRECTIONAL LSTM ARCHITECTURE
# -------------------------------------------------------------------------
class BiLSTMClassifier(nn.Module):
    """
    BiLSTM Neural Classifier:
    Embedding(100d) -> BiLSTM(hidden=64, num_layers=1) -> Dropout(0.3) -> 
    Dense(128 -> 32, ReLU) -> Dense(32 -> 1, Sigmoid output via BCEWithLogits)
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_dim: int = 64,
        embedding_matrix: np.ndarray = None,
        dropout_rate: float = 0.3
    ):
        super(BiLSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout_rate)
        # BiLSTM concatenates forward and backward hidden states: 64 * 2 = 128
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        embedded = self.embedding(x)          # (batch_size, seq_len, 100)
        lstm_out, (hn, cn) = self.lstm(embedded)

        # Concatenate final hidden states from forward and backward passes
        out = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1) # (batch_size, 128)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))         # (batch_size, 32)
        out = self.dropout(out)
        logits = self.fc2(out).squeeze(1)     # (batch_size)
        return logits

# -------------------------------------------------------------------------
# 3. TRAINING & EARLY STOPPING PIPELINE
# -------------------------------------------------------------------------
def train_bilstm(
    data_dir: str = "data/processed",
    models_dir: str = "models",
    figures_dir: str = "reports/figures",
    max_vocab: int = 25000,
    max_len: int = 300,
    batch_size: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    patience: int = 2,
    glove_path: str = None
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"BiLSTM Training Pipeline | Device: {DEVICE}")
    print("Loading data splits...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv")).dropna(subset=["clean_text", "label"])
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv")).dropna(subset=["clean_text", "label"])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv")).dropna(subset=["clean_text", "label"])

    print(f"Tokenizing sequences (max_vocab={max_vocab}, max_len={max_len})...")
    tokenizer = SimpleSequenceTokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(train_df["clean_text"])

    # Serialize tokenizer for production
    joblib.dump(tokenizer, os.path.join(models_dir, "dl_tokenizer.pkl"))

    X_train = tokenizer.texts_to_sequences(train_df["clean_text"], maxlen=max_len)
    X_val = tokenizer.texts_to_sequences(val_df["clean_text"], maxlen=max_len)
    X_test = tokenizer.texts_to_sequences(test_df["clean_text"], maxlen=max_len)

    y_train = train_df["label"].values.astype(np.float32)
    y_val = val_df["label"].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    train_loader = DataLoader(FakeNewsSequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FakeNewsSequenceDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FakeNewsSequenceDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    embedding_matrix = load_glove_matrix(tokenizer.word_index, glove_path=glove_path, embedding_dim=100)
    vocab_size = len(tokenizer.word_index) + 1

    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=64,
        embedding_matrix=embedding_matrix,
        dropout_rate=0.3
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_model_path = os.path.join(models_dir, "bilstm_model.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for seqs, labels in train_loader:
            seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
                logits = model(seqs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{epoch_time:.1f}s] - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best checkpoint (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EarlyStopping triggered after {epoch} epochs]")
                break

    # Plot loss and accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train Loss", color="#e74c3c", lw=2)
    ax1.plot(history["val_loss"], label="Val Loss", color="#3498db", lw=2, linestyle="--")
    ax1.set_title("BiLSTM Loss Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Accuracy", color="#2ecc71", lw=2)
    ax2.plot(history["val_acc"], label="Val Accuracy", color="#9b59b6", lw=2, linestyle="--")
    ax2.set_title("BiLSTM Accuracy Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    curve_path = os.path.join(figures_dir, "bilstm_learning_curves.png")
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f" Saved learning curves to {curve_path}")

    # Evaluate on held-out test split
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(DEVICE)
            probs = torch.sigmoid(model(seqs)).cpu().numpy()
            all_preds.extend((probs >= 0.5).astype(int))
            all_targets.extend(labels.numpy().astype(int))

    from sklearn.metrics import accuracy_score, f1_score
    test_acc = accuracy_score(all_targets, all_preds)
    test_f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"\n=== BiLSTM Test Split Performance ===")
    print(f"Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Macro F1 Score: {test_f1:.4f}")

    return model, history

if __name__ == "__main__":
    train_bilstm()
