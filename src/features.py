import os
import re
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# -------------------------------------------------------------------------
# 1. BAG-OF-WORDS & TF-IDF VECTORIZER BUILDERS
# -------------------------------------------------------------------------

def build_bow_vectorizer(max_features: int = 5000) -> CountVectorizer:
    """Creates a standard CountVectorizer baseline."""
    return CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 1),
        token_pattern=r"(?u)\b\w+\b"
    )

def build_tfidf_vectorizer(
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Creates an optimized TF-IDF vectorizer:
    - (1, 2) n-grams capture both key terms and two-word deceptive phrases (e.g. 'breaking news')
    - min_df=2 ignores one-off typos
    - sublinear_tf=True applies 1 + log(tf) scaling to temper extreme frequency outliers
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        token_pattern=r"(?u)\b\w+\b"
    )

def fit_and_save_tfidf(
    train_texts: pd.Series,
    save_path: str = "models/tfidf_vectorizer.pkl",
    max_features: int = 50000
) -> TfidfVectorizer:
    """Fits TF-IDF exclusively on training text and serializes it with joblib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    vectorizer = build_tfidf_vectorizer(max_features=max_features)
    vectorizer.fit(train_texts)
    joblib.dump(vectorizer, save_path)
    print(f"TF-IDF vectorizer fit on {len(train_texts)} samples and saved to {save_path}")
    return vectorizer

def load_tfidf_vectorizer(load_path: str = "models/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    """Loads a pre-fit TF-IDF vectorizer."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Vectorizer file not found at: {load_path}")
    return joblib.load(load_path)

# -------------------------------------------------------------------------
# 2. HANDCRAFTED STYLISTIC & PUNCTUATION FEATURES
# -------------------------------------------------------------------------

def extract_handcrafted_features(df: pd.DataFrame, text_col: str = "content") -> pd.DataFrame:
    """
    Extracts numerical stylistic markers that distinguish fake from real news:
    1. word_count: Total word count
    2. char_count: Total character count
    3. uppercase_ratio: Proportion of ALL-CAPS words (clickbait/outrage indicator)
    4. exclamation_ratio: Frequency of '!' per 100 words
    5. question_ratio: Frequency of '?' per 100 words
    6. avg_word_length: Average characters per token
    7. avg_sentence_len: Average words per sentence
    """
    features = pd.DataFrame(index=df.index)
    texts = df[text_col].fillna("").astype(str)

    # Word & Char counts
    words_series = texts.apply(lambda s: s.split())
    features["word_count"] = words_series.apply(len)
    features["char_count"] = texts.apply(len)

    # Uppercase ratio
    def get_upper_ratio(words):
        alpha_words = [w for w in words if w.isalpha() and len(w) >= 2]
        if not alpha_words:
            return 0.0
        return sum(1 for w in alpha_words if w.isupper()) / len(alpha_words)

    features["uppercase_ratio"] = words_series.apply(get_upper_ratio)

    # Exclamation & question ratios per 100 words
    safe_word_count = features["word_count"].replace(0, 1)
    features["exclamation_ratio"] = (texts.apply(lambda s: s.count('!')) / safe_word_count) * 100
    features["question_ratio"] = (texts.apply(lambda s: s.count('?')) / safe_word_count) * 100

    # Average word length
    features["avg_word_length"] = features["char_count"] / safe_word_count

    # Average sentence length
    def get_avg_sent_len(text):
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 0]
        if not sentences:
            return 0.0
        return float(np.mean([len(s.split()) for s in sentences]))

    features["avg_sentence_len"] = texts.apply(get_avg_sent_len)

    return features

# -------------------------------------------------------------------------
# 3. SEQUENCE TOKENIZER & GLOVE EMBEDDING MATRIX (FOR DL / LSTM)
# -------------------------------------------------------------------------

class SimpleSequenceTokenizer:
    """
    Lightweight, self-contained sequence tokenizer for LSTM sequence padding.
    Works natively without requiring TensorFlow/Keras overhead.
    """
    def __init__(self, num_words: int = 30000, oov_token: str = "<UNK>"):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {self.oov_token: 1}
        self.word_counts = {}

    def fit_on_texts(self, texts):
        for text in texts:
            for word in str(text).split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for idx, (word, _) in enumerate(sorted_words[:self.num_words - 2], start=2):
            self.word_index[word] = idx

    def texts_to_sequences(self, texts, maxlen: int = 300) -> np.ndarray:
        """Converts texts to token ID sequences with post-padding/truncation."""
        sequences = np.zeros((len(texts), maxlen), dtype=np.int32)
        for row_idx, text in enumerate(texts):
            tokens = str(text).split()[:maxlen]
            for col_idx, token in enumerate(tokens):
                token_id = self.word_index.get(token, self.word_index[self.oov_token])
                sequences[row_idx, col_idx] = token_id
        return sequences

def load_glove_matrix(
    word_index: Dict[str, int],
    glove_path: Optional[str] = None,
    embedding_dim: int = 100
) -> np.ndarray:
    """
    Builds the embedding matrix from GloVe pre-trained vectors.
    If GloVe weights file is not present locally, initializes standard normal weights.
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.random.normal(scale=0.05, size=(vocab_size, embedding_dim)).astype(np.float32)
    embedding_matrix[0] = 0  # Zero padding

    if glove_path and os.path.exists(glove_path):
        print(f"Loading GloVe embeddings from {glove_path}...")
        with open(glove_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                word = parts[0]
                if word in word_index:
                    idx = word_index[word]
                    embedding_matrix[idx] = np.asarray(parts[1:], dtype=np.float32)
        print("GloVe matrix populated.")
    else:
        print(f"[INFO] GloVe file not found at '{glove_path}'. Using initialized embedding weights.")
        
    return embedding_matrix
