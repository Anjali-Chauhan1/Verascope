import os
import sys
import numpy as np
import pandas as pd

# Add repo root to path
sys.path.append(os.path.abspath("."))
from src.features import build_bow_vectorizer, build_tfidf_vectorizer, extract_handcrafted_features, SimpleSequenceTokenizer

print("Testing Feature Extraction Pipelines...")

sample_data = {
    "title": [
        "BREAKING: SHOCKING Scandal Rocks Capital City!!!",
        "Economic Growth Report Shows Moderate 2.1 Percent Increase in Q3",
        "YOU WON'T BELIEVE What Secret Agents Found In The Basement???",
        "Federal Reserve Decides To Keep Benchmark Interest Rates Unchanged"
    ],
    "text": [
        "Unbelievable leaked documents reveal explosive allegations that the mainstream media refuses to cover! Share this now before it gets banned!",
        "The national statistics bureau released quarterly estimates today, indicating steady consumer spending and moderate industrial production across major sectors.",
        "Anonymous sources claim high-level officials held covert meetings in underground bunker. Experts warn democracy is in immediate danger! Click here for video!",
        "In a scheduled policy statement following their two-day meeting, central bank officials voted unanimously to maintain target borrowing costs between 5.0 and 5.25 percent."
    ],
    "content": [
        "BREAKING: SHOCKING Scandal Rocks Capital City!!! Unbelievable leaked documents reveal explosive allegations that the mainstream media refuses to cover! Share this now before it gets banned!",
        "Economic Growth Report Shows Moderate 2.1 Percent Increase in Q3 The national statistics bureau released quarterly estimates today, indicating steady consumer spending and moderate industrial production across major sectors.",
        "YOU WON'T BELIEVE What Secret Agents Found In The Basement??? Anonymous sources claim high-level officials held covert meetings in underground bunker. Experts warn democracy is in immediate danger! Click here for video!",
        "Federal Reserve Decides To Keep Benchmark Interest Rates Unchanged In a scheduled policy statement following their two-day meeting, central bank officials voted unanimously to maintain target borrowing costs between 5.0 and 5.25 percent."
    ],
    "label": [0, 1, 0, 1]
}

df = pd.DataFrame(sample_data)

# 1. BoW
bow_vec = build_bow_vectorizer(max_features=20)
X_bow = bow_vec.fit_transform(df["content"])
print(f"BoW Shape: {X_bow.shape} (Vocabulary size: {len(bow_vec.vocabulary_)})")

# 2. TF-IDF
tfidf_vec = build_tfidf_vectorizer(max_features=25, ngram_range=(1, 2))
X_tfidf = tfidf_vec.fit_transform(df["content"])
print(f"TF-IDF Shape: {X_tfidf.shape}")
print("Top TF-IDF feature names sample:", tfidf_vec.get_feature_names_out()[:6])

# 3. Handcrafted features
handcrafted = extract_handcrafted_features(df, text_col="content")
print("\nHandcrafted Stylistic Features:")
print(handcrafted[["word_count", "uppercase_ratio", "exclamation_ratio", "question_ratio", "avg_sentence_len"]].round(2))

# 4. Sequence Tokenizer
seq_tok = SimpleSequenceTokenizer(num_words=100)
seq_tok.fit_on_texts(df["content"])
seqs = seq_tok.texts_to_sequences(df["content"], maxlen=15)
print(f"\nSequence Array Shape: {seqs.shape}")
print("Sample Sequence Row 0:", seqs[0])

print("\nAll Feature Tests Passed Successfully!")
