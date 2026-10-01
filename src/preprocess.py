import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure required NLTK resources are available
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def remove_leakage(text: str) -> str:
    """
    Strips source metadata and publisher signatures that cause artificial leakage.
    Specifically targets ISOT Reuters datelines like 'WASHINGTON (Reuters) - '.
    """
    text = str(text)
    # Remove datelines at the beginning of the text: e.g. "WASHINGTON (Reuters) - "
    text = re.sub(r"^\s*[\w\s,./\(\)]*?\(reuters\)\s*[-—–]\s*", "", text, flags=re.IGNORECASE)
    # Remove standalone mentions of 'reuters' anywhere in text
    text = re.sub(r"\breuters\b", "", text, flags=re.IGNORECASE)
    return text

def clean_text_classic(text: str) -> str:
    """
    Full normalization pipeline for Classic ML (BoW, TF-IDF) and sequence models (LSTM):
    1. Remove publisher leakage
    2. Lowercase
    3. Remove URLs and HTML tags
    4. Remove non-alphabet characters (digits, symbols, punctuation)
    5. Tokenize, remove stopwords, and lemmatize tokens (> 2 characters)
    """
    text = remove_leakage(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)       # Strip URLs
    text = re.sub(r"<.*?>", " ", text)                      # Strip HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                   # Strip punctuation and numbers
    
    tokens = text.split()
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]
    return " ".join(cleaned_tokens)

def clean_text_transformer(text: str) -> str:
    """
    Lightweight normalization for contextual transformers (BERT / RoBERTa):
    1. Remove publisher leakage
    2. Strip URLs and raw HTML
    3. Normalize excessive whitespace
    Preserves casing, punctuation, and stopwords needed by WordPiece/BPE tokenizers.
    """
    text = remove_leakage(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
