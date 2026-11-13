import os
import re
import json
from collections import Counter
import numpy as np
from datasets import load_dataset
from utils.preprocessing import clean_text

def build_vocab(texts, max_vocab=10000):
    """Builds frequency-ranked word-to-index dictionary."""
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

def prepare_data():
    os.makedirs("data/processed", exist_ok=True)
    print("[INFO] Downloading IMDb dataset...")
    
    # FIX: Use namespaced path 'stanfordnlp/imdb' instead of just 'imdb'
    raw_dataset = load_dataset("stanfordnlp/imdb")

    train_texts = [clean_text(t) for t in raw_dataset["train"]["text"]]
    train_labels = list(raw_dataset["train"]["label"])
    test_texts = [clean_text(t) for t in raw_dataset["test"]["text"]]
    test_labels = list(raw_dataset["test"]["label"])

    # Shuffle & split train into train (20k) and validation (5k)
    np.random.seed(42)
    perm = np.random.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in perm]
    train_labels = [train_labels[i] for i in perm]

    val_texts, val_labels = train_texts[20000:], train_labels[20000:]
    train_texts, train_labels = train_texts[:20000], train_labels[:20000]

    print(f"[INFO] Building vocabulary (Size: 10,000)...")
    vocab = build_vocab(train_texts, max_vocab=10000)

    # Save processed data splits and vocabulary
    data_bundle = {
        "train_texts": train_texts, "train_labels": train_labels,
        "val_texts": val_texts, "val_labels": val_labels,
        "test_texts": test_texts, "test_labels": test_labels
    }
    
    with open("data/processed/dataset_splits.json", "w") as f:
        json.dump(data_bundle, f)
        
    with open("data/processed/vocab.json", "w") as f:
        json.dump(vocab, f)

    print("[SUCCESS] Data downloaded, preprocessed, and saved in data/processed/")

if __name__ == "__main__":
    prepare_data()