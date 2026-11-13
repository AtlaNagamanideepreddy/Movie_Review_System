import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import IMDBDataset
from models.model import SentimentBiLSTM

def train_model():
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # Load preprocessed data and vocab
    with open("data/processed/dataset_splits.json", "r") as f:
        data = json.load(f)
    with open("data/processed/vocab.json", "r") as f:
        vocab = json.load(f)

    train_ds = IMDBDataset(data["train_texts"], data["train_labels"], vocab)
    val_ds = IMDBDataset(data["val_texts"], data["val_labels"], vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = SentimentBiLSTM(vocab_size=len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    epochs = 6
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n[INFO] Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss += loss.item() * x_batch.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            t_correct += (preds == y_batch).sum().item()
            t_total += y_batch.size(0)

        train_loss = t_loss / t_total
        train_acc = t_correct / t_total

        # Validation phase
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                v_loss += loss.item() * x_batch.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                v_correct += (preds == y_batch).sum().item()
                v_total += y_batch.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch} -> Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/bilstm_sentiment.pt")

    with open("results/training_history.json", "w") as f:
        json.dump(history, f)
    print("[SUCCESS] Model training complete and best checkpoint saved to results/bilstm_sentiment.pt")

if __name__ == "__main__":
    train_model()