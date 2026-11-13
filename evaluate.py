import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score

from utils.dataset import IMDBDataset
from models.model import SentimentBiLSTM

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    with open("data/processed/dataset_splits.json", "r") as f:
        data = json.load(f)
    with open("data/processed/vocab.json", "r") as f:
        vocab = json.load(f)

    test_ds = IMDBDataset(data["test_texts"], data["test_labels"], vocab)
    test_loader = DataLoader(test_ds, batch_size=64)

    model = SentimentBiLSTM(vocab_size=len(vocab)).to(device)
    model.load_state_dict(torch.load("results/bilstm_sentiment.pt", map_location=device))
    model.eval()

    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_batch.numpy().astype(int))

    acc = accuracy_score(all_targets, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary")
    auc = roc_auc_score(all_targets, all_probs)

    print("\n================ EVALUATION METRICS ================")
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Precision     : {prec:.4f}")
    print(f"Recall        : {rec:.4f}")
    print(f"F1-Score      : {f1:.4f}")
    print(f"ROC-AUC Score : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=["Negative", "Positive"]))

    # Generate Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    plt.title("Confusion Matrix — Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close()

    # Generate Training Curves Plot if history exists
    if os_path_exists := os.path.exists("results/training_history.json"):
        with open("results/training_history.json", "r") as f:
            h = json.load(f)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        epochs = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(epochs, h["train_loss"], marker="o", label="Train Loss")
        axes[0].plot(epochs, h["val_loss"], marker="s", label="Val Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, h["train_acc"], marker="o", label="Train Acc")
        axes[1].plot(epochs, h["val_acc"], marker="s", label="Val Acc")
        axes[1].set_title("Accuracy Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/training_curves.png", dpi=300)
        plt.close()

    print("[SUCCESS] Plots saved in results/")

if __name__ == "__main__":
    import os
    evaluate_model()