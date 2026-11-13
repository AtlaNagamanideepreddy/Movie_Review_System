import json
import torch
from models.model import SentimentBiLSTM
from utils.preprocessing import clean_text
from utils.dataset import encode_and_pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("data/processed/vocab.json", "r") as f:
    vocab = json.load(f)

model = SentimentBiLSTM(vocab_size=len(vocab)).to(device)
model.load_state_dict(torch.load("results/bilstm_sentiment.pt", map_location=device, weights_only=True))
model.eval()

def predict(review: str):
    cleaned = clean_text(review)
    seq = encode_and_pad(cleaned, vocab, max_len=250)
    tensor_x = torch.tensor([seq], dtype=torch.long).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor_x)).item()
    sentiment = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    confidence = prob if prob >= 0.5 else 1 - prob
    return sentiment, confidence * 100

if __name__ == "__main__":
    print("=== Interactive Sentiment Analyzer (Type 'exit' to quit) ===")
    while True:
        text = input("\nEnter movie review: ").strip()
        if text.lower() == "exit":
            break
        if not text:
            continue
        sentiment, conf = predict(text)
        print(f"Result: {sentiment} ({conf:.2f}% confidence)")