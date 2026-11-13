import torch
from torch.utils.data import Dataset

def encode_and_pad(text: str, vocab: dict, max_len: int) -> list:
    """Converts text string to fixed-length integer sequence."""
    tokens = text.split()[:max_len]
    ids = [vocab.get(tok, 1) for tok in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=250):
        self.sequences = [encode_and_pad(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )