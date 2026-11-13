# 🎬 Movie Review Sentiment Analysis — How to Run & How It Works

> **Model:** Bidirectional LSTM (BiLSTM) | **Framework:** PyTorch | **Dataset:** Stanford IMDb (50,000 reviews)  
> **Best Accuracy:** 83.84% | **F1-Score:** 84.60% | **ROC-AUC:** 0.9184

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [How the Project Works](#-how-the-project-works)
3. [File Structure](#-file-structure)
4. [Setup & Installation](#-setup--installation)
5. [Step-by-Step: Running the Project](#-step-by-step-running-the-project)
6. [Model Architecture Explained](#-model-architecture-explained)
7. [Understanding Each Script](#-understanding-each-script)
8. [Sample Output](#-sample-output)
9. [Performance Results](#-performance-results)
10. [Common Issues & Fixes](#-common-issues--fixes)

---

## 🎯 Project Overview

This project builds a **deep learning pipeline** that reads a movie review (plain English text) and predicts whether it is **Positive** or **Negative** sentiment — with a confidence score.

It uses a **Bidirectional LSTM (BiLSTM)** neural network, which reads sentences **both left-to-right and right-to-left**, capturing context that standard (one-direction) models miss — such as negations like *"not worth watching"*.

---

## 🔄 How the Project Works

The pipeline has **5 main stages**:

```
Raw Text Review
      │
      ▼
┌─────────────────────┐
│  1. Preprocessing   │  Clean HTML, URLs, special chars → lowercase
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  2. Tokenization    │  Split into words → map to integer IDs via vocab.json
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Padding         │  Fix sequence length to 250 tokens (pad/truncate)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  4. BiLSTM Model    │  Embedding → Dropout → 2-layer BiLSTM → Dropout → Linear
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  5. Output          │  Sigmoid activation → Probability → POSITIVE / NEGATIVE
└─────────────────────┘
```

### Data Flow in Detail

| Stage | Input | Output | File Responsible |
|-------|-------|--------|-----------------|
| Text Cleaning | Raw review string | Cleaned lowercase string | `utils/preprocessing.py` |
| Encoding | Cleaned string | List of token IDs | `utils/dataset.py` |
| Padding | Token ID list | Fixed-length tensor [250] | `utils/dataset.py` |
| Embedding | Token ID tensor | Dense vectors [250 × 128] | `models/model.py` |
| BiLSTM | Embedded vectors | Context-aware hidden states | `models/model.py` |
| Classification | Concatenated hidden states | Sentiment probability (0–1) | `models/model.py` |

---

## 📁 File Structure

```
deeplearning/
│
├── 📄 train.py               ← Trains the BiLSTM model for 6 epochs
├── 📄 evaluate.py            ← Evaluates on 25,000 test reviews
├── 📄 predict.py             ← Interactive CLI: type a review, get sentiment
├── 📄 generate_report_pdf.py ← Generates a formatted PDF report
├── 📄 requirements.txt       ← All Python dependencies
│
├── 📂 data/
│   ├── download_data.py      ← Downloads & preprocesses IMDb dataset from HuggingFace
│   └── processed/
│       └── vocab.json        ← 10,000-word vocabulary (word → integer ID)
│
├── 📂 models/
│   └── model.py              ← SentimentBiLSTM class definition (PyTorch)
│
├── 📂 utils/
│   ├── preprocessing.py      ← clean_text() — strips HTML, URLs, punctuation
│   └── dataset.py            ← IMDBDataset class + encode_and_pad() function
│
└── 📂 results/
    ├── bilstm_sentiment.pt   ← ✅ Pre-trained model weights (ready to use!)
    ├── training_history.json ← Epoch-wise loss & accuracy logs
    ├── training_curves.png   ← Loss & accuracy plots
    ├── confusion_matrix.png  ← Test set confusion matrix
    ├── workflow_diagram.png  ← Pipeline schematic
    └── model_architecture.png← BiLSTM architecture diagram
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python **3.9 or higher**
- pip (Python package manager)
- ~2 GB free disk space (for dataset download)
- GPU optional — CPU works fine for prediction

### Step 1 — Create a Virtual Environment *(Recommended)*

```bash
# Windows
python -m venv ml_env
ml_env\Scripts\activate

# Linux / macOS
python3 -m venv ml_env
source ml_env/bin/activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.1.0 | Deep learning framework (PyTorch) |
| `datasets` | ≥ 2.18.0 | Load IMDb dataset from HuggingFace Hub |
| `scikit-learn` | ≥ 1.4.0 | Metrics: accuracy, F1, confusion matrix |
| `matplotlib` | ≥ 3.8.0 | Plotting training curves |
| `seaborn` | ≥ 0.13.0 | Styled confusion matrix heatmap |
| `numpy` | ≥ 1.26.0 | Numerical array operations |
| `tqdm` | ≥ 4.66.0 | Progress bars during training |

---

## 🚀 Step-by-Step: Running the Project

### ▶️ Option A — Just Run Predictions (Fastest — Model Already Trained!)

The repo includes pre-trained weights (`results/bilstm_sentiment.pt`), so you can **skip training** and start predicting immediately:

```bash
cd "C:\Users\nagam\OneDrive\Desktop\deeplearning"
python predict.py
```

---

### ▶️ Option B — Full Pipeline (Train from Scratch)

Follow these steps in order:

#### Step 1: Download & Preprocess the Dataset

```bash
python data/download_data.py
```

- Downloads the **Stanford IMDb dataset** (50,000 reviews) via HuggingFace `datasets`
- Splits into: **20,000 train** / **5,000 validation** / **25,000 test**
- Builds a **10,000-word vocabulary** (`data/processed/vocab.json`)
- Saves all splits to `data/processed/dataset_splits.json`
- ⏱ Takes ~2–5 minutes depending on internet speed

#### Step 2: Train the Model

```bash
python train.py
```

- Trains the BiLSTM for **6 epochs** using Adam optimizer
- Batch size: **64** | Learning rate: **0.001**
- Automatically saves the **best checkpoint** (by validation loss) to `results/bilstm_sentiment.pt`
- ⏱ ~5–15 minutes on CPU | ~1–2 minutes on GPU

**Training output looks like:**
```
[INFO] Training on device: cpu
[INFO] Starting training loop...
Epoch 1/6: 100%|████████████| 313/313
Epoch 1 -> Train Loss: 0.5821 | Train Acc: 0.6912 || Val Loss: 0.4723 | Val Acc: 0.7788
...
Epoch 6 -> Train Loss: 0.2834 | Train Acc: 0.8891 || Val Loss: 0.3812 | Val Acc: 0.8456
[SUCCESS] Model training complete and best checkpoint saved to results/bilstm_sentiment.pt
```

#### Step 3: Evaluate the Model

```bash
python evaluate.py
```

- Loads the saved model weights
- Runs inference on all **25,000 test reviews**
- Prints: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Saves `results/confusion_matrix.png` and `results/training_curves.png`

#### Step 4: Interactive Predictions

```bash
python predict.py
```

- Launches an interactive terminal session
- Type any movie review → get **sentiment + confidence %**
- Type `exit` to quit

---

## 🧠 Model Architecture Explained

The model is defined in `models/model.py` as `SentimentBiLSTM`:

```
Input: Token IDs  [Batch × 250]
        │
        ▼
  ┌─────────────────────────────────┐
  │  Embedding Layer                │
  │  vocab_size=10000, dim=128      │  → [Batch × 250 × 128]
  └───────────────┬─────────────────┘
                  │  Dropout (p=0.3)
                  ▼
  ┌─────────────────────────────────┐
  │  Bidirectional LSTM             │
  │  hidden=128, layers=2           │
  │  → Forward LSTM  (left→right)   │  hidden[-1]  [Batch × 128]
  │  → Backward LSTM (right→left)   │  hidden[-2]  [Batch × 128]
  └───────────────┬─────────────────┘
                  │  Concatenate forward + backward hidden states
                  │  → [Batch × 256]
                  │  Dropout (p=0.3)
                  ▼
  ┌─────────────────────────────────┐
  │  Fully Connected Layer          │
  │  Linear(256 → 1)                │  → [Batch × 1]  (raw logit)
  └───────────────┬─────────────────┘
                  │  Sigmoid (at inference)
                  ▼
           Probability (0.0 – 1.0)
           ≥ 0.5 → POSITIVE
           < 0.5 → NEGATIVE
```

**Why Bidirectional?**  
A standard LSTM reads left-to-right only. A BiLSTM also reads right-to-left, then concatenates both hidden states — giving the model full sentence context. This is critical for negations:  
- *"The movie was **not** bad at all"* — the word "not" only makes sense when the full sentence is seen.

**Total Parameters:** 1,939,713 (~7.7 MB)

---

## 📄 Understanding Each Script

### `utils/preprocessing.py` — Text Cleaner
```python
clean_text("I loved <br/> this film!! Visit http://imdb.com")
# → "i loved  this film"
```
Removes: HTML tags, URLs, special characters, extra whitespace → lowercase

---

### `utils/dataset.py` — Tokenizer & Dataset
```python
encode_and_pad("i loved this film", vocab, max_len=250)
# → [45, 892, 231, 1104, 0, 0, ..., 0]  # 250 integers
```
- Maps each word to its integer ID from `vocab.json`
- Unknown words → ID `1` (UNK token)
- Pads short sequences with `0` (PAD token)
- `IMDBDataset` wraps this into a PyTorch `Dataset`

---

### `models/model.py` — The Neural Network
Defines `SentimentBiLSTM(nn.Module)` with:
- `Embedding` → converts token IDs to dense vectors
- `LSTM(bidirectional=True)` → processes sequences both directions
- `Linear(256, 1)` → outputs a single logit score

---

### `train.py` — Training Loop
- Uses `BCEWithLogitsLoss` (binary cross-entropy with built-in sigmoid)
- **Gradient clipping** (`max_norm=1.0`) prevents exploding gradients
- Saves only the **best model** (lowest validation loss) — not the last epoch

---

### `evaluate.py` — Metrics Reporter
Runs the saved model on the full 25k test set and reports:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Generates confusion matrix + training curve plots

---

### `predict.py` — Inference CLI
```python
# How it works internally:
review → clean_text() → encode_and_pad() → model() → sigmoid() → sentiment + confidence
```

---

## 💬 Sample Output

```
=== Interactive Sentiment Analyzer (Type 'exit' to quit) ===

Enter movie review: The cinematography was breathtaking and the direction was sublime!
Result: POSITIVE (94.20% confidence)

Enter movie review: Boring plot, terrible acting, complete waste of time.
Result: NEGATIVE (98.60% confidence)

Enter movie review: Not bad, but not great either. Some parts were confusing.
Result: NEGATIVE (61.30% confidence)

Enter movie review: exit
```

---

## 📊 Performance Results

Evaluated on **25,000 unseen test reviews**:

| Metric | Score | What It Means |
|--------|-------|---------------|
| **Accuracy** | **83.84%** | 20,960 / 25,000 reviews correctly classified |
| **Precision** | **80.75%** | Of reviews predicted Positive, 80.75% actually were |
| **Recall** | **88.87%** | Correctly found 88.87% of all actual Positive reviews |
| **F1-Score** | **84.60%** | Balanced precision-recall score |
| **ROC-AUC** | **0.9184** | Near-perfect separation between classes |

**Confusion Matrix (25k Test Set):**

```
                Predicted Negative   Predicted Positive
Actual Negative       10,851               1,649
Actual Positive        1,391              11,109
```

---

## 🛠 Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: dataset_splits.json` | Dataset not downloaded yet | Run `python data/download_data.py` first |
| `FileNotFoundError: bilstm_sentiment.pt` | Model not trained / weights missing | Run `python train.py` or restore from repo |
| `ModuleNotFoundError: torch` | PyTorch not installed | Run `pip install -r requirements.txt` |
| Slow training | No GPU | Normal on CPU — takes ~10–15 min; GPU reduces to ~2 min |
| `CUDA out of memory` | GPU memory too small | Reduce batch size in `train.py`: change `batch_size=64` to `32` |
| Low accuracy on custom reviews | Very short / ambiguous text | BiLSTM needs context — write full sentences for best results |

---

## 📌 Quick Reference Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset (only needed if training from scratch)
python data/download_data.py

# Train the model (optional — weights already included)
python train.py

# Evaluate on test set
python evaluate.py

# Run interactive predictions ← START HERE
python predict.py
```

---

*Built with PyTorch · Stanford IMDb Dataset · BiLSTM Architecture*

