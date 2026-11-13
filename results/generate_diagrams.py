import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs("results", exist_ok=True)

def create_workflow_diagram():
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=300)
    ax.axis("off")

    steps = [
        "1. Raw Movie Reviews\n(IMDb 50,000 Dataset)",
        "2. Text Preprocessing\n(Cleaning, Lowercasing,\nRegex & Tokenization)",
        "3. Vocab & Encoding\n(Top 10k Vocab,\nPadding to Len 250)",
        "4. Deep Learning Model\n(Embedding + 2-Layer\nBidirectional LSTM)",
        "5. Output & Evaluation\n(Sigmoid Probability,\nBinary Classification)"
    ]

    n = len(steps)
    box_w = 1.45
    box_h = 1.6
    gap = 0.55
    start_x = 0.2
    y = 0.8

    colors = ["#E3F2FD", "#EDE7F6", "#E8F5E9", "#FFF3E0", "#FCE4EC"]
    border_colors = ["#1976D2", "#512DA8", "#388E3C", "#F57C00", "#C2185B"]

    for i, step in enumerate(steps):
        x = start_x + i * (box_w + gap)
        # Draw box
        rect = patches.FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.1,rounding_size=0.15",
            facecolor=colors[i], edgecolor=border_colors[i], linewidth=2.0
        )
        ax.add_patch(rect)
        ax.text(
            x + box_w/2, y + box_h/2, step,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color="#212121", family="sans-serif", multialignment="center"
        )

        # Draw arrow to next step
        if i < n - 1:
            arrow_x = x + box_w + 0.05
            arrow_w = gap - 0.1
            ax.annotate(
                "", xy=(arrow_x + arrow_w, y + box_h/2), xytext=(arrow_x, y + box_h/2),
                arrowprops=dict(facecolor="#424242", edgecolor="#424242", arrowstyle="-|>", lw=2, mutation_scale=15)
            )

    ax.set_xlim(0, start_x + n * box_w + (n - 1) * gap + 0.2)
    ax.set_ylim(0.2, 2.9)
    plt.title("Figure 1: End-to-End Sentiment Analysis Workflow", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("results/workflow_diagram.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[SUCCESS] Saved results/workflow_diagram.png")

def create_model_architecture_diagram():
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=300)
    ax.axis("off")

    layers = [
        ("Input Tokens", "Sequence of Word IDs\n(Shape: [Batch, 250])", "#ECEFF1", "#455A64"),
        ("Embedding Layer", "Learned Word Embeddings\n(Vocab: 10,000 -> Dim: 128)", "#E1F5FE", "#0288D1"),
        ("Dropout (p=0.3)", "Spatial / Feature Regularization", "#FFF8E1", "#FFA000"),
        ("Bidirectional LSTM\n(Layer 1 & 2)", "2 Layers, Hidden Dim: 128\nForward + Backward States (256)", "#E8EAF6", "#3F51B5"),
        ("Concatenation & Dropout", "Final Hidden Vectors [h_fwd; h_bwd]\nDropout (p=0.3)", "#F3E5F5", "#7B1FA2"),
        ("Dense / Linear Layer", "Linear Projection (256 -> 1 Logit)", "#E0F2F1", "#00796B"),
        ("Sigmoid Activation", "Probability Score in [0, 1]\nThreshold = 0.5 (Positive / Negative)", "#FFEBEE", "#D32F2F")
    ]

    y_start = 3.6
    y_step = 0.52
    w = 6.2
    h = 0.38
    x_center = 4.25

    for i, (title, desc, bg, border) in enumerate(layers):
        y = y_start - i * y_step
        rect = patches.FancyBboxPatch(
            (x_center - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.1",
            facecolor=bg, edgecolor=border, linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(x_center - w/2 + 0.2, y, title, ha="left", va="center", fontsize=9, fontweight="bold", color="#1A237E")
        ax.text(x_center + w/2 - 0.2, y, desc.replace("\n", " | "), ha="right", va="center", fontsize=8, color="#37474F")

        if i < len(layers) - 1:
            ax.annotate(
                "", xy=(x_center, y - h/2 - 0.1), xytext=(x_center, y - h/2),
                arrowprops=dict(facecolor="#37474F", edgecolor="#37474F", arrowstyle="-|>", lw=1.5, mutation_scale=10)
            )

    ax.set_xlim(0.5, 8.0)
    ax.set_ylim(y_start - (len(layers) - 1) * y_step - 0.4, y_start + 0.6)
    plt.title("Figure 2: Bidirectional LSTM Deep Learning Neural Network Architecture", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig("results/model_architecture.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("[SUCCESS] Saved results/model_architecture.png")

if __name__ == "__main__":
    create_workflow_diagram()
    create_model_architecture_diagram()
