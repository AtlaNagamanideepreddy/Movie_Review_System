import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that performs a two-pass calculation of total pages for 'Page X of Y' footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        if self._pageNumber == 1:
            # Suppress header and footer on cover page
            return

        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setStrokeColor(colors.HexColor("#B0BEC5"))
        self.setFillColor(colors.HexColor("#546E7A"))
        self.setLineWidth(0.5)

        # Header
        self.line(54, A4[1] - 45, A4[0] - 54, A4[1] - 45)
        self.drawString(54, A4[1] - 40, "Vishnu Institute of Technology :: Dept. of CSE | Deep Learning Mini Project")
        self.drawRightString(A4[0] - 54, A4[1] - 40, "Movie Review Sentiment Analysis")

        # Footer
        self.line(54, 45, A4[0] - 54, 45)
        self.drawString(54, 32, "Team 3 | IV Year CSE - Section A (2026-2027)")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(A4[0] - 54, 32, page_text)
        self.restoreState()


def build_pdf(filename="REPORT.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_cover_inst = ParagraphStyle(
        'CoverInst',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=15,
        leading=20,
        alignment=1, # Center
        textColor=colors.HexColor("#0D47A1")
    )
    style_cover_dept = ParagraphStyle(
        'CoverDept',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#1565C0")
    )
    style_cover_title = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=18,
        leading=24,
        alignment=1,
        textColor=colors.HexColor("#B71C1C")
    )
    style_cover_sub = ParagraphStyle(
        'CoverSub',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#37474F")
    )

    style_ch_heading = ParagraphStyle(
        'ChapterHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=14,
        leading=18,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#0D47A1"),
        keepWithNext=True
    )
    style_sec_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=12,
        leading=16,
        spaceBefore=10,
        spaceAfter=4,
        textColor=colors.HexColor("#1A237E"),
        keepWithNext=True
    )
    style_body = ParagraphStyle(
        'BodyJustified',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=15,
        alignment=4, # Justified
        spaceAfter=6,
        textColor=colors.HexColor("#212121")
    )
    style_caption = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13,
        alignment=1, # Center
        spaceBefore=4,
        spaceAfter=10,
        textColor=colors.HexColor("#424242")
    )
    style_table_text = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#212121")
    )
    style_table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.white
    )

    story = []

    # ================= COVER PAGE =================
    story.append(Spacer(1, 20))
    story.append(Paragraph("VISHNU INSTITUTE OF TECHNOLOGY (AUTONOMOUS)", style_cover_inst))
    story.append(Spacer(1, 4))
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", style_cover_dept))
    story.append(Spacer(1, 20))

    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D47A1"), spaceBefore=5, spaceAfter=20))

    story.append(Paragraph("DEEP LEARNING MINI-PROJECT REPORT", ParagraphStyle('ReportBadge', parent=styles['Normal'], fontName='Times-Bold', fontSize=13, leading=16, alignment=1, textColor=colors.HexColor("#455A64"))))
    story.append(Spacer(1, 10))
    story.append(Paragraph("MOVIE REVIEW SENTIMENT ANALYSIS USING BIDIRECTIONAL LSTM", style_cover_title))
    story.append(Spacer(1, 8))
    story.append(Paragraph("A Deep Learning sequence classification system evaluating user sentiment on IMDb benchmarks", style_cover_sub))

    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0D47A1"), spaceBefore=20, spaceAfter=25))

    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Course:</b> Deep Learning Mini-Project | <b>Academic Year:</b> 2026–2027", style_cover_sub))
    story.append(Paragraph("<b>Class / Section:</b> IV Year B.Tech – Section A | <b>Team No.:</b> 3", style_cover_sub))
    story.append(Spacer(1, 20))

    # Team Members Table
    team_data = [
        [Paragraph("<b>S. No.</b>", style_table_header), Paragraph("<b>Student Name</b>", style_table_header), Paragraph("<b>Roll Number</b>", style_table_header), Paragraph("<b>Contribution Role</b>", style_table_header)],
        [Paragraph("1", style_table_text), Paragraph("A. Lokesh Sai Srinivas", style_table_text), Paragraph("23PA1A512", style_table_text), Paragraph("Architecture Design & Pipeline Lead", style_table_text)],
        [Paragraph("2", style_table_text), Paragraph("Team Member 2", style_table_text), Paragraph("23PA1A513", style_table_text), Paragraph("Dataset Preparation & Text Preprocessing", style_table_text)],
        [Paragraph("3", style_table_text), Paragraph("Team Member 3", style_table_text), Paragraph("23PA1A514", style_table_text), Paragraph("Model Training & Hyperparameter Tuning", style_table_text)],
        [Paragraph("4", style_table_text), Paragraph("Team Member 4", style_table_text), Paragraph("23PA1A515", style_table_text), Paragraph("Testing, Validation & Metrics Analysis", style_table_text)],
        [Paragraph("5", style_table_text), Paragraph("Team Member 5", style_table_text), Paragraph("23PA1A516", style_table_text), Paragraph("Inference Engine & Documentation", style_table_text)]
    ]
    t_team = Table(team_data, colWidths=[40, 150, 95, 200])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#90CAF9")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F5F9FF"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_team)

    story.append(Spacer(1, 35))
    story.append(Paragraph("<b>Submission Date:</b> November 16, 2026", ParagraphStyle('SubDate', parent=styles['Normal'], fontName='Times-Roman', fontSize=10.5, alignment=1, textColor=colors.HexColor("#424242"))))
    story.append(Paragraph("Vishnu Institute of Technology (Autonomous), Vishnupur, Bhimavaram - 534202", ParagraphStyle('Address', parent=styles['Normal'], fontName='Times-Italic', fontSize=9.5, alignment=1, textColor=colors.HexColor("#757575"))))

    story.append(PageBreak())

    # ================= CHAPTER 1 =================
    story.append(Paragraph("CHAPTER 1 – INTRODUCTION", style_ch_heading))
    story.append(Paragraph("1.1 Background and Overview", style_sec_heading))
    story.append(Paragraph(
        "With the explosive growth of web-based streaming platforms, e-commerce storefronts, and entertainment aggregation portals (such as IMDb, Rotten Tomatoes, Netflix, and Amazon Prime Video), millions of viewers express opinions, reactions, and critiques in natural language. These textual movie reviews represent an invaluable repository of consumer feedback. However, manual inspection and qualitative summarization of this vast, unstructured corpus are physically infeasible for production studios, streaming curators, and film critics.",
        style_body
    ))
    story.append(Paragraph(
        "Sentiment analysis—also widely known as opinion mining—is a core domain of Natural Language Processing (NLP) that aims to computationally identify, extract, and categorize subjective opinions expressed in source text. The primary objective is to classify text into predefined emotional polarity bins (Positive or Negative). Automated sentiment classification empowers production houses to evaluate audience reception, informs recommendation engines to suggest culturally resonant titles, and assists consumers in making informed viewing choices.",
        style_body
    ))
    story.append(Paragraph(
        "Historically, sentiment classification relied upon bag-of-words (BoW) representations combined with linear classifiers such as Naive Bayes or Support Vector Machines (SVM). While computationally lightweight, these traditional approaches suffer from severe structural drawbacks: they completely discard syntactic sequence, word ordering, long-range dependencies, contextual polysemy, and grammatical negation (for instance, misinterpreting 'not good' due to presence of the isolated token 'good'). Deep learning architectures, especially Recurrent Neural Networks (RNNs) and Bidirectional Long Short-Term Memory (BiLSTM) networks, resolve these limitations by processing sequences sequentially while maintaining memory gates that capture past and future context simultaneously.",
        style_body
    ))

    # ================= CHAPTER 2 =================
    story.append(Paragraph("CHAPTER 2 – PROBLEM STATEMENT AND OBJECTIVES", style_ch_heading))
    story.append(Paragraph("2.1 Problem Statement", style_sec_heading))
    story.append(Paragraph(
        "Natural language movie reviews frequently contain complex syntactic nuances, informal vernacular, sarcasm, double negations, and colloquial expressions that obscure the author's true sentiment. The core problem addressed in this project is to develop an automated, robust, and end-to-end Deep Learning sequence classification system capable of accurately predicting whether an unconstrained textual movie review conveys a <b>Positive</b> or <b>Negative</b> sentiment.",
        style_body
    ))
    story.append(Paragraph("2.2 Objectives", style_sec_heading))
    story.append(Paragraph("The concrete objectives established for this mini-project are:", style_body))
    story.append(Paragraph("1. <b>Data Pipeline Construction:</b> Acquire and preprocess the benchmark IMDb Large Movie Review dataset (50,000 samples) by stripping HTML artifacts, standardizing casing, eliminating non-alphanumeric noise, and tokenizing input text.", style_body))
    story.append(Paragraph("2. <b>Vocabulary & Embedding Generation:</b> Build a frequency-ranked word vocabulary capped at 10,000 terms with dedicated special tokens (<PAD>, <UNK>), and transform variable-length text into standardized fixed-length numerical tensors (max sequence length = 250).", style_body))
    story.append(Paragraph("3. <b>Deep Neural Network Design:</b> Architect a multi-layer Bidirectional Long Short-Term Memory (BiLSTM) network equipped with a learned word embedding layer (128 dimensions), stacked LSTM layers with dropout regularization (0.3), and a linear classification head.", style_body))
    story.append(Paragraph("4. <b>Model Training & Optimization:</b> Train the model with Binary Cross-Entropy with Logits loss (BCEWithLogitsLoss) and Adam optimization, implementing gradient clipping and validation-loss checkpointing.", style_body))
    story.append(Paragraph("5. <b>Quantitative Evaluation:</b> Rigorously evaluate model performance on an unseen test partition of 25,000 samples across classification metrics including Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.", style_body))
    story.append(Paragraph("6. <b>Interactive Inference Deployment:</b> Develop an interactive command-line inference engine (`predict.py`) enabling real-time sentiment scoring and confidence estimation on arbitrary user reviews.", style_body))

    # ================= CHAPTER 3 =================
    story.append(Paragraph("CHAPTER 3 – LITERATURE SURVEY", style_ch_heading))
    story.append(Paragraph("3.1 Review of Related Work", style_sec_heading))
    story.append(Paragraph(
        "<b>Maas et al. (2011)</b> introduced the canonical benchmark dataset comprising 50,000 IMDb movie reviews specifically targeted for sentiment analysis. The authors proposed a probabilistic model that learned continuous word vector representations combining both unsupervised semantic distributional information and supervised sentiment labels. Their baseline linear support vector machine (SVM) achieved an accuracy of approximately 88.89% using combined bag-of-words and unsupervised vector features. However, their model was inherently bag-of-words based, failing to capture token sequencing or compositional semantics.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Hochreiter & Schmidhuber (1997)</b> pioneered the Long Short-Term Memory (LSTM) architecture to overcome the catastrophic vanishing and exploding gradient problems inherent in standard recurrent neural networks (RNNs) during backpropagation through time. By introducing memory cells governed by input, forget, and output gates, LSTMs effectively retain information across hundreds of timesteps. Later, <b>Graves et al. (2005)</b> extended this framework to Bidirectional LSTMs (BiLSTMs), allowing neural networks to access both past (left-to-right) and future (right-to-left) context simultaneously, which proved crucial for text understanding where succeeding words alter preceding meanings.",
        style_body
    ))
    story.append(Paragraph("3.2 Summary / Gap", style_sec_heading))
    story.append(Paragraph(
        "Existing studies reveal that while conventional machine learning classifiers (BoW + Naive Bayes/SVM) are computationally simple, they struggle with sentence semantics and complex negation structures. Conversely, massive Transformer models (e.g., BERT, RoBERTa) achieve state-of-the-art results but require substantial GPU compute, large memory footprints, and prolonged training times that are impractical for lightweight, real-time edge environments. A Bidirectional LSTM architecture strikes the ideal trade-off: it natively processes contextual sequences in forward and reverse directions, requires minimal parameter storage (~1.94M parameters), trains efficiently on consumer hardware, and provides strong semantic classification accuracy.",
        style_body
    ))

    # ================= CHAPTER 4 =================
    story.append(Paragraph("CHAPTER 4 – DATASET DESCRIPTION", style_ch_heading))
    story.append(Paragraph("4.1 Dataset Source", style_sec_heading))
    story.append(Paragraph(
        "The project utilizes the globally recognized <b>IMDb Large Movie Review Dataset</b>, originally published by Andrew L. Maas et al. (Stanford AI Lab). The dataset is distributed under an open research license and was retrieved via the Hugging Face `stanfordnlp/imdb` repository.",
        style_body
    ))
    story.append(Paragraph("4.2 Dataset Details", style_sec_heading))
    story.append(Paragraph(
        "The dataset contains a total of <b>50,000 highly polarized movie reviews</b>. The raw dataset is strictly balanced with a 50:50 ratio between positive and negative labels. Positive reviews originate from ratings ≥ 7 out of 10, whereas negative reviews originate from ratings ≤ 4 out of 10 (neutral reviews with scores 5–6 were excluded during dataset curation to prevent ambiguous boundary labeling).",
        style_body
    ))
    story.append(Paragraph("4.3 Sample Data and Subset Selection", style_sec_heading))
    story.append(Paragraph(
        "To ensure robust generalization and prevent data leakage, the standard IMDb split was organized into 25,000 training samples and 25,000 test samples. The training partition was further randomly split into <b>20,000 training reviews (80%)</b> and <b>5,000 validation reviews (20%)</b> using a fixed random seed (seed = 42). The test set of 25,000 samples was held out entirely for unbiased post-training evaluation.",
        style_body
    ))

    # Dataset distribution table
    ds_dist_data = [
        [Paragraph("<b>Dataset Split</b>", style_table_header), Paragraph("<b>Sample Count</b>", style_table_header), Paragraph("<b>Positive Labels</b>", style_table_header), Paragraph("<b>Negative Labels</b>", style_table_header), Paragraph("<b>Purpose</b>", style_table_header)],
        [Paragraph("Training Set", style_table_text), Paragraph("20,000", style_table_text), Paragraph("10,000 (50%)", style_table_text), Paragraph("10,000 (50%)", style_table_text), Paragraph("Model weights optimization", style_table_text)],
        [Paragraph("Validation Set", style_table_text), Paragraph("5,000", style_table_text), Paragraph("2,500 (50%)", style_table_text), Paragraph("2,500 (50%)", style_table_text), Paragraph("Hyperparameter monitoring & checkpointing", style_table_text)],
        [Paragraph("Test Set (Unseen)", style_table_text), Paragraph("25,000", style_table_text), Paragraph("12,500 (50%)", style_table_text), Paragraph("12,500 (50%)", style_table_text), Paragraph("Final independent performance evaluation", style_table_text)],
        [Paragraph("<b>Total</b>", style_table_text), Paragraph("<b>50,000</b>", style_table_text), Paragraph("<b>25,000</b>", style_table_text), Paragraph("<b>25,000</b>", style_table_text), Paragraph("Balanced 1:1 distribution", style_table_text)]
    ]
    t_ds = Table(ds_dist_data, colWidths=[90, 75, 85, 85, 150])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#283593")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#C5CAE9")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.HexColor("#F8F9FA"), colors.white]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#E8EAF6")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_ds)
    story.append(Paragraph("Table 1: IMDb Dataset Partition and Class Distribution", style_caption))

    # Sample reviews table
    sample_reviews_data = [
        [Paragraph("<b>Sample Review Text (Truncated)</b>", style_table_header), Paragraph("<b>Ground Truth Label</b>", style_table_header)],
        [Paragraph('"A brilliant masterpiece with flawless acting, crisp dialogue, and an unforgettable musical score."', style_table_text), Paragraph("1 (Positive)", style_table_text)],
        [Paragraph('"An utter disaster. Boring characters, nonsensical plot holes, and painful pacing made this unwatchable."', style_table_text), Paragraph("0 (Negative)", style_table_text)],
        [Paragraph('"Visually compelling cinematography and strong direction made this an emotionally resonant experience."', style_table_text), Paragraph("1 (Positive)", style_table_text)],
        [Paragraph('"Complete waste of money and time. Terribly directed and horribly written from beginning to end."', style_table_text), Paragraph("0 (Negative)", style_table_text)]
    ]
    t_samples = Table(sample_reviews_data, colWidths=[385, 100])
    t_samples.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#37474F")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CFD8DC")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ECEFF1"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_samples)
    story.append(Paragraph("Table 2: Sample Movie Reviews and Polarity Labels", style_caption))

    # ================= CHAPTER 5 =================
    story.append(Paragraph("CHAPTER 5 – PROPOSED METHODOLOGY", style_ch_heading))
    story.append(Paragraph("5.1 Workflow", style_sec_heading))
    story.append(Paragraph(
        "The proposed deep learning pipeline encompasses five interconnected stages: (1) Raw dataset ingestion from Stanford IMDb repositories; (2) Text cleaning, noise filtering, and canonical tokenization; (3) Vocabulary construction, integer indexing, sequence truncation, and zero-padding; (4) Deep Bidirectional LSTM neural network training with backpropagation through time; and (5) Evaluation on unseen test data accompanied by real-time interactive inference. Figure 1 delineates this end-to-end architecture.",
        style_body
    ))

    if os.path.exists("results/workflow_diagram.png"):
        story.append(Spacer(1, 4))
        story.append(Image("results/workflow_diagram.png", width=6.5*inch, height=2.1*inch))
        story.append(Paragraph("Figure 1: End-to-End Workflow Diagram of Movie Review Sentiment Analysis", style_caption))

    story.append(Paragraph("5.2 Preprocessing", style_sec_heading))
    story.append(Paragraph(
        "Textual movie reviews contain substantial noise that impairs neural convergence if left unaddressed. The preprocessing module (`utils/preprocessing.py`) executes the following operations:",
        style_body
    ))
    story.append(Paragraph("• <b>HTML Tag Removal:</b> Regular expressions (`<br\\s*/?>`) eliminate line-break and markup tags introduced during web scraping.", style_body))
    story.append(Paragraph("• <b>URL & Hyperlink Stripping:</b> Protocols and web addresses (`http\\S+|www\\S+`) are sanitized.", style_body))
    story.append(Paragraph("• <b>Alphanumeric Filtering & Lowercasing:</b> Punctuation and non-standard symbols are pruned, preserving internal apostrophes, and all tokens are converted to lowercase (`text.lower()`).", style_body))
    story.append(Paragraph("• <b>Vocabulary Construction:</b> Word frequencies across the 20,000 training reviews are counted using `collections.Counter`. The top 9,998 most frequent words are indexed alongside two essential reserved tokens: `<PAD>` (index 0) for sequence padding, and `<UNK>` (index 1) for out-of-vocabulary terms.", style_body))
    story.append(Paragraph("• <b>Sequence Encoding & Fixed Padding:</b> Each review is tokenized and mapped to its numerical identifiers. Reviews exceeding 250 tokens are truncated to retain the dense introductory and concluding arguments, while reviews shorter than 250 tokens are zero-padded to create uniform tensor batches (`[Batch_Size, 250]`).", style_body))

    story.append(Paragraph("5.3 Model Selection and Training", style_sec_heading))
    story.append(Paragraph(
        "A <b>Bidirectional LSTM</b> was selected because language sentiment depends strongly on bidirectional syntactic context. For example, in the phrase 'Although the acting was slow, the breathtaking conclusion made it extraordinary', a forward-only RNN perceives negative connotation until the very end, whereas a BiLSTM aggregates both clauses simultaneously. Training is performed using mini-batch gradient descent (batch size 64) over 6 epochs with gradient clipping (norm = 1.0) to prevent destabilizing gradient spikes.",
        style_body
    ))

    # ================= CHAPTER 6 =================
    story.append(Paragraph("CHAPTER 6 – DEEP LEARNING MODEL ARCHITECTURE", style_ch_heading))
    story.append(Paragraph("6.1 Model Architecture", style_sec_heading))
    story.append(Paragraph(
        "The model is implemented in PyTorch (`models/model.py`) as `SentimentBiLSTM`. It comprises an Embedding layer, two stacked Bidirectional LSTM layers, spatial dropout, hidden-state concatenation, and a final linear projection:",
        style_body
    ))
    story.append(Paragraph("1. <b>Embedding Layer:</b> Maps discrete word indices $x_t \\in [0, 9999]$ into continuous dense vector representations $\\mathbf{e}_t \\in \\mathbb{R}^{128}$. Padding token 0 is assigned a non-trainable zero gradient.", style_body))
    story.append(Paragraph("2. <b>Spatial Dropout:</b> A dropout rate of $p=0.3$ is applied to embedding vectors to regularize against co-adaptation.", style_body))
    story.append(Paragraph("3. <b>2-Layer Bidirectional LSTM:</b> Configured with `hidden_dim = 128`, `num_layers = 2`, `batch_first = True`, and inter-layer dropout of 0.3. At each timestep, the forward LSTM computes $\\overrightarrow{\\mathbf{h}}_t$ and the reverse LSTM computes $\\overleftarrow{\\mathbf{h}}_t$.", style_body))
    story.append(Paragraph("4. <b>Hidden State Concatenation:</b> The final forward hidden state $\\overrightarrow{\\mathbf{h}}_{T}$ and reverse hidden state $\\overleftarrow{\\mathbf{h}}_{1}$ from the topmost LSTM layer are concatenated into a single feature vector $\\mathbf{h}_{\\text{cat}} \\in \\mathbb{R}^{256}$.", style_body))
    story.append(Paragraph("5. <b>Classification Head:</b> A Linear layer maps $\\mathbf{h}_{\\text{cat}}$ to a single scalar logit $z \\in \\mathbb{R}$. During inference, a Sigmoid function $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ yields the posterior probability of positive sentiment.", style_body))

    if os.path.exists("results/model_architecture.png"):
        story.append(Spacer(1, 4))
        story.append(Image("results/model_architecture.png", width=6.2*inch, height=2.8*inch))
        story.append(Paragraph("Figure 2: Architectural Diagram of the Proposed Bidirectional LSTM Network", style_caption))

    story.append(Paragraph("6.2 Model Configuration", style_sec_heading))
    story.append(Paragraph("The exact hyperparameters utilized during model development and training are documented below:", style_body))

    # Model configuration table
    config_data = [
        [Paragraph("<b>Hyperparameter / Setting</b>", style_table_header), Paragraph("<b>Configured Value</b>", style_table_header), Paragraph("<b>Rationale & Description</b>", style_table_header)],
        [Paragraph("Vocabulary Size", style_table_text), Paragraph("10,000 tokens", style_table_text), Paragraph("Captures >95% of common vocabulary while pruning rare misspellings", style_table_text)],
        [Paragraph("Max Sequence Length", style_table_text), Paragraph("250 tokens", style_table_text), Paragraph("Covers >85% of full reviews without excessive zero-padding overhead", style_table_text)],
        [Paragraph("Embedding Dimension", style_table_text), Paragraph("128", style_table_text), Paragraph("Dense representation dimension for lexical features", style_table_text)],
        [Paragraph("LSTM Hidden Dimension", style_table_text), Paragraph("128 per direction (256 total)", style_table_text), Paragraph("Adequate capacity for sequence state tracking without overfitting", style_table_text)],
        [Paragraph("Number of LSTM Layers", style_table_text), Paragraph("2 (Stacked)", style_table_text), Paragraph("Hierarchical temporal feature extraction", style_table_text)],
        [Paragraph("Dropout Rate", style_table_text), Paragraph("0.3 (30%)", style_table_text), Paragraph("Prevents neuron co-adaptation and over-reliance on specific tokens", style_table_text)],
        [Paragraph("Optimizer", style_table_text), Paragraph("Adam (Adaptive Moment)", style_table_text), Paragraph("Fast convergence with adaptive learning rates ($\beta_1=0.9, \beta_2=0.999$)", style_table_text)],
        [Paragraph("Initial Learning Rate", style_table_text), Paragraph("0.001 (1e-3)", style_table_text), Paragraph("Standard robust learning rate for Adam with LSTM sequence models", style_table_text)],
        [Paragraph("Loss Function", style_table_text), Paragraph("BCEWithLogitsLoss", style_table_text), Paragraph("Combines Sigmoid layer and Binary Cross-Entropy with numerical stability", style_table_text)],
        [Paragraph("Batch Size", style_table_text), Paragraph("64", style_table_text), Paragraph("Balances memory throughput and stable stochastic gradient updates", style_table_text)],
        [Paragraph("Training Epochs", style_table_text), Paragraph("6 Epochs", style_table_text), Paragraph("Empirically chosen; minimum validation loss reached at epoch 6", style_table_text)],
        [Paragraph("Gradient Clipping", style_table_text), Paragraph("max_norm = 1.0", style_table_text), Paragraph("Mitigates exploding gradients across deep recurrent unrolls", style_table_text)]
    ]
    t_cfg = Table(config_data, colWidths=[130, 110, 245])
    t_cfg.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#004D40")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#B2DFDB")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#E0F2F1"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_cfg)
    story.append(Paragraph("Table 3: Hyperparameter Specifications for the Deep Learning Model", style_caption))

    # ================= CHAPTER 7 =================
    story.append(Paragraph("CHAPTER 7 – IMPLEMENTATION DETAILS", style_ch_heading))
    story.append(Paragraph("7.1 Software and Libraries", style_sec_heading))
    story.append(Paragraph(
        "The project was implemented in Python 3.12 within a virtualized execution environment using modern open-source deep learning and scientific computing libraries:",
        style_body
    ))
    story.append(Paragraph("• <b>Python 3.12:</b> Core high-level object-oriented programming runtime.", style_body))
    story.append(Paragraph("• <b>PyTorch 2.5.1:</b> Core deep learning framework utilized for tensor computation, dynamic computation graphs, automatic differentiation (`autograd`), GPU acceleration, and recurrent layers (`torch.nn.LSTM`).", style_body))
    story.append(Paragraph("• <b>Hugging Face Datasets:</b> Efficient programmatic streaming and caching of the Stanford IMDb dataset.", style_body))
    story.append(Paragraph("• <b>Scikit-Learn 1.9.1:</b> Evaluation utilities including `accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`, and `roc_auc_score`.", style_body))
    story.append(Paragraph("• <b>Matplotlib 3.11 & Seaborn 0.13:</b> Publication-grade visualization for confusion matrices, training/validation loss curves, and architectural schematics.", style_body))
    story.append(Paragraph("• <b>NumPy & tqdm:</b> Fast numerical array operations and dynamic training progress logging.", style_body))

    story.append(Paragraph("7.2 Implementation and Execution", style_sec_heading))
    story.append(Paragraph(
        "The project repository is architected modularly into distinct operational components:",
        style_body
    ))
    story.append(Paragraph("• `data/download_data.py`: Downloads raw IMDb reviews, cleans text via regex, constructs vocabulary dictionary (`vocab.json`), and exports 20k/5k/25k splits (`dataset_splits.json`).", style_body))
    story.append(Paragraph("• `utils/preprocessing.py`: Standalone text sanitization function for training and runtime inference.", style_body))
    story.append(Paragraph("• `utils/dataset.py`: PyTorch `Dataset` subclass performing token encoding, length truncation, and zero-padding.", style_body))
    story.append(Paragraph("• `models/model.py`: Neural network class `SentimentBiLSTM` definition.", style_body))
    story.append(Paragraph("• `train.py`: Full training loop with validation checks, gradient clipping, history logging, and checkpoint saving to `results/bilstm_sentiment.pt`.", style_body))
    story.append(Paragraph("• `evaluate.py`: Test set evaluation script that generates numerical reports, confusion matrix heatmap, and training curves.", style_body))
    story.append(Paragraph("• `predict.py`: Interactive CLI script allowing real-time review input and immediate sentiment prediction.", style_body))

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Steps to Execute the Project:</b>", style_sec_heading))
    story.append(Paragraph("<code>1. Clone repository: git clone https://github.com/Lokesh-Sai-Srinivas/Movie-Review-Sentiment-Analysis.git</code>", style_body))
    story.append(Paragraph("<code>2. Create & activate virtual environment: python -m venv ml_env && ml_env\\Scripts\\activate</code>", style_body))
    story.append(Paragraph("<code>3. Install dependencies: pip install -r requirements.txt</code>", style_body))
    story.append(Paragraph("<code>4. Download & preprocess dataset: python data/download_data.py</code>", style_body))
    story.append(Paragraph("<code>5. Train Deep Learning model: python train.py</code>", style_body))
    story.append(Paragraph("<code>6. Evaluate performance on 25k test set: python evaluate.py</code>", style_body))
    story.append(Paragraph("<code>7. Run interactive predictions: python predict.py</code>", style_body))

    # ================= CHAPTER 8 =================
    story.append(Paragraph("CHAPTER 8 – RESULTS AND PERFORMANCE ANALYSIS", style_ch_heading))
    story.append(Paragraph("8.1 Execution and Training Summary", style_sec_heading))
    story.append(Paragraph(
        "The model was trained for 6 epochs over the 20,000 training reviews with validation monitoring across 5,000 reviews per epoch. Table 4 records the convergence progression across all training epochs.",
        style_body
    ))

    # Training progress table
    epoch_data = [
        [Paragraph("<b>Epoch</b>", style_table_header), Paragraph("<b>Train Loss</b>", style_table_header), Paragraph("<b>Train Accuracy</b>", style_table_header), Paragraph("<b>Validation Loss</b>", style_table_header), Paragraph("<b>Validation Accuracy</b>", style_table_header), Paragraph("<b>Checkpoint Status</b>", style_table_header)],
        [Paragraph("1", style_table_text), Paragraph("0.6588", style_table_text), Paragraph("59.86%", style_table_text), Paragraph("0.6253", style_table_text), Paragraph("66.88%", style_table_text), Paragraph("Saved initial model", style_table_text)],
        [Paragraph("2", style_table_text), Paragraph("0.5760", style_table_text), Paragraph("71.18%", style_table_text), Paragraph("0.5956", style_table_text), Paragraph("67.12%", style_table_text), Paragraph("Saved better checkpoint", style_table_text)],
        [Paragraph("3", style_table_text), Paragraph("0.5002", style_table_text), Paragraph("76.91%", style_table_text), Paragraph("0.4493", style_table_text), Paragraph("80.20%", style_table_text), Paragraph("Saved better checkpoint", style_table_text)],
        [Paragraph("4", style_table_text), Paragraph("0.4247", style_table_text), Paragraph("81.35%", style_table_text), Paragraph("0.3959", style_table_text), Paragraph("83.80%", style_table_text), Paragraph("Saved better checkpoint", style_table_text)],
        [Paragraph("5", style_table_text), Paragraph("0.3727", style_table_text), Paragraph("83.98%", style_table_text), Paragraph("0.5325", style_table_text), Paragraph("81.52%", style_table_text), Paragraph("Checkpoint retained (Loss: 0.3959)", style_table_text)],
        [Paragraph("6", style_table_text), Paragraph("0.3231", style_table_text), Paragraph("86.67%", style_table_text), Paragraph("0.3699", style_table_text), Paragraph("85.10%", style_table_text), Paragraph("<b>Best model saved (Loss: 0.3699)</b>", style_table_text)]
    ]
    t_ep = Table(epoch_data, colWidths=[40, 75, 85, 85, 95, 105])
    t_ep.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#303F9F")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#C5CAE9")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F5F9FF"), colors.white]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#E8EAF6")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_ep)
    story.append(Paragraph("Table 4: Epoch-by-Epoch Training and Validation Performance", style_caption))

    story.append(Paragraph("8.2 Performance Metrics", style_sec_heading))
    story.append(Paragraph(
        "The optimal model weights saved at epoch 6 were loaded to evaluate the unseen test partition of <b>25,000 reviews</b>. Table 5 presents the quantitative evaluation metrics along with their technical interpretation.",
        style_body
    ))

    metrics_data = [
        [Paragraph("<b>Metric</b>", style_table_header), Paragraph("<b>Obtained Value</b>", style_table_header), Paragraph("<b>Brief Interpretation</b>", style_table_header)],
        [Paragraph("<b>Accuracy</b>", style_table_text), Paragraph("<b>83.84% (0.8384)</b>", style_table_text), Paragraph("Proportion of correctly classified reviews (20,960 correct out of 25,000 total test samples).", style_table_text)],
        [Paragraph("<b>Precision</b>", style_table_text), Paragraph("<b>80.75% (0.8075)</b>", style_table_text), Paragraph("Positive predictive value: out of all reviews predicted positive, 80.75% were genuinely positive.", style_table_text)],
        [Paragraph("<b>Recall (Sensitivity)</b>", style_table_text), Paragraph("<b>88.87% (0.8887)</b>", style_table_text), Paragraph("True positive rate: the model captured 88.87% of all actual positive reviews in the test corpus.", style_table_text)],
        [Paragraph("<b>F1-Score</b>", style_table_text), Paragraph("<b>84.60% (0.8460)</b>", style_table_text), Paragraph("Harmonic mean of precision and recall, demonstrating well-balanced classification capability.", style_table_text)],
        [Paragraph("<b>ROC-AUC Score</b>", style_table_text), Paragraph("<b>0.9184</b>", style_table_text), Paragraph("Area under ROC Curve; confirms high discriminatory power across variable decision thresholds.", style_table_text)]
    ]
    t_met = Table(metrics_data, colWidths=[100, 105, 280])
    t_met.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#90CAF9")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#E3F2FD"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_met)
    story.append(Paragraph("Table 5: Quantitative Test-Set Performance Metrics", style_caption))

    # Confusion matrix breakdown table
    cm_data = [
        [Paragraph("<b>Category</b>", style_table_header), Paragraph("<b>Predicted Negative</b>", style_table_header), Paragraph("<b>Predicted Positive</b>", style_table_header), Paragraph("<b>Total Actual</b>", style_table_header)],
        [Paragraph("<b>Actual Negative</b>", style_table_text), Paragraph("<b>9,851 (True Negatives)</b>", style_table_text), Paragraph("2,649 (False Positives)", style_table_text), Paragraph("12,500", style_table_text)],
        [Paragraph("<b>Actual Positive</b>", style_table_text), Paragraph("1,391 (False Negatives)", style_table_text), Paragraph("<b>11,109 (True Positives)</b>", style_table_text), Paragraph("12,500", style_table_text)],
        [Paragraph("<b>Total Predicted</b>", style_table_text), Paragraph("11,242", style_table_text), Paragraph("13,758", style_table_text), Paragraph("<b>25,000</b>", style_table_text)]
    ]
    t_cm = Table(cm_data, colWidths=[110, 125, 125, 125])
    t_cm.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#263238")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#B0BEC5")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#ECEFF1"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_cm)
    story.append(Paragraph("Table 6: Confusion Matrix Breakdown on 25,000 Test Reviews", style_caption))

    story.append(Paragraph("8.3 Graphs and Sample Predictions", style_sec_heading))
    story.append(Paragraph(
        "Figure 3 presents the training and validation curves for loss and accuracy across the 6 epochs. The training loss decreases smoothly from 0.6588 to 0.3231, while validation loss converges to 0.3699. Validation accuracy reaches 85.10%, tracking training accuracy closely and confirming the absence of pathological overfitting due to dropout regularization.",
        style_body
    ))

    if os.path.exists("results/training_curves.png"):
        story.append(Spacer(1, 4))
        story.append(Image("results/training_curves.png", width=6.5*inch, height=2.4*inch))
        story.append(Paragraph("Figure 3: Training and Validation Loss and Accuracy Curves across 6 Epochs", style_caption))

    story.append(Paragraph(
        "Figure 4 displays the normalized confusion matrix heatmap on the 25,000 test reviews. The dominant diagonal (9,851 TN and 11,109 TP) demonstrates the strong classification fidelity of the Bidirectional LSTM model.",
        style_body
    ))

    if os.path.exists("results/confusion_matrix.png"):
        story.append(Spacer(1, 4))
        story.append(Image("results/confusion_matrix.png", width=4.5*inch, height=3.5*inch))
        story.append(Paragraph("Figure 4: Confusion Matrix Heatmap on Unseen IMDb Test Set (25,000 Samples)", style_caption))

    # Sample Predictions Table
    story.append(Paragraph("<b>Qualitative Sample Predictions:</b>", style_sec_heading))
    pred_data = [
        [Paragraph("<b>Movie Review Input</b>", style_table_header), Paragraph("<b>Ground Truth</b>", style_table_header), Paragraph("<b>Predicted</b>", style_table_header), Paragraph("<b>Confidence</b>", style_table_header)],
        [Paragraph('"The cinematic direction, brilliant acting, and deeply emotional storyline made this an absolute masterpiece of modern cinema."', style_table_text), Paragraph("Positive", style_table_text), Paragraph("<font color='#2E7D32'><b>Positive</b></font>", style_table_text), Paragraph("93.0%", style_table_text)],
        [Paragraph('"An absolute waste of two hours. The script was hollow, characters were unconvincing, and the pacing was utterly unbearable."', style_table_text), Paragraph("Negative", style_table_text), Paragraph("<font color='#C62828'><b>Negative</b></font>", style_table_text), Paragraph("98.6%", style_table_text)],
        [Paragraph('"A visually stunning film with groundbreaking visual effects, though the plot was a bit predictable at times. Overall an enjoyable experience!"', style_table_text), Paragraph("Positive", style_table_text), Paragraph("<font color='#2E7D32'><b>Positive</b></font>", style_table_text), Paragraph("77.1%", style_table_text)],
        [Paragraph('"Terrible acting and nonsensical plot twists ruined what could have been a promising premise. Extremely disappointed."', style_table_text), Paragraph("Negative", style_table_text), Paragraph("<font color='#C62828'><b>Negative</b></font>", style_table_text), Paragraph("97.4%", style_table_text)],
        [Paragraph('"Incredible performance by the lead cast. A thrilling, heart-wrenching roller coaster ride from start to finish!"', style_table_text), Paragraph("Positive", style_table_text), Paragraph("<font color='#2E7D32'><b>Positive</b></font>", style_table_text), Paragraph("97.3%", style_table_text)]
    ]
    t_pred = Table(pred_data, colWidths=[265, 75, 75, 70])
    t_pred.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4A148C")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E1BEE7")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor("#F3E5F5"), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_pred)
    story.append(Paragraph("Table 7: Real-Time Actual vs. Predicted Sentiment Inferences with Confidence Scores", style_caption))

    # ================= CHAPTER 9 =================
    story.append(Paragraph("CHAPTER 9 – CONCLUSION AND FUTURE ENHANCEMENT", style_ch_heading))
    story.append(Paragraph("9.1 Conclusion", style_sec_heading))
    story.append(Paragraph(
        "This project successfully developed, trained, and thoroughly evaluated an end-to-end Deep Learning sequence classification pipeline for Movie Review Sentiment Analysis using a Bidirectional LSTM network in PyTorch. The system achieved a <b>Test Accuracy of 83.84%</b>, <b>Precision of 80.75%</b>, <b>Recall of 88.87%</b>, <b>F1-Score of 84.60%</b>, and <b>ROC-AUC of 0.9184</b> across 25,000 unseen benchmark reviews.",
        style_body
    ))
    story.append(Paragraph(
        "By ingesting sequential text in both forward and reverse directions, the BiLSTM network effectively resolved complex linguistic negation patterns and preserved long-range contextual relationships far beyond the capability of classical bag-of-words classifiers. The interactive CLI tool provides fast, deterministic inference (< 20ms per review on CPU), proving the practical viability of the proposed architecture for production review analysis.",
        style_body
    ))
    story.append(Paragraph("9.2 Limitations and Future Work", style_sec_heading))
    story.append(Paragraph("While the model delivers robust classification performance, certain inherent limitations provide clear avenues for future enhancement:", style_body))
    story.append(Paragraph("• <b>Fixed Vocabulary Limitation:</b> Setting the vocabulary size to 10,000 maps rare words, specialized jargon, and emerging slang to `<UNK>`. Future iterations can incorporate subword tokenization (Byte-Pair Encoding / WordPiece) to handle arbitrary out-of-vocabulary terms gracefully.", style_body))
    story.append(Paragraph("• <b>Subtle Sarcasm & Irony:</b> Reviews containing high-context irony (e.g., 'What a stunning feat of absolute incompetence!') occasionally mislead the classifier due to high-weight positive lexical tokens ('stunning', 'feat'). Integrating self-attention layers can help the network weight contradictory clauses appropriately.", style_body))
    story.append(Paragraph("• <b>Pre-trained Transformer Fine-Tuning:</b> Transitioning to contextual transformer backbones such as BERT, RoBERTa, or DeBERTa will allow deeper transfer learning from billions of pre-trained tokens, potentially elevating classification accuracy beyond 93%.", style_body))
    story.append(Paragraph("• <b>Aspect-Based Sentiment Analysis (ABSA):</b> Extending the binary polarity classification to multi-aspect granular scoring (e.g., separating sentiments for Direction, Acting, Screenplay, and Soundtrack) will yield richer insights for film producers.", style_body))

    # ================= CHAPTER 10 =================
    story.append(Paragraph("CHAPTER 10 – REFERENCES", style_ch_heading))
    story.append(Paragraph("10.1 References", style_sec_heading))
    refs = [
        "[1] A. L. Maas, R. E. Daly, P. T. Pham, D. Huang, A. Y. Ng, and C. Potts, 'Learning word vectors for sentiment analysis,' in <i>Proc. 49th Annu. Meeting Assoc. Comput. Linguistics (ACL)</i>, Portland, OR, USA, 2011, pp. 142–150.",
        "[2] S. Hochreiter and J. Schmidhuber, 'Long short-term memory,' <i>Neural Comput.</i>, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.",
        "[3] A. Graves, S. Fernández, and J. Schmidhuber, 'Bidirectional LSTM networks for improved phoneme classification and recognition,' in <i>Proc. Int. Conf. Artif. Neural Netw. (ICANN)</i>, Warsaw, Poland, 2005, pp. 799–804.",
        "[4] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, 'BERT: Pre-training of deep bidirectional transformers for language understanding,' <i>arXiv preprint arXiv:1810.04805</i>, 2018.",
        "[5] A. Paszke et al., 'PyTorch: An imperative style, high-performance deep learning library,' in <i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, Vancouver, Canada, vol. 32, 2019, pp. 8024–8035.",
        "[6] F. Pedregosa et al., 'Scikit-learn: Machine learning in Python,' <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, Oct. 2011.",
        "[7] J. D. Hunter, 'Matplotlib: A 2D graphics environment,' <i>Comput. Sci. Eng.</i>, vol. 9, no. 3, pp. 90–95, 2007.",
        "[8] Stanford NLP Group, 'IMDb Large Movie Review Dataset,' Stanford AI Lab, 2011. [Online]. Available: https://ai.stanford.edu/~amaas/data/sentiment/."
    ]
    for r in refs:
        story.append(Paragraph(r, style_body))
        story.append(Spacer(1, 2))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Generated complete formatted report PDF: {filename}")

if __name__ == "__main__":
    build_pdf("REPORT.pdf")
