# 🎯 Fine_Tuning — Multilingual Transformer Fine-Tuning Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-zone)

A modular, production-grade deep learning fine-tuning framework for **Sentiment Analysis** and **18-Category Topic Classification** across **Arabic**, **English**, and **French** news corpora.

This repository consumes the human-annotated dataset (`data/global_data_libelised.csv`, 1.48M+ articles) exported from **`Data_Exploring`**, fine-tunes domain-specific transformer architectures with focal loss, class weighting, sliding-window tokenisation, auto-resume mechanisms, and cross-lingual translation pipelines, then produces comprehensive evaluation artifacts and learning curves.

---

## 📑 Table of Contents

- [Dataset Specifications & Taxonomy](#dataset-specifications--taxonomy)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Fine-Tuning Models & Configurations](#fine-tuning-models--configurations)
  - [1. Arabic Sentiment Classification](#1-arabic-sentiment-classification)
  - [2. Arabic Topic Classification (18 Categories)](#2-arabic-topic-classification-18-categories)
  - [3. English Sentiment Classification](#3-english-sentiment-classification)
  - [4. English Topic Classification (18 Categories)](#4-english-topic-classification-18-categories)
  - [5. French Sentiment Classification](#5-french-sentiment-classification)
  - [6. French Topic Classification (18 Categories)](#6-french-topic-classification-18-categories)
- [Training Methodology & Optimizations](#training-methodology--optimizations)
- [Quickstart & Execution Guide](#quickstart--execution-guide)
  - [Running Fine-Tuning Pipelines](#running-fine-tuning-pipelines)
  - [Experiment Scanning & Leaderboard](#experiment-scanning--leaderboard)
- [Experiment Output Structure](#experiment-output-structure)
- [Installation & Setup](#installation--setup)

---

## 📊 Dataset Specifications & Taxonomy

The entire fine-tuning pipeline relies on a single consolidated, human-annotated dataset:

| Field | Path | Columns | Description |
| :--- | :--- | :--- | :--- |
| **Master Dataset** | `data/global_data_libelised.csv` | `id`, `text`, `language`, `sentiment`, `topic` | Clean preprocessed and human-annotated news corpus across Modern Standard Arabic (`ar`), Dialectal Arabic (`da`), English (`en`), and French (`fr`). |

### 1. Sentiment Classes (3)
| Class ID | Label | Interpretation |
| :---: | :--- | :--- |
| `0` | `NEGATIVE` | Negative tone, critical news, economic crisis, conflict |
| `1` | `NEUTRAL` | Factual reporting, official statements, announcements |
| `2` | `POSITIVE` | Positive developments, agreements, growth, recovery |

### 2. Topic Categories (18)
| ID | English | French | Arabic | ID | English | French | Arabic |
| :---: | :--- | :--- | :--- | :---: | :--- | :--- | :--- |
| `0` | Politics | Politique | السياسة | `9` | Sports | Sports | الرياضة |
| `1` | Economy | Économie | الاقتصاد | `10` | Culture | Culture | الثقافة |
| `2` | Security | Sécurité | الأمن | `11` | Education | Éducation | التعليم |
| `3` | Energy | Énergie | الطاقة | `12` | Technology | Technologie | التكنولوجيا |
| `4` | Conflict | Conflit | النزاع | `13` | Environment | Environnement | البيئة |
| `5` | Elections | Élections | الانتخابات | `14` | Diplomacy | Diplomatie | الدبلوماسية |
| `6` | Justice | Justice | العدالة | `15` | Religion | Religion | الدين |
| `7` | Health | Santé | الصحة | `16` | Migration | Migration | الهجرة |
| `8` | Weather | Météo | الطقس | `17` | General | Général | عام |

---

## 🏗️ Project Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────┐
│           data/global_data_libelised.csv (1.48M+ rows)                         │
│           [Columns: id, text, language, sentiment, topic]                      │
└──────────────────────┬──────────────────────────────┬──────────────────────────┘
                       │                              │
        ┌──────────────┼──────────────────┐           │
        ▼              ▼                  ▼           │
  [Arabic (ar / da)] [English (en)]   [French (fr)]   │
        │              │                  │           │
  ┌─────┴─────┐  ┌─────┴─────┐    ┌──────┴──────┐   │
  ▼           ▼  ▼           ▼    ▼             ▼   │
[Sentiment] [Topic] [Sentiment] [Topic] [Sentiment] [Topic]
CAMeL BERT  CAMeL  Twitter-   RoBERTa  CamemBERT  CamemBERT
+Focal+     BERT   RoBERTa-   -Base    -large      -large +
Sliding     +LLRD  large +    +Focal   +Focal      Translation
Window      +Focal TextAug    +Label   +Augment    Pipeline
                   +Resume    Smooth   (>80% Acc)  (18 Classes)
        │              │                  │
        └──────────────┴──────────────────┘
                       │
                       ▼  experiments/<task>/<timestamp>/
┌────────────────────────────────────────────────────────────────────────────────┐
│  ├── checkpoints/    (Periodic training checkpoints)                           │
│  ├── best_model/     (Safetensors, tokenizer, config)                          │
│  ├── eval_results.json  (Macro F1, accuracy, per-group scores)                 │
│  ├── hyperparameters.json                                                      │
│  └── plots/          (Train/eval loss curves, confusion matrices, ROC curves)  │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼ reports/report_search.py
┌────────────────────────────────────────────────────────────────────────────────┐
│                      Automated Experiment Leaderboard                          │
│       Auto-scans all runs & ranks checkpoints by Macro F1 / Accuracy           │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Fine_Tuning/
├── README.md                              # Master project documentation
├── fine_tune_strategy.txt                 # In-depth rationale for every design choice
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git rules (excluding large CSVs & weights)
│
├── data/
│   └── global_data_libelised.csv              # Master human-annotated dataset (from Data_Exploring)
│
├── training/                              # Fine-tuning pipelines
│   ├── arabic/
│   │   ├── __init__.py
│   │   ├── fine_tune_arabic_sentiment.py  # Arabic Sentiment — CAMeL BERT + Sliding Window + Focal Loss
│   │   └── fine_tune_arabic_topic.py      # Arabic Topic — CAMeL BERT + LLRD + Focal Loss (18 classes)
│   │
│   ├── english/
│   │   ├── __init__.py
│   │   ├── fine_tune_english_sentiment.py # English Sentiment — Twitter-RoBERTa-large + Text Augmentation
│   │   └── fine_tune_english_topic.py     # English Topic — RoBERTa-Base + Focal Loss + Label Smoothing
│   │
│   └── french/
│       ├── __init__.py
│       ├── fine_tune_french_sentiment.py  # French Sentiment — CamemBERT-large + Focal Loss (target >80%)
│       └── fine_tune_french_topic.py      # French Topic — CamemBERT-large + Translation Pipeline (18 classes)
│
├── reports/                               # Experiment run ranking & reporting
│   ├── __init__.py
│   └── report_search.py                   # Automated run scanner & Macro F1 leaderboard
│
└── experiments/                           # Persistent run outputs, checkpoints & metrics
    ├── arabic_sentiment/                  # Arabic sentiment training runs
    ├── arabic_topic/                      # Arabic topic training runs
    ├── english_sentiment/                 # English sentiment training runs
    ├── english_topic/                     # English topic training runs
    ├── french_sentiment/                  # French sentiment training runs
    └── french_topic/                      # French topic training runs
```

---

## 🔬 Fine-Tuning Models & Configurations

### 1. Arabic Sentiment Classification
- **Script**: [`training/arabic/fine_tune_arabic_sentiment.py`](training/arabic/fine_tune_arabic_sentiment.py)
- **Base Model**: `CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment`
- **Architecture**: 12-layer Transformer, 768 hidden, 12 heads
- **Language Variants Supported**: Modern Standard Arabic (`ar`), Dialectal Arabic (`da`)
- **Key Techniques**:
  - **Sliding Window** tokenisation (`MAX_LEN=512`, `STRIDE=256`) for long documents
  - **Focal Loss** (`α=0.45`, `γ=2.5`) with dynamic class weights
  - **POSITIVE class augmentation** with Arabic intensifiers (`جداً`, `حقاً`, …)
  - **MSA (`ar`) / Dialect (`da`) stratified splits** for balanced evaluation
  - **Auto-resume** from latest valid checkpoint
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `STRIDE`: 256
  - `BATCH_SIZE`: 4 (Effective: 16 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 1e-5 (Cosine-with-restarts)
  - `EPOCHS`: 20 | `WARMUP_RATIO`: 0.2 | `WEIGHT_DECAY`: 0.03
  - `LABEL_SMOOTHING`: 0.15 | `MAX_GRAD_NORM`: 0.5
  - `EARLY_STOPPING`: Patience=20 | Threshold=0.005
- **Best Achieved**: ~87.7% Accuracy

---

### 2. Arabic Topic Classification (18 Categories)
- **Script**: [`training/arabic/fine_tune_arabic_topic.py`](training/arabic/fine_tune_arabic_topic.py)
- **Base Model**: `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- **Loss Function**: `FocalLoss(gamma=1.5)` + Dynamic Class Weights
- **Optimizer**: AdamW with **Layerwise Learning Rate Decay** (`decay=0.95`)
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `BATCH_SIZE`: 8
  - `LEARNING_RATE`: 1e-5 | `EPOCHS`: 15
  - `EARLY_STOPPING`: Patience=10 epochs

---

### 3. English Sentiment Classification
- **Script**: [`training/english/fine_tune_english_sentiment.py`](training/english/fine_tune_english_sentiment.py)
- **Base Model**: `cardiffnlp/twitter-roberta-large-topic-sentiment-latest`
- **Architecture**: RoBERTa-large (355M parameters)
- **Key Techniques**:
  - **TextAugmenter**: synonym replacement, random deletion, random swap (`AUGMENT_PROB=0.3`)
  - **Enhanced Focal Loss** with label smoothing (`ε=0.05`, `γ=2.0`)
  - **Auto-Resume** from latest checkpoint in `sentiment_experiments/`
  - **OOM guard**: auto-reduces batch size on CUDA out-of-memory errors
  - **ROC, Precision-Recall, per-class F1, and confidence analysis** plots
- **Hyperparameters**:
  - `MAX_LEN`: 160 | `BATCH_SIZE`: 8 (Effective: 32 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 5e-6 (Cosine) | `EPOCHS`: 12
  - `WARMUP_RATIO`: 0.1 | `WEIGHT_DECAY`: 0.01
  - `EARLY_STOPPING`: Patience=10 | Threshold=0.001

---

### 4. English Topic Classification (18 Categories)
- **Script**: [`training/english/fine_tune_english_topic.py`](training/english/fine_tune_english_topic.py)
- **Base Model**: `roberta-base` (Optimized)
- **Loss Function**: `FocalLoss(alpha=0.25, gamma=2.0)` + Label Smoothing (`0.25`)
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `BATCH_SIZE`: 8 (Effective: 32 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 1e-5 | `EPOCHS`: 6
  - `REGULARIZATION`: Dropout (`0.35`), Weight Decay (`0.08`), Max Grad Norm (`0.3`)

---

### 5. French Sentiment Classification
- **Script**: [`training/french/fine_tune_french_sentiment.py`](training/french/fine_tune_french_sentiment.py)
- **Base Model**: `camembert/camembert-large` (435M parameters)
- **Data Strategy**: Combines **real French articles** from the master dataset with **translated data** (from `translated_data/translated_french_data_all.parquet`)
- **Key Techniques**:
  - **Focal Loss** (`α=0.65`, `γ=4.0`) with strong POSITIVE class focus
  - **POSITIVE class augmentation** with French intensifiers (`très`, `vraiment`, `extrêmement`, …)
  - **Enhanced class weights**: POSITIVE × 2.5, NEUTRAL × 0.7, NEGATIVE × 1.1
  - 80/10/10 train/validation/test split
  - **Gradient checkpointing DISABLED** for speed on RTX 3050
- **Hyperparameters**:
  - `MAX_LEN`: 384 | `BATCH_SIZE`: 4 (Effective: 32 via `GRAD_ACCUM=8`)
  - `LEARNING_RATE`: 1e-5 | `EPOCHS`: 15
  - `WARMUP_RATIO`: 0.2 | `WEIGHT_DECAY`: 0.05
  - `EARLY_STOPPING`: Patience=10 | Threshold=0.003
- **Target**: >80% Accuracy

> **⚠️ Pre-requisite**: Run the translation script first to populate `translated_data/` before launching this pipeline.

---

### 6. French Topic Classification (18 Categories)
- **Script**: [`training/french/fine_tune_french_topic.py`](training/french/fine_tune_french_topic.py)
- **Base Model**: `camembert/camembert-large`
- **Translation Pipeline**: Uses `Helsinki-NLP/opus-mt-en-fr` (MarianMT) to automatically translate English articles to French; translated data is cached in `translated_data/french_topic_translated_data.csv`
- **Key Techniques**:
  - **Custom `LargeClassifierFocal` model** with CLS-token pooling + dropout head
  - **Focal Loss with class-specific gammas** (`γ` ranges 2.0–4.0 based on class frequency)
  - **Soft inverse-sqrt class weights** (clipped 0.8–2.0)
  - **Environment class oversampling** (class 13 boosted to ≥500 samples)
  - **Auto-resume** from latest valid checkpoint
  - **ROC curves + AUC bar charts** per class
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `BATCH_SIZE`: 4 (Effective: 16 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 5e-6 | `EPOCHS`: 6
  - `WEIGHT_DECAY`: 0.01 | `MAX_GRAD_NORM`: 0.3
  - `EARLY_STOPPING`: Patience=5 | Threshold=0.001
  - `EVAL_STEPS`: 3000 | `SEED`: 42

---

## ⚡ Training Methodology & Optimizations

1. **Sliding Window for Long Documents** _(Arabic Sentiment)_:
   - Instead of hard truncation at 512 tokens, Arabic texts are split into overlapping chunks (`STRIDE=256`). Each chunk is classified independently, allowing the model to process arbitrarily long documents.

2. **Focal Loss for Extreme Imbalance**:
   - Addresses high frequency discrepancy between major news topics (e.g. *Politics*, *General*) and niche topics (e.g. *Weather*, *Environment*, *Migration*).
   - FL(p_t) = -(1 - p_t)^γ · log(p_t) — downweights easy examples, focuses on hard ones.
   - Class-specific gammas (2.0–4.0) used in French topic to penalise rare classes more aggressively.

3. **Layerwise Learning Rate Decay (LLRD)** _(Arabic Topic)_:
   - Lower Transformer layers train with smaller LRs (preserving pre-trained morphological features); upper layers adapt rapidly. Decay=0.95 per layer.

4. **Cross-Lingual Translation Pipeline** _(French Topic)_:
   - English articles in the dataset are automatically translated to French using MarianMT (`Helsinki-NLP/opus-mt-en-fr`), expanding the French training corpus. Results are cached to avoid re-translation.

5. **Text Augmentation** _(English Sentiment)_:
   - Training samples undergo random synonym replacement, word deletion, and word swap at `AUGMENT_PROB=0.3` to improve generalisation and reduce overfitting on majority classes.

6. **POSITIVE Class Augmentation** _(Arabic & French Sentiment)_:
   - The underrepresented POSITIVE class is boosted by appending language-appropriate intensifiers to existing positive samples, creating synthetic training examples.

7. **Mixed Precision (`fp16` / `bf16`) & Gradient Checkpointing**:
   - Reduces VRAM consumption by ~50%, enabling larger batch sizes on consumer GPUs (RTX 3050/4060).

8. **Resilient Auto-Resume**:
   - All scripts scan their experiment directories for the latest valid checkpoint (`trainer_state.json` + `model.safetensors`) and resume seamlessly — protecting progress against crashes or deliberate interruptions.

---

## 🚀 Quickstart & Execution Guide

### Running Fine-Tuning Pipelines

Launch any fine-tuning task from the project root:

```powershell
# ── Arabic ──────────────────────────────────────────────────────────────────
# Train Arabic Sentiment model (MSA + Dialects, Sliding Window)
python training/arabic/fine_tune_arabic_sentiment.py

# Train Arabic Topic model (18 categories, LLRD + Focal Loss)
python training/arabic/fine_tune_arabic_topic.py

# ── English ─────────────────────────────────────────────────────────────────
# Train English Sentiment model (RoBERTa-large + Text Augmentation)
python training/english/fine_tune_english_sentiment.py

# Train English Topic model (18 categories, Focal Loss + Label Smoothing)
python training/english/fine_tune_english_topic.py

# ── French ──────────────────────────────────────────────────────────────────
# NOTE: French sentiment requires pre-translated data in translated_data/
# Ensure translated_data/translated_french_data_all.parquet exists first.
python training/french/fine_tune_french_sentiment.py

# French topic auto-translates English articles on first run (cached afterwards)
python training/french/fine_tune_french_topic.py
```

### Offline Inference & Confusion Matrix Evaluation

Evaluate a saved checkpoint against a held-out test split:

```powershell
python analysis/test_inference.py experiments/arabic_topic/<run_timestamp>/checkpoint-XXXXXX
```

### Experiment Scanning & Leaderboard

Automatically scan all completed runs in `experiments/` and rank by Macro F1:

```powershell
# Scan all tasks
python reports/report_search.py --task all

# Scan per language & task
python reports/report_search.py --task arabic_sentiment   --metric f1_macro
python reports/report_search.py --task arabic_topic       --metric f1_macro
python reports/report_search.py --task english_sentiment  --metric f1_macro
python reports/report_search.py --task english_topic      --metric f1_macro
python reports/report_search.py --task french_sentiment   --metric f1_macro
python reports/report_search.py --task french_topic       --metric f1_macro
```

---

## 📦 Experiment Output Structure

Every run creates a timestamped folder inside `experiments/<task>/<timestamp>/`:

```
experiments/<task>/<timestamp>/
├── best_model/                         # Exported model weights & tokenizer
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── label_mapping.json              # id→label mapping (French & English)
├── checkpoints/                        # Step-based checkpoints (checkpoint-XXXX)
├── hyperparameters.json                # Immutable run configuration recorded at start
├── eval_results.json                   # Final evaluation metrics (F1 Macro, accuracy)
├── test_results.json                   # Test-set results (MSA, Dialect, or combined)
├── summary.json                        # Comprehensive experiment report
├── predictions_<split>.csv             # Per-sample predictions & confidence scores
└── plots/                              # High-resolution visual artifacts
    ├── train_loss.png
    ├── eval_loss.png
    ├── eval_f1_accuracy.png
    ├── learning_curve.png
    ├── confusion_matrix_<split>.png
    ├── roc_curves_<split>.png          # French topic & English sentiment
    ├── auc_barchart_<split>.png        # French topic
    ├── per_class_f1_bar.png            # English sentiment
    └── confidence_analysis.png         # English sentiment
```

---

## 🛠️ Installation & Setup

1. **Clone repository & enter directory**:
   ```powershell
   git clone <repo_url>
   cd Fine_Tuning
   ```

2. **Create & activate Python virtual environment**:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify GPU Acceleration**:
   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
   ```

5. **Prepare translated data for French Sentiment** _(first-time only)_:
   ```powershell
   # The French topic script auto-generates its translation cache on first run.
   # For French sentiment, ensure the parquet files exist in translated_data/:
   #   translated_data/translated_french_data_all.parquet
   #   translated_data/real_french_data.parquet
   ```
