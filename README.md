# 🎯 Fine_Tuning — Multilingual Transformer Fine-Tuning Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![CUDA Enabled](https://img.shields.io/badge/CUDA-Supported-green.svg)](https://developer.nvidia.com/cuda-zone)

A modular, production-grade deep learning fine-tuning framework for **Sentiment Analysis** (3 classes) and **Topic Classification** (18 categories) across **Arabic**, **English**, and **French** news corpora.

This repository consumes the human-annotated dataset (`data/global_data_libelised.csv`, 1.48M+ articles) exported from **`Data_Exploring`**, fine-tunes domain-specific transformer architectures with focal loss, dynamic class weighting, sliding-window tokenisation, auto-resume mechanisms, and cross-lingual translation pipelines, then produces comprehensive evaluation artifacts and learning curves.

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

The fine-tuning pipelines consume a consolidated, human-annotated dataset:

| Field | Path | Columns | Description |
| :--- | :--- | :--- | :--- |
| **Master Dataset** | `data/global_data_libelised.csv` | `id`, `text`, `language`, `sentiment`, `topic` | Clean preprocessed and human-annotated news corpus (1,484,846 rows) across Modern Standard Arabic (`ar`), Dialectal Arabic (`da`), English (`en`), and French (`fr`). |

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
                       ▼  Experiment Outputs & Artifacts
┌────────────────────────────────────────────────────────────────────────────────┐
│  ├── checkpoints/       (Step or epoch checkpoints)                            │
│  ├── best_model/        (Safetensors, tokenizer, config)                       │
│  ├── eval_results.json  (Macro F1, accuracy, per-split scores)                 │
│  ├── hyperparameters.json                                                      │
│  └── plots/             (Train/eval loss curves, confusion matrices, ROC)      │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼ reports/report_search.py
┌────────────────────────────────────────────────────────────────────────────────┐
│                      Automated Experiment Leaderboard                          │
│       Auto-scans runs in experiments/ & ranks models by Macro F1 / Accuracy    │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
Fine_Tuning/
├── README.md                              # Master project documentation
├── fine_tune_strategy.txt                 # Architectural decisions & methodology rationale
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git exclusion rules
│
├── data/
│   └── global_data_libelised.csv          # Master human-annotated dataset (1.48M+ rows)
│
├── training/                              # Fine-tuning pipelines
│   ├── __init__.py
│   │
│   ├── arabic/
│   │   ├── __init__.py
│   │   ├── fine_tune_arabic_sentiment.py  # CAMeL BERT + Sliding Window (512/256) + Focal Loss
│   │   └── fine_tune_arabic_topic.py      # CAMeL BERT + LLRD + Focal Loss (18 classes)
│   │
│   ├── english/
│   │   ├── __init__.py
│   │   ├── fine_tune_english_sentiment.py # Twitter-RoBERTa-large + Text Augmentation + Focal Loss
│   │   └── fine_tune_english_topic.py     # RoBERTa-Base + Focal Loss + Label Smoothing (18 classes)
│   │
│   └── french/
│       ├── __init__.py
│       ├── fine_tune_french_sentiment.py  # CamemBERT-large + Focal Loss (γ=4.0) + Augmentation
│       └── fine_tune_french_topic.py      # CamemBERT-large + MarianMT Translation + Focal Loss
│
├── reports/                               # Run scanning & leaderboards
│   ├── __init__.py
│   └── report_search.py                   # Automated experiment scanner & Macro F1 leaderboard
│
└── experiments/                           # Persistent run outputs, checkpoints & metrics
    ├── arabic_sentiment/                  # Arabic sentiment runs
    ├── arabic_topic/                      # Arabic topic runs
    └── english_topic/                     # English topic runs
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
  - **POSITIVE class augmentation** with Arabic intensifiers (`جداً`, `حقاً`, `للغاية`, …)
  - **MSA (`ar`) / Dialect (`da`) stratified splits** (80% train, 10% validation, 10% test)
  - **Auto-resume** from latest valid checkpoint in `experiments/arabic_sentiment/`
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `STRIDE`: 256
  - `BATCH_SIZE`: 4 (Effective: 16 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 1e-5 (Cosine-with-restarts)
  - `EPOCHS`: 20 | `WARMUP_RATIO`: 0.2 | `WEIGHT_DECAY`: 0.03
  - `LABEL_SMOOTHING`: 0.15 | `MAX_GRAD_NORM`: 0.5
  - `EARLY_STOPPING`: Patience=20 | Threshold=0.005
- **Output Directory**: `experiments/arabic_sentiment/<timestamp>/`

---

### 2. Arabic Topic Classification (18 Categories)
- **Script**: [`training/arabic/fine_tune_arabic_topic.py`](training/arabic/fine_tune_arabic_topic.py)
- **Base Model**: `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- **Loss Function**: `FocalLoss(gamma=1.5)` + Dynamic Class Weights
- **Optimizer**: AdamW with **Layerwise Learning Rate Decay (LLRD)** (`decay=0.95`)
- **Key Techniques**:
  - Stratified MSA & Dialect splits with combined evaluation
  - Step-based evaluation with confusion matrix callbacks
  - Auto-saving best model based on `eval_f1_macro`
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `BATCH_SIZE`: 8 (Effective: 8 via `GRAD_ACCUM=1`)
  - `LEARNING_RATE`: 1e-5 | `EPOCHS`: 15
  - `WARMUP_RATIO`: 0.15 | `WEIGHT_DECAY`: 0.01 | `MAX_GRAD_NORM`: 0.3
  - `EARLY_STOPPING`: Patience=10 | Threshold=0.001
- **Output Directory**: `experiments/arabic_topic/<timestamp>/`

---

### 3. English Sentiment Classification
- **Script**: [`training/english/fine_tune_english_sentiment.py`](training/english/fine_tune_english_sentiment.py)
- **Base Model**: `cardiffnlp/twitter-roberta-large-topic-sentiment-latest`
- **Architecture**: RoBERTa-large (355M parameters)
- **Key Techniques**:
  - **TextAugmenter**: synonym replacement, random deletion, random swap (`AUGMENT_PROB=0.3`)
  - **Enhanced Focal Loss** with label smoothing (`ε=0.05`, `γ=2.0`)
  - **Auto-Resume** from latest checkpoint in `sentiment_experiments/`
  - **OOM Guard**: auto-reduces batch size on CUDA out-of-memory errors
  - **Diagnostic plots**: ROC, Precision-Recall, per-class F1, and confidence analysis
- **Hyperparameters**:
  - `MAX_LEN`: 160 | `BATCH_SIZE`: 8 (Effective: 32 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 5e-6 (Cosine) | `EPOCHS`: 12
  - `WARMUP_RATIO`: 0.1 | `WEIGHT_DECAY`: 0.01
  - `EARLY_STOPPING`: Patience=10 | Threshold=0.001
- **Output Directory**: `sentiment_experiments/english_sentiment_enhanced/run_<timestamp>/`

---

### 4. English Topic Classification (18 Categories)
- **Script**: [`training/english/fine_tune_english_topic.py`](training/english/fine_tune_english_topic.py)
- **Base Model**: `roberta-base` (Optimized)
- **Loss Function**: `FocalLoss(alpha=0.25, gamma=2.0)` + Label Smoothing (`0.25`)
- **Key Techniques**:
  - Custom `EnhancedTrainer` with mixed precision autocast
  - Regularized classifier head with dropout (`0.35`)
  - Comprehensive per-split evaluation (Train, Validation, Test)
  - Auto-resume functionality from latest checkpoint
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `BATCH_SIZE`: 8 (Effective: 32 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 1e-5 (Cosine) | `EPOCHS`: 6
  - `WEIGHT_DECAY`: 0.08 | `MAX_GRAD_NORM`: 0.3 | `DROPOUT`: 0.35
- **Output Directory**: `experiments/english_topic/run_<timestamp>/`

---

### 5. French Sentiment Classification
- **Script**: [`training/french/fine_tune_french_sentiment.py`](training/french/fine_tune_french_sentiment.py)
- **Base Model**: `camembert/camembert-large` (435M parameters)
- **Data Strategy**: Combines human-annotated French articles with cached translated data (`training/french/translated_data/`)
- **Key Techniques**:
  - **Focal Loss** (`α=0.65`, `γ=4.0`) with strong POSITIVE class focus
  - **POSITIVE class augmentation** with French intensifiers (`très`, `vraiment`, `extrêmement`, …)
  - **Enhanced class weights**: POSITIVE × 2.5, NEUTRAL × 0.7, NEGATIVE × 1.1
  - 80/10/10 train/validation/test split
  - Target: >80% Accuracy
- **Hyperparameters**:
  - `MAX_LEN`: 384 | `BATCH_SIZE`: 4 (Effective: 32 via `GRAD_ACCUM=8`)
  - `LEARNING_RATE`: 1e-5 | `EPOCHS`: 15
  - `WARMUP_RATIO`: 0.2 | `WEIGHT_DECAY`: 0.05
  - `EARLY_STOPPING`: Patience=10 | Threshold=0.003
- **Output Directory**: `training/french/experiments/french_sentiment_camembert_large_final/`

> **📌 Note on Data Cache**: Requires pre-translated parquet files in `training/french/translated_data/`:
> - `translated_french_data_all.parquet`
> - `real_french_data.parquet`

---

### 6. French Topic Classification (18 Categories)
- **Script**: [`training/french/fine_tune_french_topic.py`](training/french/fine_tune_french_topic.py)
- **Base Model**: `camembert/camembert-large`
- **Translation Pipeline**: Uses `Helsinki-NLP/opus-mt-en-fr` (MarianMT) to translate English articles to French; cached in `translated_data/french_topic_translated_data.csv`
- **Key Techniques**:
  - **Custom `LargeClassifierFocal` model** with CLS-token pooling + dropout head
  - **Focal Loss with class-specific gammas** (`γ` ranges 2.0–4.0 based on class frequency)
  - **Soft inverse-sqrt class weights** (clipped 0.8–2.0)
  - **Environment class oversampling** (class 13 boosted to ≥500 samples)
  - **Auto-resume** from latest valid checkpoint
- **Hyperparameters**:
  - `MAX_LEN`: 512 | `BATCH_SIZE`: 4 (Effective: 16 via `GRAD_ACCUM=4`)
  - `LEARNING_RATE`: 5e-6 | `EPOCHS`: 6
  - `WEIGHT_DECAY`: 0.01 | `MAX_GRAD_NORM`: 0.3
  - `EARLY_STOPPING`: Patience=5 | Threshold=0.001
  - `EVAL_STEPS`: 3000 | `SEED`: 42
- **Output Directory**: `training/french/experiments/topic_french/<timestamp>/`

---

## ⚡ Training Methodology & Optimizations

1. **Sliding Window for Long Documents** _(Arabic Sentiment)_:
   - Instead of hard truncation at 512 tokens, Arabic texts are split into overlapping chunks (`STRIDE=256`). Each chunk is classified independently, allowing the model to process arbitrarily long documents.

2. **Focal Loss for Extreme Imbalance**:
   - Addresses high frequency discrepancy between major news topics (e.g. *Politics*, *General*) and niche topics (e.g. *Weather*, *Environment*, *Migration*).
   - $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ — downweights easy examples and focuses on hard minority classes.
   - Class-specific gammas ($2.0 \le \gamma \le 4.0$) are used in French topic classification.

3. **Layerwise Learning Rate Decay (LLRD)** _(Arabic Topic)_:
   - Lower Transformer layers train with smaller learning rates (preserving pre-trained representations); upper layers adapt rapidly. Decay factor: `0.95` per layer.

4. **Cross-Lingual Translation Augmentation** _(French Pipelines)_:
   - English news articles in the dataset are translated to French using MarianMT (`Helsinki-NLP/opus-mt-en-fr`), expanding the French training corpus. Results are cached to disk to avoid redundant translation.

5. **Text Augmentation** _(English Sentiment)_:
   - Training samples undergo random synonym replacement, word deletion, and word swap (`AUGMENT_PROB=0.3`) to improve generalization.

6. **POSITIVE Class Augmentation** _(Arabic & French Sentiment)_:
   - The underrepresented POSITIVE class is boosted by appending language-appropriate intensifiers to existing positive samples.

7. **Mixed Precision (`fp16` / `bf16`) & Memory Management**:
   - Reduces VRAM consumption by ~50%, enabling larger batch sizes on consumer GPUs (e.g., RTX 3050/4060).

8. **Resilient Auto-Resume**:
   - Training scripts automatically scan their experiment directories for valid checkpoints and resume seamlessly upon restart.

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
# Train English Sentiment model (Twitter-RoBERTa-large + Text Augmentation)
python training/english/fine_tune_english_sentiment.py

# Train English Topic model (18 categories, Focal Loss + Label Smoothing)
python training/english/fine_tune_english_topic.py

# ── French ──────────────────────────────────────────────────────────────────
# Train French Sentiment model (CamemBERT-large + Focal Loss)
# Note: Ensure cached parquet files exist in training/french/translated_data/
python training/french/fine_tune_french_sentiment.py

# Train French Topic model (CamemBERT-large + Auto-Translation)
# Auto-translates English articles on first run and caches to translated_data/
python training/french/fine_tune_french_topic.py
```

### Experiment Scanning & Leaderboard

The leaderboard utility [`reports/report_search.py`](reports/report_search.py) scans completed runs in `experiments/` and ranks them by Macro F1 or Accuracy:

```powershell
# Scan all tasks in experiments/
python reports/report_search.py --task all

# Scan specific task and rank by Macro F1
python reports/report_search.py --task arabic_sentiment   --metric f1_macro
python reports/report_search.py --task arabic_topic       --metric f1_macro
python reports/report_search.py --task english_topic      --metric f1_macro

# Rank by accuracy
python reports/report_search.py --task all                --metric accuracy
```

---

## 📦 Experiment Output Structure

Standard run outputs contain the following artifacts:

```
<experiment_directory>/
├── best_model/                         # Saved model weights & tokenizer
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   └── label_mapping.json              # Class ID to string label mapping
├── checkpoints/                        # Periodic training checkpoints (checkpoint-XXXX)
├── hyperparameters.json                # Immutable run configuration recorded at launch
├── eval_results.json                   # Final validation / test metrics (F1 Macro, accuracy)
├── test_results.json                   # Detailed test set metrics
├── summary.json                        # Comprehensive run summary (if generated)
├── predictions_<split>.csv             # Per-sample predictions & ground truth
└── plots/                              # Visual diagnostics
    ├── train_loss.png
    ├── eval_loss.png
    ├── eval_f1_accuracy.png
    ├── learning_curve.png
    ├── confusion_matrix_<split>.png
    ├── roc_curves_<split>.png          # French topic & English sentiment
    ├── auc_barchart_<split>.png        # French topic
    ├── per_class_f1_bar.png            # English sentiment / Arabic topic
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

