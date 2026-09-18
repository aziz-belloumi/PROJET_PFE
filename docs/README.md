# Multilingual Transformer Fine-Tuning Suite Documentation

Welcome to the central documentation hub for the Multilingual Transformer Fine-Tuning codebase. This folder contains in-depth documentation for all fine-tuning scripts across **Arabic**, **English**, and **French** for both **Sentiment Analysis** (3 classes) and **Topic Classification** (18 categories).

---

## 📁 Documentation Index

```
docs/
├── README.md                          # Master Overview & Hyperparameter Matrix (this file)
├── README_arabic_sentiment.md         # Arabic Sentiment Analysis (CamelBERT-Mix Sentiment)
├── README_arabic_topic.md             # Arabic Topic Classification (CamelBERT-Mix + LLRD)
├── README_english_sentiment.md        # English Sentiment Analysis (Twitter-RoBERTa-Large)
├── README_english_topic.md            # English Topic Classification (RoBERTa-Base Optimized)
├── README_french_sentiment.md         # French Sentiment Analysis (CamemBERT-Large)
└── README_french_topic.md             # French Topic Classification (CamemBERT-Large + Focal Loss)
```

---

## 🗺️ Quick Access by Task

| Language | Task | Backbone Model | Full Documentation | Target Classes | Key Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🇸🇦 **Arabic** | **Sentiment** | `camelbert-mix-sentiment` | [**Arabic Sentiment README**](file:///c:/Users/bello/Desktop/Fine_Tuning/docs/README_arabic_sentiment.md) | 3 (`NEG`, `NEU`, `POS`) | Sliding Window (512 len, 256 stride), Positive Intensifiers, Weighted Focal Loss ($\gamma=2.5, \alpha=0.45$) |
| 🇸🇦 **Arabic** | **Topic** | `camelbert-mix` | [**Arabic Topic README**](file:///c:/Users/bello/Desktop/Fine_Tuning/docs/README_arabic_topic.md) | 18 Topics | Layerwise Learning Rate Decay ($0.95$ across 12 layers), Median-Clipped Class Weights, Focal Loss ($\gamma=1.5$) |
| 🇬🇧 **English** | **Sentiment** | `twitter-roberta-large` | [**English Sentiment README**](file:///c:/Users/bello/Desktop/Fine_Tuning/docs/README_english_sentiment.md) | 3 (`NEG`, `NEU`, `POS`) | Online Stochastic Text Augmentation, Focal Loss ($\gamma=2.0$) + Label Smoothing ($0.05$), Dynamic OOM Guard |
| 🇬🇧 **English** | **Topic** | `roberta-base` (Optimized) | [**English Topic README**](file:///c:/Users/bello/Desktop/Fine_Tuning/docs/README_english_topic.md) | 18 Topics | 2-Stage Curriculum Loss (Smoothed CE $\to$ Focal Loss), Dropout ($0.35$), Label Smoothing ($0.25$) |
| 🇫🇷 **French** | **Sentiment** | `camembert-large` | [**French Sentiment README**](file:///c:/Users/bello/Desktop/Fine_Tuning/docs/README_french_sentiment.md) | 3 (`NEG`, `NEU`, `POS`) | CamemBERT-Large (435M), Extreme Focal Loss ($\gamma=4.0$), French Intensifier Augmentation (50%) |
| 🇫🇷 **French** | **Topic** | `camembert-large` | [**French Topic README**](file:///c:/Users/bello/Desktop/Fine_Tuning/docs/README_french_topic.md) | 18 Topics | `LargeClassifierFocal`, Dynamic Class Gammas ($\gamma_c \in [2.0, 4.0]$), Offline MarianMT Translation Pipeline |

---

## 📊 Cross-Model Hyperparameter & Strategy Matrix

| Configuration Parameter | Arabic Sentiment | Arabic Topic | English Sentiment | English Topic | French Sentiment | French Topic |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Script Path** | [`fine_tune_arabic_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/arabic/fine_tune_arabic_sentiment.py) | [`fine_tune_arabic_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/arabic/fine_tune_arabic_topic.py) | [`fine_tune_english_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/english/fine_tune_english_sentiment.py) | [`fine_tune_english_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/english/fine_tune_english_topic.py) | [`fine_tune_french_sentiment.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/french/fine_tune_french_sentiment.py) | [`fine_tune_french_topic.py`](file:///c:/Users/bello/Desktop/Fine_Tuning/training/french/fine_tune_french_topic.py) |
| **Base Backbone** | `camelbert-mix-sentiment` | `camelbert-mix` | `twitter-roberta-large` | `roberta-base` | `camembert-large` | `camembert-large` |
| **Parameter Count** | 110M | 110M | 355M | 125M | 435M | 435M |
| **Target Classes** | 3 | 18 | 3 | 18 | 3 | 18 |
| **Max Sequence Length** | 512 (256-stride window) | 512 | 160 | 512 | 384 | 512 |
| **Physical / Effective Batch** | 4 / **16** | 8 / **8** | 8 / **32** | 8 / **32** | 4 / **32** | 4 / **16** |
| **Learning Rate** | $1 \times 10^{-5}$ | $1 \times 10^{-5}$ (LLRD $0.95$) | $5 \times 10^{-6}$ | $1 \times 10^{-5}$ | $1 \times 10^{-5}$ | $5 \times 10^{-6}$ |
| **Scheduler** | Cosine w/ Restarts | Cosine | Cosine | Cosine | Cosine | Linear Warmup |
| **Loss Function** | Weighted Focal ($\gamma=2.5$) | Weighted Focal ($\gamma=1.5$) | Focal ($\gamma=2.0$) + Smooth | 2-Stage Smooth CE $\to$ Focal | Extreme Focal ($\gamma=4.0$) | Class-Specific Gammas ($2.0–4.0$) |
| **Label Smoothing** | $\epsilon = 0.15$ | None | $\epsilon = 0.05$ | $\epsilon = 0.25$ | None | None |
| **Weight Decay** | $0.03$ | $0.01$ | $0.01$ | $0.08$ | $0.05$ | $0.01$ |
| **Augmentation** | Arabic Intensifiers | Resampling | Online Synonym/Delete/Swap | Resampling | French Intensifiers (50%) | MarianMT + Class 13 Oversample |
| **Early Stopping Patience** | 20 | 10 | 10 (post-warmup) | 2 (epoch-level) | 10 | 5 |
| **Hardware Setup** | RTX 3050 (fp16 + GC) | RTX 3050 (fp16/bf16) | RTX 3050 (fp16 + GC + OOM Guard) | RTX 3050 (fp16 + AMP) | RTX 3050 (fp16, GC Disabled) | RTX 3050 (bf16/fp16, GC Disabled) |

---

## 🧠 Architectural Insights Across the Suite

### 1. Language-Specific Backbones over Generic Multilingual Models
Generic multilingual models (mBERT, XLM-R) distribute parameter capacity across 100+ languages, causing suboptimal tokenization and morphological degradation. Each pipeline uses dedicated monolingual/dialectal backbones:
* **Arabic**: CamelBERT-Mix handles morphological richness across Modern Standard Arabic (MSA) and dialects (`da`).
* **English**: Twitter-RoBERTa-Large is adapted to informal tone, concise syntax, and nuanced headline sentiment.
* **French**: CamemBERT-Large (435M) provides the representational capacity necessary for complex journalistic French.

### 2. Multi-Faceted Class Imbalance Mitigation
News datasets feature severe skew towards `NEUTRAL` and dense categories (`Politics`, `General`). Three complementary strategies are applied:
* **Loss Functions**: Customized Focal Losses with focusing exponents ranging from $\gamma=1.5$ to class-specific dynamic gammas up to $\gamma=4.0$.
* **Lexical Augmentation**: Synthetically injecting culturally and linguistically authentic intensifiers for rare positive sentiment samples.
* **Structural Oversampling**: Hard integer repetition ensuring a minimum sample floor (e.g. Environment class).

### 3. Hardware Optimization for RTX 3050
All scripts are configured for laptop GPU hardware (4GB–6GB VRAM):
* **Gradient Accumulation**: Simulates effective batch sizes up to 32 while keeping physical batches within 4–8 samples.
* **Mixed Precision (`fp16` / `bf16` / `tf32`)**: Accelerates matrix multiplications on Tensor Cores.
* **Strategic Gradient Checkpointing**: Enabled where memory constraints dominate (Arabic sliding window, Twitter-RoBERTa-large) and disabled where compute throughput dominates (CamemBERT-Large).
* **Automated OOM Recovery**: Dynamically catches memory errors and scales batch sizes down to prevent process crashes.

---

## 🚀 Execution Commands

```powershell
# Arabic Models
python training/arabic/fine_tune_arabic_sentiment.py
python training/arabic/fine_tune_arabic_topic.py

# English Models
python training/english/fine_tune_english_sentiment.py
python training/english/fine_tune_english_topic.py

# French Models
python training/french/fine_tune_french_sentiment.py
python training/french/fine_tune_french_topic.py
```
