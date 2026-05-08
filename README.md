# Multilingual NLP Pipeline for News Analysis

This repository contains the codebase for a comprehensive Multilingual Natural Language Processing (NLP) pipeline, developed as a Projet de Fin d'Études (PFE). 

## 📌 Project Overview
The primary goal of this project is to process, analyze, and extract deep insights from a large corpus of multilingual news articles (primarily Arabic, French, and English). It is designed to automatically understand the subject matter, extract key entities (people, organizations, locations), gauge the sentiment of the text, and perform trend analysis over time.

## 🎯 Current Phase: Baseline & Ground Truth Collection (Zero-Shot)
In this **current version**, our focus is heavily on data preparation, pipeline hardening, and baseline evaluations. We are currently:
1. **Relying on Pre-Trained Models (Zero-Shot):** Using existing, highly capable out-of-the-box models (like GLiNER for NER and LLMs for sentiment/topics) without specific training on our news dataset.
2. **Generating Ground Truth:** Exporting separated language datasets (e.g., `french_texts.csv`) and running manual annotator CLI tools to validate model outputs. This creates a high-quality, human-verified dataset.
3. **Benchmarking:** Evaluating the baseline accuracy, resource consumption, and inference speed of these models.

### 🔮 Future Phase: Fine-Tuning
The primary purpose of what we are doing right now is data collection for the future. We are going to **Fine-Tune** smaller, highly efficient domain-specific Transformer models (like AraBERT, CamemBERT, or specialized RoBERTa variants) later. The annotations and datasets we are generating and cleaning right now will serve as the training data for these fine-tuned models, allowing us to eventually replace heavy, general-purpose LLMs with faster, specialized alternatives.

## 🚀 Key Features

### 1. Advanced Multilingual Support
- **Language Detection:** Uses `fasttext` to automatically detect the language of incoming raw texts.
- **Language-Specific Preprocessing:** Features dedicated preprocessing routers (`src/preprocessing/router.py`).
  - **Arabic (`arabic.py`):** Includes specialized normalization for Arabic text, such as removing decorative characters (Franco-Arabic text, spam strings, block-drawing characters), standardizing "laughter" repetitions, and unifying brackets and punctuation.
  - **Latin (`latin.py`):** Handles French and English text normalization and noise reduction.

### 2. Core NLP AI Models
- **Named Entity Recognition (NER):** Leverages `GLiNER` (Generalist and Lightweight Model for Named Entity Recognition, specifically `urchade/gliner_multi-v2.1`) to accurately identify entities across different languages. Supports integration with AraBERT and CAMEL for specialized Arabic NER.
- **Topic Generation & Extraction:** Utilizes Large Language Models (LLMs) to dynamically extract the main topics discussed in an article.
- **Sentiment Analysis:** Analyzes the emotional tone of the articles using state-of-the-art transformer and LLM models, categorizing them and scoring the sentiment.

### 3. High-Performance Architecture
- **Two-Stage Execution Pipeline:** 
  - **CPU Pass (`pipeline/cpu_pass.py`):** Handles lightweight tasks like language detection, preprocessing, and timing benchmarks.
  - **GPU Pass (`pipeline/gpu_pass.py`):** Offloads heavy deep-learning inferences (Transformers, LLMs) to the GPU, operating model-by-model to prevent Out-Of-Memory (OOM) crashes.
- **Benchmarking Tools:** Built-in benchmarking to track CPU and GPU inference times, ensuring models are running efficiently and identifying hardware bottlenecks (e.g., handling CUDA assertions).

### 4. Advanced Analytics & Trend Detection
- **Analytics Engine (`analysis/`):** Generates reports on global entity frequencies, tracks the top entities by month (`entities_by_month.py`), and breaks them down by country (`top_entities_by_country.py`).
- **Peak Detection (`topic_peaks.py`):** Identifies statistical "peaks" (using z-score thresholds) in entity or topic mentions over rolling time windows, enabling the discovery of trending news stories or sudden global events.

## 📂 Codebase Structure

```text
PROJET_PFE/
│
├── main.py                     # Entry point of the application
├── requirements.txt            # Python dependencies (PyTorch, Transformers, GLiNER, Spacy, etc.)
│
├── src/                        # Core Logic & Utilities
│   ├── config.py / db_config.py# Application and database configuration (SQLAlchemy)
│   ├── ner_extraction.py       # Wrapper for GLiNER and other NER models
│   ├── sentiment_analysis.py   # Wrapper for Sentiment LLMs/Transformers
│   ├── topic_generation.py     # Wrapper for Topic Extraction models
│   ├── language_detection.py   # FastText-based language identification
│   └── preprocessing/          # Language-specific text cleaning rules (arabic.py, latin.py, router.py)
│
├── pipeline/                   # Execution Workflow
│   ├── sampler.py              # Fetches and samples articles from the database
│   ├── cpu_pass.py             # Execution stage for lightweight operations
│   ├── gpu_pass.py             # Execution stage for heavy AI inference
│   └── exporter.py             # Exports results to CSV for manual validation
│
├── analysis/                   # Reporting and Statistical Analysis
│   ├── report_generator.py     # Main generator for analytics CSVs
│   ├── entities_by_month.py    # Tracks entity frequency over time
│   ├── topic_peaks.py          # Detects anomalies/spikes in topic trends
│   └── top_entities_by_country.py
│
├── data_seperation/            # Scripts for separating data by language for annotation
└── results/                    # Directory where output CSVs and benchmark reports are saved
```

## 🛠️ Technology Stack
- **Deep Learning / NLP:** PyTorch, HuggingFace Transformers, GLiNER, SpaCy, NLTK, FastText.
- **Data Processing:** Pandas, NumPy, Scikit-learn.
- **Database:** MySQL, SQLAlchemy.
- **Logging & System:** Python `logging`, `platform`, `psutil`.

## ⚙️ How It Works (The Pipeline Flow)
1. **Sampling:** `main.py` initiates a run by pulling a sample (e.g., 1000 articles) from the database via `sampler.py`.
2. **Preprocessing:** Text is routed to `preprocessing/router.py` based on its detected language, where it is thoroughly cleaned.
3. **Inference (GPU):** The cleaned text is passed to NER, Sentiment, and Topic models sequentially in `gpu_pass.py`.
4. **Export & Results:** The extracted entities, sentiments, and topics are written back to the database. Additionally, `exporter.py` outputs CSVs (`gliner_ner_results.csv`, `topic_sentiment_results.csv`) to a timestamped directory in `results/`.
5. **Analytics:** The `analysis/report_generator.py` aggregates the database results to find trends and anomalies.
