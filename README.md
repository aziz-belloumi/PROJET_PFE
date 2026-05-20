# Multilingual NLP Pipeline for News Analysis

This repository contains the codebase for a comprehensive Multilingual Natural Language Processing (NLP) pipeline, developed as a Projet de Fin d'Études (PFE).

## 📌 Project Overview
The primary goal of this project is to process, analyze, and extract deep insights from a large corpus of multilingual news articles (primarily Arabic, French, and English). It is designed to automatically understand the subject matter, extract key entities (people, organizations, locations), gauge the sentiment of the text, and perform trend analysis over time.

---

## 🎯 Current Phase: Baseline & Ground Truth Collection (Zero-Shot)
In this **current version**, our focus is heavily on data preparation, pipeline hardening, and baseline evaluations:
1. **Relying on Pre-Trained Models (Zero-Shot):** Using highly capable out-of-the-box models (like GLiNER for NER and LLMs for sentiment/topics) without specific training on our news dataset.
2. **Generating Ground Truth:** Exporting separated language datasets (e.g., `french_texts.csv`) and running manual annotator CLI tools to validate model outputs. This creates a high-quality, human-verified dataset.
3. **Benchmarking:** Evaluating the baseline accuracy, resource consumption, and inference speed of these models.

### 🔮 Future Phase: Fine-Tuning
The primary purpose of what we are doing right now is data collection for the future. We are going to **Fine-Tune** smaller, highly efficient domain-specific Transformer models (like AraBERT, CamemBERT, or specialized RoBERTa variants) later. The annotations and datasets we are generating and cleaning right now will serve as the training data for these fine-tuned models, allowing us to eventually replace heavy, general-purpose LLMs with faster, specialized alternatives.

---

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

---

## 📊 Database Schema & Enriched Data Tables

The pipeline stores processed results in a series of relational MySQL tables managed via SQLAlchemy. These tables link the original raw dataset with the NLP model predictions.

### 1. Enriched Pipeline Tables
* **`articles_enriched`**: Serves as the central repository for the pipeline results.
  * **Columns:** `article_id` (Primary Key), `language`, `sentiment_label`, and multiple CPU/GPU processing latency metrics (`cpu_time_ner`, `gpu_time_sentiment`, etc.).
* **`entities`**: The global unique entity vocabulary list.
  * **Columns:** `entity_id` (Primary Key), `entity_name`, `entity_type`, `normalized_name`, and a global `frequency` count. It enforces unique constraints on the combination of `(entity_type, normalized_name)`.
* **`article_entities`**: A relational junction table connecting articles and their extracted entities.
  * **Columns:** `(article_id, entity_id, model_version)` as a Composite Primary Key, and `confidence_score`.
* **`article_topics`**: Stores the LLM-extracted topic for each article.
  * **Columns:** `article_id` (Primary Key) and `topic_label`.

### 2. Original Raw Table Integrations
The pipeline reads raw input data from original tables:
* **`article`**: The source table representing original articles. The pipeline relies on its metadata columns:
  * `id` — Unique identifier.
  * `crawl_date` — Primary timestamp for trend bucketing.
  * `year` / `month` — Fallback date indicators if `crawl_date` is empty or corrupted.
  * `id_countries` — Relational mapping to identify geographical distributions.
* **`country`**: Map-reference table supplying names for country-based reporting:
  * `id` — Country ID.
  * `label_en` / `label_fr` / `label_ar` — Multilingual name labels.

---

## 📈 Analytical & Benchmarking Modules (`analysis/`)

The `analysis/` folder aggregates raw article metadata and pipeline outputs to generate CSV reports.

* **`entities_by_month.py`**: Buckets articles chronologically based on `crawl_date` (or `year`/`month` fallback). Extracts top-K entities for each month based on distinct article counts and mention frequencies.
* **`top_entities_by_country.py`**: Explodes the `id_countries` values from the original article table, joins them against the `country` table, and computes top-K entities globally per country.
* **`topics_by_month.py`**: Computes the distribution share of each topic dynamically per month, identifying the single "Dominant Topic" of each month.
* **`topic_peaks.py`**: Detects statistical anomalies in topic shares using a rolling z-score. An anomaly/peak is flagged when a topic's share rises above a rolling average by `2.5` standard deviations.
* **`ner_comparison.py`**: Evaluates agreement between different NER models (like AraBERT vs CamelBERT in Arabic) and produces a label distribution (PER, ORG, LOC, DAT, EVE, MIS, PRO, COM) and confidence stats.
* **`sentiment_comparison.py`**: Computes sentiment distributions (POS, NEG, NEU, MIX) per model and language, tracking model agreement percentages.
* **`timing_comparison.py`**: Aggregates CPU and GPU inference times per model, outputting speedup ratios (e.g., how many times faster GPU processing was vs CPU).

---

## 🗂️ Language Separation & Data Processing Workflow (`data_seperation/`)

To support annotation campaigns and dialectal research, the repository contains tools in the `data_seperation/` folder to clean, filter, and balance datasets:

* **`fetch_texts.py`**: Separates raw articles into individual language files:
  * **Language Identification:** Uses FastText to categorize text into English, French, Arabic, Mixed, or Rejected.
  * **Quality Gates:** Filters out articles that are too short (<= 2 words), have low language confidence scores, or fail alpha-ratio threshold checks.
  * **Arabic Ratio Check:** Measures the ratio of Arabic Unicode characters to classify pure Arabic texts vs Mixed scripts.
  * **Resume Logic:** Automatically tracks already-processed lines to resume separation if interrupted.
* **`split_dataset.py`**: Splits separated language files (like `arabic_texts.csv`) into balanced, equal-sized chunks to distribute workloads among annotators.
* **`check_dialect/ensemble_annotator.py`**: A multi-model ensemble system classifying Arabic text as Modern Standard Arabic (MSA) or Dialectal Arabic (Egyptian, Levantine, Gulf, Maghrebi).

---

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
│   ├── top_entities_by_country.py # Segment entities by geographical origin
│   ├── ner_comparison.py       # Computes NER model metrics and label distributions
│   ├── sentiment_comparison.py # Compares sentiment labels and agreement percentages
│   └── timing_comparison.py    # Analyzes CPU/GPU timing and speedups
│
├── data_seperation/            # Scripts for separating data by language for annotation
│   ├── fetch_texts.py          # FastText separation and resume pipeline
│   ├── split_dataset.py        # Balances datasets into workload chunks
│   └── check_dialect/          # Ensemble Arabic dialect annotator (MSA vs Dialectal)
│
└── results/                    # Directory where output CSVs and benchmark reports are saved
```

---

## 🛠️ Technology Stack
- **Deep Learning / NLP:** PyTorch, HuggingFace Transformers, GLiNER, SpaCy, NLTK, FastText.
- **Data Processing:** Pandas, NumPy, Scikit-learn.
- **Database:** MySQL, SQLAlchemy (PyMySQL).
- **Logging & System:** Python `logging`, `platform`, `psutil`.

---

## ⚙️ How It Works (The Pipeline Flow)
1. **Sampling:** `main.py` initiates a run by pulling a sample (e.g., 1000 articles) from the database via `sampler.py`.
2. **Preprocessing:** Text is routed to `preprocessing/router.py` based on its detected language, where it is thoroughly cleaned.
3. **Inference (GPU):** The cleaned text is passed to NER, Sentiment, and Topic models sequentially in `gpu_pass.py`.
4. **Export & Results:** The extracted entities, sentiments, and topics are written back to the database. Additionally, `exporter.py` outputs CSVs (`gliner_ner_results.csv`, `topic_sentiment_results.csv`) to a timestamped directory in `results/`.
5. **Analytics:** The `analysis/report_generator.py` aggregates the database results to find trends and anomalies.

---

## Detailed Preprocessing (Arabic & Latin)

This project contains robust, language-specific preprocessing implemented in `src/preprocessing/arabic.py` and `src/preprocessing/latin.py`. Both are flag-driven preprocessors that can be tuned for NER, topic, or sentiment tasks.

### Arabic preprocessing highlights (`src/preprocessing/arabic.py`)
- Unicode normalization and Arabic letter canonicalization (`أ/إ/آ -> ا`, `ة -> ه`, `ى -> ي`).
- Diacritics and Tatweel removal to reduce noise.
- URL/email removal and HTTP fragment filtering.
- Optional removal of Arabic-Indic and Western digits.
- Decorative noise removal (box drawing, dingbats, replacement chars, Arabic zero `٠`).
- Social spam line detection/removal using Franco‑Arabic decorative char heuristics.
- Loose bracket collapse and unmatched bracket stripping while preserving balanced pairs.
- `fix_merged_keywords()` splits merged tokens for keywords like `ليبيا`, `للبيع`, `للإيجار`, and now `طرابلس` (Tripoli), avoiding accidental splits of single-letter prefixes.
- Punctuation normalization with de-duplication and enforced single-space around internal periods (`.`).

### Latin preprocessing highlights (`src/preprocessing/latin.py`)
- Normalizes typographic quotes and dashes, deduplicates repeated punctuation.
- Decorative and unicode junk removal.
- Two modes for social content: placeholders for sentiment (`http`, `@user`, `email`) or full removal for NER/topic.
- CamelCase hashtag splitting for long tokens.
- Enforces the same one-space-around-internal-period rule as Arabic for consistency.

These preprocessors are designed to preserve entity spans and linguistic signals while removing social/decorative noise that harms extraction quality.

---

## Qwen (Ollama) compatibility — important note

- Qwen is integrated via Ollama HTTP calls (see `BenchLLM` in `scripts/ressources/master_benchmark.py` and `src/topic_generation.py`). Ollama runs as a separate server process and manages model VRAM itself.
- Consequences:
  - The Python benchmark measures only local process memory (`psutil`) and `torch.cuda.memory_allocated()` inside the Python process. If Qwen runs inside Ollama, you will see `Peak GPU = 0.0MB` for the Python process even though the model uses GPU on the Ollama server.
  - `Load Time = 0.00s` for Qwen in the Python benchmark indicates the Python-side object initialization is trivial; the model lifecycle is handled by Ollama.

Conclusion: the `analysis/` scripts WILL run with Qwen provided the Ollama service is running and the Qwen model is available on the server. If Ollama is not running or the model is missing, Qwen-backed steps will either be skipped or appear as zero/placeholder metrics in the generated reports.

---

## Quick Run Commands

Activate your Python environment, install dependencies, then run:

```bash
python main.py
```

For benchmarks only:

```bash
python scripts/ressources/master_benchmark.py
python scripts/run_all_models.py
```

For analysis reports:

```bash
python analysis/report_generator.py
```

---

## Recommended Next Steps

- Add unit tests for `fix_merged_keywords()` and the dot-space normalization.
- If using Ollama locally, add an optional `nvidia-smi` sampler or Ollama API integration to capture server-side GPU memory for more accurate benchmarks.
- Add example input/output snippets to `README.md` or `docs/` to show the effect of preprocessing on typical tweets/articles.
