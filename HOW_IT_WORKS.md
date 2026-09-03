# How the NLP Pipeline Works

> A multilingual NLP enrichment pipeline for Arabic, English, and French news articles, performing **Named Entity Recognition (NER)**, **Sentiment Analysis**, **Topic Classification**, and **LLM-based analysis** using Qwen.

---

## Table of Contents

1. [Quick Start (Command Sequence)](#1-quick-start-command-sequence)
2. [High-Level Overview](#2-high-level-overview)
3. [Project Structure & File Organization](#3-project-structure--file-organization)
4. [Pipeline Stages](#4-pipeline-stages)
   - [Stage 1 — Sampling & Language Detection](#stage-1--sampling--language-detection)
   - [Stage 2 — GPU Inference Pass](#stage-2--gpu-inference-pass)
   - [Stage 3 — Qwen (LLM) Pass](#stage-3--qwen-llm-pass)
   - [Stage 4 — Analytics & Report Generation](#stage-4--analytics--report-generation)
5. [Key Modules](#5-key-modules)
6. [Models Used](#6-models-used)
7. [Database Schema](#7-database-schema)
8. [Configuration Reference](#8-configuration-reference)

---

## 1. Quick Start (Command Sequence)

Follow these step-by-step commands to get the pipeline running from scratch:

### Step 1: Create & Activate Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Or on Windows (CMD)
.venv\Scripts\activate.bat

# Or on Linux / macOS
source .venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Create a file named `.env.local` in the project root for database and hardware settings:
```env
# Database Settings
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=nlp_mena
RAW_TABLE=article

# Hardware
GPU_DEVICE=0
CPU_DEVICE=-1
```

> **Pipeline Controls:** Pipeline execution toggles (`SAMPLE_SIZE`, `RUN_NER`, `RUN_SENTIMENT`, `RUN_TOPIC`, `RUN_QWEN`, `EXPORT_ANALYTICS_CSV`, etc.) are configured directly in Python code in [`src/config/pipeline.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/pipeline.py).


### Step 4: (Optional) Setup Qwen LLM with Ollama
If you wish to enable the Qwen pass (`RUN_QWEN=true`):
1. Install [Ollama](https://ollama.com/).
2. Pull the Qwen 2.5 7B model:
   ```bash
   ollama pull qwen2.5:7b
   ```
3. Start the Ollama background service/server (keep this running in a separate terminal):
   ```bash
   ollama serve
   ```
4. Set `RUN_QWEN=true` in your `.env.local`.

### Step 5: Run the Pipeline
```bash
python main.py
```

---

## 2. High-Level Overview

```
MySQL DB (raw articles)
        │
        ▼
 ┌──────────────────────┐
 │  Stage 1: Sampling   │  ← Fetch unprocessed articles
 │  & Language Detect   │  ← fastText language detection (ar / en / fr)
 │  (pipeline/sampler)  │  ← Task-specific text preprocessing
 └──────────┬───────────┘
            │ work_df (valid articles)
            ▼
 ┌──────────────────────┐
 │  Stage 2: GPU Pass   │  ← NER  → article_entities / entities
 │  (pipeline/gpu_pass) │  ← Sentiment → article_sentiments
 └──────────┬───────────┘  ← Topic → article_topics
            │
            ▼
 ┌──────────────────────┐
 │  Stage 3: Qwen Pass  │  ← Ollama-hosted Qwen2.5:7b LLM
 │  (src/qwen/)         │  ← qwen_article_sentiments
 └──────────┬───────────┘  ← qwen_article_topics
            │
            ▼
 ┌──────────────────────┐
 │  Stage 4: Analytics  │  ← MySQL analytics tables (stored in DB)
 │  (analysis/)         │  ← analytics_topics_by_month
 └──────────────────────┘  ← analytics_topic_peaks
                            ← analytics_entities_by_month
                            ← analytics_top_entities_by_country
                            ← Optional summary CSV: analytics_summary.csv
```

The pipeline is **incremental**: it only processes articles that haven't been analyzed yet (using `article_sentiments.sentiment_label IS NULL` as a tracking condition). Articles in unsupported languages are marked as `SKIPPED` so they are never reprocessed.

---

## 3. Project Structure & File Organization

The repository is structured modularly to isolate each concern across pipeline stage execution, core NLP engine models, text preprocessing routines, database persistence, comparative LLM evaluation, and analytical SQL aggregations.

### Directory Tree

```
PROJET_PFE/
├── main.py                         # Master entry point — orchestrates Stages 1 to 4
├── requirements.txt                # Pinned Python package dependencies
├── .env.local                      # Local database credentials & hardware configuration
├── .gitignore                      # Git ignore rules (venv, model caches, temporary logs)
├── README.md                       # Comprehensive engineering documentation & benchmark findings
├── HOW_IT_WORKS.md                 # Detailed operational & architectural reference guide
│
├── pipeline/                       # Pipeline Execution Stages
│   ├── __init__.py                 # Package marker
│   ├── sampler.py                  # Stage 1: Batch sampling, fastText language detection, task preprocessing
│   ├── gpu_pass.py                 # Stage 2: Sequential GPU inference for NER, Sentiment, & Topic models
│   └── cpu_pass.py                 # Legacy / CPU-only inference runner and benchmark harness
│
├── src/                            # Core Application Modules & NLP Engines
│   ├── __init__.py                 # Package marker
│   ├── db_config.py                # SQLAlchemy engine, connection pooling, migrations, atomic upserts
│   ├── language_detection.py       # FastTextLanguageDetector & ArticleClassifier
│   ├── ner_extraction.py           # GLiNERNER & TransformersNER engines with unified 8-class taxonomy
│   ├── sentiment_extraction.py     # LLMSentiment classifier with multi-chunk voting & heuristic overrides
│   ├── topic_extraction.py         # TransformerTopic (fine-tuned) & LLMTopic (zero-shot) classifiers
│   ├── chunking.py                 # Token-aware sliding window chunking utility
│   ├── text_utils.py               # UTF-8 stdout configuration & BiDi/Arabic ligature terminal reshaping
│   │
│   ├── config/                     # Modular Configuration Sub-Package
│   │   ├── __init__.py             # Aggregated Config class combining all configuration modules
│   │   ├── base.py                 # Base paths, getenv() helper, and device resolution
│   │   ├── db.py                   # MySQL connection parameters, pool sizes, and table name definitions
│   │   ├── models.py               # Model names, Hugging Face repo IDs, local paths, and MODEL_ID_MAP
│   │   ├── pipeline.py             # Feature toggles (RUN_NER, RUN_SENTIMENT, RUN_TOPIC, RUN_QWEN, etc.)
│   │   ├── hyperparameters.py      # NER confidence thresholds, chunk token limits, overlap sizes
│   │   ├── heuristics.py           # Post-classification sentiment domain cues & 18 topic category mappings
│   │   ├── preprocessing.py        # Preprocessing flag presets per language and task
│   │   └── qwen.py                 # Ollama REST endpoint configuration, timeouts, and model tags
│   │
│   ├── preprocessing/              # Text Cleaning & Normalization Engine
│   │   ├── __init__.py             # Exposes PreprocessRouter, ArabicPreprocessor, LatinPreprocessor
│   │   ├── router.py               # Routes text to language- and task-specific cleaning pipelines
│   │   ├── arabic.py               # Tashkeel, tatweel, ligature, and Franco-Arabic normalization
│   │   └── latin.py                # English & French NFKC normalization, accents, URLs, noise stripping
│   │
│   └── qwen/                       # Comparative LLM Evaluation Pass (Ollama API)
│       ├── __init__.py             # Package marker
│       ├── run_qwen_pass.py        # Stage 3: Orchestration loop for Qwen 2.5 LLM inference
│       ├── sentiment_extraction.py # Prompt builder & response parser for Qwen 3-class sentiment
│       └── topic_extraction.py     # Prompt builder & response parser for Qwen 18-class topic mapping
│
├── finetuned_models/               # Local Fine-Tuned PyTorch / Hugging Face Checkpoints
│   ├── ARABIC SENTIMENT/           # Arabic 3-class sentiment model weights & tokenizer
│   ├── ENGLISH SENTIMENT/          # English 3-class sentiment model weights & tokenizer
│   ├── FRENSH SENTIMENT/           # French 3-class sentiment model weights & tokenizer (CamemBERT)
│   ├── ARABIC TOPIC/               # Arabic 18-class topic classification model weights & tokenizer
│   ├── ENGLSIH TOPIC/              # English 18-class topic classification model weights & tokenizer
│   └── FRENSH TOPIC/               # French 18-class topic classification model weights & tokenizer
│
├── analysis/                       # Analytics & Automated SQL Aggregations (Stage 4)
│   ├── __init__.py                 # Package marker
│   ├── report_generator.py         # Master coordinator executing all SQL analytics & exporting CSV
│   ├── topics_by_month.py          # Monthly topic volume and percentage share distributions
│   ├── topic_peaks.py              # Time-series rolling Z-score statistical anomaly & spike detection
│   ├── entities_by_month.py        # Temporal Top-K entity tracking per calendar month
│   ├── top_entities_by_country.py  # Geographic Top-K entity distributions by article country metadata
│   └── reports/                    # Target output directory for generated CSV reports (analytics_summary.csv)
│
└── scripts/                        # Standalone Utility & Benchmarking Scripts
    ├── __init__.py                 # Package marker
    └── ressources/                 # Benchmark harnesses and evaluation notes
        ├── master_benchmark.py     # Multi-device standalone latency and peak VRAM benchmark harness
        └── score_remarques.md      # Performance benchmarks, evaluation notes, and observations
```

---

### Detailed Breakdown by Directory & File

#### 1. Root Directory

- [`main.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/main.py):
  - **Purpose:** The central orchestrator for the entire NLP enrichment pipeline.
  - **Responsibilities:**
    1. Initializes database connections and creates target tables if absent via [`DatabaseConnection.init_result_tables()`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/db_config.py).
    2. Invokes **Stage 1** ([`pipeline/sampler.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/pipeline/sampler.py)) to fetch, language-detect, filter, and preprocess articles.
    3. Invokes **Stage 2** ([`pipeline/gpu_pass.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/pipeline/gpu_pass.py)) to run sequential GPU inference across NER, Sentiment, and Topic models.
    4. Optionally triggers **Stage 3** ([`src/qwen/run_qwen_pass.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/qwen/run_qwen_pass.py)) when `RUN_QWEN=True`.
    5. Updates global entity frequencies via [`DatabaseConnection.update_entity_frequencies()`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/db_config.py).
    6. Invokes **Stage 4** ([`analysis/report_generator.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/report_generator.py)) to generate SQL analytics tables and export `analytics_summary.csv`.
- [`requirements.txt`](file:///c:/Users/bello/Desktop/PROJET_PFE/requirements.txt):
  - **Purpose:** Pinned project dependencies (`torch`, `transformers`, `gliner`, `fast-langdetect`, `sqlalchemy`, `pymysql`, `pandas`, `arabic-reshaper`, `python-bidi`, etc.).
- [`.env.local`](file:///c:/Users/bello/Desktop/PROJET_PFE/.env.local):
  - **Purpose:** Local environment configuration for database connectivity (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `RAW_TABLE`), device indices, and connection pooling parameters.
- [`README.md`](file:///c:/Users/bello/Desktop/PROJET_PFE/README.md):
  - **Purpose:** Production engineering manual with benchmark findings, entity linking rules, and detailed database table specifications.
- [`HOW_IT_WORKS.md`](file:///c:/Users/bello/Desktop/PROJET_PFE/HOW_IT_WORKS.md):
  - **Purpose:** Step-by-step operational handbook documenting workflow stages, model architectures, configuration options, and module interactions.

---

#### 2. `pipeline/` — Pipeline Execution Stages

Contains the stage-specific runner scripts executed sequentially by `main.py`:

- [`pipeline/sampler.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/pipeline/sampler.py) (**Stage 1 — Sampling & Language Detection**):
  - Queries `RAW_TABLE` for unprocessed articles (`sentiment_label IS NULL`) up to `SAMPLE_SIZE`.
  - Runs language detection via [`FastTextLanguageDetector`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/language_detection.py) with Arabic dialect aggregation and Arabic-script character ratio checks.
  - Splits articles into `work_df` (supported: `ar`, `en`, `fr`) and `skipped_df` (unsupported or low confidence).
  - Immediately writes `SKIPPED` placeholder records to `article_sentiments`, `article_topics`, `qwen_article_sentiments`, and `qwen_article_topics` for skipped articles to prevent re-processing.
  - Pre-inserts placeholder rows for valid articles to lock them as in-progress.
  - Prepares task-specific cleaned texts (`text_ner`, `text_sentiment`, `text_topic`) via [`PreprocessRouter`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/router.py).
- [`pipeline/gpu_pass.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/pipeline/gpu_pass.py) (**Stage 2 — GPU Inference Pass**):
  - Coordinates sequential, model-by-model GPU batch inference to avoid VRAM Out-of-Memory (OOM) errors:
    1. **NER Inference:** Runs [`GLiNERNER`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/ner_extraction.py) on `text_ner`, applies adjacent entity merging and deduplication, and bulk-inserts into `article_entities` and `entities`.
    2. **Sentiment Inference:** Runs [`LLMSentiment`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/sentiment_extraction.py) on `text_sentiment` with sliding-window chunking, majority vote aggregation, and heuristic rule overrides. Writes results to `article_sentiments`.
    3. **Topic Inference:** Runs [`TransformerTopic`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/topic_extraction.py) on `text_topic` with chunk voting. Writes results to `article_topics`.
  - Records execution metrics (latency per document, total wall-clock time, peak VRAM allocated in MB) into `benchmark_results`.
  - Cleans VRAM using `torch.cuda.empty_cache()` and garbage collection between models.
- [`pipeline/cpu_pass.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/pipeline/cpu_pass.py):
  - Standalone CPU benchmarking runner for measuring inference throughput on CPU hardware.

---

#### 3. `src/` — Core Modules & NLP Engines

Houses the core algorithms, model wrappers, preprocessing pipelines, configuration modules, and database operations:

##### `src/config/` — Modular Configuration Sub-Package
Centralizes all settings into structured sub-modules aggregated into a single unified [`Config`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/__init__.py) class:
- [`src/config/base.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/base.py): Base class resolving repository paths, loading `.env.local`, and managing CPU/GPU device selection.
- [`src/config/db.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/db.py): MySQL connection parameters, pool sizing (`DB_POOL_SIZE`, `DB_MAX_OVERFLOW`), and table name definitions.
- [`src/config/models.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/models.py): Hugging Face repo IDs, local model directory paths, and the numeric `MODEL_ID_MAP` (IDs 0–11).
- [`src/config/pipeline.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/pipeline.py): Pipeline execution controls (`RUN_NER`, `RUN_SENTIMENT`, `RUN_TOPIC`, `RUN_QWEN`, `SAMPLE_SIZE`, `LANG_THRESHOLD`, `EXPORT_ANALYTICS_CSV`).
- [`src/config/hyperparameters.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/hyperparameters.py): Chunk limits (e.g., 512 tokens), overlap sizes, NER confidence thresholds, and unified entity types.
- [`src/config/heuristics.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/heuristics.py): Sentiment post-rule keywords (question markers, negative override cues) and the 18-class multilingual topic taxonomy.
- [`src/config/preprocessing.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/preprocessing.py): Preprocessing configuration presets for Arabic and Latin text pipelines across tasks.
- [`src/config/qwen.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/config/qwen.py): Ollama REST endpoint configurations (`http://localhost:11434`), model tags (`qwen2.5:7b`), and timeout limits.

##### `src/preprocessing/` — Text Normalization Engine
- [`src/preprocessing/router.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/router.py): Defines [`PreprocessRouter`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/router.py), which routes text to the appropriate cleaner based on detected ISO language and task.
- [`src/preprocessing/arabic.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/arabic.py): Implements [`ArabicPreprocessor`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/arabic.py) for Arabic normalization (Unicode decomposition, diacritics/tashkeel stripping, tatweel removal, alef normalization, Franco-Arabic cleaning, noise stripping).
- [`src/preprocessing/latin.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/latin.py): Implements [`LatinPreprocessor`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/latin.py) for English and French text (NFKC normalization, accent preservation/stripping, smart quotes, hashtag/URL/mention removal, case folding).

##### `src/qwen/` — Comparative LLM Evaluation Pass
- [`src/qwen/run_qwen_pass.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/qwen/run_qwen_pass.py) (**Stage 3 — Qwen Pass**): Iterates over sampled articles and invokes Qwen 2.5 via Ollama REST API.
- [`src/qwen/sentiment_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/qwen/sentiment_extraction.py): Constructs language-specific prompts and parses structured JSON responses for 3-class sentiment (`POSITIVE`, `NEGATIVE`, `NEUTRAL`).
- [`src/qwen/topic_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/qwen/topic_extraction.py): Prompts Qwen for zero-shot topic classification and maps returned categories to the 18 standardized topics.

##### Core Standalone Source Files
- [`src/db_config.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/db_config.py):
  - Implements [`DatabaseConnection`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/db_config.py) with SQLAlchemy engine pooling (`QueuePool`).
  - Contains database schema migration & table initialization ([`init_result_tables`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/db_config.py)).
  - Provides atomic upsert operations for sentiment, topic, and Qwen tables.
  - Implements entity deduplication via exact matching and bounded fuzzy similarity (`SequenceMatcher`), plus global entity frequency re-indexing.
- [`src/language_detection.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/language_detection.py):
  - Implements [`FastTextLanguageDetector`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/language_detection.py) using `fast-langdetect` / fastText `lid.176.bin`.
  - Aggregates Arabic dialect probabilities (`ar`, `arz`, `ary`, `arq`, etc.) and performs Arabic-script character ratio checks.
  - Exposes [`ArticleClassifier`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/language_detection.py) for batch DataFrame language categorization.
- [`src/ner_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/ner_extraction.py):
  - Implements [`GLiNERNER`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/ner_extraction.py) (zero-shot multilingual token extraction) and [`TransformersNER`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/ner_extraction.py).
  - Handles token chunking with character offset tracking, maps raw labels into 8 unified categories (`PER`, `ORG`, `LOC`, `DAT`, `EVE`, `PRO`, `COM`, `MIS`), expands entity word boundaries, and merges adjacent same-type entities with linker words (`of`, `de`, `و`).
- [`src/sentiment_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/sentiment_extraction.py):
  - Implements [`LLMSentiment`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/sentiment_extraction.py) managing fine-tuned BERT models per language.
  - Executes sliding-window chunk inference with majority voting aggregation.
  - Converts model-specific outputs (5-star ratings for CamemBERT, 3-class logits for RoBERTa and Arabic BERT) into unified `POSITIVE`/`NEGATIVE`/`NEUTRAL` labels.
  - Applies post-inference heuristics (question neutralizer, negative domain cue overrides).
- [`src/topic_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/topic_extraction.py):
  - Implements [`TransformerTopic`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/topic_extraction.py) (fine-tuned 18-class transformer classifiers) and [`LLMTopic`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/topic_extraction.py) (zero-shot NLI fallback).
  - Implements chunk-level scoring and vote aggregation with tie-breaking.
  - Maps numerical class indices to standardized multilingual category names.
- [`src/chunking.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/chunking.py):
  - Implements [`token_chunks()`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/chunking.py), splitting long texts into overlapping token windows using Hugging Face tokenizer offset mappings.
- [`src/text_utils.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/text_utils.py):
  - Implements [`init_console_encoding()`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/text_utils.py) for UTF-8 standard streams on Windows.
  - Implements [`format_arabic_for_console()`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/text_utils.py) using `arabic_reshaper` and `python-bidi` for proper RTL rendering in terminals.

---

#### 4. `finetuned_models/` — Local Model Checkpoints

This directory stores locally trained PyTorch / Hugging Face model weights, tokenizer vocabularies, and configuration files:
- `ARABIC SENTIMENT/`: Fine-tuned Arabic BERT 3-class sentiment classifier.
- `ENGLISH SENTIMENT/`: Fine-tuned RoBERTa English 3-class sentiment classifier.
- `FRENSH SENTIMENT/`: Fine-tuned CamemBERT French 3-class sentiment classifier.
- `ARABIC TOPIC/`: Fine-tuned Arabic 18-class topic classifier.
- `ENGLSIH TOPIC/`: Fine-tuned English 18-class topic classifier.
- `FRENSH TOPIC/`: Fine-tuned French 18-class topic classifier.

---

#### 5. `analysis/` — Analytics & Automated SQL Reporting (Stage 4)

Contains analytical SQL aggregation scripts and report generators executed automatically in Stage 4:
- [`analysis/report_generator.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/report_generator.py): Master runner coordinating all sub-analysis modules. Populates analytics tables in MySQL and exports a consolidated summary CSV to [`analysis/reports/analytics_summary.csv`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/reports/analytics_summary.csv).
- [`analysis/topics_by_month.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/topics_by_month.py): Computes monthly topic volume, dominant monthly topics, and percentage share across the corpus into `analytics_topics_by_month`.
- [`analysis/topic_peaks.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/topic_peaks.py): Implements rolling 3-month Z-score anomaly detection to identify statistical surges ($Z > 2.5$) into `analytics_topic_peaks`.
- [`analysis/entities_by_month.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/entities_by_month.py): Aggregates monthly Top-K entities based on mention counts, distinct article coverage, and mean confidence into `analytics_entities_by_month`.
- [`analysis/top_entities_by_country.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/top_entities_by_country.py): Groups Top-K entity occurrences by article country metadata (`id_countries`) into `analytics_top_entities_by_country`.
- `analysis/reports/`: Destination folder storing exported analytics CSV reports.

---

#### 6. `scripts/` — Utility Scripts & Benchmarks

- [`scripts/ressources/master_benchmark.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/scripts/ressources/master_benchmark.py): Standalone evaluation suite measuring per-document latency (ms), throughput (docs/sec), and peak VRAM allocation across CUDA, CPU, and Ollama.
- [`scripts/ressources/score_remarques.md`](file:///c:/Users/bello/Desktop/PROJET_PFE/scripts/ressources/score_remarques.md): Notes and qualitative observations regarding model performance and classification accuracy.

---

### Component Responsibility & Data Flow Matrix

| File / Component | Stage | Primary Input | Primary Output / Target Table | Core Responsibility |
| :--- | :--- | :--- | :--- | :--- |
| [`main.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/main.py) | Master | CLI / User execution | Pipeline execution logs | Orchestrates stages 1–4 sequentially |
| [`pipeline/sampler.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/pipeline/sampler.py) | Stage 1 | MySQL `article` table | `work_df`, `skipped_df`, DB locks | Batch sampling, fastText language detection, task text cleaning |
| [`src/preprocessing/router.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/preprocessing/router.py) | Stage 1 | Raw article strings | Cleaned `text_ner`, `text_sentiment`, `text_topic` | Language- and task-aware text normalization |
| [`pipeline/gpu_pass.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/pipeline/gpu_pass.py) | Stage 2 | `work_df` | `article_entities`, `article_sentiments`, `article_topics`, `benchmark_results` | Sequential GPU batch inference with VRAM management |
| [`src/ner_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/ner_extraction.py) | Stage 2 | `text_ner` | `List[NEREntity]` | Zero-shot GLiNER extraction, label unification, deduplication |
| [`src/sentiment_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/sentiment_extraction.py) | Stage 2 | `text_sentiment` | `SentimentResult` | Chunked BERT inference, majority vote, heuristic overrides |
| [`src/topic_extraction.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/topic_extraction.py) | Stage 2 | `text_topic` | `TopicResult` | 18-class transformer topic classification and voting |
| [`src/qwen/run_qwen_pass.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/qwen/run_qwen_pass.py) | Stage 3 | `work_df` | `qwen_article_sentiments`, `qwen_article_topics`, `qwen_benchmark_results` | Comparative Ollama LLM inference pass |
| [`src/db_config.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/src/db_config.py) | All | DataFrames / Predictions | MySQL Schema & Persistence | Connection pooling, atomic upserts, entity deduplication |
| [`analysis/report_generator.py`](file:///c:/Users/bello/Desktop/PROJET_PFE/analysis/report_generator.py) | Stage 4 | Enriched MySQL tables | 4 Analytics DB tables & `analytics_summary.csv` | Automated SQL analytical aggregation & time-series report creation |

---

## 4. Pipeline Stages

### Stage 1 — Sampling & Language Detection

**File:** `pipeline/sampler.py`

1. **Fetch unprocessed articles** from the `article` table (configurable via `RAW_TABLE`). Only articles without an existing `sentiment_label` are selected (up to `SAMPLE_SIZE`, default 1000).

2. **Language Detection** using `FastTextLanguageDetector` (backed by `fast-langdetect` / fastText `lid.176.bin`):
   - The top-5 language predictions are retrieved.
   - For **Arabic**, confidence scores are *aggregated* across all Arabic dialect codes (e.g., `ar`, `arz`, `ary`, `arq`, etc.) because fastText splits probability among dialects for colloquial text.
   - Articles containing Arabic-script characters are additionally validated by their **Arabic character ratio**.
   - Supported languages: `ar`, `en`, `fr`.
   - Confidence threshold: `LANG_THRESHOLD` (default **0.51**).

3. **Filtering:**
   - Articles that pass the language filter → `work_df`
   - Articles that fail (unsupported language or low confidence) → `skipped_df` (marked as `SKIPPED` in DB)

4. **Task-specific preprocessing** on `work_df`:
   - `text_ner` — cleaned text for NER model input
   - `text_sentiment` — cleaned text for sentiment model input
   - `text_topic` — cleaned text for topic model input
   - Each uses a language-specific `PreprocessRouter` with different cleaning rules.

5. **Pre-insert placeholder rows** in `article_sentiments`, `qwen_article_sentiments`, `article_topics`, and `qwen_article_topics` to mark articles as "in progress."

**Returns:** `(work_df, skipped_df)` DataFrames.

---

### Stage 2 — GPU Inference Pass

**File:** `pipeline/gpu_pass.py`

Runs only when CUDA is available. Models are loaded and run **one at a time** to avoid GPU Out-of-Memory (OOM) errors.

#### NER (Named Entity Recognition)

- Uses **GLiNER** (`urchade/gliner_multi-v2.1`) — a zero-shot multilingual NER model.
- Articles are split into word-based chunks (default 450 words, 100-word overlap).
- Extracted entities are normalized to unified label types:

  | Code | Meaning |
  |------|---------|
  | `PER` | Person |
  | `ORG` | Organization |
  | `LOC` | Location / Place |
  | `DAT` | Date / Time |
  | `EVE` | Event |
  | `PRO` | Product / Brand |
  | `COM` | Competition / League |
  | `MIS` | Miscellaneous |

- Adjacent same-type entities are **merged** if separated by allowed gap characters or multilingual linker words (`of`, `و`, `de`, `et`, `ابن`, etc.).
- Duplicates are removed, keeping the highest-confidence prediction.
- Results written to: `article_entities` and `entities` tables.

#### Sentiment Analysis

- Uses **fine-tuned BERT-based models** stored locally in `finetuned_models/` (one per language).
- Long articles are split into character chunks (`CHUNK_SIZE` chars) with overlap.
- Each chunk is scored independently; the final label is determined by **majority vote** among chunks.
- **Post-heuristics** are applied:
  - Short ambiguous questions → forced `NEUTRAL`
  - `NEUTRAL` predictions containing negative domain cues (e.g., "attack", "crisis", "هجوم") → overridden to `NEGATIVE`
- Output labels: `POSITIVE`, `NEGATIVE`, `NEUTRAL`, or `UNK`
- Results written to: `article_sentiments` table.

#### Topic Classification

- Uses **fine-tuned transformer classifiers** stored locally in `finetuned_models/` (one per language).
- Articles are chunked and each chunk classified independently.
- Final topic chosen by **vote aggregation** (most frequent label, tie-broken by total score).
- Category names are multilingual — configured in `Config.CATEGORY_DISPLAY` with `ar`/`en`/`fr` variants.
- Results written to: `article_topics` table.

---

### Stage 3 — Qwen (LLM) Pass

**Files:** `src/qwen/`

- Calls a locally-running **Ollama** instance serving `qwen2.5:7b`.
- Performs **sentiment** and **topic** classification using structured LLM prompts in the article's language.
- Also handles articles from `skipped_df` (marks them as `SKIPPED`).
- Results written to: `qwen_article_sentiments` and `qwen_article_topics` tables.
- Can be toggled via `RUN_QWEN=True/False`.

---

### Stage 4 — Analytics & Report Generation

**Files:** `analysis/`

Runs automatically after GPU inference. Before generating reports, entity frequency aggregation runs first (`db.update_entity_frequencies()`) to recompute global mention counts in the `entities` table.

Then, analytics are computed and stored directly into dedicated MySQL tables. Optionally, a single consolidated summary CSV file (`analytics_summary.csv`) is exported:

| DB Table | Source Script | Description |
|---|---|---|
| `analytics_topics_by_month` | `topics_by_month.py` | Topic distribution and volume per month |
| `analytics_topic_peaks` | `topic_peaks.py` | Statistical detection of topic surge events (Z-score > 2.5) |
| `analytics_entities_by_month` | `entities_by_month.py` | Top-K entity mention trends over time |
| `analytics_top_entities_by_country` | `top_entities_by_country.py` | Top-K entities per country |

**Optional CSV Export:**
When `EXPORT_ANALYTICS_CSV=true` (or `export_summary_csv=True`), a single unified summary CSV file is saved to `analysis/reports/analytics_summary.csv` synthesizing all analytics tables (KPIs, dominant monthly topics, surges, and top monthly & country entities).

---

## 5. Key Modules

### Configuration (`src/config/`)

The `Config` class merges all sub-configs via Python multiple inheritance:

| Sub-Config | Controls |
|-----------|---------|
| `BaseConfig` | Project root path, `getenv()` helper |
| `DatabaseConfig` | MySQL host, port, credentials, table names |
| `ModelsConfig` | Model names, local paths, numeric model IDs |
| `PipelineConfig` | `RUN_NER`, `RUN_SENTIMENT`, `RUN_TOPIC`, `RUN_QWEN`, `SAMPLE_SIZE`, `LANG_THRESHOLD` |
| `HyperparametersConfig` | Chunk sizes, score thresholds, overlap sizes |
| `HeuristicsConfig` | Negative domain cues, question starter words per language |
| `PreprocessingConfig` | Per-language/task preprocessing presets |
| `QwenConfig` | Ollama URL, model name, timeouts |

All values can be overridden via environment variables (loaded from `.env.local`).

---

### Language Detection (`src/language_detection.py`)

**Class:** `FastTextLanguageDetector`

- Uses `fast_langdetect` with the `lid.176.bin` fastText model (supports 176 languages).
- Top-5 predictions retrieved and Arabic dialect probabilities aggregated to avoid split-confidence issues.
- Arabic character-ratio heuristic prevents misclassification of colloquial Arabic as Persian/Urdu.
- Returns a `LanguageDetection(lang, score, raw_label)` dataclass.

**Class:** `ArticleClassifier` — higher-level wrapper used in batch classification scripts.

---

### Preprocessing (`src/preprocessing/`)

A `PreprocessRouter` dispatches text to language-specific and task-specific preprocessors:

- **Arabic:** Normalizes Unicode, removes tashkeel (diacritics), strips non-Arabic noise.
- **Latin (EN/FR):** Lowercasing, punctuation normalization, noise removal.
- **Task presets:** Separate rules for `lang_detect`, `ner`, `sentiment`, and `topic`.

---

### NER Extraction (`src/ner_extraction.py`)

Two NER implementations sharing the same interface:

| Class | Backend | Notes |
|-------|---------|-------|
| `GLiNERNER` | `urchade/gliner_multi-v2.1` | Zero-shot, multilingual |
| `TransformersNER` | Hugging Face token-classification | Language-specific models |

Both expose `.predict(text, language) -> List[NEREntity]`.

`NEREntity` fields: `text`, `label`, `start`, `end`, `score`

Key processing steps in `TransformersNER`:
1. Split text into overlapping token chunks
2. Run the HF pipeline on each chunk
3. Normalize labels using `LABEL_UNIFICATION` map
4. Filter by score threshold and minimum entity length
5. Optionally expand short entities to word boundaries
6. Deduplicate and merge adjacent same-type entities

---

### Sentiment Extraction (`src/sentiment_extraction.py`)

**Class:** `LLMSentiment`

- Lazy-loads language-specific sentiment pipelines on first use (thread-safe via `Lock`).
- Chunked inference with majority vote aggregation.
- Label normalization handles model-specific output formats:
  - CamemBERT-based FR model: 5-star ratings collapsed to `POSITIVE`/`NEUTRAL`/`NEGATIVE`
  - RoBERTa EN model: direct 3-class output
  - Arabic BERT: direct 3-class output
- Post-rule heuristics for edge cases.

`SentimentResult` fields: `label`, `score`, `probs` (dict of label → probability)

---

### Topic Extraction (`src/topic_extraction.py`)

Two topic classifiers:

| Class | Method | Notes |
|-------|--------|-------|
| `TransformerTopic` | Fine-tuned text classification | Primary, used in production |
| `LLMTopic` | Zero-shot classification with hypothesis templates | Benchmarking / fallback |

Both expose `.predict(text, lang) -> TopicResult`.

`TopicResult` fields: `label` (human-readable category), `score`

---

### Qwen Pass (`src/qwen/`)

| File | Purpose |
|------|---------|
| `run_qwen_pass.py` | Orchestrates the Qwen inference pass |
| `sentiment_extraction.py` | Prompts Qwen for sentiment classification |
| `topic_extraction.py` | Prompts Qwen for topic classification |

Communicates with Ollama REST API. Supports all three languages with language-specific prompts.

---

### Database Layer (`src/db_config.py`)

**Class:** `DatabaseConnection`

- SQLAlchemy + PyMySQL driver connecting to MySQL.
- Connection pooling: `DB_POOL_SIZE=10`, `DB_MAX_OVERFLOW=20`, `DB_POOL_RECYCLE=3600s`.

Key methods:

| Method | Description |
|--------|-------------|
| `init_result_tables()` | Creates output tables if they don't exist |
| `upsert_article_sentiments(...)` | Insert/update BERT sentiment result |
| `upsert_qwen_article_sentiments(...)` | Insert/update Qwen sentiment result |
| `upsert_article_topic(...)` | Insert/update BERT topic result |
| `upsert_qwen_article_topic(...)` | Insert/update Qwen topic result |
| `insert_entities(...)` | Batch-write NER entities |
| `update_entity_frequencies()` | Recompute global entity frequency counts |

---

## 6. Models Used

| Model | Task | Language | Type |
|-------|------|----------|------|
| `urchade/gliner_multi-v2.1` | NER | ar/en/fr | Zero-shot (GLiNER) |
| Fine-tuned (local, `ARABIC SENTIMENT/`) | Sentiment | ar | BERT-based |
| Fine-tuned (local, `ENGLISH SENTIMENT/`) | Sentiment | en | BERT-based |
| Fine-tuned (local, `FRENSH SENTIMENT/`) | Sentiment | fr | CamemBERT-based |
| Fine-tuned (local, `ARABIC TOPIC/`) | Topic | ar | BERT-based |
| Fine-tuned (local, `ENGLSIH TOPIC/`) | Topic | en | BERT-based |
| Fine-tuned (local, `FRENSH TOPIC/`) | Topic | fr | BERT-based |
| `qwen2.5:7b` (Ollama local) | Sentiment + Topic | ar/en/fr | LLM |
| `lid.176.bin` (fastText) | Language Detection | 176 langs | fastText |

Numeric model IDs (used in DB tables):

| ID | Model |
|----|-------|
| 0 | `hatmimoha/arabic-ner` |
| 1 | `CAMeL-Lab/bert-base-arabic-camelbert-msa-ner` |
| 2 | `dslim/bert-base-NER` |
| 3 | `Jean-Baptiste/camembert-ner` |
| 4 | `urchade/gliner_multi-v2.1` |
| 5 | Arabic Sentiment (fine-tuned) |
| 6 | English Sentiment (fine-tuned) |
| 7 | French Sentiment (fine-tuned) |
| 8 | Arabic Topic (fine-tuned) |
| 9 | English Topic (fine-tuned) |
| 10 | French Topic (fine-tuned) |
| 11 | `qwen2.5:7b` |

---

## 7. Database Schema

Output tables written to `nlp_mena` database (configurable via `DB_NAME`):

| Table | Contents |
|-------|----------|
| `article_sentiments` | BERT sentiment predictions per article (label, score, language, model_id) |
| `article_topics` | BERT topic predictions per article (label, confidence, language, model_id) |
| `article_entities` | NER entity mentions per article (entity_id, article_id, model_id) |
| `entities` | Unique entity registry (text, type, frequency count) |
| `qwen_article_sentiments` | Qwen LLM sentiment predictions |
| `qwen_article_topics` | Qwen LLM topic predictions |
| `benchmark_results` | BERT model timing benchmarks |
| `qwen_benchmark_results` | Qwen model timing benchmarks |
| `analytics_topics_by_month` | Monthly topic frequency, share, and total article volume |
| `analytics_topic_peaks` | Statistical anomaly and surge detection for topic trends |
| `analytics_entities_by_month` | Top-K dominant entities per month |
| `analytics_top_entities_by_country` | Top-K dominant entities broken down per country |

Articles are linked by `article_id` (foreign key to the `article` table).

---

## 8. Configuration Reference

### Pipeline Controls (in code: `src/config/pipeline.py`)

| Setting | Default | Description |
|---|---|---|
| `SAMPLE_SIZE` | `1000` | Max articles to fetch and process per run |
| `RUN_NER` | `True` | Enable Named Entity Recognition extraction |
| `RUN_SENTIMENT` | `True` | Enable sentiment analysis |
| `RUN_TOPIC` | `True` | Enable topic classification |
| `RUN_QWEN` | `False` | Enable Qwen LLM pass |
| `LANG_THRESHOLD` | `0.51` | Minimum fastText confidence threshold |
| `TOPIC_MIN_TEXT_CHARS` | `30` | Minimum character length for topic classification |
| `TOPIC_MIN_TEXT_WORDS` | `5` | Minimum word count for topic classification |
| `EXPORT_ANALYTICS_CSV` | `True` | Optionally export consolidated `analytics_summary.csv` |

### Environment Variables (`.env.local`)

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | MySQL server hostname |
| `DB_PORT` | `3306` | MySQL server port |
| `DB_USER` | `root` | MySQL username |
| `DB_PASSWORD` | _(empty)_ | MySQL password |
| `DB_NAME` | `nlp_mena` | MySQL database name |
| `RAW_TABLE` | `article` | Source articles table name |
| `DB_POOL_SIZE` | `10` | SQLAlchemy connection pool size |

---

### Run Output & Logging

Run logs are printed to the console via Python's standard `logging` module (logger name: `nlp_pipeline`, level: `INFO`).
Each run logs timing benchmarks, model progress, language filtering statistics, and database write summaries.
