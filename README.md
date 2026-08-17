# Multilingual NLP Pipeline for News Analysis

A Projet de Fin d'Etudes (PFE) - automated NLP enrichment pipeline for Arabic, English, and French news articles using fine-tuned BERT-based models and GPU-accelerated inference.

---

## Project Overview

This pipeline reads raw news articles from a MySQL database, detects their language, preprocesses them, and runs three NLP tasks:

| Task | Model Type | Output |
|------|-----------|--------|
| Sentiment Analysis | Fine-tuned BERT (per-language) | POSITIVE / NEGATIVE / NEUTRAL |
| Topic Classification | Fine-tuned BERT (per-language) | 18 categories (Politics, Conflict, Economy...) |
| Named Entity Recognition | GLiNER gliner_multi-v2.1 | Entities + type + confidence |

All results are written back to MySQL. A second optional pass using Qwen 2.5 (via Ollama) runs sentiment and topic inference for cross-model comparison.

---

## Architecture Overview

`
main.py
  |
  |-- STAGE 1 - pipeline/sampler.py
  |     |-- Fetch N unprocessed articles from article table
  |     |     (WHERE sentiment_label IS NULL in articles_enriched)
  |     |-- Language detection via fast-langdetect (fastText)
  |     |-- Filter: supported langs [ar, en, fr] + score >= 0.51
  |     |     \-- Unsupported / low-score -> SKIPPED (language = NULL)
  |     |-- Preprocess text for NER / sentiment / topic
  |     \-- Empty-after-preprocess -> also SKIPPED (language = NULL)
  |
  |-- STAGE 2 - pipeline/gpu_pass.py  (model-by-model to prevent OOM)
  |     |-- NER  -> GLiNER (multilingual)
  |     |-- Sentiment -> Fine-tuned BERT [ar / en / fr]
  |     |-- Topic     -> Fine-tuned BERT [ar / en / fr]
  |     \-- Writes to DB: articles_enriched, article_entities,
  |                        article_topics, benchmark_results
  |
  |-- STAGE 3 - Qwen pass  (optional, RUN_QWEN=True)
  |     \-- Ollama HTTP -> qwen_articles_enriched, qwen_article_topics
  |
  \-- STAGE 4 - analysis/report_generator.py  (optional)
        \-- CSV reports: entity trends, topic peaks, sentiment comparison
`

---

## Fine-Tuned Models

All fine-tuned models live under finetuned_models/ and are loaded via HuggingFace transformers:

| Folder | Task | Language | Architecture |
|--------|------|----------|-------------|
| ARABIC SENTIMENT | Sentiment | Arabic | BERT (3-class) |
| ENGLISH SENTIMENT | Sentiment | English | BERT (3-class) |
| FRENSH SENTIMENT | Sentiment | French | BERT (3-class) |
| ARABIC TOPIC | Topic | Arabic | BERT (18-class) |
| ENGLSIH TOPIC | Topic | English | BERT (18-class) |
| FRENSH TOPIC | Topic | French | BERT (18-class) |

### Topic Categories (18)
Politics, Conflict, Economy, Diplomacy, Security, Elections, Religion, Sports, Health, Technology, Energy, Environment, Migration, Culture, Weather, Infrastructure, Justice, General

### Sentiment Labels
- POSITIVE - Favourable / optimistic tone
- NEGATIVE - Critical / alarming tone
- NEUTRAL  - Informational / balanced tone
- SKIPPED  - Filtered before inference (unsupported language, low confidence, or empty after preprocessing). language column is NULL.

---

## Database Schema

### articles_enriched - Central results table
| Column | Type | Description |
|--------|------|-------------|
| article_id | BIGINT PK | Links to raw article.id |
| language | VARCHAR(10) | ar / en / fr / NULL (SKIPPED) |
| sentiment_label | VARCHAR(10) | POSITIVE / NEGATIVE / NEUTRAL / NULL |
| sentiment_score | FLOAT | Inference confidence score (0.0 to 1.0) |
| gpu_time_ner | BIGINT | NER inference time (ms) |
| gpu_time_sentiment | BIGINT | Sentiment inference time (ms) |
| gpu_time_topic | BIGINT | Topic inference time (ms) |

### benchmark_results - Per-article inference benchmarks (BERT models)
| Column | Type | Description |
|--------|------|-------------|
| article_id | BIGINT | Article reference |
| task | VARCHAR(20) | sentiment / topic / ner |
| model_id | TINYINT | Numeric model ID (see Config.MODEL_ID_MAP) |
| language | VARCHAR(10) | ar / en / fr |
| device | VARCHAR(10) | GPU |
| total_inf_time_sec | DOUBLE | Inference time for this article (seconds) |
| avg_ms_per_doc | DOUBLE | Average ms across articles of that lang in the batch |
| peak_gpu_mb | DOUBLE | Peak GPU memory (if measured) |

Model ID mapping (Config.MODEL_ID_MAP):

| ID | Key | Task |
|----|-----|------|
| 0 | hatmimoha/arabic-ner | NER (Arabic AraBERT) |
| 1 | CAMeL-Lab/bert-base-arabic-camelbert-msa-ner | NER (Arabic CAMeL) |
| 2 | dslim/bert-base-NER | NER (English) |
| 3 | Jean-Baptiste/camembert-ner | NER (French) |
| 4 | urchade/gliner_multi-v2.1 | NER (Multilingual GLiNER) |
| 5 | ar_sentiment_ft | Sentiment (Arabic Fine-tuned) |
| 6 | en_sentiment_ft | Sentiment (English Fine-tuned) |
| 7 | fr_sentiment_ft | Sentiment (French Fine-tuned) |
| 8 | ar_topic_ft | Topic (Arabic Fine-tuned) |
| 9 | en_topic_ft | Topic (English Fine-tuned) |
| 10 | fr_topic_ft | Topic (French Fine-tuned) |
| 11 | qwen2.5:7b | Qwen (Ollama) |

### article_topics - Topic per article
| Column | Type | Description |
|--------|------|-------------|
| article_id | BIGINT PK | Links to article.id |
| topic_label | VARCHAR(100) | One of the 18 categories |
| topic_score | FLOAT | Inference confidence score (0.0 to 1.0) |

### entities - Global entity vocabulary
| Column | Type | Description |
|--------|------|-------------|
| entity_id | BIGINT PK | Auto-increment |
| entity_name | VARCHAR(1024) | Raw extracted text |
| entity_type | VARCHAR(20) | PER / ORG / LOC / UNK |
| normalized_name | VARCHAR(512) | Lowercased + stripped form |
| frequency | INT | Total mentions across all articles |

### article_entities - Article to Entity links
| Column | Type | Description |
|--------|------|-------------|
| article_id | BIGINT | Article reference |
| entity_id | BIGINT | Entity reference |
| model_version | TINYINT | NER model version used |
| confidence_score | FLOAT | GLiNER confidence |

### Qwen parallel tables
- qwen_articles_enriched - mirrors articles_enriched for Qwen results
- qwen_article_topics - mirrors article_topics for Qwen results
- qwen_benchmark_results - mirrors benchmark_results for Qwen

---

## Codebase Structure

`
PROJET_PFE/
|
|-- main.py                        # Entry point - configures and runs the full pipeline
|-- requirements.txt               # Python dependencies
|-- .env.local                     # DB credentials (not committed)
|
|-- finetuned_models/              # Fine-tuned BERT models (loaded by transformers)
|   |-- ARABIC SENTIMENT/
|   |-- ENGLISH SENTIMENT/
|   |-- FRENSH SENTIMENT/
|   |-- ARABIC TOPIC/
|   |-- ENGLSIH TOPIC/
|   \-- FRENSH TOPIC/
|
|-- src/                           # Core logic and utilities
|   |-- config/                    # Centralized modular configuration package
|   |   |-- base.py                # Environment loading, project paths, logging/hardware defaults
|   |   |-- db.py                  # Database connection settings & table definitions
|   |   |-- models.py              # Model paths, registry maps & global MODEL_ID_MAP
|   |   |-- pipeline.py            # Execution switches & quality gates
|   |   |-- hyperparameters.py     # Task-specific hyperparameters
|   |   |-- heuristics.py          # Domain cues, question starters, topic taxonomy
|   |   |-- preprocessing.py       # Arabic & Latin preprocessing presets
|   |   |-- qwen.py                # Ollama & Qwen LLM settings
|   |   \-- __init__.py            # Aggregated Config class & re-exports
|   |-- db_config.py               # SQLAlchemy connection, table creation, all upsert methods
|   |-- ner_extraction.py          # GLiNER + TransformersNER wrappers
|   |-- sentiment_extraction.py    # LLMSentiment - per-language BERT pipeline wrapper
|   |-- topic_extraction.py        # TransformerTopic + LLMTopic wrappers
|   |-- language_detection.py      # fast-langdetect wrapper
|   |-- text_utils.py              # Arabic console reshaping (arabic_reshaper + python-bidi)
|   \-- preprocessing/
|       |-- router.py              # Routes text to Arabic or Latin preprocessor
|       |-- arabic.py              # Arabic normalization (diacritics, ligatures, noise)
|       \-- latin.py              # French/English normalization
|
|-- pipeline/                      # Execution workflow
|   |-- sampler.py                 # Stage 1: fetch, lang-detect, filter, preprocess
|   \-- gpu_pass.py               # Stage 2: NER + Sentiment + Topic GPU inference + DB writes
|
|-- src/qwen/                      # Qwen (Ollama) inference pass
|   |-- run_qwen_pass.py
|   |-- sentiment_extraction.py
|   \-- topic_extraction.py
|
|-- analysis/                      # Reporting and analytics
|   |-- report_generator.py        # Orchestrates all report scripts
|   |-- entities_by_month.py       # Top entities per month
|   |-- top_entities_by_country.py # Top entities per country
|   |-- topics_by_month.py         # Topic distribution over time
|   |-- topic_peaks.py             # Z-score spike detection on topic shares
|   \-- sentiment_comparison.py   # BERT vs Qwen sentiment agreement
|
\-- scripts/                      # Standalone utilities
    \-- ressources/
        \-- master_benchmark.py   # Standalone benchmark runner
`

---

## Technology Stack

| Category | Libraries |
|----------|-----------|
| Deep Learning | PyTorch 2.5+, HuggingFace Transformers 5.x, GLiNER |
| NLP Utilities | fast-langdetect (fastText), arabic-reshaper, python-bidi |
| Data | Pandas 3.x, NumPy |
| Database | MySQL, SQLAlchemy 2.x, PyMySQL |
| Inference server | Ollama (Qwen 2.5:7b, optional) |
| System | Python 3.11, CUDA 12.1 |

---

## Pipeline Flow - Step by Step

1. Sampling (sampler.py): pulls up to sample_size rows from article where articles_enriched.sentiment_label IS NULL (unprocessed). Articles with unsupported language, low confidence, or empty body after preprocessing are immediately written as SKIPPED with language = NULL and will not be re-fetched.

2. GPU Inference (gpu_pass.py): models are loaded and unloaded one at a time (model-by-model strategy) to avoid VRAM OOM. Order: NER then Sentiment then Topic. A 1-second GPU cooldown runs between model groups.

3. DB Writes: after inference, results are upserted into articles_enriched, article_topics, article_entities, and benchmark_results.

4. Qwen Pass (optional, RUN_QWEN = True in main.py): sends the same preprocessed texts to a local Ollama server for independent sentiment + topic prediction stored in qwen_* tables.

5. Analytics (optional, GENERATE_REPORTS = True in main.py): aggregates DB results to produce trend and comparison CSVs in analysis/reports/.

---

## Sentinel Labels

| sentiment_label | language | Meaning |
|-----------------|---------|---------|
| POSITIVE | ar/en/fr | Valid positive prediction |
| NEGATIVE | ar/en/fr | Valid negative prediction |
| NEUTRAL | ar/en/fr | Valid neutral prediction |
| SKIPPED | NULL | Filtered before inference (unsupported lang / empty text) |
| NULL (DB) | ar/en/fr | Placeholder - processing queued but not yet run |

Console alerts: if a model ever produces a label outside [POSITIVE, NEGATIVE, NEUTRAL], a [!] SENTIMENT UNK DETECTED block is printed. Arabic text is reshaped with arabic_reshaper + python-bidi for correct RTL display in the Windows console.

---

## Key Configuration (main.py flags)

`python
RUN_NER           = True    # Enable/disable NER pass
RUN_SENTIMENT     = True    # Enable/disable Sentiment pass
RUN_TOPIC         = True    # Enable/disable Topic pass
RUN_QWEN          = False   # Enable Qwen (Ollama) comparison pass
GENERATE_REPORTS  = False   # Generate analytics CSVs after inference

sample_size       = 500     # Number of articles per run
`

---

## Quick Start

`ash
# 1. Activate the virtual environment
.env\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure DB credentials in .env.local
# DB_HOST=localhost  DB_PORT=3306  DB_USER=root  DB_PASSWORD=...  DB_NAME=webradar_libya_nlp_tmp

# 4. Run the pipeline (adjust sample_size in main.py)
python main.py

# 5. (Optional) Qwen pass - requires Ollama running with qwen2.5:7b
# ollama serve  &&  ollama pull qwen2.5:7b
# Then set RUN_QWEN = True in main.py

# 6. (Optional) Generate analytics reports
# Set GENERATE_REPORTS = True in main.py
`

---

## Qwen (Ollama) - Important Note

Qwen runs via Ollama HTTP (localhost:11434), not inside the Python process. This means:
- peak_gpu_mb will be 0.0 in qwen_benchmark_results - VRAM is managed by the Ollama server, invisible to torch.cuda.memory_allocated().
- load_time_sec is near 0 - Ollama manages model lifecycle; the Python object is a lightweight HTTP client.
- If Ollama is not running, the Qwen pass is skipped with a logged error. The BERT pipeline results are unaffected.

---

## Preprocessing Details

### Arabic (src/preprocessing/arabic.py)
- Unicode normalization and Arabic letter canonicalization (alef variants -> alef, teh marbuta -> ha, alef maqsura -> ya)
- Diacritics (tashkeel) and Tatweel removal
- URL/email/HTTP fragment removal
- Decorative noise removal (box drawing, dingbats, Arabic zero)
- Social spam line detection using Franco-Arabic heuristics
- fix_merged_keywords() splits merged tokens for common keywords (Libya, Tripoli, for sale, etc.)
- Punctuation normalization and de-duplication

### Latin (src/preprocessing/latin.py)
- Typographic quote and dash normalization
- Decorative and unicode junk removal
- Social content placeholders for sentiment (http, @user, email) or full removal for NER/topic
- CamelCase hashtag splitting
- Consistent single-space around internal periods
